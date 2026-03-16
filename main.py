#!/usr/bin/env python3
# main.py — egocentric capture pipeline orchestrator
#
# Flow:
#   1. Select or create session (task name)
#   2. Wrist check (5s, Orbbec color, YOLOv8n-pose)
#      - If session is wrist-check optional: ENTER to skip
#      - If check fails: retry prompt
#      - On first pass: session marked wrist-check optional for future runs
#   3. 30 × 1-min recording loop
#      - Each chunk recorded to NVMe as .bag
#      - Immediately queued for background S3 multipart upload
#      - Local file auto-deleted after verified upload
#   4. Wait for all queued uploads to finish
import sys
import time
import select
import logging
from pathlib import Path

import config
import session_manager as sm
from wrist_check import WristChecker
from recorder import record_chunk
from uploader import UploadQueue

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Session selection ─────────────────────────────────────────────────────────

def select_or_create_session() -> str:
    sessions = sm.list_sessions()

    print("\n" + "─" * 55)
    print("  EGOCENTRIC CAPTURE — Session Select")
    print("─" * 55)

    if sessions:
        for i, name in enumerate(sessions, 1):
            info     = sm.session_info(name)
            optional = "  [wrist ✓ optional]" if info.get("wrist_check_optional") else ""
            chunks   = info.get("total_chunks_recorded", 0)
            print(f"  [{i}]  {name}  ({chunks} chunks recorded){optional}")
    else:
        print("  No existing sessions.")

    print("  [N]  New session")
    print("─" * 55)

    while True:
        try:
            choice = input("  Select [1..N] or N: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

        if choice.upper() == "N":
            task = _prompt_task_name()
            sm.create_session(task)
            return task

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]

        print("  Invalid choice — try again.")


def _prompt_task_name() -> str:
    while True:
        raw = input("  Task name (e.g. pouring, grasping): ").strip().lower()
        task = raw.replace(" ", "_")
        if task:
            return task
        print("  Task name cannot be empty.")


# ── Wrist check ───────────────────────────────────────────────────────────────

def run_wrist_check_with_retry(checker: WristChecker) -> bool:
    """Run wrist check; offer retry on failure. Returns True if passed."""
    while True:
        print("\n[wrist_check] Starting 5-second check — show both wrists to the Orbbec camera...")
        passed, fraction = checker.run()

        if passed:
            print(f"[wrist_check] ✓ PASSED  ({fraction:.0%} of frames had both wrists)")
            return True

        print(f"[wrist_check] ✗ FAILED  ({fraction:.0%} — need ≥{config.WRIST_PASS_FRACTION:.0%})")
        try:
            retry = input("  Retry? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if retry == "n":
            return False


def maybe_skip_wrist_check() -> bool:
    """
    For wrist-check-optional sessions: prompt user to press ENTER to skip.
    Returns True if user skipped, False if they want the check to run.
    Uses a 8-second window; no input → proceed with check.
    """
    print()
    print("[wrist_check] This session is wrist-check optional.")
    print("  Press ENTER within 8 seconds to skip, or wait to run the check...")

    # Non-blocking stdin read with 8s timeout
    rlist, _, _ = select.select([sys.stdin], [], [], 8.0)
    if rlist:
        sys.stdin.readline()
        print("[wrist_check] Skipping wrist check.")
        return True

    print("[wrist_check] No input received — running wrist check.")
    return False


# ── Recording loop ────────────────────────────────────────────────────────────

def recording_loop(task: str, uploader: UploadQueue):
    config.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    failed_chunks = []

    print()
    print("─" * 55)
    log.info(f"Session start: {task} / {session_ts}")
    log.info(f"Total: {config.TOTAL_CHUNKS} chunks × {config.CHUNK_DURATION_SEC}s "
             f"= {config.TOTAL_CHUNKS * config.CHUNK_DURATION_SEC // 60} min")
    log.info(f"Storage: {config.RECORDING_DIR}")
    log.info(f"S3 prefix: s3://{config.S3_BUCKET}/{sm.s3_prefix(task)}/")
    print("─" * 55)

    for chunk_idx in range(config.TOTAL_CHUNKS):
        chunk_name = f"{task}_{session_ts}_chunk{chunk_idx:02d}"
        local_path = config.RECORDING_DIR / f"{chunk_name}.bag"
        s3_key     = f"{sm.s3_prefix(task)}/{chunk_name}.bag"

        elapsed_min = chunk_idx * config.CHUNK_DURATION_SEC // 60
        log.info(
            f"[record] Chunk {chunk_idx + 1:2d}/{config.TOTAL_CHUNKS}  "
            f"(+{elapsed_min}min)  → {chunk_name}"
        )

        bag_path = record_chunk(local_path)

        if bag_path:
            sm.increment_chunks(task)
            uploader.enqueue(bag_path, s3_key)
            log.info(
                f"[record] ✓ Chunk {chunk_idx + 1} ready  "
                f"({bag_path.stat().st_size / 1e6:.0f} MB)  "
                f"upload queue depth: {uploader.pending_count()}"
            )
        else:
            log.error(f"[record] ✗ Chunk {chunk_idx + 1} FAILED — no valid bag produced")
            failed_chunks.append(chunk_idx + 1)

    return failed_chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # ── 1. Session ────────────────────────────────────────
    task = select_or_create_session()
    log.info(f"Session: '{task}'")

    # ── 2. Wrist check ────────────────────────────────────
    checker = WristChecker()

    if sm.is_wrist_check_optional(task):
        skipped = maybe_skip_wrist_check()
        if not skipped:
            passed = run_wrist_check_with_retry(checker)
            if not passed:
                log.error("Wrist check failed. Aborting session.")
                sys.exit(1)
    else:
        passed = run_wrist_check_with_retry(checker)
        if not passed:
            log.error("Wrist check failed. Aborting session.")
            sys.exit(1)
        sm.mark_wrist_check_optional(task)

    # Release model + Orbbec device before recorder takes over
    del checker
    log.info(f"[wrist_check] Device cooldown ({config.DEVICE_COOLDOWN_SEC}s)...")
    time.sleep(config.DEVICE_COOLDOWN_SEC)

    # ── 3. Recording + upload ─────────────────────────────
    uploader     = UploadQueue()
    failed       = recording_loop(task, uploader)

    # ── 4. Drain upload queue ─────────────────────────────
    pending = uploader.pending_count()
    if pending:
        log.info(f"\n[upload] Recording complete. Draining {pending} pending uploads...")
        log.info("[upload] (Safe to leave running — recording is done, uploads continue in background)")
        uploader.wait_all()

    # ── Summary ───────────────────────────────────────────
    print()
    print("─" * 55)
    log.info("SESSION COMPLETE")
    log.info(f"  Task:           {task}")
    log.info(f"  Chunks recorded: {config.TOTAL_CHUNKS - len(failed)}/{config.TOTAL_CHUNKS}")
    if failed:
        log.warning(f"  Failed chunks:  {failed}")
    log.info(f"  S3 location:    s3://{config.S3_BUCKET}/{sm.s3_prefix(task)}/")
    print("─" * 55)

    uploader.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[main] Interrupted by user. Uploads that are in-flight will be abandoned.")
        print("[main] Files on NVMe are safe — re-run to resume uploads.")
        sys.exit(0)
