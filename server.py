#!/usr/bin/env python3
"""
server.py — Web dashboard + egocentric capture pipeline
Run: python3 server.py
Dashboard: http://<pi-ip>:8000
"""
import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import config
import session_manager as sm
import state as S
from recorder import record_chunk
from uploader import UploadQueue
from wrist_check import WristChecker

# ── Logging setup ─────────────────────────────────────────────────────────────

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
_sh  = logging.StreamHandler()
_sh.setFormatter(_fmt)
_wh  = S.StateLogHandler()
_wh.setFormatter(_fmt)

root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
root_log.addHandler(_sh)
root_log.addHandler(_wh)

log = logging.getLogger(__name__)

# ── Pipeline command queue (thread → pipeline thread) ─────────────────────────
_cmd: queue.Queue = queue.Queue()
_uploader: Optional[UploadQueue] = None


# ── Pipeline thread ───────────────────────────────────────────────────────────

def _drain_cmd() -> Optional[dict]:
    """Non-blocking: return next cmd or None."""
    try:
        return _cmd.get_nowait()
    except queue.Empty:
        return None


def _run_wrist(checker: WristChecker) -> tuple[bool, float]:
    """Run wrist check and publish progress to state."""
    S.update(wrist_status="running")

    def _cb(frames_total: int, frames_detected: int, fraction: float):
        S.update(
            wrist_frames_total=frames_total,
            wrist_frames_detected=frames_detected,
            wrist_fraction=round(fraction, 3),
        )

    passed, fraction = checker.run(progress_cb=_cb)
    S.update(wrist_status="passed" if passed else "failed", wrist_fraction=round(fraction, 3))
    return passed, fraction


def _pipeline():
    global _uploader
    _uploader = UploadQueue()

    while True:
        # ── IDLE ──────────────────────────────────────────────────────────
        S.update(
            phase="idle",
            task="", session_id="",
            wrist_status="idle", wrist_fraction=0.0,
            chunk_current=0, chunk_elapsed_sec=0.0, chunks_failed=0,
            sessions=sm.list_sessions(),
            error="",
        )

        cmd = _cmd.get()   # blocks until a command arrives
        if cmd["cmd"] != "start":
            continue

        task        = cmd["task"]
        session_id  = time.strftime("%Y%m%d_%H%M%S")
        wrist_opt   = sm.is_wrist_check_optional(task)

        sm.create_session(task)
        S.update(
            phase="wrist_check",
            task=task, session_id=session_id,
            wrist_optional=wrist_opt,
            chunk_current=0,
            chunk_total=config.TOTAL_CHUNKS,
            chunks_failed=0,
            upload_done=0, upload_pending=0,
        )
        log.info(f"[pipeline] Session: {task} / {session_id}  wrist_optional={wrist_opt}")

        # ── WRIST CHECK ───────────────────────────────────────────────────
        checker = WristChecker()
        passed  = False

        # Handle skip command that arrived before we even started
        preload = _drain_cmd()
        if preload and preload["cmd"] == "skip_wrist":
            log.info("[pipeline] Wrist check pre-skipped via UI")
            S.update(wrist_status="passed")
            passed = True
        elif preload and preload["cmd"] == "stop":
            continue
        else:
            passed, _ = _run_wrist(checker)
            if passed:
                sm.mark_wrist_check_optional(task)
                time.sleep(config.DEVICE_COOLDOWN_SEC)

        # If failed, wait for skip / retry / stop
        while not passed:
            S.update(phase="wrist_wait")
            log.info("[pipeline] Wrist check failed — waiting for retry or skip")
            cmd = _cmd.get()
            if cmd["cmd"] == "stop":
                break
            elif cmd["cmd"] == "skip_wrist":
                S.update(wrist_status="passed")
                passed = True
                break
            elif cmd["cmd"] == "retry_wrist":
                S.update(phase="wrist_check")
                passed, _ = _run_wrist(checker)
                if passed:
                    sm.mark_wrist_check_optional(task)
                    time.sleep(config.DEVICE_COOLDOWN_SEC)

        if not passed:
            log.info("[pipeline] Session aborted at wrist check")
            continue

        # ── RECORDING LOOP ────────────────────────────────────────────────
        config.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        S.update(phase="recording")
        prefix       = f"{task}_{session_id}"
        failed_count = 0

        log.info(f"[pipeline] Starting recording: {config.TOTAL_CHUNKS} × {config.CHUNK_DURATION_SEC}s")
        log.info(f"[pipeline] S3: s3://{config.S3_BUCKET}/{sm.s3_prefix(task)}/")

        for idx in range(config.TOTAL_CHUNKS):
            # Check for stop
            c = _drain_cmd()
            if c and c["cmd"] == "stop":
                log.info("[pipeline] Stop requested — ending session early")
                break

            chunk_name = f"{prefix}_chunk{idx:02d}"
            local_path = config.RECORDING_DIR / f"{chunk_name}.bag"
            S.update(chunk_current=idx + 1, chunk_elapsed_sec=0.0)
            log.info(f"[pipeline] Chunk {idx+1}/{config.TOTAL_CHUNKS} → {chunk_name}")

            # Record in sub-thread so we can update elapsed time and check for stop
            result: list = []
            t = threading.Thread(target=lambda: result.append(record_chunk(local_path)), daemon=True)
            t.start()
            t_start = time.time()
            stop_mid_chunk = False
            while t.is_alive():
                S.update(chunk_elapsed_sec=round(time.time() - t_start, 1))
                c = _drain_cmd()
                if c and c["cmd"] == "stop":
                    log.info("[pipeline] Stop requested mid-chunk — will stop after this chunk")
                    stop_mid_chunk = True
                    break
                time.sleep(0.5)
            t.join()

            bag = result[0] if result else None
            if bag:
                s3_key = f"{sm.s3_prefix(task)}/{bag.name}"
                _uploader.enqueue(bag, s3_key)
                sm.increment_chunks(task)
            else:
                failed_count += 1
                S.update(chunks_failed=failed_count)
                log.error(f"[pipeline] ✗ Chunk {idx+1} failed ({failed_count} total)")

            if stop_mid_chunk:
                log.info("[pipeline] Stopping after chunk as requested")
                break

        # ── DONE ──────────────────────────────────────────────────────────
        S.update(phase="done")
        log.info(f"[pipeline] Session complete. Chunks: {config.TOTAL_CHUNKS - failed_count} ok, {failed_count} failed")
        time.sleep(3)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    S.set_loop(asyncio.get_event_loop())
    threading.Thread(target=_pipeline, daemon=True, name="pipeline").start()
    yield


app = FastAPI(lifespan=_lifespan)

_STATIC = Path(__file__).parent / "static" / "index.html"


@app.get("/")
async def index():
    return HTMLResponse(_STATIC.read_text())


@app.get("/api/state")
async def api_state():
    return S.snapshot()


@app.post("/api/session/start")
async def api_start(body: dict):
    task = (body.get("task") or "").strip()
    if not task:
        return JSONResponse({"error": "task name required"}, status_code=400)
    _cmd.put({"cmd": "start", "task": task})
    return {"ok": True}


@app.post("/api/session/stop")
async def api_stop():
    _cmd.put({"cmd": "stop"})
    return {"ok": True}


@app.post("/api/wrist/skip")
async def api_wrist_skip():
    _cmd.put({"cmd": "skip_wrist"})
    return {"ok": True}


@app.post("/api/wrist/retry")
async def api_wrist_retry():
    _cmd.put({"cmd": "retry_wrist"})
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = S.subscribe()
    try:
        await ws.send_text(json.dumps(S.snapshot()))
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                msg = json.dumps({**S.snapshot(), "_hb": True})
            await ws.send_text(msg)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        S.unsubscribe(q)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
