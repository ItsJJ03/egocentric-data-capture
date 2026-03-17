# recorder.py — 1-minute bag chunk recorder via ob_device_record_nogui + PTY
#
# The binary is INTERACTIVE:
#   1. Prompts for a filename (waits for "filename" in output)
#   2. We send the bag path + newline
#   3. It confirms recording started (waits for "started" in output)
#   4. We record for chunk duration
#   5. We send "q\n" for clean bag finalisation
#
# Uses LD_LIBRARY_PATH from config.OB_LIB_PATH so the SDK .so files are found.
import os
import pty
import select
import subprocess
import time
import logging
import threading
from pathlib import Path
import config

log = logging.getLogger(__name__)

device_lock = threading.Lock()


def _read_until(fd: int, keyword: str, timeout: float) -> bool:
    buf      = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            r, _, _ = select.select([fd], [], [], min(0.1, remaining))
        except (ValueError, OSError):
            return False
        if r:
            try:
                chunk = os.read(fd, 4096)
                buf  += chunk
                txt   = chunk.decode("utf-8", errors="ignore").strip()
                if txt:
                    log.debug(f"[orbbec] {txt}")
                if keyword.encode() in buf:
                    return True
            except OSError:
                return False
    return False


def _drain(fd: int, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            r, _, _ = select.select([fd], [], [], min(0.1, remaining))
        except (ValueError, OSError):
            return
        if r:
            try:
                chunk = os.read(fd, 65536)
                txt   = chunk.decode("utf-8", errors="ignore").strip()
                if txt:
                    log.debug(f"[orbbec] {txt}")
            except OSError:
                return


def _close_fd(fd: int):
    try:
        os.close(fd)
    except OSError:
        pass


def record_chunk(output_path: Path, duration_sec: int = config.CHUNK_DURATION_SEC) -> Path | None:
    """
    Record one chunk using ob_device_record_nogui's interactive protocol.
    Returns the .bag Path on success, None on failure.
    """
    with device_lock:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        # Add the Orbbec lib path if configured, otherwise inherit from environment
        lib_path = getattr(config, "OB_LIB_PATH", None)
        if lib_path:
            env["LD_LIBRARY_PATH"] = lib_path

        master_fd, slave_fd = pty.openpty()

        proc = subprocess.Popen(
            [config.OB_RECORD_BIN],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, close_fds=True,
        )
        os.close(slave_fd)

        log.debug(f"[recorder] pid={proc.pid}")

        # Wait for filename prompt
        if not _read_until(master_fd, "filename", timeout=15):
            log.error("[recorder] No filename prompt from ob_device_record_nogui")
            proc.kill(); proc.wait(); _close_fd(master_fd)
            return None

        # Send bag path
        os.write(master_fd, f"{str(output_path)}\n".encode())

        # Wait for recording started
        if not _read_until(master_fd, "started", timeout=15):
            log.error("[recorder] Recorder did not confirm start")
            proc.kill(); proc.wait(); _close_fd(master_fd)
            return None

        log.info(f"[recorder] Recording {duration_sec}s → {output_path.name}")

        # Record for chunk duration
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            _drain(master_fd, timeout=min(1.0, deadline - time.time()))

        # Clean stop — try 'q' without newline first (raw keypress), then with newline
        stopped = False
        for stop_seq in [b"q", b"q\n", b"Q\n"]:
            try:
                os.write(master_fd, stop_seq)
            except OSError:
                break
            try:
                proc.wait(timeout=10)
                stopped = True
                log.info(f"[recorder] Clean stop with {stop_seq!r}")
                break
            except subprocess.TimeoutExpired:
                pass

        if not stopped:
            # SIGTERM gives the binary a chance to flush and close the bag
            log.warning("[recorder] q did not stop process — sending SIGTERM")
            import signal as _signal
            try:
                proc.send_signal(_signal.SIGTERM)
                proc.wait(timeout=config.OB_STOP_TIMEOUT_SEC)
                stopped = True
            except subprocess.TimeoutExpired:
                log.warning("[recorder] SIGTERM timeout — killing (bag may be corrupt)")
                proc.kill()
                proc.wait()

        _drain(master_fd, timeout=2.0)
        _close_fd(master_fd)

        # Resolve actual bag file — accept any non-empty file
        for p in [output_path, output_path.with_suffix(".bag"), Path(str(output_path) + ".bag")]:
            if p.exists() and p.stat().st_size >= config.S3_MIN_VALID_SIZE:
                log.info(f"[recorder] ✓ {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
                return p

        # Log what's actually in the recording dir for debugging
        existing = list(config.RECORDING_DIR.glob("*.bag"))
        log.error(f"[recorder] ✗ No valid bag found at {output_path} — dir contains: {[f.name for f in existing]}")
        return None
