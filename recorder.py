# recorder.py — 1-minute bag chunk recorder via ob_device_record_nogui + PTY
#
# Uses the proven PTY/termios approach from the existing project:
#   - Clears only OPOST flag (not full tty.setraw) to preserve libusb TTY context
#   - Sends 'q' for clean bag finalisation
#   - Device lock prevents concurrent Orbbec access
import os
import pty
import select
import subprocess
import termios
import time
import logging
import threading
from pathlib import Path
import config

log = logging.getLogger(__name__)

# Shared lock — wrist_check.py also imports this to coordinate device access
device_lock = threading.Lock()


def record_chunk(output_path: Path, duration_sec: int = config.CHUNK_DURATION_SEC) -> Path | None:
    """
    Record one chunk to output_path using ob_device_record_nogui.

    Blocks for duration_sec, then sends 'q' for a clean stop.
    Acquires device_lock for the full duration.

    Returns the actual .bag Path on success, None on failure.
    """
    with device_lock:
        master_fd, slave_fd = pty.openpty()

        # ── Selective termios: clear only OPOST ──────────────────────
        # Full tty.setraw() breaks libusb's TTY context on Pi 5.
        # Clearing just OPOST prevents output-processing corruption
        # while keeping the context intact for libusb device enumeration.
        attrs      = termios.tcgetattr(master_fd)
        attrs[1]  &= ~termios.OPOST
        termios.tcsetattr(master_fd, termios.TCSANOW, attrs)

        proc = subprocess.Popen(
            [config.OB_RECORD_BIN, str(output_path)],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

        log.debug(f"[recorder] ob_device_record_nogui pid={proc.pid}")

        # ── Fast-exit detection: if binary crashes immediately, bail ──
        time.sleep(0.5)
        if proc.poll() is not None:
            log.error(f"[recorder] ob_device_record_nogui exited immediately (rc={proc.returncode})")
            _drain(master_fd, timeout=1.0)
            _close_fd(master_fd)
            return None

        # ── Drain startup output ──────────────────────────────────────
        _drain(master_fd, timeout=config.OB_STARTUP_DRAIN_SEC)
        log.info(f"[recorder] Recording {duration_sec}s chunk → {output_path.name}")

        # ── Record for chunk duration ─────────────────────────────────
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            remaining = deadline - time.time()
            # Drain output while waiting to prevent master_fd buffer filling
            _drain(master_fd, timeout=min(1.0, remaining))

        # ── Clean stop: send 'q' ──────────────────────────────────────
        try:
            os.write(master_fd, b'q')
            log.debug("[recorder] Sent 'q' to ob_device_record_nogui")
        except OSError as e:
            log.warning(f"[recorder] Could not send 'q': {e}")

        # ── Drain exit output and wait for process ────────────────────
        _drain(master_fd, timeout=config.OB_STOP_TIMEOUT_SEC)
        try:
            proc.wait(timeout=config.OB_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            log.warning("[recorder] Process did not exit cleanly — killing")
            proc.kill()
            proc.wait()

        _close_fd(master_fd)

        # ── Resolve actual bag file path ──────────────────────────────
        # ob_device_record_nogui may or may not append .bag depending on version
        candidates = [
            output_path,
            output_path.with_suffix(".bag"),
            Path(str(output_path) + ".bag"),
        ]
        for p in candidates:
            if p.exists() and p.stat().st_size >= config.S3_MIN_VALID_SIZE:
                log.info(f"[recorder] ✓ Bag saved: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
                return p

        log.error(f"[recorder] ✗ No valid bag found at {output_path} (checked {candidates})")
        return None


def _drain(fd: int, timeout: float):
    """Drain readable data from fd until timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            r, _, _ = select.select([fd], [], [], min(0.1, remaining))
        except (ValueError, OSError):
            return
        if r:
            try:
                os.read(fd, 65536)
            except OSError:
                return


def _close_fd(fd: int):
    try:
        os.close(fd)
    except OSError:
        pass
