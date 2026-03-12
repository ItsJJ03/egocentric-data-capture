"""
FOV Check — pre-capture validation.
Launches orbbec_stream via PTY with OPOST disabled,
detects two distinct skin regions (both wrists must be in FOV).
"""
import cv2, time, logging, threading, os, select, pty, termios
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import subprocess

log = logging.getLogger(__name__)

SKIN_LOWER      = np.array([0,   20,  70], dtype=np.uint8)
SKIN_UPPER      = np.array([20, 255, 255], dtype=np.uint8)
MIN_REGION_PX   = 800    # min pixels per skin region to count as a wrist
MIN_REGIONS     = 2      # need at least 2 distinct regions (both wrists)
ORBBEC_STREAM   = "/mnt/ssd/OrbbecSDK_Pi5/orbbec_stream"
ORBBEC_LIB      = "/opt/OrbbecSDK/lib"
STARTUP_SECS    = 3
DEVICE_RELEASE_S = 2.5   # wait for Orbbec to release between checks

_last_stop_time = 0.0    # module-level, tracks when last orbbec_stream stopped


@dataclass
class FOVResult:
    passed:            bool
    frames_checked:    int
    frames_with_hands: int
    message:           str


class FOVChecker:
    def __init__(self, duration_sec=5, min_detection_frames=10, frame_cb=None):
        self.duration_sec         = duration_sec
        self.min_detection_frames = min_detection_frames
        self.frame_cb             = frame_cb
        self._cancel              = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self, orbbec_rgb_device=None) -> FOVResult:
        global _last_stop_time

        # Wait for device to release if called too quickly
        since_last = time.time() - _last_stop_time
        if since_last < DEVICE_RELEASE_S:
            wait = DEVICE_RELEASE_S - since_last
            log.info(f"Waiting {wait:.1f}s for Orbbec device to release...")
            time.sleep(wait)

        total_secs = self.duration_sec + STARTUP_SECS
        log.info(f"FOV check — {total_secs}s total")

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ORBBEC_LIB
        env["QT_QPA_PLATFORM"] = "offscreen"

        master_fd, slave_fd = pty.openpty()

        # Disable OPOST on slave — stops \n→\r\n corruption of binary JPEG data
        try:
            attrs = termios.tcgetattr(slave_fd)
            attrs[1] = attrs[1] & ~termios.OPOST
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        except Exception as e:
            log.warning(f"Could not set PTY attrs: {e}")

        try:
            proc = subprocess.Popen(
                [ORBBEC_STREAM],
                stdout=slave_fd, stderr=slave_fd, stdin=slave_fd,
                env=env, close_fds=True)
            os.close(slave_fd)
        except Exception as e:
            try: os.close(master_fd)
            except: pass
            return FOVResult(False, 0, 0, f"Failed to launch orbbec_stream: {e}")

        frames_checked    = 0
        frames_with_hands = 0
        t_start           = time.time()
        deadline          = t_start + total_secs
        buf               = b""

        try:
            while time.time() < deadline and not self._cancel.is_set():
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if not r:
                    continue
                try:
                    chunk = os.read(master_fd, 131072)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk

                if len(buf) > 4 * 1024 * 1024:
                    last = buf.rfind(b"FRAME ", len(buf) - 512 * 1024)
                    buf  = buf[last:] if last > 0 else buf[-2 * 1024 * 1024:]

                while True:
                    idx = buf.find(b"FRAME COLOR ")
                    if idx == -1:
                        break
                    nl = buf.find(b"\n", idx)
                    if nl == -1:
                        buf = buf[idx:]; break
                    header = buf[idx:nl].decode("utf-8", errors="ignore").strip()
                    try:
                        parts     = header.split()
                        data_size = int(parts[6])
                    except (IndexError, ValueError):
                        buf = buf[nl + 1:]; continue

                    frame_end = nl + 1 + data_size
                    if frame_end > len(buf):
                        buf = buf[idx:]; break

                    data = buf[nl + 1:frame_end]
                    buf  = buf[frame_end:]

                    frame = cv2.imdecode(
                        np.frombuffer(data, dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    detected, n_regions, vis = self._detect_skin(frame)
                    if detected:
                        frames_with_hands += 1
                    frames_checked += 1

                    if frames_checked % 10 == 1:
                        log.info(f"FOV frame {frames_checked}: regions={n_regions} detected={detected}")

                    elapsed  = time.time() - t_start
                    progress = min(elapsed / total_secs, 1.0)
                    color    = (0, 220, 0) if detected else (0, 0, 220)
                    label    = f"BOTH WRISTS IN FOV" if detected else f"WRISTS NOT IN FOV ({n_regions}/2)"

                    cv2.putText(vis, label, (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
                    cv2.putText(vis, label, (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(vis, f"{frames_with_hands}/{frames_checked} frames",
                                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

                    h, w = vis.shape[:2]
                    cv2.rectangle(vis, (0, h - 10), (int(w * progress), h), color, -1)

                    if self.frame_cb:
                        self.frame_cb(vis, detected)

        finally:
            proc.terminate()
            try: proc.wait(timeout=3)
            except Exception: proc.kill()
            try: os.close(master_fd)
            except: pass
            _last_stop_time = time.time()

        passed  = frames_with_hands >= self.min_detection_frames
        message = (
            f"PASS — both wrists detected in {frames_with_hands}/{frames_checked} frames"
            if passed else
            f"FAIL — wrists only in {frames_with_hands}/{frames_checked} frames "
            f"(need {self.min_detection_frames}). Ensure both wrist cameras are in frame."
        )
        log.info(f"FOV check: {message}")
        return FOVResult(passed, frames_checked, frames_with_hands, message)

    def _detect_skin(self, frame):
        """Returns (detected, n_qualifying_regions, annotated_frame).
        Requires MIN_REGIONS distinct skin blobs — prevents false positives
        from uniform background lighting."""
        vis  = frame.copy()
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        qualifying = [c for c in contours if cv2.contourArea(c) >= MIN_REGION_PX]
        n_regions  = len(qualifying)
        detected   = n_regions >= MIN_REGIONS

        for cnt in qualifying:
            x, y, w, h = cv2.boundingRect(cnt)
            c = (0, 220, 0) if detected else (0, 120, 220)
            cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)

        return detected, n_regions, vis
