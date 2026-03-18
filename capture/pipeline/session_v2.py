"""
Session V2 — 30-min session orchestrator.

Flow:
  1. Initial FOV check (handled by server before session start)
  2. Loop until session time exhausted:
     a. Grab single frame from Orbbec → wrist check
        (skipped for first segment since FOV just passed)
     b. Record 1-min segment (Orbbec bag + Kreo MP4s)
     c. Enqueue segment files for S3 upload
     d. Optionally convert bag → MCAP
  3. Session complete

Outputs per segment:
  orbbec_<session>_seg<N>.bag
  kreo1_<session>_seg<N>.mp4
  kreo2_<session>_seg<N>.mp4
  timestamps_<session>_seg<N>.csv
  (optional) orbbec_<session>_seg<N>.mcap
"""
import os, csv, time, threading, logging, json, select, pty, termios, subprocess
import cv2, numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, List

from capture.config import (
    OUTPUT_DIR, ORBBEC_REC, ORBBEC_LIB,
    KREO1_DEVICE, KREO2_DEVICE, KREO_W, KREO_H,
    FPS, SEGMENT_DURATION, SESSION_DURATION,
    ORBBEC_STREAM, ORBBEC_STREAM_LIB,
)
from capture.cameras.orbbec import OrbbecRecorder
from capture.cameras.kreo   import KreoCamera, probe
from capture.cameras.fov_check import single_frame_check, _last_stop_time

log = logging.getLogger(__name__)

FRAME_GRAB_TIMEOUT = 15   # seconds to wait for a frame from orbbec_stream
DEVICE_RELEASE_S   = 3.0  # wait after stopping a recording before frame grab


def _grab_orbbec_frame(timeout: int = FRAME_GRAB_TIMEOUT) -> Optional[np.ndarray]:
    """
    Briefly launch orbbec_stream, grab a single color frame, kill it.
    Returns the frame or None on failure.
    """
    from capture.cameras.fov_check import _last_stop_time as lst
    since = time.time() - lst
    if since < DEVICE_RELEASE_S:
        wait = DEVICE_RELEASE_S - since
        log.info(f"Waiting {wait:.1f}s for Orbbec to release before frame grab...")
        time.sleep(wait)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ORBBEC_STREAM_LIB
    env["QT_QPA_PLATFORM"] = "offscreen"

    master_fd, slave_fd = pty.openpty()
    try:
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] = attrs[1] & ~termios.OPOST
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    except Exception:
        pass

    try:
        proc = subprocess.Popen(
            [ORBBEC_STREAM],
            stdout=slave_fd, stderr=slave_fd, stdin=slave_fd,
            env=env, close_fds=True)
        os.close(slave_fd)
    except Exception as e:
        try: os.close(master_fd)
        except: pass
        log.error(f"Failed to launch orbbec_stream: {e}")
        return None

    frame = None
    buf   = b""
    deadline = time.time() + timeout

    try:
        while time.time() < deadline:
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

            # Trim buffer
            if len(buf) > 2 * 1024 * 1024:
                last = buf.rfind(b"FRAME ", len(buf) - 512 * 1024)
                buf  = buf[last:] if last > 0 else buf[-1024 * 1024:]

            idx = buf.find(b"FRAME COLOR ")
            if idx == -1:
                continue
            nl = buf.find(b"\n", idx)
            if nl == -1:
                continue

            header = buf[idx:nl].decode("utf-8", errors="ignore").strip()
            try:
                parts     = header.split()
                data_size = int(parts[6])
            except (IndexError, ValueError):
                buf = buf[nl + 1:]
                continue

            frame_end = nl + 1 + data_size
            if frame_end > len(buf):
                continue

            data = buf[nl + 1:frame_end]
            buf  = buf[frame_end:]

            decoded = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                frame = decoded
                break
    finally:
        proc.terminate()
        try: proc.wait(timeout=3)
        except: proc.kill()
        try: os.close(master_fd)
        except: pass
        # Update global last stop time
        import capture.cameras.fov_check as fov_mod
        fov_mod._last_stop_time = time.time()

    return frame


@dataclass
class SegmentInfo:
    index:     int
    status:    str = "pending"  # pending | recording | uploading | complete | failed
    files:     Dict[str, str] = field(default_factory=dict)
    wrist_ok:  Optional[bool] = None
    start_time: Optional[float] = None
    end_time:   Optional[float] = None


class SessionV2:
    """
    30-minute session with sequential 1-min segments.
    """

    def __init__(self,
                 operator_id: str = "",
                 activity_label: str = "",
                 segment_duration: int = SEGMENT_DURATION,
                 session_duration: int = SESSION_DURATION,
                 mcap_enabled: bool = False,
                 on_state_change: Callable = None,
                 on_segment_update: Callable = None,
                 on_frame_check: Callable = None,
                 on_complete: Callable = None,
                 upload_queue = None):

        self.operator_id      = operator_id
        self.activity_label   = activity_label
        self.segment_duration = segment_duration
        self.session_duration = session_duration
        self.mcap_enabled     = mcap_enabled

        self.on_state_change  = on_state_change
        self.on_segment_update = on_segment_update
        self.on_frame_check   = on_frame_check
        self.on_complete      = on_complete
        self.upload_queue     = upload_queue

        self._stop       = threading.Event()
        self._thread     = None
        self.session_id  = None
        self.session_dir = None
        self.segments:   List[SegmentInfo] = []
        self.max_segments = session_duration // segment_duration

    def start(self):
        self._stop.clear()
        self.segments.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(OUTPUT_DIR, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_early(self):
        self._stop.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def get_state(self) -> dict:
        return {
            "session_id":       self.session_id,
            "operator_id":      self.operator_id,
            "activity_label":   self.activity_label,
            "segment_duration": self.segment_duration,
            "session_duration": self.session_duration,
            "mcap_enabled":     self.mcap_enabled,
            "max_segments":     self.max_segments,
            "segments": [
                {
                    "index":    s.index,
                    "status":   s.status,
                    "wrist_ok": s.wrist_ok,
                }
                for s in self.segments
            ],
        }

    def _state(self, status: str, detail: str = "", **extra):
        log.info(f"[session-v2] {status}: {detail}")
        if self.on_state_change:
            self.on_state_change(status, detail, **extra)

    def _run(self):
        sid = self.session_id
        session_start = time.time()
        session_deadline = session_start + self.session_duration

        # Probe cameras once
        kreo1_ok = probe(KREO1_DEVICE)
        kreo2_ok = probe(KREO2_DEVICE)
        log.info(f"Kreo 1: {'OK' if kreo1_ok else 'N/A'} | Kreo 2: {'OK' if kreo2_ok else 'N/A'}")

        if not kreo1_ok and not kreo2_ok:
#       log.warning
            ("No Kreo cameras found — recording Orbbec only")

        self._state("session_active", f"Session {sid} started — {self.max_segments} segments planned")

        seg_idx = 0
        consecutive_wrist_fails = 0
        MAX_CONSECUTIVE_FAILS   = 3

        while not self._stop.is_set() and time.time() < session_deadline and seg_idx < self.max_segments:

            seg = SegmentInfo(index=seg_idx)
            self.segments.append(seg)

            # ── Inter-segment wrist check (skip for first segment) ────
            if seg_idx > 0:
                self._state("frame_check", f"Wrist check before segment {seg_idx + 1}...")
                log.info(f"Grabbing frame for inter-segment wrist check #{seg_idx}")
                frame = _grab_orbbec_frame()

                if frame is not None:
                    result = single_frame_check(frame)
                    seg.wrist_ok = result.passed

                    if self.on_frame_check:
                        self.on_frame_check(seg_idx, result)

                    if not result.passed:
                        consecutive_wrist_fails += 1
                        log.warning(f"Wrist check FAILED ({consecutive_wrist_fails}/{MAX_CONSECUTIVE_FAILS})")
                        self._state("wrist_check_failed",
                                    f"Wrist check failed — adjust position. "
                                    f"Retrying... ({consecutive_wrist_fails}/{MAX_CONSECUTIVE_FAILS})")

                        if consecutive_wrist_fails >= MAX_CONSECUTIVE_FAILS:
                            self._state("error", "Too many consecutive wrist check failures. Session paused.")
                            # Wait for operator to fix or stop
                            self._stop.wait(timeout=60)
                            if self._stop.is_set():
                                break
                            consecutive_wrist_fails = 0

                        # Retry this segment
                        self.segments.pop()
                        continue
                    else:
                        consecutive_wrist_fails = 0
                else:
                    log.warning("Could not grab frame for wrist check, proceeding anyway")
                    seg.wrist_ok = None
            else:
                seg.wrist_ok = True  # Initial FOV already passed

            # ── Check remaining time ──────────────────────────────────
            remaining = session_deadline - time.time()
            if remaining < 10:
                log.info("Less than 10s remaining, ending session")
                self.segments.pop()
                break
            actual_duration = min(self.segment_duration, remaining)

            # ── Record segment ────────────────────────────────────────
            seg.status = "recording"
            seg.start_time = time.time()
            self._notify_segment(seg)
            self._state("recording", f"Segment {seg_idx + 1}/{self.max_segments} — {int(actual_duration)}s",
                        segment_idx=seg_idx)

            files = self._record_segment(
                sid, seg_idx, actual_duration, kreo1_ok, kreo2_ok)

            seg.end_time = time.time()
            seg.files = files
            seg.status = "uploading"
            self._notify_segment(seg)

            # ── Enqueue upload ────────────────────────────────────────
            if self.upload_queue:
                self.upload_queue.enqueue_segment_files(sid, seg_idx, files)

            # ── Optional MCAP conversion ──────────────────────────────
            if self.mcap_enabled:
                bag_path = files.get("bag")
                if bag_path and os.path.exists(bag_path):
                    mcap_out = bag_path.replace(".bag", ".mcap")
                    self._state("converting", f"Converting segment {seg_idx + 1} bag → MCAP...")
                    try:
                        from capture.pipeline.postprocess import convert_bag_to_mcap
                        ok = convert_bag_to_mcap(bag_path, mcap_out)
                        if ok:
                            files["mcap"] = mcap_out
                            if self.upload_queue:
                                fname = os.path.basename(mcap_out)
                                s3_key = f"captures/{sid}/seg_{seg_idx:03d}/{fname}"
                                self.upload_queue.enqueue(mcap_out, s3_key, seg_idx)
                    except Exception as e:
                        log.error(f"MCAP conversion failed: {e}")

            seg.status = "complete"
            self._notify_segment(seg)

            seg_idx += 1

        # ── Session complete ──────────────────────────────────────────
        elapsed = time.time() - session_start
        n_complete = sum(1 for s in self.segments if s.status == "complete")

        # Write session manifest
        manifest = {
            "session_id":       sid,
            "operator_id":      self.operator_id,
            "activity_label":   self.activity_label,
            "segments_complete": n_complete,
            "segments_planned":  self.max_segments,
            "duration_actual":  round(elapsed, 1),
            "mcap_enabled":     self.mcap_enabled,
            "segments": [
                {"index": s.index, "status": s.status, "files": s.files, "wrist_ok": s.wrist_ok}
                for s in self.segments
            ],
        }
        manifest_path = os.path.join(self.session_dir, f"manifest_{sid}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self._state("complete",
                    f"Session {sid} complete — {n_complete} segments in {elapsed:.0f}s")
        if self.on_complete:
            self.on_complete(sid, n_complete, manifest)

    def _record_segment(self, session_id: str, seg_idx: int,
                        duration: float, kreo1_ok: bool, kreo2_ok: bool) -> dict:
        """Record one segment — Orbbec + Kreos for `duration` seconds."""
        prefix = f"{self.session_dir}/{session_id}_seg{seg_idx:03d}"
        bag_out   = f"{prefix}_orbbec.bag"
        kreo1_out = f"{prefix}_kreo1.mp4"
        kreo2_out = f"{prefix}_kreo2.mp4"
        ts_csv    = f"{prefix}_timestamps.csv"

        ts_lock = threading.Lock()
        ts_rows = []
        stop_ev = threading.Event()

        def log_ts(cam, ns, idx):
            with ts_lock:
                ts_rows.append([cam, ns, idx])

        # Barrier for sync
        n_cams  = 1 + sum([kreo1_ok, kreo2_ok])
        barrier = threading.Barrier(n_cams)
        t0_ns   = [0]

        def t0_once():
            if t0_ns[0] == 0:
                t0_ns[0] = time.time_ns()

        orbbec = OrbbecRecorder(bag_out, ORBBEC_REC, ORBBEC_LIB)

        kreos = []
        if kreo1_ok:
            k1 = KreoCamera(KREO1_DEVICE, "Kreo1", kreo1_out, KREO_W, KREO_H, FPS)
            k1.log_ts_cb = log_ts
            kreos.append(k1)
        if kreo2_ok:
            k2 = KreoCamera(KREO2_DEVICE, "Kreo2", kreo2_out, KREO_W, KREO_H, FPS)
            k2.log_ts_cb = log_ts
            kreos.append(k2)

        orbbec_started = threading.Event()
        orbbec_ok      = [False]

        def orbbec_thread():
            ok = orbbec.start()
            orbbec_ok[0] = ok
            if ok:
                t0_once()
                barrier.wait()
                orbbec_started.set()
                stop_ev.wait(timeout=duration)
                stop_ev.set()
                orbbec.stop()
            else:
                orbbec_started.set()
                stop_ev.set()

        threading.Thread(target=orbbec_thread, daemon=True).start()

        for k in kreos:
            k.start(barrier=barrier)

        if not orbbec_started.wait(timeout=30):
            log.error("Orbbec did not start for segment")
            stop_ev.set()
            for k in kreos:
                k.stop(); k.join()
            return {"bag": bag_out}

        if not orbbec_ok[0]:
            log.error("Orbbec recorder failed for segment")
            for k in kreos:
                k.stop(); k.join()
            return {"bag": bag_out}

        t0_once()

        # Wait for segment to finish or session stop
        end_time = time.time() + duration + 2
        while not stop_ev.is_set() and not self._stop.is_set() and time.time() < end_time:
            time.sleep(0.5)

        stop_ev.set()
        for k in kreos:
            k.stop()
        for k in kreos:
            k.join()

        # Save timestamps
        with ts_lock:
            rows = list(ts_rows)
        rows.sort(key=lambda r: r[1])
        with open(ts_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["camera", "unix_ns", "offset_ns", "frame_idx"])
            w.writerows([[r[0], r[1], r[1] - t0_ns[0], r[2]] for r in rows])

        files = {
            "bag":        bag_out,
            "timestamps": ts_csv,
        }
        if kreo1_ok:
            files["kreo1"] = kreo1_out
        if kreo2_ok:
            files["kreo2"] = kreo2_out

        log.info(f"Segment {seg_idx} recorded — {len(rows)} timestamp rows")
        return files

    def _notify_segment(self, seg: SegmentInfo):
        if self.on_segment_update:
            self.on_segment_update(seg.index, seg.status, seg.wrist_ok)
