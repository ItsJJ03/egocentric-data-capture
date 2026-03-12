"""
Session orchestrator — coordinates Orbbec + 2x Kreo recording.
Handles barrier sync, timestamps, and post-processing trigger.
"""
import os, csv, time, threading, logging
from datetime import datetime
from pathlib import Path

from capture.config import (
    OUTPUT_DIR, ORBBEC_REC, ORBBEC_LIB,
    KREO1_DEVICE, KREO2_DEVICE, KREO_W, KREO_H,
    FPS, CAPTURE_DURATION
)
from capture.cameras.orbbec import OrbbecRecorder
from capture.cameras.kreo   import KreoCamera, probe
from capture.pipeline.postprocess import convert_bag, make_combined

log = logging.getLogger(__name__)


class CaptureSession:
    """One complete recording session — start → record → postprocess."""

    def __init__(self, on_state_change=None, on_complete=None):
        """
        on_state_change(state: str, detail: str) — UI callback
        on_complete(session_id: str, files: dict) — called when fully done
        States: 'recording' | 'converting' | 'complete' | 'error'
        """
        self.on_state_change = on_state_change
        self.on_complete     = on_complete

        self._stop      = threading.Event()
        self._ts_lock   = threading.Lock()
        self._ts_rows   = []
        self._thread    = None
        self.session_id = None
        self.files      = {}

    def start(self):
        """Launch session in background thread."""
        self._stop.clear()
        self._ts_rows.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_early(self):
        """Request early stop (stop button pressed)."""
        self._stop.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _state(self, state: str, detail: str = ""):
        log.info(f"[session] {state}: {detail}")
        if self.on_state_change:
            self.on_state_change(state, detail)

    def _log_ts(self, cam: str, ns: int, idx: int):
        with self._ts_lock:
            self._ts_rows.append([cam, ns, idx])

    def _run(self):
        s   = self.session_id
        out = OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        bag_out      = f"{out}/orbbec_{s}.bag"
        kreo1_out    = f"{out}/kreo1_{s}.mp4"
        kreo2_out    = f"{out}/kreo2_{s}.mp4"
        obc_color    = f"{out}/orbbec_color_{s}.mp4"
        obc_depth    = f"{out}/orbbec_depth_{s}.mp4"
        combined_out = f"{out}/combined_{s}.mp4"
        ts_csv       = f"{out}/timestamps_{s}.csv"
        meta_out     = f"{out}/session_{s}.txt"

        # ── Probe cameras ─────────────────────────────────────────
        kreo1_ok = probe(KREO1_DEVICE)
        kreo2_ok = probe(KREO2_DEVICE)

        if not kreo1_ok and not kreo2_ok:
            self._state('error', "No Kreo cameras found")
            return

        log.info(f"Kreo 1: {'OK' if kreo1_ok else 'not found'}")
        log.info(f"Kreo 2: {'OK' if kreo2_ok else 'not found'}")

        # ── Barrier — sync all streams to same t0 ─────────────────
        n_cams  = 1 + sum([kreo1_ok, kreo2_ok])  # orbbec always counts
        barrier = threading.Barrier(n_cams)
        t0_ns   = [0]

        def _t0_once():
            if t0_ns[0] == 0: t0_ns[0] = time.time_ns()

        # ── Start cameras ─────────────────────────────────────────
        orbbec = OrbbecRecorder(bag_out, ORBBEC_REC, ORBBEC_LIB)

        kreos = []
        if kreo1_ok:
            k1 = KreoCamera(KREO1_DEVICE, "Kreo1", kreo1_out, KREO_W, KREO_H, FPS)
            k1.log_ts_cb = self._log_ts
            kreos.append(k1)
        if kreo2_ok:
            k2 = KreoCamera(KREO2_DEVICE, "Kreo2", kreo2_out, KREO_W, KREO_H, FPS)
            k2.log_ts_cb = self._log_ts
            kreos.append(k2)

        # Start Orbbec in its own thread (PTY blocks)
        orbbec_started = threading.Event()
        orbbec_ok      = [False]

        def _orbbec_thread():
            ok = orbbec.start()
            orbbec_ok[0] = ok
            if ok:
                _t0_once()
                barrier.wait()
                orbbec_started.set()        # unblock main thread
                self._stop.wait(timeout=CAPTURE_DURATION)
                self._stop.set()
                orbbec.stop()
            else:
                orbbec_started.set()        # unblock on failure too
                self._stop.set()

        threading.Thread(target=_orbbec_thread, daemon=True).start()

        # Start Kreos (they wait at barrier)
        for k in kreos:
            k.start(barrier=barrier)

        # Wait for Orbbec to confirm start before declaring recording
        if not orbbec_started.wait(timeout=30):
            self._state('error', "Orbbec did not start in time")
            self._stop.set()
            for k in kreos: k.stop(); k.join()
            return

        if not orbbec_ok[0]:
            self._state('error', "Orbbec recorder failed to start")
            for k in kreos: k.stop(); k.join()
            return

        _t0_once()
        self._state('recording', f"Session {s} — {CAPTURE_DURATION}s")

        # Wait for natural end or early stop
        self._stop.wait(timeout=CAPTURE_DURATION + 5)
        self._stop.set()

        # Stop Kreos
        for k in kreos: k.stop()
        for k in kreos: k.join()

        # ── Save timestamps ───────────────────────────────────────
        with self._ts_lock: rows = list(self._ts_rows)
        rows.sort(key=lambda r: r[1])
        with open(ts_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["camera", "unix_ns", "offset_ns", "frame_idx"])
            w.writerows([[r[0], r[1], r[1] - t0_ns[0], r[2]] for r in rows])
        log.info(f"Timestamps saved: {len(rows)} rows -> {ts_csv}")

        # ── Post-processing ───────────────────────────────────────
        self._state('converting', "Converting bag → MP4...")

        bag_ok = convert_bag(bag_out, obc_color, obc_depth)

        self._state('converting', "Building combined grid...")
        make_combined(
            orbbec_color = obc_color  if bag_ok    else None,
            orbbec_depth = obc_depth  if bag_ok    else None,
            kreo1        = kreo1_out  if kreo1_ok  else None,
            kreo2        = kreo2_out  if kreo2_ok  else None,
            out_path     = combined_out,
        )

        # ── Meta file ─────────────────────────────────────────────
        self.files = {
            'bag':      bag_out,
            'kreo1':    kreo1_out  if kreo1_ok else None,
            'kreo2':    kreo2_out  if kreo2_ok else None,
            'color':    obc_color  if bag_ok   else None,
            'depth':    obc_depth  if bag_ok   else None,
            'combined': combined_out,
            'timestamps': ts_csv,
        }
        with open(meta_out, 'w') as f:
            for k, v in self.files.items():
                f.write(f"{k}={v or 'skipped'}\n")
            f.write(f"t0_ns={t0_ns[0]}\n")
            f.write(f"duration_sec={CAPTURE_DURATION}\n")

        self._state('complete', f"Session {s} done")
        if self.on_complete:
            self.on_complete(s, self.files)
