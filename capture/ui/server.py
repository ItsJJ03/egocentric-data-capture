"""
FastAPI backend V2 — 30-min session orchestrator with modern dashboard.

Endpoints:
  GET  /                → UI dashboard
  POST /fov-check       → start initial FOV check
  GET  /fov-stream      → MJPEG stream of FOV check
  POST /session/start   → start 30-min session (after FOV pass)
  POST /session/stop    → stop session early
  GET  /status          → full state JSON
  GET  /upload-status   → upload queue status
  GET  /history         → today's session history
  POST /settings        → update runtime settings
  GET  /settings        → get current settings
  GET  /frame-check-img → latest frame check image (JPEG)
  WS   /ws              → real-time state updates
"""
import asyncio, base64, cv2, json, logging, threading, time, os, glob
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, date

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

from capture.config import (
    UI_HOST, UI_PORT, FOV_CHECK_SECS, FOV_MIN_DETECTION_FRAMES,
    SEGMENT_DURATION, SESSION_DURATION, MCAP_ENABLED_DEFAULT, OUTPUT_DIR,
)
from capture.cameras.fov_check import FOVChecker, FOVResult
from capture.pipeline.session_v2 import SessionV2
from capture.pipeline.uploader import UploadQueue

log = logging.getLogger(__name__)
app = FastAPI()

# ── Runtime settings (mutable from UI) ────────────────────────────
settings = {
    "segment_duration": SEGMENT_DURATION,
    "session_duration": SESSION_DURATION,
    "mcap_enabled":     MCAP_ENABLED_DEFAULT,
    "operator_id":      "",
    "activity_label":   "",
}

# ── Global state ──────────────────────────────────────────────────
state = {
    "status":          "idle",
    "message":         "Ready — run FOV check to begin",
    "fov_result":      None,
    "session_id":      None,
    "current_segment": -1,
    "max_segments":    SESSION_DURATION // SEGMENT_DURATION,
    "segments":        [],
    "progress":        0,
    "frame_check":     None,   # latest frame check result
    "detection_method": None,
}

_fov_checker:      Optional[FOVChecker]   = None
_session:          Optional[SessionV2]    = None
_fov_frame:        Optional[np.ndarray]   = None
_frame_check_img:  Optional[np.ndarray]   = None
_fov_lock          = threading.Lock()
_fc_lock           = threading.Lock()
_ws_clients        = set()
_ws_lock           = threading.Lock()

_upload_queue = UploadQueue()

# ── Session history ───────────────────────────────────────────────
_session_history = []


def _set_state(**kwargs):
    state.update(kwargs)
    _broadcast()


def _broadcast():
    """Push state + upload status to all WebSocket clients."""
    data = state.copy()
    data["upload_status"] = _upload_queue.get_status()
    data["settings"]      = settings.copy()

    msg = json.dumps(data, default=str)
    with _ws_lock:
        dead = set()
        for ws in _ws_clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)
            except:
                dead.add(ws)
        _ws_clients.difference_update(dead)


_loop: asyncio.AbstractEventLoop = None


# ── WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _loop
    _loop = asyncio.get_event_loop()
    await ws.accept()
    with _ws_lock:
        _ws_clients.add(ws)
    try:
        # Send initial state
        data = state.copy()
        data["upload_status"] = _upload_queue.get_status()
        data["settings"]      = settings.copy()
        await ws.send_text(json.dumps(data, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        with _ws_lock:
            _ws_clients.discard(ws)


# ── FOV Check ─────────────────────────────────────────────────────
@app.post("/fov-check")
def start_fov_check():
    global _fov_checker
    if state["status"] in ("recording", "converting", "session_active", "frame_check"):
        return JSONResponse({"error": "Cannot run FOV check during active session"}, 400)

    _set_state(status="fov_checking", message="Running FOV check...", fov_result=None)

    def _run():
        global _fov_checker, _fov_frame

        def _frame_cb(frame, detected):
            global _fov_frame
            with _fov_lock:
                _fov_frame = frame.copy()

        _fov_checker = FOVChecker(
            duration_sec=FOV_CHECK_SECS,
            min_detection_frames=FOV_MIN_DETECTION_FRAMES,
            frame_cb=_frame_cb)

        result = _fov_checker.run()

        if result.passed:
            _set_state(status="fov_passed", message=result.message,
                       fov_result=result.message, detection_method=result.method)
        else:
            _set_state(status="fov_failed", message=result.message,
                       fov_result=result.message, detection_method=result.method)

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


@app.get("/fov-stream")
def fov_stream():
    """MJPEG stream of FOV check frames."""
    def _gen():
        while state["status"] == "fov_checking":
            with _fov_lock:
                frame = _fov_frame
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
            time.sleep(1 / 15)
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Session Control ───────────────────────────────────────────────
@app.post("/session/start")
def start_session():
    global _session
    if state["status"] != "fov_passed":
        return JSONResponse({"error": "FOV check must pass first"}, 400)
    if _session and _session.is_running():
        return JSONResponse({"error": "Session already running"}, 400)

    if not settings["operator_id"].strip():
        return JSONResponse({"error": "Operator ID required"}, 400)

    # Start upload queue
    _upload_queue.on_status_change = lambda s: _broadcast()
    _upload_queue.start()

    def _on_state(status, detail, **extra):
        seg_idx = extra.get("segment_idx", state.get("current_segment", -1))
        _set_state(status=status, message=detail, current_segment=seg_idx)

    def _on_segment_update(seg_idx, seg_status, wrist_ok):
        segs = state.get("segments", [])
        # Update or append
        found = False
        for s in segs:
            if s["index"] == seg_idx:
                s["status"] = seg_status
                s["wrist_ok"] = wrist_ok
                found = True
                break
        if not found:
            segs.append({"index": seg_idx, "status": seg_status, "wrist_ok": wrist_ok})
        _set_state(segments=segs, current_segment=seg_idx)

    def _on_frame_check(seg_idx, result):
        global _frame_check_img
        with _fc_lock:
            _frame_check_img = result.annotated.copy() if result.annotated is not None else None
        _set_state(frame_check={
            "segment": seg_idx,
            "passed":  result.passed,
            "message": result.message,
            "method":  result.method,
        })

    def _on_complete(session_id, n_segments, manifest):
        _session_history.append({
            "session_id":  session_id,
            "operator_id": settings["operator_id"],
            "activity":    settings["activity_label"],
            "segments":    n_segments,
            "timestamp":   datetime.now().isoformat(),
            "mcap":        settings["mcap_enabled"],
        })
        _set_state(status="complete",
                   message=f"Session {session_id} — {n_segments} segments complete")

    max_segs = settings["session_duration"] // settings["segment_duration"]
    _set_state(
        segments=[], current_segment=-1, max_segments=max_segs,
        frame_check=None,
    )

    _session = SessionV2(
        operator_id=settings["operator_id"],
        activity_label=settings["activity_label"],
        segment_duration=settings["segment_duration"],
        session_duration=settings["session_duration"],
        mcap_enabled=settings["mcap_enabled"],
        on_state_change=_on_state,
        on_segment_update=_on_segment_update,
        on_frame_check=_on_frame_check,
        on_complete=_on_complete,
        upload_queue=_upload_queue,
    )
    _session.start()
    return {"ok": True, "session_id": _session.session_id}


@app.post("/session/stop")
def stop_session():
    if _session and _session.is_running():
        _session.stop_early()
        return {"ok": True}
    return JSONResponse({"error": "No active session"}, 400)


# ── Frame check image ────────────────────────────────────────────
@app.get("/frame-check-img")
def frame_check_image():
    """Return latest frame check annotated image as JPEG."""
    with _fc_lock:
        img = _frame_check_img
    if img is None:
        return JSONResponse({"error": "No frame check image"}, 404)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(
        iter([buf.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


# ── Settings ──────────────────────────────────────────────────────
@app.get("/settings")
def get_settings():
    return settings


@app.post("/settings")
async def update_settings(request: Request):
    try:
        req = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, 400)

    for key in ("segment_duration", "session_duration", "mcap_enabled",
                "operator_id", "activity_label"):
        if key in req:
            settings[key] = req[key]

    # Update max_segments in state
    state["max_segments"] = settings["session_duration"] // settings["segment_duration"]
    _broadcast()
    log.info(f"Settings updated: {settings}")
    return {"ok": True, "settings": settings}


# ── Upload Status ─────────────────────────────────────────────────
@app.get("/upload-status")
def upload_status():
    return _upload_queue.get_status()


# ── Session History ───────────────────────────────────────────────
@app.get("/history")
def get_history():
    """Return today's sessions + any persisted from disk."""
    today = date.today().isoformat()
    # Include in-memory history
    today_sessions = [h for h in _session_history
                      if h["timestamp"].startswith(today)]

    # Also scan disk for manifests
    if os.path.exists(OUTPUT_DIR):
        for d in sorted(glob.glob(os.path.join(OUTPUT_DIR, "session_*")), reverse=True):
            manifests = glob.glob(os.path.join(d, "manifest_*.json"))
            for m in manifests:
                try:
                    with open(m) as f:
                        data = json.load(f)
                    sid = data.get("session_id", "")
                    # Check if already in memory
                    if not any(h["session_id"] == sid for h in today_sessions):
                        today_sessions.append({
                            "session_id":  sid,
                            "operator_id": data.get("operator_id", ""),
                            "activity":    data.get("activity_label", ""),
                            "segments":    data.get("segments_complete", 0),
                            "timestamp":   sid,  # approximate
                            "mcap":        data.get("mcap_enabled", False),
                        })
                except Exception:
                    pass

    return {"sessions": today_sessions[:20]}  # limit to 20


# ── Status ────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    data = state.copy()
    data["upload_status"] = _upload_queue.get_status()
    data["settings"]      = settings.copy()
    return data


# ── GPIO bridge ───────────────────────────────────────────────────
@app.post("/gpio/fov")
def gpio_fov():
    return start_fov_check()

@app.post("/gpio/start")
def gpio_start():
    return start_session()


# ── Serve UI ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    return HTMLResponse(ui_path.read_text())


# ── Startup / Shutdown ────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown_event():
    _upload_queue.stop()


def run():
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="warning")
