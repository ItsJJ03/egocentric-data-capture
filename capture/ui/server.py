"""
FastAPI backend — serves web UI and handles capture control.

Endpoints:
  GET  /              → UI
  POST /fov-check     → start 5s FOV check
  GET  /fov-stream    → MJPEG stream of FOV check frames
  POST /start         → start 60s recording (only if FOV passed)
  POST /stop          → stop recording early
  GET  /status        → current state JSON
  WS   /ws            → real-time state updates to UI
"""
import asyncio, base64, cv2, logging, threading, time
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

from capture.config import UI_HOST, UI_PORT, FOV_CHECK_SECS, FOV_MIN_DETECTION_FRAMES
from capture.cameras.fov_check import FOVChecker, FOVResult
from capture.pipeline.session  import CaptureSession

log = logging.getLogger(__name__)
app = FastAPI()

# ── Global state ──────────────────────────────────────────────────
state = {
    "status":       "idle",       # idle | fov_checking | fov_passed | fov_failed | recording | converting | complete | error
    "message":      "Ready",
    "fov_result":   None,         # FOVResult
    "session_id":   None,
    "files":        {},
    "progress":     0,            # 0-100
}

_fov_checker:  Optional[FOVChecker]      = None
_session:      Optional[CaptureSession]  = None
_fov_frame:    Optional[np.ndarray]      = None
_fov_lock      = threading.Lock()
_ws_clients    = set()
_ws_lock       = threading.Lock()


def _set_state(**kwargs):
    state.update(kwargs)
    _broadcast(state.copy())


def _broadcast(data: dict):
    """Push state to all connected WebSocket clients."""
    import json
    msg = json.dumps({k: str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v
                      for k, v in data.items()})
    with _ws_lock:
        dead = set()
        for ws in _ws_clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)
            except: dead.add(ws)
        _ws_clients.difference_update(dead)


_loop: asyncio.AbstractEventLoop = None


# ── WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _loop
    _loop = asyncio.get_event_loop()
    await ws.accept()
    with _ws_lock: _ws_clients.add(ws)
    try:
        await ws.send_text(__import__('json').dumps(state))
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        with _ws_lock: _ws_clients.discard(ws)


# ── FOV check ─────────────────────────────────────────────────────
@app.post("/fov-check")
def start_fov_check():
    global _fov_checker
    if state["status"] in ("recording", "converting"):
        return JSONResponse({"error": "Cannot run FOV check during recording"}, 400)

    _set_state(status="fov_checking", message="Running 5s FOV check...", fov_result=None)

    def _run():
        global _fov_checker, _fov_frame
        def _frame_cb(frame, detected):
            global _fov_frame
            with _fov_lock: _fov_frame = frame.copy()

        _fov_checker = FOVChecker(
            duration_sec=FOV_CHECK_SECS,
            min_detection_frames=FOV_MIN_DETECTION_FRAMES,
            frame_cb=_frame_cb)

        result = _fov_checker.run()

        if result.passed:
            _set_state(status="fov_passed", message=result.message,
                       fov_result=result.message)
        else:
            _set_state(status="fov_failed", message=result.message,
                       fov_result=result.message)

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
            time.sleep(1/15)
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Recording ─────────────────────────────────────────────────────
@app.post("/start")
def start_recording():
    global _session
    if state["status"] != "fov_passed":
        return JSONResponse({"error": "FOV check must pass before recording"}, 400)
    if state["status"] == "recording":
        return JSONResponse({"error": "Already recording"}, 400)

    def _on_state(s, detail):
        _set_state(status=s, message=detail)

    def _on_complete(session_id, files):
        _set_state(status="complete", message=f"Session {session_id} complete",
                   session_id=session_id, files=files)

    _session = CaptureSession(on_state_change=_on_state, on_complete=_on_complete)
    _session.start()
    return {"ok": True}


@app.post("/stop")
def stop_recording():
    if _session and _session.is_running():
        _session.stop_early()
        return {"ok": True}
    return JSONResponse({"error": "Not recording"}, 400)


# ── Status ────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    return state


# ── GPIO bridge (called by capture_daemon.py) ─────────────────────
@app.post("/gpio/fov")
def gpio_fov():
    return start_fov_check()

@app.post("/gpio/start")
def gpio_start():
    return start_recording()


# ── Serve UI ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    return HTMLResponse(ui_path.read_text())


def run():
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="warning")
