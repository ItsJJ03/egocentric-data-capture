# state.py — shared mutable state + WebSocket broadcast
#
# uploader.py uses:  update(**kwargs)  and  get("key")
# server.py uses:    subscribe() / unsubscribe() / snapshot()
# logging uses:      StateLogHandler / append_log()

import asyncio
import json
import logging
import threading
import time
from copy import deepcopy
from typing import Any, Optional

import config

# ── Internal state dict ───────────────────────────────────────────────────────

_state: dict = {
    "phase":               "idle",   # idle|wrist_check|wrist_wait|recording|done
    "task":                "",
    "session_id":          "",
    "operator":            config.OPERATOR_ID,

    # Wrist check
    "wrist_status":        "idle",   # idle|running|passed|failed
    "wrist_fraction":      0.0,
    "wrist_frames_total":  0,
    "wrist_frames_detected": 0,
    "wrist_optional":      False,

    # Recording
    "chunk_current":       0,
    "chunk_total":         config.TOTAL_CHUNKS,
    "chunk_elapsed_sec":   0.0,
    "chunk_duration_sec":  config.CHUNK_DURATION_SEC,
    "chunks_failed":       0,

    # Upload
    "upload_pending":      0,
    "upload_done":         0,
    "upload_failed":       0,
    "upload_current_file": "",
    "upload_current_pct":  0,

    # Sessions list (for UI)
    "sessions":            [],

    # Log ring-buffer
    "log":                 [],

    "error":               "",
}

_lock:    threading.Lock = threading.Lock()
_clients: set            = set()   # asyncio.Queue instances
_loop:    Optional[asyncio.AbstractEventLoop] = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


# ── Read ──────────────────────────────────────────────────────────────────────

def get(key: str = None) -> Any:
    """get() → full dict copy.  get("key") → single value."""
    with _lock:
        if key is not None:
            return deepcopy(_state.get(key))
        return deepcopy(_state)


def snapshot() -> dict:
    """JSON-serialisable copy with a computed chunk_remaining_sec field."""
    with _lock:
        s = deepcopy(_state)
    elapsed = s.get("chunk_elapsed_sec", 0.0)
    s["chunk_remaining_sec"] = max(0.0, s["chunk_duration_sec"] - elapsed)
    s["server_time"]         = time.time()
    return s


# ── Write ─────────────────────────────────────────────────────────────────────

def update(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)
        msg = json.dumps(_state)
    _broadcast(msg)


def append_log(line: str) -> None:
    with _lock:
        _state["log"].append(line)
        if len(_state["log"]) > 100:
            _state["log"] = _state["log"][-100:]
        msg = json.dumps(_state)
    _broadcast(msg)


# ── WebSocket pub/sub ─────────────────────────────────────────────────────────

def subscribe() -> "asyncio.Queue[str]":
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _clients.add(q)
    return q


def unsubscribe(q: "asyncio.Queue[str]") -> None:
    _clients.discard(q)


def _broadcast(msg: str) -> None:
    if _loop is None or not _clients:
        return
    for q in list(_clients):
        _loop.call_soon_threadsafe(_safe_put, q, msg)


def _safe_put(q: "asyncio.Queue[str]", msg: str) -> None:
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        pass


# ── Log handler that feeds into state ────────────────────────────────────────

class StateLogHandler(logging.Handler):
    """Captures log records into state["log"] for the web dashboard."""
    def emit(self, record: logging.LogRecord):
        try:
            append_log(self.format(record))
        except Exception:
            pass
