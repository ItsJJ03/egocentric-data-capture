# session_manager.py — persist session metadata and wrist-check state
import json
import time
import logging
from pathlib import Path
import config

log = logging.getLogger(__name__)


def _load() -> dict:
    if config.SESSION_REGISTRY.exists():
        try:
            return json.loads(config.SESSION_REGISTRY.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[session] Registry corrupt, resetting: {e}")
    return {}


def _save(data: dict):
    config.SESSION_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via temp file
    tmp = config.SESSION_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(config.SESSION_REGISTRY)


def list_sessions() -> list[str]:
    return list(_load().keys())


def create_session(task_name: str):
    data = _load()
    if task_name not in data:
        data[task_name] = {
            "wrist_check_optional": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_chunks_recorded": 0,
        }
        _save(data)
        log.info(f"[session] Created new session: '{task_name}'")


def mark_wrist_check_optional(task_name: str):
    data = _load()
    if task_name in data:
        data[task_name]["wrist_check_optional"] = True
        _save(data)
        log.info(f"[session] '{task_name}' marked wrist-check optional for future runs")


def is_wrist_check_optional(task_name: str) -> bool:
    return _load().get(task_name, {}).get("wrist_check_optional", False)


def increment_chunks(task_name: str, n: int = 1):
    data = _load()
    if task_name in data:
        data[task_name]["total_chunks_recorded"] = (
            data[task_name].get("total_chunks_recorded", 0) + n
        )
        _save(data)


def s3_prefix(task_name: str) -> str:
    """S3 key prefix matching intern's path: <operator>/<DD-MM-YY>/<task>
    Aligns with the autocapture dashboard bucket layout."""
    date = time.strftime('%d-%m-%y')   # same format as intern's server.js
    return f"{config.OPERATOR_ID}/{date}/{task_name}"


def session_info(task_name: str) -> dict:
    return _load().get(task_name, {})
