# uploader.py — background S3 multipart upload queue with auto-delete
#
# Design goals:
#   - Recording NEVER waits on upload
#   - Files accumulate safely on NVMe if WiFi drops
#   - Delete only after S3 ContentLength verified
#   - Multipart upload handles GB-scale bag files on flaky WiFi
#   - Exponential backoff on failure, S3_MAX_RETRIES before giving up
import os
import queue
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

import config

log = logging.getLogger(__name__)


def _update_dynamo_stats(uploaded: int = 0, sessions: int = 0, recorded: int = 0):
    """
    Fire-and-forget DynamoDB write to the shared autocapture-operator-stats table.
    Matches the schema used by the intern's mobile capture system so both data
    sources appear in the same dashboard.
    """
    if not config.DYNAMODB_ENABLED:
        return
    try:
        client = boto3.client(
            "dynamodb",
            region_name          = config.AWS_REGION,
            aws_access_key_id    = getattr(config, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key= getattr(config, "AWS_SECRET_ACCESS_KEY", None),
        )
        expressions, values = [], {
            ":zero": {"N": "0"},
            ":ts":   {"S": time.strftime("%Y-%m-%dT%H:%M:%S")},
        }
        if uploaded > 0:
            expressions.append("totalVideosUploaded = if_not_exists(totalVideosUploaded, :zero) + :u")
            values[":u"] = {"N": str(uploaded)}
        if sessions > 0:
            expressions.append("totalSessions = if_not_exists(totalSessions, :zero) + :s")
            values[":s"] = {"N": str(sessions)}
        if recorded > 0:
            expressions.append("totalVideosRecorded = if_not_exists(totalVideosRecorded, :zero) + :r")
            values[":r"] = {"N": str(recorded)}
        if not expressions:
            return
        expressions.append("lastActive = :ts")
        client.update_item(
            TableName=config.DYNAMODB_TABLE,
            Key={"operatorId": {"S": config.OPERATOR_ID}},
            UpdateExpression="SET " + ", ".join(expressions),
            ExpressionAttributeValues=values,
        )
        log.debug(f"[dynamo] stats updated: operator={config.OPERATOR_ID} +uploaded={uploaded}")
    except Exception as e:
        log.warning(f"[dynamo] stats write failed (non-fatal): {e}")


@dataclass
class UploadJob:
    local_path: Path
    s3_key:     str
    attempt:    int = 0


class UploadQueue:
    """
    Thread-safe upload queue. Enqueue bags; a background worker uploads
    them via S3 multipart and deletes on verified success.
    """

    def __init__(self):
        self._q            = queue.Queue()
        self._stop_event   = threading.Event()
        self._in_flight    = 0
        self._lock         = threading.Lock()
        self._s3           = boto3.client(
            "s3",
            region_name          = config.AWS_REGION,
            aws_access_key_id    = getattr(config, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key= getattr(config, "AWS_SECRET_ACCESS_KEY", None),
        )
        self._transfer_cfg = TransferConfig(
            multipart_threshold = config.S3_MULTIPART_CHUNK,
            multipart_chunksize = config.S3_MULTIPART_CHUNK,
            max_concurrency     = config.S3_MAX_CONCURRENCY,
            use_threads         = True,
        )
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="s3-uploader"
        )
        self._thread.start()
        log.info("[uploader] Background upload thread started")

    # ── Public API ────────────────────────────────────────────────────

    def enqueue(self, local_path: Path, s3_key: str):
        with self._lock:
            self._in_flight += 1
        self._q.put(UploadJob(local_path=local_path, s3_key=s3_key))
        import state as _state
        _state.update(upload_pending=self._in_flight)
        log.info(
            f"[uploader] Queued: {local_path.name} "
            f"({local_path.stat().st_size / 1e6:.0f} MB) "
            f"-> s3://{config.S3_BUCKET}/{s3_key}"
        )

    def pending_count(self) -> int:
        with self._lock:
            return self._in_flight

    def wait_all(self, poll_interval: float = 5.0):
        """Block until all enqueued jobs are done (success or exhausted retries)."""
        while True:
            with self._lock:
                if self._in_flight == 0:
                    return
            log.info(f"[uploader] Waiting — {self.pending_count()} uploads remaining...")
            time.sleep(poll_interval)

    def stop(self):
        self._stop_event.set()

    # ── Worker ────────────────────────────────────────────────────────

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                job: UploadJob = self._q.get(timeout=2.0)
            except queue.Empty:
                continue

            success = self._upload(job)

            if success:
                self._delete_local(job.local_path)
            else:
                job.attempt += 1
                if job.attempt < config.S3_MAX_RETRIES:
                    backoff = config.S3_RETRY_BACKOFF ** job.attempt
                    log.warning(
                        f"[uploader] Retry {job.attempt}/{config.S3_MAX_RETRIES} "
                        f"for {job.local_path.name} in {backoff}s"
                    )
                    threading.Timer(backoff, self._q.put, args=(job,)).start()
                    # Don't decrement in_flight — still pending
                    self._q.task_done()
                    continue
                else:
                    log.error(
                        f"[uploader] ✗ GAVE UP on {job.local_path.name} after "
                        f"{config.S3_MAX_RETRIES} attempts. "
                        f"File preserved at {job.local_path}"
                    )
                    with self._lock:
                        self._in_flight -= 1
                    import state as _state
                    _state.update(
                        upload_pending=self._in_flight,
                        upload_failed=(_state.get("upload_failed") or 0) + 1,
                    )

            self._q.task_done()

    def _upload(self, job: UploadJob) -> bool:
        if not job.local_path.exists():
            log.error(f"[uploader] File missing: {job.local_path}")
            return False

        file_size = job.local_path.stat().st_size

        # ── Deduplication: skip if already in S3 (matches intern's logic) ─
        try:
            existing = self._s3.head_object(Bucket=config.S3_BUCKET, Key=job.s3_key)
            if existing["ContentLength"] == file_size:
                log.info(f"[uploader] ⚠ Duplicate skipped — {job.s3_key} already in S3")
                with self._lock:
                    self._in_flight -= 1
                return True  # treat as success so local file is deleted
        except Exception:
            pass  # key doesn't exist → proceed with upload
        log.info(
            f"[uploader] Uploading {job.local_path.name} "
            f"({file_size / 1e6:.0f} MB) attempt {job.attempt + 1}"
        )
        import state as _state
        _state.update(upload_current_file=job.local_path.name, upload_current_pct=0)

        try:
            self._s3.upload_file(
                Filename = str(job.local_path),
                Bucket   = config.S3_BUCKET,
                Key      = job.s3_key,
                Config   = self._transfer_cfg,
                Callback = _ProgressCallback(job.local_path.name, file_size),
            )
        except Exception as e:
            log.warning(f"[uploader] Upload error for {job.local_path.name}: {e}")
            return False

        # ── Verify via HeadObject ─────────────────────────────────────
        try:
            head = self._s3.head_object(Bucket=config.S3_BUCKET, Key=job.s3_key)
            remote_size = head["ContentLength"]
            if remote_size != file_size:
                log.warning(
                    f"[uploader] Size mismatch for {job.local_path.name}: "
                    f"local={file_size}, remote={remote_size}"
                )
                return False
            log.info(f"[uploader] Verified: {job.s3_key} ({remote_size / 1e6:.0f} MB)")
            with self._lock:
                self._in_flight -= 1
            import state as _state
            _state.update(
                upload_pending=self._in_flight,
                upload_done=_state.get("upload_done") + 1,
                upload_current_file="",
                upload_current_pct=0,
            )
            # Write to shared DynamoDB stats table (same as intern's mobile system)
            threading.Thread(
                target=_update_dynamo_stats, kwargs={"uploaded": 1}, daemon=True
            ).start()
            return True
        except Exception as e:
            log.warning(f"[uploader] HeadObject failed for {job.s3_key}: {e}")
            return False

    def _delete_local(self, path: Path):
        try:
            path.unlink()
            log.info(f"[uploader] 🗑  Deleted local: {path.name}")
        except OSError as e:
            log.warning(f"[uploader] Could not delete {path}: {e}")


class _ProgressCallback:
    """Logs upload progress at 10% intervals."""

    def __init__(self, name: str, total: int):
        self.name      = name
        self.total     = total
        self.uploaded  = 0
        self._last_pct = -1
        self._lock     = threading.Lock()

    def __call__(self, bytes_transferred: int):
        with self._lock:
            self.uploaded += bytes_transferred
            pct = int(100 * self.uploaded / self.total) if self.total else 0
            if pct >= self._last_pct + 10:
                self._last_pct = pct
                log.info(f"[uploader]   {self.name}: {pct}%")
                import state as _state
                _state.update(upload_current_pct=pct)
