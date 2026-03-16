# Egocentric Capture Pipeline

## System Flow

```
main.py
  │
  ├─ 1. Session select / create
  │     └─ ~/.egocentric_sessions.json persists task metadata
  │
  ├─ 2. Wrist check (5s, Orbbec color, YOLOv8n-pose)
  │     ├─ Both wrists in ≥50% of inferred frames → PASS
  │     ├─ Fail → retry prompt
  │     ├─ First pass → session marked wrist-check optional
  │     └─ Optional sessions → ENTER to skip (8s window)
  │
  ├─ 3. Recording loop (30 × 1-min chunks)
  │     ├─ ob_device_record_nogui → .bag on NVMe PCIe
  │     ├─ Clean stop via PTY 'q' (preserves bag integrity)
  │     └─ Each chunk immediately enqueued for upload
  │
  └─ 4. Background S3 upload (concurrent with recording)
        ├─ boto3 multipart (50MB parts, 4 concurrent)
        ├─ HeadObject size verification before delete
        ├─ Exponential backoff retry (max 5 attempts)
        └─ Auto-delete local file after verified upload
```

## S3 Structure

```
s3://egocentric-datacollection1/
  └─ <task_name>/
       └─ <YYYY-MM-DD>/
            ├─ <task>_<timestamp>_chunk00.bag
            ├─ <task>_<timestamp>_chunk01.bag
            └─ ...
```

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt --break-system-packages

# 2. Ensure AWS credentials are configured
aws configure  # or use IAM role / env vars

# 3. Ensure NVMe is mounted
sudo mount /dev/nvme0n1p1 /mnt/nvme

# 4. Run
python3 main.py
```

## Storage Budget

| Stream         | Rate        | Per chunk (1 min) | Per session (30 min) |
|---------------|-------------|-------------------|----------------------|
| Orbbec depth  | ~15 MB/s    | ~900 MB           | ~27 GB               |
| Orbbec color  | ~2–5 MB/s   | ~150 MB           | ~4.5 GB              |
| **Total**     | **~17 MB/s**| **~1 GB**         | **~30 GB**           |

NVMe write throughput (~400–900 MB/s) provides >20× headroom.
WiFi upload (~20–40 MB/s sustained) will lag ~25–50s per chunk — this is expected.
The upload queue safely absorbs this; recording never waits on upload.

## Failure Modes

| Scenario                  | Behaviour                                      |
|--------------------------|------------------------------------------------|
| WiFi drops during upload  | Retry with backoff; file safe on NVMe          |
| WiFi drops entirely       | All chunks accumulate on NVMe; re-run to drain |
| Recorder crash mid-chunk  | Chunk skipped; session continues               |
| Ctrl-C during recording   | Uploads abandoned; NVMe files preserved        |
| S3 size mismatch          | Re-upload, file NOT deleted                    |
| Max retries exhausted     | Error logged; file preserved for manual upload |

## Key Design Decisions

- **PTY OPOST-only**: `tty.setraw()` breaks libusb TTY context on Pi 5.
  Clearing only `OPOST` flag prevents binary corruption while preserving
  libusb device enumeration context (from existing project learnings).

- **Device cooldown (5s)**: After wrist check releases the Orbbec via SDK,
  a 5s sleep ensures the device is fully de-initialised before
  `ob_device_record_nogui` re-opens it.

- **Multipart S3**: Mandatory for ~1GB files on flaky WiFi. Each 50MB part
  can be independently retried without re-uploading the whole bag.

- **Delete-after-verify**: `head_object` ContentLength check before `unlink`.
  Prevents data loss from partial uploads being mistaken as complete.
