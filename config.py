# config.py — egocentric capture pipeline
from pathlib import Path

# ─── Orbbec color stream (wrist check only) ───────────────
ORBBEC_COLOR_WIDTH      = 320
ORBBEC_COLOR_HEIGHT     = 180
ORBBEC_COLOR_FPS        = 30

# ─── Wrist check ──────────────────────────────────────────
WRIST_CHECK_DURATION    = 5        # seconds
WRIST_PASS_FRACTION     = 0.50     # ≥50% of inferred frames must have both wrists
YOLO_CONF               = 0.4
YOLO_MODEL              = "yolov8n-pose.pt"
KP_LEFT_WRIST           = 9        # COCO keypoint index
KP_RIGHT_WRIST          = 10
YOLO_EVERY_N_FRAMES     = 3        # run inference every Nth frame (~3.6 FPS effective)
DEVICE_COOLDOWN_SEC     = 5        # wait after wrist check before recorder takes device

# ─── Recording ────────────────────────────────────────────
CHUNK_DURATION_SEC      = 60                                  # 1 min per bag
TOTAL_CHUNKS            = 30                                  # 30 min session
RECORDING_DIR           = Path("/mnt/nvme/egocentric")        # NVMe mount
OB_RECORD_BIN           = "/usr/local/bin/ob_device_record_nogui"
OB_STARTUP_DRAIN_SEC    = 3.0     # seconds to drain startup output before timing chunk
OB_STOP_TIMEOUT_SEC     = 15      # max seconds to wait for clean stop after 'q'

# ─── Session registry ─────────────────────────────────────
SESSION_REGISTRY        = Path.home() / ".egocentric_sessions.json"

# ─── Operator identity ────────────────────────────────────
# Matches the 'operator' field in the intern's DynamoDB table.
# Set via env var EGOCENTRIC_OPERATOR or falls back to hostname.
import os, socket
OPERATOR_ID             = os.environ.get("EGOCENTRIC_OPERATOR", socket.gethostname())

# ─── S3 ───────────────────────────────────────────────────
S3_BUCKET               = "egocentric-datacollection1"
S3_MULTIPART_CHUNK      = 50  * 1024 * 1024   # 50 MB per part
S3_MAX_CONCURRENCY      = 4                    # parallel part uploads
S3_MAX_RETRIES          = 5
S3_RETRY_BACKOFF        = 2                    # exponential base (2^attempt seconds)
S3_MIN_VALID_SIZE       = 1 * 1024 * 1024      # bags must be >1MB to be considered valid

# ─── DynamoDB (shared with intern's autocapture dashboard) ─
AWS_REGION              = "ap-south-1"
DYNAMODB_TABLE          = "autocapture-operator-stats"
DYNAMODB_ENABLED        = True   # set False to disable stats writes without code change
