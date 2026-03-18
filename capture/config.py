"""
Central config — edit this file only for hardware changes.
V2: 30-min session → sequential 1-min segments with wrist checks.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────
OUTPUT_DIR   = "/mnt/ssd/recordings"
ORBBEC_REC   = os.path.expanduser("~/ob_examples_build/bin/ob_device_record_nogui")
ORBBEC_LIB   = os.path.expanduser(
    "~/OrbbecSDK_fresh/OrbbecSDK_v2.7.6_202602021228_d712cda_linux_arm64/lib")

# ── Camera devices ────────────────────────────────────────────────
KREO1_DEVICE = "/dev/video_kreo_1"   # left wrist
KREO2_DEVICE = "/dev/video_kreo_2"   # right wrist
KREO_W       = 1280
KREO_H       = 720

# ── Capture settings ──────────────────────────────────────────────
FPS                  = 30
SEGMENT_DURATION     = 60          # seconds per video segment (default 1 min)
SESSION_DURATION     = 1800        # seconds total session (default 30 min)
FOV_CHECK_SECS       = 5           # duration of initial FOV check

# ── FOV / wrist check ────────────────────────────────────────────
FOV_MIN_DETECTION_FRAMES = 10      # ~30% of 5s at 30fps for initial check
YOLO_MODEL_PATH    = os.path.expanduser("~/models/yolov8n-pose.pt")
YOLO_CONF_THRESH   = 0.3
# Wrist keypoint indices in COCO-pose: 9=left_wrist, 10=right_wrist
WRIST_KP_INDICES   = [9, 10]
WRIST_KP_CONF      = 0.3          # min keypoint confidence

# HSV fallback skin detection
SKIN_LOWER      = (0,   20,  70)
SKIN_UPPER      = (20, 255, 255)
MIN_REGION_PX   = 800
MIN_REGIONS     = 2

# ── GPIO pins ─────────────────────────────────────────────────────
PIN_FOV_CHECK = 17
PIN_START_REC = 27
DEBOUNCE_S    = 0.05

# ── Web UI ────────────────────────────────────────────────────────
UI_HOST = "0.0.0.0"
UI_PORT = 8080

# ── Orbbec stream (FOV check binary) ─────────────────────────────
ORBBEC_STREAM    = "/mnt/ssd/OrbbecSDK_Pi5/orbbec_stream"
ORBBEC_STREAM_LIB = "/opt/OrbbecSDK/lib"

# ── S3 upload ─────────────────────────────────────────────────────
S3_BUCKET        = "egocentric-datacollection1"
S3_PREFIX        = "captures/"
S3_MAX_RETRIES   = 3
S3_RETRY_DELAY   = 10             # seconds between retries

# ── Combined grid tile size ───────────────────────────────────────
TILE_W = 640
TILE_H = 360

# ── MCAP conversion ──────────────────────────────────────────────
MCAP_ENABLED_DEFAULT = False       # toggle from UI
