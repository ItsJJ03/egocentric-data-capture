"""
Central config — edit this file only for hardware changes.
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
FPS              = 30
CAPTURE_DURATION = 60          # seconds — fixed 1-minute sessions
FOV_CHECK_SECS   = 5           # duration of pre-capture FOV check

# ── FOV check ─────────────────────────────────────────────────────
# Minimum number of frames (out of FOV_CHECK_SECS * FPS) where
# at least one hand must be detected to pass the check
FOV_MIN_DETECTION_FRAMES = 10  # ~30% of 5s at 30fps

# ── GPIO pins ─────────────────────────────────────────────────────
PIN_FOV_CHECK = 17    # button 1 — start FOV check  (GPIO17, Pin 11)
PIN_START_REC = 27    # button 2 — start recording  (GPIO27, Pin 13)
DEBOUNCE_S    = 0.05

# ── Web UI ────────────────────────────────────────────────────────
UI_HOST = "0.0.0.0"
UI_PORT = 8080

# ── Combined grid tile size ───────────────────────────────────────
TILE_W = 640
TILE_H = 360
ORBBEC_RGB_DEVICE = '/dev/video_kreo_1'
