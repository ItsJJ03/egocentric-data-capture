# wrist_check.py — 5-second wrist presence check via YOLOv8n-pose on Orbbec color stream
import time
import logging
import numpy as np
import config

log = logging.getLogger(__name__)


class WristChecker:
    def __init__(self):
        log.info("[wrist_check] Loading YOLOv8n-pose model...")
        from ultralytics import YOLO
        self.model = YOLO(config.YOLO_MODEL)
        log.info("[wrist_check] Model loaded.")

    def run(self) -> tuple[bool, float]:
        """
        Stream Orbbec color for WRIST_CHECK_DURATION seconds.
        Runs YOLO pose inference every YOLO_EVERY_N_FRAMES frames.
        Returns (passed: bool, fraction: float) where fraction =
        (frames with both wrists detected) / (total inferred frames).
        """
        try:
            import pyorbbecsdk as ob
        except ImportError:
            log.error("[wrist_check] pyorbbecsdk not found — cannot run wrist check")
            return False, 0.0

        pipeline = ob.Pipeline()
        cfg      = ob.Config()

        try:
            profile_list  = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(
                config.ORBBEC_COLOR_WIDTH,
                config.ORBBEC_COLOR_HEIGHT,
                ob.OBFormat.RGB,
                config.ORBBEC_COLOR_FPS,
            )
            cfg.enable_stream(color_profile)
            pipeline.start(cfg)
        except Exception as e:
            log.error(f"[wrist_check] Orbbec pipeline start failed: {e}")
            return False, 0.0

        total_inferred = 0
        both_wrists    = 0
        frame_idx      = 0
        deadline       = time.time() + config.WRIST_CHECK_DURATION
        hsv_fallback_count = 0

        log.info(f"[wrist_check] Running {config.WRIST_CHECK_DURATION}s check — show both wrists to camera...")

        try:
            while time.time() < deadline:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                frame_idx += 1
                if frame_idx % config.YOLO_EVERY_N_FRAMES != 0:
                    continue

                # Build numpy array from frame buffer
                raw  = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                img  = raw.reshape((color_frame.get_height(), color_frame.get_width(), 3))

                total_inferred += 1
                detected = self._infer_wrists(img)

                if detected:
                    both_wrists += 1

                # Live progress bar
                frac      = both_wrists / total_inferred if total_inferred else 0.0
                remaining = max(0.0, deadline - time.time())
                bar_len   = 20
                filled    = int(bar_len * frac)
                bar       = "█" * filled + "░" * (bar_len - filled)
                status    = "✓" if frac >= config.WRIST_PASS_FRACTION else "…"
                print(
                    f"\r  [{bar}] {frac:5.1%}  {both_wrists}/{total_inferred} frames  "
                    f"{remaining:.1f}s remaining  {status}",
                    end="",
                    flush=True,
                )

        finally:
            pipeline.stop()
            print()  # newline after \r

        if total_inferred == 0:
            log.warning("[wrist_check] No frames inferred — camera issue?")
            return False, 0.0

        fraction = both_wrists / total_inferred
        passed   = fraction >= config.WRIST_PASS_FRACTION
        return passed, fraction

    def _infer_wrists(self, img: np.ndarray) -> bool:
        """
        Returns True if both wrists (KP 9 & 10) detected above confidence threshold.
        Falls back to HSV skin detection silently if YOLO finds no person.
        """
        results = self.model(img, verbose=False, conf=config.YOLO_CONF)

        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                kp   = kps_data[0]   # first detected person
                lw_c = float(kp[config.KP_LEFT_WRIST][2])
                rw_c = float(kp[config.KP_RIGHT_WRIST][2])
                if lw_c > config.YOLO_CONF and rw_c > config.YOLO_CONF:
                    return True

        # Silent HSV fallback — checks for skin-tone pixels in left/right quadrants
        return self._hsv_skin_fallback(img)

    @staticmethod
    def _hsv_skin_fallback(img: np.ndarray) -> bool:
        """
        Rough check: significant skin-tone presence in both left and right
        horizontal thirds of frame. Not used for scoring — only as fallback.
        """
        try:
            import cv2
            hsv   = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            mask  = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
            h, w  = mask.shape
            left  = mask[:, :w // 3]
            right = mask[:, 2 * w // 3:]
            threshold = 0.05 * (h * w // 3)
            return (left.sum() > threshold) and (right.sum() > threshold)
        except Exception:
            return False

    def __del__(self):
        # Ensure model is released before recorder takes the Orbbec device
        try:
            del self.model
        except Exception:
            pass
