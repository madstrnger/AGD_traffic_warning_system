#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       AGD307 Speed Camera — RADAR SIMULATOR MODE                           ║
║       Tests the full ML pipeline WITHOUT physical radar hardware            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE                                                                     ║
║    Replaces the AGD307 RS-422 radar with a software simulator so you can    ║
║    verify the complete pipeline end-to-end:                                  ║
║      USB webcam capture  →  YOLOv8 plate detection  →  EasyOCR  →  log      ║
║                                                                              ║
║  TWO SIMULATION MODES  (set SIM_MODE below)                                  ║
║                                                                              ║
║  AUTO   Generates a realistic traffic stream automatically.                  ║
║         Mostly sub-limit speeds with periodic violations fired at a         ║
║         configurable interval so you can walk in front of the camera        ║
║         holding a printed plate.                                             ║
║                                                                              ║
║  MANUAL Press ENTER to fire a violation at the default speed, or            ║
║         type any integer speed and press ENTER.                              ║
║         Perfect for single-shot testing one frame at a time.                ║
║                                                                              ║
║  WHAT IS IDENTICAL TO PRODUCTION                                             ║
║    • CameraCapture   — real USB webcam, CAP_V4L2, warm-up                   ║
║    • PlateDetector   — real YOLOv8 + EasyOCR, class filtering,              ║
║                        imgsz=640, bottom-45% fallback crop                  ║
║    • PlateLogger     — real detected_plates.txt log format                  ║
║    • ViolationWorker — real async worker thread + queue                     ║
║                                                                              ║
║  WHAT IS REPLACED                                                            ║
║    • RadarInterface  → SimulatedRadar  (no serial port opened)              ║
║                                                                              ║
║  HOW TO RUN                                                                  ║
║    source ~/speed_trap_env/bin/activate                                      ║
║    python3 speed_camera_sim.py                    # default AUTO mode        ║
║    python3 speed_camera_sim.py --manual           # MANUAL mode             ║
║    python3 speed_camera_sim.py --auto             # explicit AUTO mode       ║
║                                                                              ║
║  OUTPUT FILES  (same locations as production)                               ║
║    captures/         violation JPEG images, annotated                        ║
║    detected_plates.txt  plate log                                            ║
║    app.log           full application log                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import os
import queue
import random
import re
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
# Lazy-imported inside classes (allows startup even without GPU available)
# from ultralytics import YOLO
# import easyocr


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — mirror of production speed_camera.py
#  Edit these to match your Pi setup exactly.
# ══════════════════════════════════════════════════════════════════════════════

SPEED_LIMIT_KPH  = 20               # Trigger threshold (kph) — same as production
CAMERA_INDEX     = 0                # USB webcam device index
CAMERA_WIDTH     = 1280
CAMERA_HEIGHT    = 720
COOLDOWN_SEC     = 5.0              # Min seconds between consecutive captures

PROJECT_DIR      = Path(__file__).resolve().parent
BASE_DIR         = PROJECT_DIR
CAPTURE_DIR      = BASE_DIR / "captures"
PLATE_LOG        = BASE_DIR / "detected_plates.txt"
APP_LOG          = BASE_DIR / "app.log"

YOLO_MODEL_PATH  = PROJECT_DIR / "models" / "yolov8_plate.pt"
YOLO_FALLBACK    = PROJECT_DIR / "models" / "yolov8n.pt"
OCR_LANGUAGES    = ["en"]

INDIAN_PLATE_RE  = re.compile(
    r"[A-Z]{2}[\s\-]?"
    r"\d{2}[\s\-]?"
    r"[A-Z]{1,3}[\s\-]?"
    r"\d{4}",
    re.IGNORECASE,
)

LOG_LEVEL = logging.INFO

# ── Simulator-specific config ─────────────────────────────────────────────────

# Default mode — overridden by --manual / --auto CLI flags
SIM_MODE = "AUTO"

# AUTO mode settings
AUTO_RADAR_HZ        = 10           # Speed readings per second (matches *MS=5)
AUTO_VIOLATION_EVERY = 15.0         # Fire one violation every N seconds
AUTO_VIOLATION_SPEED = 35           # kph — the simulated violation speed
AUTO_SUBLIMIT_RANGE  = (5, 18)      # kph range for non-violation background traffic

# MANUAL mode settings
MANUAL_DEFAULT_SPEED = 35           # kph used when user just presses Enter


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(
        level=LOG_LEVEL,
        format=fmt,
        handlers=[
            logging.FileHandler(APP_LOG),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("SpeedCamera")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATED RADAR  — drops in for RadarInterface with identical public API
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedRadar:
    """
    Replaces RadarInterface.  No serial port is opened.

    AUTO mode
    ─────────
    A background thread generates speed readings into an internal queue at
    AUTO_RADAR_HZ.  Most readings are sub-limit background traffic.  Every
    AUTO_VIOLATION_EVERY seconds the thread injects exactly one reading at
    AUTO_VIOLATION_SPEED, giving you time to position in front of the camera.

    MANUAL mode
    ───────────
    read_speed_blocking() reads one line from stdin.
      • Just press Enter       → MANUAL_DEFAULT_SPEED kph
      • Type  35  then Enter   → 35 kph
      • Type  q   then Enter   → clean shutdown
    """

    def __init__(self, log: logging.Logger, mode: str):
        self.log  = log.getChild("SimRadar")
        self.mode = mode.upper()
        self._q:    queue.Queue   = queue.Queue(maxsize=200)
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API (mirrors RadarInterface) ──────────────────────────────────

    def connect(self) -> bool:
        self.log.info(
            f"[SIM] RadarInterface bypassed — SimulatedRadar active "
            f"(mode={self.mode})"
        )
        return True

    def initialise(self) -> bool:
        cmds = [
            f"*LOWSPEED={SPEED_LIMIT_KPH - 10}  → ACK (simulated)",
            "*MS=5                              → ACK (simulated)",
            "*BIDI=0                            → ACK (simulated)",
            "*SAVE!                             → ACK (simulated)",
        ]
        for c in cmds:
            self.log.info(f"[SIM] CMD {c}")
            time.sleep(0.05)

        if self.mode == "AUTO":
            self._thread = threading.Thread(
                target=self._auto_generator,
                daemon=True,
                name="SimRadar-Auto",
            )
            self._thread.start()
            self.log.info(
                f"[SIM] AUTO mode — violation every {AUTO_VIOLATION_EVERY}s "
                f"at {AUTO_VIOLATION_SPEED} kph.  "
                f"Background traffic: {AUTO_SUBLIMIT_RANGE[0]}–"
                f"{AUTO_SUBLIMIT_RANGE[1]} kph."
            )
        else:
            self.log.info(
                "[SIM] MANUAL mode — press Enter to fire a violation "
                f"(default {MANUAL_DEFAULT_SPEED} kph), "
                "or type a speed integer, or 'q' to quit."
            )
        return True

    def read_speed_blocking(self) -> float | None:
        """Return next speed value.  Blocks until one is available."""
        if self.mode == "AUTO":
            return self._auto_read()
        else:
            return self._manual_read()

    def close(self):
        self._stop.set()
        self.log.info("[SIM] SimulatedRadar closed.")

    # ── AUTO mode internals ──────────────────────────────────────────────────

    def _auto_generator(self):
        """
        Runs in a daemon thread.  Pushes speed readings into self._q.

        Timeline per AUTO_VIOLATION_EVERY-second cycle:
          • Continuous sub-limit readings at AUTO_RADAR_HZ
          • One violation reading injected at the cycle midpoint
        """
        interval    = 1.0 / AUTO_RADAR_HZ
        cycle_start = time.monotonic()
        violation_fired = False

        while not self._stop.is_set():
            now     = time.monotonic()
            elapsed = now - cycle_start

            if elapsed >= AUTO_VIOLATION_EVERY:
                # Reset cycle
                cycle_start     = now
                elapsed         = 0.0
                violation_fired = False

            # Fire violation at the midpoint of the cycle so the camera
            # has time to warm-up and the user has time to get in position.
            midpoint = AUTO_VIOLATION_EVERY / 2.0
            if not violation_fired and elapsed >= midpoint:
                speed           = float(AUTO_VIOLATION_SPEED)
                violation_fired = True
                self.log.debug(f"[SIM] Injecting violation: {speed} kph")
            else:
                # Sub-limit background traffic — slightly randomised
                speed = float(random.randint(*AUTO_SUBLIMIT_RANGE))

            try:
                self._q.put_nowait(speed)
            except queue.Full:
                pass   # drop silently — main loop is processing

            time.sleep(interval)

    def _auto_read(self) -> float | None:
        try:
            return self._q.get(timeout=2.0)
        except queue.Empty:
            return None

    # ── MANUAL mode internals ────────────────────────────────────────────────

    def _manual_read(self) -> float | None:
        """
        Block on stdin.  Called from the main loop thread each iteration.
        Prints a prompt only once per call.
        """
        print(
            f"\n[SIM] Press Enter to fire {MANUAL_DEFAULT_SPEED} kph violation  "
            f"|  type speed (int) + Enter  |  'q' + Enter to quit",
            end=" > ",
            flush=True,
        )
        try:
            raw = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not raw:
            self.log.info(f"[SIM] Manual trigger: {MANUAL_DEFAULT_SPEED} kph")
            return float(MANUAL_DEFAULT_SPEED)

        if raw.lower() == "q":
            self.log.info("[SIM] User requested quit.")
            raise KeyboardInterrupt

        try:
            speed = float(raw)
            self.log.info(f"[SIM] Manual trigger: {speed} kph")
            return speed
        except ValueError:
            self.log.warning(f"[SIM] Unrecognised input {raw!r} — ignoring.")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA CAPTURE  ← IDENTICAL to production speed_camera.py
# ══════════════════════════════════════════════════════════════════════════════

class CameraCapture:
    """Manages USB webcam via OpenCV."""

    def __init__(self, log: logging.Logger):
        self.log = log.getChild("Camera")
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        # Prefer V4L2 backend explicitly — more reliable for USB webcams on Pi.
        # Falls back to OpenCV auto-detection if V4L2 is unavailable.
        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self.log.warning(
                "V4L2 backend failed — retrying with auto-detection backend."
            )
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.log.error(
                f"Cannot open camera at index {CAMERA_INDEX}. "
                "Run: v4l2-ctl --list-devices  or  ls /dev/video* "
                "to confirm the webcam is detected."
            )
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # Warm-up: discard first few frames (auto-exposure settling)
        for _ in range(5):
            self._cap.read()
        self.log.info(
            f"Camera {CAMERA_INDEX} opened ({CAMERA_WIDTH}×{CAMERA_HEIGHT})"
        )
        return True

    def capture(self) -> np.ndarray | None:
        """Grab a single frame. Returns BGR ndarray or None on failure."""
        if self._cap is None or not self._cap.isOpened():
            self.log.error("Camera not open.")
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.log.error("Frame grab failed.")
            return None
        return frame

    def save_frame(self, frame: np.ndarray, speed: float) -> Path:
        """Save frame to CAPTURE_DIR and return its path."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"violation_{ts}_{int(speed)}kph.jpg"
        path = CAPTURE_DIR / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.log.info(f"Frame saved → {path}")
        return path

    def release(self):
        if self._cap:
            self._cap.release()
            self.log.info("Camera released.")


# ══════════════════════════════════════════════════════════════════════════════
#  PLATE DETECTOR  ← IDENTICAL to production speed_camera.py
# ══════════════════════════════════════════════════════════════════════════════

class PlateDetector:
    """
    Two-stage pipeline:
      Stage 1 — YOLOv8 (license plate detection model) → bounding boxes
      Stage 2 — EasyOCR → text from each cropped plate region
    Falls back to full-frame OCR if no plate box is detected.
    """

    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD  = 0.45
    PADDING_PX     = 8

    def __init__(self, log: logging.Logger):
        self.log           = log.getChild("Detector")
        self._yolo:        object | None = None
        self._ocr:         object | None = None
        self._is_fallback: bool = False

    def load(self) -> bool:
        return self._load_yolo() and self._load_ocr()

    # ── YOLO ─────────────────────────────────────────────────────────────────

    def _load_yolo(self) -> bool:
        from ultralytics import YOLO

        if YOLO_MODEL_PATH.exists():
            self.log.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            self._yolo = YOLO(str(YOLO_MODEL_PATH))
        else:
            self.log.warning(
                f"ANPR model not found at {YOLO_MODEL_PATH}. "
                "Attempting to re-download from GitHub …"
            )
            import urllib.request
            _ANPR_URL = (
                "https://github.com/Muhammad-Zeerak-Khan/"
                "Automatic-License-Plate-Recognition-using-YOLOv8/"
                "raw/main/license_plate_detector.pt"
            )
            try:
                YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.log.info(f"Downloading from {_ANPR_URL} …")
                urllib.request.urlretrieve(_ANPR_URL, YOLO_MODEL_PATH)
                self._yolo = YOLO(str(YOLO_MODEL_PATH))
                self.log.info(f"ANPR weights saved → {YOLO_MODEL_PATH}")
            except Exception as exc:
                self.log.warning(
                    f"Download failed ({exc}). "
                    f"Falling back to YOLOv8n at {YOLO_FALLBACK}."
                )
                if YOLO_FALLBACK.exists():
                    self._yolo = YOLO(str(YOLO_FALLBACK))
                else:
                    self._yolo = YOLO("yolov8n.pt")
                self._is_fallback = True

        self.log.info(
            f"YOLO model ready  (fallback={self._is_fallback})"
        )
        return True

    # ── EasyOCR ──────────────────────────────────────────────────────────────

    def _load_ocr(self) -> bool:
        import easyocr
        self.log.info(f"Loading EasyOCR (languages: {OCR_LANGUAGES}) …")
        self._ocr = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=False,
            model_storage_directory=str(BASE_DIR / "easyocr_models"),
            download_enabled=True,
        )
        self.log.info("EasyOCR ready.")
        return True

    # ── Detection pipeline ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = []
        plate_crops, bboxes = self._detect_plates(frame)

        if not plate_crops:
            self.log.debug("No plate boxes detected; running OCR on full frame.")
            plate_crops = [frame]
            bboxes      = [(0, 0, frame.shape[1], frame.shape[0])]

        for crop, bbox in zip(plate_crops, bboxes):
            plate_text, conf = self._ocr_crop(crop)
            if plate_text:
                results.append({
                    "plate_text": plate_text,
                    "confidence": conf,
                    "bbox":       bbox,
                })
        return results

    def _detect_plates(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple]]:
        crops, bboxes = [], []
        _VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck (COCO)
        try:
            preds = self._yolo.predict(
                frame,
                conf    = self.CONF_THRESHOLD,
                iou     = self.IOU_THRESHOLD,
                imgsz   = 640,
                verbose = False,
            )
            h, w = frame.shape[:2]
            for result in preds:
                for box in result.boxes:
                    cls_id = int(box.cls[0])

                    if self._is_fallback:
                        if cls_id not in _VEHICLE_CLASSES:
                            continue
                    else:
                        if cls_id != 0:
                            continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if self._is_fallback:
                        # Only bottom 45% of vehicle — where plates live
                        y1 = y1 + int((y2 - y1) * 0.55)

                    x1 = max(0, x1 - self.PADDING_PX)
                    y1 = max(0, y1 - self.PADDING_PX)
                    x2 = min(w, x2 + self.PADDING_PX)
                    y2 = min(h, y2 + self.PADDING_PX)
                    crops.append(frame[y1:y2, x1:x2])
                    bboxes.append((x1, y1, x2, y2))
        except Exception as exc:
            self.log.error(f"YOLO inference error: {exc}")
        return crops, bboxes

    def _ocr_crop(self, crop: np.ndarray) -> tuple[str, float]:
        try:
            processed = self._preprocess_for_ocr(crop)
            ocr_out   = self._ocr.readtext(
                processed,
                detail    = 1,
                paragraph = False,
                allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
            )
        except Exception as exc:
            self.log.error(f"OCR error: {exc}")
            return "", 0.0

        if not ocr_out:
            return "", 0.0

        texts   = [e[1] for e in ocr_out]
        confs   = [e[2] for e in ocr_out]
        raw     = " ".join(texts).upper().strip()
        conf    = float(np.mean(confs)) if confs else 0.0
        cleaned = self._clean_plate_text(raw)
        self.log.debug(f"OCR raw={raw!r}  cleaned={cleaned!r}  conf={conf:.2f}")
        return cleaned, conf

    @staticmethod
    def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        if h < 40 or w < 120:
            scale = max(40 / h, 120 / w, 1.0) * 2
            crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.dilate(thresh, kernel, iterations=1)

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9\- ]", "", text.upper())
        cleaned = cleaned.replace(" ", "").replace("-", "")
        match   = INDIAN_PLATE_RE.search(cleaned)
        if match:
            return match.group(0).upper()
        return cleaned if len(cleaned) >= 4 else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PLATE LOGGER  ← IDENTICAL to production speed_camera.py
# ══════════════════════════════════════════════════════════════════════════════

class PlateLogger:
    """Appends plate detections to the log file, one entry per line."""

    _HEADER = (
        "# AGD307 Speed Camera Log\n"
        "# Format: TIMESTAMP | SPEED_KPH | PLATE_TEXT | CONFIDENCE | IMAGE\n"
        "#" + "─" * 72 + "\n"
    )

    def __init__(self, log: logging.Logger):
        self.log = log.getChild("Logger")
        self._ensure_file()

    def _ensure_file(self):
        if not PLATE_LOG.exists():
            PLATE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(PLATE_LOG, "w") as f:
                f.write(self._HEADER)
            self.log.info(f"Created plate log: {PLATE_LOG}")

    def append(self, speed: float, plate_text: str,
               confidence: float, image_path: Path):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = (
            f"{ts} | {speed:6.1f} kph | {plate_text:<15} "
            f"| conf={confidence:.2f} | {image_path.name}\n"
        )
        with open(PLATE_LOG, "a") as f:
            f.write(line)
        self.log.info(f"LOGGED → {line.rstrip()}")

    def append_no_plate(self, speed: float, image_path: Path):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = (
            f"{ts} | {speed:6.1f} kph | {'NO_PLATE':<15} "
            f"| conf=0.00 | {image_path.name}\n"
        )
        with open(PLATE_LOG, "a") as f:
            f.write(line)
        self.log.info(f"LOGGED (no plate) → {line.rstrip()}")


# ══════════════════════════════════════════════════════════════════════════════
#  VIOLATION WORKER  ← IDENTICAL to production speed_camera.py
# ══════════════════════════════════════════════════════════════════════════════

class ViolationWorker(threading.Thread):
    """
    Receives (frame, speed) from a queue and runs the detector pipeline
    asynchronously so the radar read loop is never blocked.
    """

    def __init__(
        self,
        detector: PlateDetector,
        camera:   CameraCapture,
        logger:   PlateLogger,
        log:      logging.Logger,
    ):
        super().__init__(daemon=True, name="ViolationWorker")
        self._q        = queue.Queue(maxsize=4)
        self._detector = detector
        self._camera   = camera
        self._logger   = logger
        self.log       = log.getChild("Worker")
        self._stop_evt = threading.Event()

    def enqueue(self, frame: np.ndarray, speed: float):
        try:
            self._q.put_nowait((frame, speed))
        except queue.Full:
            self.log.warning("Worker queue full — dropping frame.")

    def run(self):
        self.log.info("Violation worker started.")
        while not self._stop_evt.is_set():
            try:
                frame, speed = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._process(frame, speed)

    def _process(self, frame: np.ndarray, speed: float):
        image_path = self._camera.save_frame(frame, speed)

        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"[SIM] SPEED: {int(speed)} kph  [{datetime.now().strftime('%H:%M:%S')}]",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2,  # orange for sim
        )

        detections = self._detector.detect(annotated)

        if detections:
            for det in detections:
                plate = det["plate_text"]
                conf  = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, plate,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
                self._logger.append(speed, plate, conf, image_path)
            cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            self.log.info(f"No plate detected  (speed={speed:.0f} kph)")
            self._logger.append_no_plate(speed, image_path)

    def stop(self):
        self._stop_evt.set()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class SpeedCameraSimApp:
    """Orchestrator — identical flow to production, SimulatedRadar injected."""

    def __init__(self, mode: str):
        self.log       = setup_logging()
        self._radar    = SimulatedRadar(self.log, mode)
        self._camera   = CameraCapture(self.log)
        self._detector = PlateDetector(self.log)
        self._pl_log   = PlateLogger(self.log)
        self._worker   = ViolationWorker(
            self._detector, self._camera, self._pl_log, self.log
        )
        self._last_trigger = 0.0

    def start(self):
        self.log.info("═" * 64)
        self.log.info("  AGD307 Speed Camera — SIMULATOR MODE")
        self.log.info(f"  Sim mode     : {self._radar.mode}")
        self.log.info(f"  Speed limit  : {SPEED_LIMIT_KPH} kph")
        self.log.info(f"  Plate log    : {PLATE_LOG}")
        self.log.info(f"  Captures     : {CAPTURE_DIR}")
        if self._radar.mode == "AUTO":
            self.log.info(
                f"  Auto cycle   : violation at {AUTO_VIOLATION_SPEED} kph "
                f"every {AUTO_VIOLATION_EVERY}s"
            )
        self.log.info("═" * 64)

        if not self._radar.connect():
            sys.exit(1)
        if not self._radar.initialise():
            sys.exit(1)
        if not self._camera.open():
            sys.exit(1)
        if not self._detector.load():
            sys.exit(1)

        self._worker.start()
        self.log.info(
            f"[SIM] System ARMED — monitoring for speeds above "
            f"{SPEED_LIMIT_KPH} kph …"
        )
        self._loop()

    def _loop(self):
        try:
            while True:
                speed = self._radar.read_speed_blocking()
                if speed is None:
                    continue

                if speed > SPEED_LIMIT_KPH:
                    now = time.monotonic()
                    if now - self._last_trigger >= COOLDOWN_SEC:
                        self._last_trigger = now
                        self.log.warning(
                            f"⚡ [SIM] SPEED VIOLATION: {speed:.0f} kph "
                            f"(limit {SPEED_LIMIT_KPH} kph)"
                        )
                        frame = self._camera.capture()
                        if frame is not None:
                            self._worker.enqueue(frame, speed)
                    else:
                        remaining = COOLDOWN_SEC - (now - self._last_trigger)
                        self.log.debug(
                            f"Speed {speed:.0f} kph — cooldown "
                            f"({remaining:.1f}s left)"
                        )

        except KeyboardInterrupt:
            self.log.info("Interrupted — shutting down …")
        finally:
            self._shutdown()

    def _shutdown(self):
        self._worker.stop()
        self._worker.join(timeout=10)
        self._radar.close()
        self._camera.release()
        self.log.info("Simulator shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGD307 Speed Camera — Radar Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 speed_camera_sim.py             # AUTO mode (default)\n"
            "  python3 speed_camera_sim.py --auto      # explicit AUTO\n"
            "  python3 speed_camera_sim.py --manual    # MANUAL mode\n"
        ),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--auto",
        action="store_true",
        help=f"AUTO mode: fire violation every {AUTO_VIOLATION_EVERY}s automatically",
    )
    grp.add_argument(
        "--manual",
        action="store_true",
        help="MANUAL mode: press Enter (or type speed) to trigger each capture",
    )
    args = parser.parse_args()

    if args.manual:
        mode = "MANUAL"
    elif args.auto:
        mode = "AUTO"
    else:
        mode = SIM_MODE   # default from config at top of file

    SpeedCameraSimApp(mode=mode).start()
