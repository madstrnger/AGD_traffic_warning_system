#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AGD307 Radar Speed Camera — Raspberry Pi 5                        ║
║          RS-422 via FTDI USB  |  Indian Number Plate Detection              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL DEPENDENCIES (run once before first launch):                       ║
║                                                                              ║
║  sudo apt-get update && sudo apt-get install -y \                            ║
║      python3-pip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6           ║
║                                                                              ║
║  USE install.sh + requirements.txt (already done).  Verified versions:      ║
║      pyserial==3.5                                                           ║
║      opencv-python-headless==4.10.0.84   (4.9.x also works)                 ║
║      ultralytics==8.2.18                                                     ║
║      easyocr==1.7.1                                                          ║
║      numpy==1.24.4  (pinned; upgrade only if all deps re-tested together)    ║
║      Pillow==10.3.0                                                          ║
║      torch==2.3.0  /  torchvision==0.18.0                                   ║
║      scipy==1.13.0  /  scikit-image==0.22.0                                 ║
║                                                                              ║
║  PyTorch for Raspberry Pi 5 (aarch64 / bookworm):                           ║
║  If pip install fails for torch, use:                                        ║
║      pip3 install --break-system-packages \                                 ║
║          torch torchvision --index-url \                                     ║
║          https://download.pytorch.org/whl/cpu                               ║
║                                                                              ║
║  HARDWARE WIRING (AGD307 10-way RS422 → FTDI CA-250 → USB):                 ║
║    Orange  (RS422 TXZ) → FTDI RX-                                           ║
║    Pink    (RS422 TXY) → FTDI RX+                                           ║
║    Brown   (RS422 RXA) → FTDI TX-                                           ║
║    Violet  (RS422 RXB) → FTDI TX+                                           ║
║    Green   (Ground)    → FTDI GND                                           ║
║    Red     (12/24 Vdc) → External PSU +                                     ║
║    Black   (0V)        → External PSU −                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import sys
import time
import queue
import logging
import threading
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import serial
import cv2
import numpy as np
# Lazy-imported in respective classes to allow startup without GPU
# from ultralytics import YOLO
# import easyocr


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these values to match your setup
# ══════════════════════════════════════════════════════════════════════════════

SERIAL_PORT      = "/dev/ttyUSB0"   # FTDI adapter — verify with: ls /dev/ttyUSB*
BAUD_RATE        = 9600
DATA_BITS        = serial.EIGHTBITS      # = 8 — using named constant for clarity
PARITY           = serial.PARITY_NONE    # = 'N'
STOP_BITS        = serial.STOPBITS_ONE   # = 1
SERIAL_TIMEOUT   = 2.0              # seconds per read attempt

SPEED_LIMIT_KPH  = 20               # Trigger threshold (kph)
RADAR_LOW_SPEED  = 10               # Sent as *LOWSPEED= to radar (kph)
                                     # Radar ignores vehicles below this speed

CAMERA_INDEX     = 0                # USB webcam device index
CAMERA_WIDTH     = 1280             # Capture resolution
CAMERA_HEIGHT    = 720
COOLDOWN_SEC     = 5.0              # Min seconds between consecutive captures
                                     # (prevents burst-triggering on one vehicle)

# PROJECT_DIR = directory that contains this script (same dir where you ran
# install.sh, which created the models/ sub-folder there).
PROJECT_DIR      = Path(__file__).resolve().parent

BASE_DIR         = PROJECT_DIR                      # logs + captures live here
CAPTURE_DIR      = BASE_DIR / "captures"            # saved violation images
PLATE_LOG        = BASE_DIR / "detected_plates.txt"
APP_LOG          = BASE_DIR / "app.log"

# install.sh downloads the ANPR model to  <project>/models/yolov8_plate.pt
# and the generic fallback to             <project>/models/yolov8n.pt
# These paths must match exactly — do not rename the files.
YOLO_MODEL_PATH  = PROJECT_DIR / "models" / "yolov8_plate.pt"
YOLO_FALLBACK    = PROJECT_DIR / "models" / "yolov8n.pt"

# EasyOCR languages — 'en' covers all Latin-script Indian plates.
# Add 'hi' for Hindi text if needed (increases load time).
OCR_LANGUAGES    = ["en"]

# Indian number plate regex — XX00XX0000 or XX-00-XX-0000 variants
INDIAN_PLATE_RE  = re.compile(
    r"[A-Z]{2}[\s\-]?"          # State code  e.g. MH
    r"\d{2}[\s\-]?"              # District    e.g. 12
    r"[A-Z]{1,3}[\s\-]?"        # Series      e.g. AB
    r"\d{4}",                    # Number      e.g. 1234
    re.IGNORECASE,
)

LOG_LEVEL = logging.INFO


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
#  RADAR INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class RadarInterface:
    """
    Manages all serial communication with the AGD307.

    Start-up sequence (per product manual):
      1. Open port at 9600 8N1
      2. Send 'AGD'      → verify response contains 'AGD'
      3. *LOWSPEED=N     → set lower speed cut-off
      4. *MS=5           → 'dd' format, 10 fps  (plain integer speed per line)
      5. *SAVE!          → commit to flash
    Speed lines arrive as plain decimal integers, e.g. "45\r\n" = 45 kph.
    """

    CMD_TERMINATOR = "\r"           # AGD307 expects CR-terminated commands
    INIT_DELAY     = 0.3            # seconds between init commands
    AGD_TIMEOUT    = 5.0            # seconds to wait for AGD handshake

    def __init__(self, log: logging.Logger):
        self.log  = log.getChild("Radar")
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open serial port and verify radar responds to AGD handshake."""
        try:
            self._ser = serial.Serial(
                port      = SERIAL_PORT,
                baudrate  = BAUD_RATE,
                bytesize  = DATA_BITS,
                parity    = PARITY,
                stopbits  = STOP_BITS,
                timeout   = self.AGD_TIMEOUT,
            )
            self.log.info(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE} baud")
        except serial.SerialException as exc:
            self.log.error(f"Cannot open {SERIAL_PORT}: {exc}")
            return False

        # Flush any stale bytes
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        time.sleep(0.5)

        # Handshake — send 'AGD', radar must reply with model/version line
        self.log.info("Sending AGD handshake …")
        self._send_raw("AGD")
        response = self._read_response(lines=2, timeout=self.AGD_TIMEOUT)

        if "AGD" not in response.upper():
            self.log.error(
                f"Handshake FAILED. Got: {response!r}\n"
                "Check wiring, baud rate, and that rotary switch is set to 0 (RS422 mode)."
            )
            return False

        self.log.info(f"Radar responded: {response.strip()!r}")
        return True

    # ── Initialisation commands ─────────────────────────────────────────────

    def initialise(self) -> bool:
        """Send configuration commands per manual: LOWSPEED, MS, SAVE."""
        commands = [
            (f"*LOWSPEED={RADAR_LOW_SPEED}",
             f"Set low-speed threshold to {RADAR_LOW_SPEED} kph"),
            ("*MS=5",
             "Set speed message format to 'dd' @ 10 fps"),
            ("*BIDI=0",
             "Advance-only detection"),
            ("*SAVE!",
             "Commit settings to flash"),
        ]

        for cmd, desc in commands:
            self.log.info(f"CMD  {cmd}  ({desc})")
            self._send_raw(cmd)
            time.sleep(self.INIT_DELAY)
            reply = self._read_response(lines=1, timeout=2.0)
            self.log.debug(f"   → {reply.strip()!r}")
            if "ERROR" in reply.upper():
                self.log.warning(f"Radar returned error for '{cmd}': {reply!r}")

        self.log.info("Radar initialised — now streaming speed data.")
        return True

    # ── Speed stream ─────────────────────────────────────────────────────────

    def read_speed_blocking(self) -> float | None:
        """
        Block until one speed value arrives.
        Returns speed in kph (float), or None if the line is non-numeric.
        *MS=5 produces "dd" lines: plain integer kph, e.g. "32\\r\\n"
        """
        if self._ser is None:
            return None
        try:
            raw = self._ser.readline()          # blocks up to SERIAL_TIMEOUT
            if not raw:
                return None
            text = raw.decode("ascii", errors="ignore").strip()
            if not text:
                return None
            # Accept lines that are purely digits (the 'dd' format)
            if re.fullmatch(r"\d+", text):
                speed = float(text)
                self.log.debug(f"Speed reading: {speed} kph")
                return speed
            # Some firmware versions prefix with '#' or status text.
            # Guard against error strings like 'ERROR 04' → ghost speed.
            if "ERROR" not in text.upper():
                digits = re.search(r"\d+", text)
                if digits:
                    return float(digits.group())
        except (serial.SerialException, UnicodeDecodeError) as exc:
            self.log.error(f"Serial read error: {exc}")
        return None

    # ── Query a parameter (diagnostic helper) ────────────────────────────────

    def query(self, param: str) -> str:
        self._send_raw(f"*{param.upper()}?")
        return self._read_response(lines=1, timeout=2.0)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _send_raw(self, text: str):
        with self._lock:
            payload = (text + self.CMD_TERMINATOR).encode("ascii")
            self._ser.write(payload)
            self._ser.flush()

    def _read_response(self, lines: int = 1, timeout: float = 2.0) -> str:
        """Read up to `lines` newline-terminated lines within `timeout` s."""
        if self._ser is None:
            return ""
        self._ser.timeout = timeout
        collected = []
        for _ in range(lines):
            raw = self._ser.readline()
            if raw:
                collected.append(raw.decode("ascii", errors="ignore"))
        self._ser.timeout = SERIAL_TIMEOUT
        return "".join(collected)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.log.info("Serial port closed.")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA CAPTURE
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
#  PLATE DETECTOR  (YOLOv8 + EasyOCR)
# ══════════════════════════════════════════════════════════════════════════════

class PlateDetector:
    """
    Two-stage pipeline:
      Stage 1 — YOLOv8 (license plate detection model) → bounding boxes
      Stage 2 — EasyOCR → text from each cropped plate region
    Falls back to full-frame OCR if no plate box is detected.

    Model auto-download:
      On first run the YOLOv8 license-plate model is downloaded from
      HuggingFace (keremberke/yolov8m-license-plate-detection).
      Subsequent runs load from YOLO_MODEL_PATH.
    """

    CONF_THRESHOLD = 0.35          # YOLO detection confidence minimum
    IOU_THRESHOLD  = 0.45          # NMS IoU threshold
    PADDING_PX     = 8             # Pixels to expand each detected box

    def __init__(self, log: logging.Logger):
        self.log        = log.getChild("Detector")
        self._yolo:     object | None = None
        self._ocr:      object | None = None
        # True when the fallback generic yolov8n model is loaded instead of
        # the ANPR model.  Controls class filtering inside _detect_plates.
        self._is_fallback: bool = False

    def load(self) -> bool:
        """Load YOLO model and EasyOCR reader. Call once at startup."""
        return self._load_yolo() and self._load_ocr()

    # ── YOLO ─────────────────────────────────────────────────────────────────

    def _load_yolo(self) -> bool:
        from ultralytics import YOLO

        if YOLO_MODEL_PATH.exists():
            self.log.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            self._yolo = YOLO(str(YOLO_MODEL_PATH))
        else:
            # Primary fallback: use the yolov8n.pt that install.sh pre-cached
            # in models/yolov8n.pt (vehicle detector; lower plate accuracy).
            self.log.warning(
                f"ANPR model not found at {YOLO_MODEL_PATH}. "
                "Expected install.sh to have downloaded it to models/yolov8_plate.pt. "
                "Attempting to re-download from GitHub …"
            )
            import urllib.request, shutil as _sh
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
                    f"Falling back to YOLOv8n at {YOLO_FALLBACK}. "
                    "Plate detection accuracy will be lower."
                )
                # yolov8n.pt was pre-cached by install.sh in models/
                if YOLO_FALLBACK.exists():
                    self._yolo = YOLO(str(YOLO_FALLBACK))
                else:
                    self._yolo = YOLO("yolov8n.pt")  # last resort ultralytics auto-dl
                self._is_fallback = True

        self.log.info("YOLO model ready.")
        return True

    # ── EasyOCR ──────────────────────────────────────────────────────────────

    def _load_ocr(self) -> bool:
        import easyocr
        self.log.info(f"Loading EasyOCR (languages: {OCR_LANGUAGES}) …")
        self._ocr = easyocr.Reader(
            OCR_LANGUAGES,
            gpu          = False,     # Set True if you have a GPU
            model_storage_directory = str(BASE_DIR / "easyocr_models"),
            download_enabled = True,
        )
        self.log.info("EasyOCR ready.")
        return True

    # ── Detection pipeline ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run the full plate-detection pipeline on a BGR frame.

        Returns a list of dicts:
          { 'plate_text': str, 'confidence': float, 'bbox': (x1,y1,x2,y2) }
        """
        results = []

        # ── Stage 1: Locate plate regions with YOLO ───────────────────────
        plate_crops, bboxes = self._detect_plates(frame)

        if not plate_crops:
            # No bounding boxes found — try OCR on the full frame
            self.log.debug("No plate boxes detected; running OCR on full frame.")
            plate_crops = [frame]
            bboxes      = [(0, 0, frame.shape[1], frame.shape[0])]

        # ── Stage 2: OCR on each crop ─────────────────────────────────────
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
        """Return list of cropped plate images and their bounding boxes."""
        crops, bboxes = [], []
        # Vehicle classes in the COCO-trained yolov8n fallback model
        _VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck
        try:
            preds = self._yolo.predict(
                frame,
                conf    = self.CONF_THRESHOLD,
                iou     = self.IOU_THRESHOLD,
                imgsz   = 640,   # caps RAM+inference time; avoid native 1280×720
                verbose = False,
            )
            h, w = frame.shape[:2]
            for result in preds:
                for box in result.boxes:
                    cls_id = int(box.cls[0])

                    if self._is_fallback:
                        # Generic model: only accept vehicle classes;
                        # skip people, traffic lights, signs, etc.
                        if cls_id not in _VEHICLE_CLASSES:
                            continue
                    else:
                        # ANPR model: class 0 = license plate,
                        # class 1 (if present) = vehicle body — skip it.
                        if cls_id != 0:
                            continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if self._is_fallback:
                        # Crop only the bottom 45 % of the vehicle bbox —
                        # that is where Indian plates are always mounted.
                        y1 = y1 + int((y2 - y1) * 0.55)

                    # Add padding, clamp to frame boundaries
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
        """
        Run EasyOCR on a single image crop.
        Returns (cleaned_text, confidence) or ("", 0.0) if nothing found.
        """
        try:
            # Pre-process for better OCR on Indian plates
            processed = self._preprocess_for_ocr(crop)
            ocr_out   = self._ocr.readtext(
                processed,
                detail          = 1,
                paragraph       = False,
                allowlist       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
            )
        except Exception as exc:
            self.log.error(f"OCR error: {exc}")
            return "", 0.0

        if not ocr_out:
            return "", 0.0

        # Concatenate all text fragments
        texts = [entry[1] for entry in ocr_out]
        confs = [entry[2] for entry in ocr_out]
        raw   = " ".join(texts).upper().strip()
        conf  = float(np.mean(confs)) if confs else 0.0

        cleaned = self._clean_plate_text(raw)
        self.log.debug(f"OCR raw={raw!r}  cleaned={cleaned!r}  conf={conf:.2f}")
        return cleaned, conf

    # ── Pre-processing ────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """
        Sharpen and threshold the plate crop to improve OCR on Indian plates.
        Works well for both white and yellow background plates.
        """
        # Upscale small crops — guard against degenerate zero-pixel crops
        # that YOLO can theoretically return on edge cases.
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop  # nothing useful to process
        if h < 40 or w < 120:
            scale  = max(40 / h, 120 / w, 1.0) * 2
            crop   = cv2.resize(crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold handles uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        # Mild dilation to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph  = cv2.dilate(thresh, kernel, iterations=1)
        return morph

    # ── Text cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """
        Remove spaces, hyphens; apply common OCR confusion corrections;
        validate against Indian plate pattern.
        """
        # Strip non-alphanumeric except hyphen/space
        cleaned = re.sub(r"[^A-Z0-9\- ]", "", text.upper())
        cleaned = cleaned.replace(" ", "").replace("-", "")

        # Common OCR confusion fixes for plates
        FIXES = {"0": "O", "1": "I", "5": "S"}   # only if in letter positions
        # Simple: keep as-is and rely on regex validation below

        # Try to match Indian plate pattern
        match = INDIAN_PLATE_RE.search(cleaned)
        if match:
            return match.group(0).upper()

        # Return as-is if it looks like a partial plate (≥4 chars)
        return cleaned if len(cleaned) >= 4 else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PLATE LOGGER
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

    def append(
        self,
        speed:      float,
        plate_text: str,
        confidence: float,
        image_path: Path,
    ):
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line  = (
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
#  WORKER THREAD  (capture + detect, offloaded from serial-read loop)
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
        # Save the raw capture image
        image_path = self._camera.save_frame(frame, speed)

        # Draw speed annotation on the saved copy
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"SPEED: {int(speed)} kph  [{datetime.now().strftime('%H:%M:%S')}]",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
        )

        # Run plate detection
        detections = self._detector.detect(annotated)

        if detections:
            for det in detections:
                plate = det["plate_text"]
                conf  = det["confidence"]
                bbox  = det["bbox"]

                # Draw bbox on annotated image
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, plate,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )

                self._logger.append(speed, plate, conf, image_path)

            # Save annotated image (overwrite raw)
            cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            self.log.info(f"No plate detected in frame for speed={speed:.0f} kph")
            self._logger.append_no_plate(speed, image_path)

    def stop(self):
        self._stop_evt.set()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class SpeedCameraApp:
    """Top-level orchestrator."""

    def __init__(self):
        self.log      = setup_logging()
        self._radar   = RadarInterface(self.log)
        self._camera  = CameraCapture(self.log)
        self._detector = PlateDetector(self.log)
        self._pl_log  = PlateLogger(self.log)
        self._worker  = ViolationWorker(
            self._detector, self._camera, self._pl_log, self.log
        )
        self._last_trigger = 0.0   # epoch time of last capture

    # ── Startup ──────────────────────────────────────────────────────────────

    def start(self):
        self.log.info("═" * 60)
        self.log.info("  AGD307 Speed Camera starting …")
        self.log.info(f"  Speed limit : {SPEED_LIMIT_KPH} kph")
        self.log.info(f"  Plate log   : {PLATE_LOG}")
        self.log.info(f"  Captures    : {CAPTURE_DIR}")
        self.log.info("═" * 60)

        # Validate serial port exists
        if not Path(SERIAL_PORT).exists():
            self.log.error(
                f"Serial port {SERIAL_PORT} not found. "
                "Check USB connection and run: ls /dev/ttyUSB*"
            )
            sys.exit(1)

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
            f"System ARMED — monitoring for vehicles above "
            f"{SPEED_LIMIT_KPH} kph …"
        )
        self._loop()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        try:
            while True:
                speed = self._radar.read_speed_blocking()
                if speed is None:
                    continue

                # Check against speed limit
                if speed > SPEED_LIMIT_KPH:
                    now = time.monotonic()
                    if now - self._last_trigger >= COOLDOWN_SEC:
                        self._last_trigger = now
                        self.log.warning(
                            f"⚡ SPEED VIOLATION: {speed:.0f} kph "
                            f"(limit {SPEED_LIMIT_KPH} kph)"
                        )
                        frame = self._camera.capture()
                        if frame is not None:
                            self._worker.enqueue(frame, speed)
                    else:
                        remaining = COOLDOWN_SEC - (now - self._last_trigger)
                        self.log.debug(
                            f"Speed {speed:.0f} kph exceeded limit but "
                            f"cooldown active ({remaining:.1f}s left)"
                        )

        except KeyboardInterrupt:
            self.log.info("Interrupted — shutting down …")
        finally:
            self._shutdown()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        self._worker.stop()
        self._worker.join(timeout=10)
        self._radar.close()
        self._camera.release()
        self.log.info("Speed Camera shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick diagnostic: list available serial ports
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device}  —  {p.description}")
    else:
        print("WARNING: No serial ports found. Check USB connection.")
    print()

    SpeedCameraApp().start()
