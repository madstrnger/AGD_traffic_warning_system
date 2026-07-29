#!/usr/bin/env python3
"""
AGD307 Speed Camera -- REAL RADAR MODE
Raspberry Pi 5 | RS-422 via FTDI CA-250 | Indian ANPR (YOLOv8 + EasyOCR)
ESP32 LCD display over Wi-Fi (TCP socket)

Integrates:
  - RadarInterface  : from test_radar.py (tested on hardware, CR fix included)
  - CameraCapture   : from speed_camera.py (CAP_V4L2, warm-up, buffer flush)
  - PlateDetector   : from speed_camera.py (YOLOv8 + EasyOCR, class filter)
  - PlateLogger     : from speed_camera.py (append-only audit log)
  - ViolationWorker : from speed_camera.py (async daemon queue, maxsize=4)
  - ESP32Sender     : NEW -- sends plate + speed to ESP32 over Wi-Fi TCP

Usage:
  python3 final_test_v4.py                        # /dev/ttyUSB0 @ 9600
  python3 final_test_v4.py --port /dev/ttyUSB1
  python3 final_test_v4.py --port /dev/ttyUSB0 --baud 9600
"""

import argparse
import os
import queue
import re
import sys
import time
import socket
import threading
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
#  CONFIGURATION
# =============================================================================

SERIAL_PORT      = "/dev/ttyUSB0"      # FTDI adapter -- verify: ls /dev/ttyUSB*
BAUD_RATE        = 9600

SPEED_LIMIT_KPH  = 20                  # Capture trigger threshold (kph)
RADAR_LOW_SPEED  = 10                  # *LOWSPEED= value sent to radar (kph)

CAMERA_INDEX     = 0
CAMERA_WIDTH     = 1280
CAMERA_HEIGHT    = 720
COOLDOWN_SEC     = 5.0
CAPTURE_DELAY_MS = 0

PROJECT_DIR      = Path(__file__).resolve().parent
BASE_DIR         = PROJECT_DIR
CAPTURE_DIR      = BASE_DIR / "captures"
PLATE_LOG        = BASE_DIR / "detected_plates.txt"
APP_LOG          = BASE_DIR / "app.log"

YOLO_MODEL_PATH  = PROJECT_DIR / "models" / "yolov8_plate.pt"
YOLO_FALLBACK    = PROJECT_DIR / "models" / "yolov8n.pt"
OCR_LANGUAGES    = ["en"]

INDIAN_PLATE_RE  = re.compile(
    r"[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}",
    re.IGNORECASE,
)

LOG_LEVEL = logging.INFO

# =============================================================================
#  ESP32 CONFIGURATION  ← only thing you need to update
# =============================================================================

ESP32_IP   = "192.168.x.x"   # ← Replace with the IP shown on LCD at boot
ESP32_PORT = 8080

# =============================================================================
#  PYSERIAL IMPORT
# =============================================================================

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed.  Run:  pip install pyserial")
    sys.exit(1)

# =============================================================================
#  SPEED LINE PARSER
# =============================================================================

_MS5_RE = re.compile(r"^(\d{2,3})$")
_MS6_RE = re.compile(r"^\*S(\d{3})$")
_SPD_RE = re.compile(r"^SPD:(\d{1,3}(?:\.\d{1,2})?)$")

def parse_speed(line: str) -> float | None:
    if "ERROR" in line.upper():
        return None
    if line.startswith("#") or (line.startswith("*") and not line.startswith("*S")):
        return None
    for pattern in (_MS5_RE, _MS6_RE, _SPD_RE):
        m = pattern.match(line)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 300.0:
                    return val
            except ValueError:
                pass
    return None

# =============================================================================
#  RADAR INTERFACE
# =============================================================================

class RadarInterface:
    def __init__(self, port: str, baud: int, log: logging.Logger):
        self.port   = port
        self.baud   = baud
        self.log    = log.getChild("Radar")
        self.ser    = None
        self.buffer = ""

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port     = self.port,
                baudrate = self.baud,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = 1.0,
                xonxoff  = False,
                rtscts   = False,
                dsrdtr   = False,
            )
            self.log.info(f"Serial port {self.port} open at {self.baud} baud.")
            return True
        except serial.SerialException as exc:
            self.log.error(f"Cannot open {self.port}: {exc}")
            return False

    def initialise(self) -> bool:
        def send_cmd(cmd: str, wait: float = 0.4) -> str:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + "\r").encode("ascii"))
            self.ser.flush()
            time.sleep(wait)
            raw  = self.ser.read(self.ser.in_waiting).decode("ascii", errors="ignore")
            resp = raw.strip().replace("\r\n", " | ").replace("\r", " ").replace("\n", " ")
            self.log.info(f"  CMD {cmd:<20} -> {resp if resp else '(no response)'}")
            return resp

        self.log.info("Sending AGD identity handshake ...")
        resp = send_cmd("AGD", wait=1.0)
        if "AGD" not in resp.upper():
            self.log.error(f"Handshake FAILED. Got: {resp!r}")
            return False
        self.log.info(f"Radar identified: {resp!r}")

        send_cmd("*BAUD?")
        send_cmd("*LOWSPEED?")
        send_cmd("*MS?")
        send_cmd("*MM?")
        send_cmd(f"*LOWSPEED={RADAR_LOW_SPEED}")
        send_cmd("*MS=5")
        send_cmd("*BIDI=0")
        send_cmd("*SAVE!")

        self.log.info("Radar initialised -- speed streaming active.")
        return True

    def read_speed_blocking(self) -> float | None:
        while True:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1).decode(
                    "ascii", errors="replace"
                )
            except serial.SerialException as exc:
                self.log.error(f"Serial read error: {exc}")
                time.sleep(0.5)
                return None

            if not chunk:
                continue

            self.buffer += chunk
            self.buffer = self.buffer.replace("\r\n", "\n").replace("\r", "\n")

            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                speed = parse_speed(line)
                if speed is not None:
                    self.log.debug(f"Speed: {speed:.0f} kph")
                    return speed

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log.info("Radar serial port closed.")

# =============================================================================
#  LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
    logging.basicConfig(
        level    = LOG_LEVEL,
        format   = fmt,
        handlers = [
            logging.FileHandler(APP_LOG),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("SpeedCamera")

# =============================================================================
#  CAMERA CAPTURE
# =============================================================================

class CameraCapture:
    def __init__(self, log: logging.Logger):
        self.log         = log.getChild("Camera")
        self._cap        = None
        self._frame      = None
        self._lock       = threading.Lock()
        self._stop_evt   = threading.Event()
        self._thread     = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self.log.warning("V4L2 backend failed -- retrying with auto-detection.")
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.log.error(f"Cannot open camera at index {CAMERA_INDEX}.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(10):
            self._cap.read()

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._reader,
            daemon=True,
            name="CameraLiveReader",
        )
        self._thread.start()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.01)
        else:
            self.log.error("Camera reader thread produced no frames within 3 s.")
            return False

        self.log.info(
            f"Camera {CAMERA_INDEX} live ({CAMERA_WIDTH}x{CAMERA_HEIGHT}) -- "
            "continuous reader thread running."
        )
        return True

    def capture(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                self.log.error("Camera not ready -- no frame available yet.")
                return None
            return self._frame.copy()

    def save_frame(self, frame: np.ndarray, speed: float) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"violation_{ts}_{int(speed)}kph.jpg"
        path = CAPTURE_DIR / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.log.info(f"Frame saved -> {path}")
        return path

    def release(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self.log.info("Camera released.")

    def _reader(self):
        while not self._stop_evt.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

# =============================================================================
#  PLATE DETECTOR
# =============================================================================

class PlateDetector:
    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD  = 0.45
    PADDING_PX     = 8

    def __init__(self, log: logging.Logger):
        self.log           = log.getChild("Detector")
        self._yolo         = None
        self._ocr          = None
        self._is_fallback: bool = False

    def load(self) -> bool:
        return self._load_yolo() and self._load_ocr()

    def _load_yolo(self) -> bool:
        from ultralytics import YOLO
        if YOLO_MODEL_PATH.exists():
            self.log.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            self._yolo = YOLO(str(YOLO_MODEL_PATH))
        else:
            self.log.warning(
                f"ANPR model not found at {YOLO_MODEL_PATH}. "
                "Attempting re-download from GitHub ..."
            )
            import urllib.request
            _URL = (
                "https://github.com/Muhammad-Zeerak-Khan/"
                "Automatic-License-Plate-Recognition-using-YOLOv8/"
                "raw/main/license_plate_detector.pt"
            )
            try:
                YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(_URL, YOLO_MODEL_PATH)
                self._yolo = YOLO(str(YOLO_MODEL_PATH))
                self.log.info(f"ANPR weights saved -> {YOLO_MODEL_PATH}")
            except Exception as exc:
                self.log.warning(
                    f"Download failed ({exc}). "
                    f"Falling back to yolov8n at {YOLO_FALLBACK}."
                )
                if YOLO_FALLBACK.exists():
                    self._yolo = YOLO(str(YOLO_FALLBACK))
                else:
                    self._yolo = YOLO("yolov8n.pt")
                self._is_fallback = True
        self.log.info(f"YOLO ready (fallback={self._is_fallback})")
        return True

    def _load_ocr(self) -> bool:
        import easyocr
        self.log.info(f"Loading EasyOCR ({OCR_LANGUAGES}) ...")
        self._ocr = easyocr.Reader(
            OCR_LANGUAGES,
            gpu                     = False,
            model_storage_directory = str(BASE_DIR / "easyocr_models"),
            download_enabled        = True,
        )
        self.log.info("EasyOCR ready.")
        return True

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = []
        crops, bboxes = self._detect_plates(frame)
        if not crops:
            self.log.debug("No plate boxes -- running OCR on full frame.")
            crops  = [frame]
            bboxes = [(0, 0, frame.shape[1], frame.shape[0])]
        for crop, bbox in zip(crops, bboxes):
            text, conf = self._ocr_crop(crop)
            if text:
                results.append({"plate_text": text, "confidence": conf, "bbox": bbox})
        return results

    def _detect_plates(self, frame):
        crops, bboxes = [], []
        _VEHICLE_CLASSES = {2, 3, 5, 7}
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

    def _ocr_crop(self, crop):
        try:
            proc = self._preprocess(crop)
            out  = self._ocr.readtext(
                proc,
                detail         = 1,
                paragraph      = False,
                allowlist      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
                text_threshold = 0.6,
                width_ths      = 0.9,
                adjust_contrast= 0.5,
            )
        except Exception as exc:
            self.log.error(f"OCR error: {exc}")
            return "", 0.0
        if not out:
            return "", 0.0
        texts   = [e[1] for e in out]
        confs   = [e[2] for e in out]
        raw     = " ".join(texts).upper().strip()
        conf    = float(np.mean(confs)) if confs else 0.0
        cleaned = self._clean(raw)
        self.log.debug(f"OCR raw={raw!r} cleaned={cleaned!r} conf={conf:.2f}")
        return cleaned, conf

    @staticmethod
    def _preprocess(crop):
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        if h < 64 or w < 128:
            scale = max(64 / h, 128 / w, 1.0) * 2
            crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        lab     = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l       = clahe.apply(l)
        crop    = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return crop

    @staticmethod
    def _clean(text):
        cleaned = re.sub(r"[^A-Z0-9\- ]", "", text.upper())
        cleaned = cleaned.replace(" ", "").replace("-", "")

        DIGIT_TO_LETTER = str.maketrans("0125", "OIZS")
        LETTER_TO_DIGIT = str.maketrans("OIZS", "0125")

        m = INDIAN_PLATE_RE.search(cleaned)
        if m:
            return m.group(0).upper()

        if len(cleaned) >= 8:
            chars = list(cleaned)
            for i in (0, 1):
                if i < len(chars) and chars[i].isdigit():
                    chars[i] = chars[i].translate(DIGIT_TO_LETTER)
            for i in (2, 3):
                if i < len(chars) and chars[i].isalpha():
                    chars[i] = chars[i].translate(LETTER_TO_DIGIT)
            for i in range(max(0, len(chars) - 4), len(chars)):
                if chars[i].isalpha():
                    chars[i] = chars[i].translate(LETTER_TO_DIGIT)
            for i in range(4, max(4, len(chars) - 4)):
                if chars[i].isdigit():
                    chars[i] = chars[i].translate(DIGIT_TO_LETTER)
            corrected = "".join(chars)
            m = INDIAN_PLATE_RE.search(corrected)
            if m:
                return m.group(0).upper()

        return cleaned if len(cleaned) >= 4 else ""

# =============================================================================
#  PLATE LOGGER
# =============================================================================

class PlateLogger:
    _HEADER = (
        "# AGD307 Speed Camera Log\n"
        "# Format: TIMESTAMP | SPEED_KPH | PLATE_TEXT | CONFIDENCE | IMAGE\n"
        "#" + "-" * 72 + "\n"
    )

    def __init__(self, log: logging.Logger):
        self.log = log.getChild("Logger")
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
        self.log.info(f"LOGGED -> {line.rstrip()}")

    def append_no_plate(self, speed: float, image_path: Path):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = (
            f"{ts} | {speed:6.1f} kph | {'NO_PLATE':<15} "
            f"| conf=0.00 | {image_path.name}\n"
        )
        with open(PLATE_LOG, "a") as f:
            f.write(line)
        self.log.info(f"LOGGED (no plate) -> {line.rstrip()}")

# =============================================================================
#  ESP32 SENDER  ← NEW: sends plate number + speed to ESP32 over Wi-Fi
# =============================================================================

class ESP32Sender:
    """
    Sends plate number and speed to the ESP32 TCP server over Wi-Fi.
    The ESP32 displays the data on the JHD162A LCD.

    Message format sent:  "KL01DA3021|80\n"
    LCD row 1 shows:      "KL01 DA 3021    "   (plate number)
    LCD row 2 shows:      "80kmph SLOW DOWN"   (speed warning)
    """

    def __init__(self, ip: str, port: int, log: logging.Logger):
        self.ip   = ip
        self.port = port
        self.log  = log.getChild("ESP32")

    def send(self, plate: str, speed: float):
        """
        Call this after every successful plate detection.
        plate : string like "KL01DA3021"
        speed : float like 87.0
        """
        message = f"{plate}|{int(speed)}\n"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((self.ip, self.port))
                s.sendall(message.encode("utf-8"))
            self.log.info(f"[ESP32] Sent -> {message.strip()}")
        except socket.timeout:
            self.log.warning("[ESP32] Timeout -- ESP32 not responding. Check IP.")
        except ConnectionRefusedError:
            self.log.warning("[ESP32] Connection refused -- is ESP32 powered on?")
        except OSError as exc:
            self.log.warning(f"[ESP32] Send failed: {exc}")

# =============================================================================
#  VIOLATION WORKER  ← _process() updated to call ESP32Sender
# =============================================================================

class ViolationWorker(threading.Thread):
    """
    Async daemon thread. Receives (frame, speed) from a bounded queue
    and runs the full ANPR pipeline so the radar read loop is never blocked.
    """

    def __init__(self, detector: PlateDetector, camera: CameraCapture,
                 logger: PlateLogger, esp32: ESP32Sender, log: logging.Logger):
        super().__init__(daemon=True, name="ViolationWorker")
        self._q        = queue.Queue(maxsize=4)
        self._detector = detector
        self._camera   = camera
        self._logger   = logger
        self._esp32    = esp32          # ← NEW
        self.log       = log.getChild("Worker")
        self._stop_evt = threading.Event()

    def enqueue(self, frame: np.ndarray, speed: float):
        try:
            self._q.put_nowait((frame, speed))
        except queue.Full:
            self.log.warning("Worker queue full -- dropping frame.")

    def run(self):
        self.log.info("ViolationWorker started.")
        while not self._stop_evt.is_set():
            try:
                frame, speed = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._process(frame, speed)

    def _process(self, frame: np.ndarray, speed: float):
        image_path = self._camera.save_frame(frame, speed)
        annotated  = frame.copy()
        cv2.putText(
            annotated,
            f"SPEED: {int(speed)} kph  [{datetime.now().strftime('%H:%M:%S')}]",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,
        )
        detections = self._detector.detect(annotated)
        if detections:
            for det in detections:
                plate = det["plate_text"]
                conf  = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, plate, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self._logger.append(speed, plate, conf, image_path)

                # ── NEW: send plate + speed to ESP32 LCD ──────────────────────
                self._esp32.send(plate, speed)
                # ──────────────────────────────────────────────────────────────

            cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            self.log.info(f"No plate detected (speed={speed:.0f} kph)")
            self._logger.append_no_plate(speed, image_path)

    def stop(self):
        self._stop_evt.set()

# =============================================================================
#  MAIN APPLICATION
# =============================================================================

class SpeedCameraRealRadarApp:
    """Orchestrator -- real AGD307 radar, full ML pipeline."""

    def __init__(self, port: str, baud: int):
        self.log       = setup_logging()
        self._radar    = RadarInterface(port, baud, self.log)
        self._camera   = CameraCapture(self.log)
        self._detector = PlateDetector(self.log)
        self._pl_log   = PlateLogger(self.log)
        self._esp32    = ESP32Sender(ESP32_IP, ESP32_PORT, self.log)  # ← NEW
        self._worker   = ViolationWorker(
            self._detector, self._camera, self._pl_log,
            self._esp32, self.log                                     # ← NEW
        )
        self._last_trigger = 0.0

    def start(self):
        self.log.info("=" * 64)
        self.log.info("  AGD307 Speed Camera -- REAL RADAR MODE")
        self.log.info(f"  Port         : {self._radar.port}")
        self.log.info(f"  Baud rate    : {self._radar.baud}")
        self.log.info(f"  Speed limit  : {SPEED_LIMIT_KPH} kph")
        self.log.info(f"  Low-speed    : {RADAR_LOW_SPEED} kph (*LOWSPEED)")
        self.log.info(f"  Plate log    : {PLATE_LOG}")
        self.log.info(f"  Captures     : {CAPTURE_DIR}")
        self.log.info(f"  ESP32 target : {ESP32_IP}:{ESP32_PORT}")  # ← NEW
        self.log.info("=" * 64)

        if not Path(self._radar.port).exists():
            self.log.error(
                f"Serial port {self._radar.port} not found. "
                "Check USB connection: ls /dev/ttyUSB*"
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
            f"[REAL] System ARMED -- monitoring for speeds above "
            f"{SPEED_LIMIT_KPH} kph ..."
        )
        self._loop()

    def _loop(self):
        try:
            while True:
                speed = self._radar.read_speed_blocking()
                if speed is None:
                    time.sleep(1.0)
                    continue
                if speed > SPEED_LIMIT_KPH:
                    now = time.monotonic()
                    if now - self._last_trigger >= COOLDOWN_SEC:
                        self._last_trigger = now
                        self.log.warning(
                            f"[REAL] SPEED VIOLATION: {speed:.0f} kph "
                            f"(limit {SPEED_LIMIT_KPH} kph)"
                        )
                        if CAPTURE_DELAY_MS > 0:
                            time.sleep(CAPTURE_DELAY_MS / 1000.0)
                        frame = self._camera.capture()
                        if frame is not None:
                            self._worker.enqueue(frame, speed)
                    else:
                        remaining = COOLDOWN_SEC - (now - self._last_trigger)
                        self.log.debug(
                            f"Speed {speed:.0f} kph -- cooldown "
                            f"({remaining:.1f}s left)"
                        )
        except KeyboardInterrupt:
            self.log.info("Interrupted -- shutting down ...")
        finally:
            self._shutdown()

    def _shutdown(self):
        self._worker.stop()
        self._worker.join(timeout=10)
        self._radar.close()
        self._camera.release()
        self.log.info("System shut down cleanly.")

# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device}  --  {p.description}")
    else:
        print("WARNING: No serial ports detected. Check USB connection.")
    print()

    parser = argparse.ArgumentParser(
        description="AGD307 Speed Camera -- Real Radar Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 final_test_v4.py\n"
            "  python3 final_test_v4.py --port /dev/ttyUSB1\n"
            "  python3 final_test_v4.py --port /dev/ttyUSB0 --baud 9600\n"
        ),
    )
    parser.add_argument(
        "--port", default=SERIAL_PORT,
        help=f"Serial port (default: {SERIAL_PORT})"
    )
    parser.add_argument(
        "--baud", default=BAUD_RATE, type=int,
        help=f"Baud rate (default: {BAUD_RATE})"
    )
    args = parser.parse_args()

    SpeedCameraRealRadarApp(port=args.port, baud=args.baud).start()