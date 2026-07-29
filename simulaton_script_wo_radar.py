#!/usr/bin/env python3
"""
AGD307 Speed Camera -- MANUAL TEST MODE (no radar required)
Raspberry Pi 5 | Indian ANPR (YOLOv8 + EasyOCR) | ESP32 LCD over Wi-Fi

Identical to Final_test_v5.py except:
  - RadarInterface removed entirely -- no serial port needed
  - Press ENTER to capture the current live frame and run the full
    ANPR + ESP32 pipeline on it
  - Type a speed integer before pressing Enter to set the logged speed
    (default: TEST_SPEED_KPH defined below)
  - Type 'q' + Enter to quit cleanly

Usage:
  python3 test_no_radar.py
"""

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

CAMERA_INDEX     = 0
CAMERA_WIDTH     = 1280
CAMERA_HEIGHT    = 720

# Speed value written to the log and sent to ESP32 for each manual capture.
# Change this to whatever speed you want to appear on the display during tests.
TEST_SPEED_KPH   = 35

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
#  ESP32 CONFIGURATION
# =============================================================================

ESP32_IP   = "192.168.x.x"   # Replace with the IP shown on LCD at boot
ESP32_PORT = 8080

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
#  CAMERA CAPTURE  -- continuous live reader thread (no buffer lag)
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
        name = f"capture_{ts}_{int(speed)}kph.jpg"
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
#  PLATE DETECTOR  -- YOLOv8 + EasyOCR
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
        "# Speed Camera Log -- Manual Test Mode\n"
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
#  ESP32 SENDER
# =============================================================================

class ESP32Sender:
    """
    Sends plate number and speed to the ESP32 TCP server over Wi-Fi.
    Message format: "KL01DA3021|80\n"
    """

    def __init__(self, ip: str, port: int, log: logging.Logger):
        self.ip   = ip
        self.port = port
        self.log  = log.getChild("ESP32")

    def send(self, plate: str, speed: float):
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
#  VIOLATION WORKER
# =============================================================================

class ViolationWorker(threading.Thread):
    def __init__(self, detector: PlateDetector, camera: CameraCapture,
                 logger: PlateLogger, esp32: ESP32Sender, log: logging.Logger):
        super().__init__(daemon=True, name="ViolationWorker")
        self._q        = queue.Queue(maxsize=4)
        self._detector = detector
        self._camera   = camera
        self._logger   = logger
        self._esp32    = esp32
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

        # Run detection on the CLEAN frame -- before any text overlays are drawn.
        # If detect() ran on the annotated frame, full-frame OCR fallback would
        # read the speed overlay text and return it as a false plate match.
        detections = self._detector.detect(frame)

        # Build annotated copy for the saved output image.
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"SPEED: {int(speed)} kph  [{datetime.now().strftime('%H:%M:%S')}]",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,
        )

        if detections:
            for det in detections:
                plate = det["plate_text"]
                conf  = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, plate, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self._logger.append(speed, plate, conf, image_path)
                self._esp32.send(plate, speed)
            cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            self.log.info(f"No plate detected (speed={speed:.0f} kph)")
            cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._logger.append_no_plate(speed, image_path)

    def stop(self):
        self._stop_evt.set()

# =============================================================================
#  MAIN APPLICATION  -- manual Enter-triggered mode
# =============================================================================

class ManualTestApp:
    """
    No-radar test mode.
    Press Enter to capture the current live frame and run the full pipeline.
    Optionally type a speed integer before Enter to override TEST_SPEED_KPH.
    Type 'q' + Enter to quit.
    """

    def __init__(self):
        self.log       = setup_logging()
        self._camera   = CameraCapture(self.log)
        self._detector = PlateDetector(self.log)
        self._pl_log   = PlateLogger(self.log)
        self._esp32    = ESP32Sender(ESP32_IP, ESP32_PORT, self.log)
        self._worker   = ViolationWorker(
            self._detector, self._camera, self._pl_log,
            self._esp32, self.log
        )

    def start(self):
        self.log.info("=" * 64)
        self.log.info("  Speed Camera -- MANUAL TEST MODE  (no radar)")
        self.log.info(f"  Default speed : {TEST_SPEED_KPH} kph")
        self.log.info(f"  Plate log     : {PLATE_LOG}")
        self.log.info(f"  Captures      : {CAPTURE_DIR}")
        self.log.info(f"  ESP32 target  : {ESP32_IP}:{ESP32_PORT}")
        self.log.info("=" * 64)

        if not self._camera.open():
            sys.exit(1)
        if not self._detector.load():
            sys.exit(1)

        self._worker.start()
        self.log.info("System ready.")
        self._loop()

    def _loop(self):
        print()
        print("  Press ENTER to capture and process the current frame.")
        print("  Type a speed (integer) then ENTER to use a custom speed.")
        print("  Type 'q' then ENTER to quit.")
        print()

        try:
            while True:
                try:
                    raw = input("  > ").strip()
                except EOFError:
                    break

                if raw.lower() == "q":
                    break

                # Parse optional speed override
                speed = float(TEST_SPEED_KPH)
                if raw:
                    try:
                        speed = float(raw)
                    except ValueError:
                        print(f"  Unrecognised input '{raw}' -- "
                              f"using default {TEST_SPEED_KPH} kph.")

                # Capture instantaneous frame from live reader
                frame = self._camera.capture()
                if frame is None:
                    self.log.error("Failed to capture frame.")
                    continue

                self.log.info(
                    f"Manual capture triggered at {speed:.0f} kph -- "
                    "processing ..."
                )
                self._worker.enqueue(frame, speed)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        self.log.info("Shutting down ...")
        self._worker.stop()
        self._worker.join(timeout=10)
        self._camera.release()
        self.log.info("Shutdown complete.")

# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ManualTestApp().start()
