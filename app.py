#!/usr/bin/env python3
"""AGD radar -> overspeed trigger -> image capture -> plate detection + OCR logger.

Designed for Raspberry Pi 5 receiving AGD-307 radar data over RS-422 via USB serial.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import importlib
from typing import Any



SPEED_REGEX = re.compile(r"[-+]?\d*\.?\d+")


@dataclass
class AppConfig:
    serial_port: str
    baud_rate: int
    speed_limit: float
    trigger_cooldown_s: float
    output_text_file: Path
    image_output_dir: Path
    yolo_model_path: Path
    yolo_confidence: float
    camera_w: int
    camera_h: int
    tesseract_psm: int


class SpeedCameraApp:
    def _load_runtime_dependencies(self) -> None:
        try:
            self.serial_module = importlib.import_module("serial")
            self.cv2 = importlib.import_module("cv2")
            self.pytesseract = importlib.import_module("pytesseract")
            self.picamera2 = importlib.import_module("picamera2")
            ultralytics = importlib.import_module("ultralytics")
            self.detector = ultralytics.YOLO(str(self.cfg.yolo_model_path))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing runtime dependency: {exc.name}. Install from README and retry."
            ) from exc

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.running = True
        self.last_trigger_ts = 0.0

        self.serial_conn: Optional[Any] = None
        self.serial_module: Any = None
        self.camera: Optional[Any] = None

        self.cv2: Any = None
        self.pytesseract: Any = None
        self.detector: Any = None

    def setup(self) -> None:
        self.cfg.output_text_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.image_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.yolo_model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.cfg.yolo_model_path}. "
                "Put your trained model there or pass --yolo-model-path."
            )

        self._load_runtime_dependencies()

        logging.info("Opening serial port %s @ %d", self.cfg.serial_port, self.cfg.baud_rate)
        self.serial_conn = self.serial_module.Serial(
            port=self.cfg.serial_port,
            baudrate=self.cfg.baud_rate,
            timeout=1.0,
        )

        logging.info("Initializing Pi camera")
        self.camera = self.picamera2.Picamera2()
        cam_cfg = self.camera.create_still_configuration(
            main={"size": (self.cfg.camera_w, self.cfg.camera_h), "format": "RGB888"}
        )
        self.camera.configure(cam_cfg)
        self.camera.start()
        time.sleep(1.0)

    def shutdown(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None
        if self.serial_conn is not None:
            self.serial_conn.close()
            self.serial_conn = None

    def run(self) -> None:
        self.setup()
        logging.info("App started. Waiting for speed data...")

        while self.running:
            assert self.serial_conn is not None
            try:
                raw_line = self.serial_conn.readline()
                if not raw_line:
                    continue

                decoded = raw_line.decode("utf-8", errors="ignore").strip()
                speed = self.extract_speed(decoded)
                if speed is None:
                    continue

                logging.info("Radar speed: %.2f km/h | raw='%s'", speed, decoded)

                if speed > self.cfg.speed_limit:
                    self.handle_overspeed(speed)
            except Exception as exc:
                if self.serial_module and isinstance(exc, self.serial_module.SerialException):
                    logging.exception("Serial error")
                    time.sleep(1.0)
                    continue
                logging.exception("Unhandled exception in main loop")
                time.sleep(0.5)

        self.shutdown()

    def extract_speed(self, line: str) -> Optional[float]:
        """Extract numeric speed from radar line.

        Current rule: use the last numeric token in the line.
        """
        matches = SPEED_REGEX.findall(line)
        if not matches:
            return None

        try:
            return float(matches[-1])
        except ValueError:
            return None

    def handle_overspeed(self, speed: float) -> None:
        now = time.time()
        if now - self.last_trigger_ts < self.cfg.trigger_cooldown_s:
            logging.debug("Skipping trigger due to cooldown")
            return

        self.last_trigger_ts = now

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.cfg.image_output_dir / f"overspeed_{ts}_{int(speed)}.jpg"

        frame = self.capture_frame()
        self.cv2.imwrite(str(image_path), self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
        logging.info("Captured image: %s", image_path)

        plate_text = self.detect_and_read_plate(frame)
        if not plate_text:
            plate_text = "<NO_PLATE_FOUND>"

        self.append_result(speed=speed, image_path=image_path, plate_text=plate_text)
        logging.info("Logged plate: %s", plate_text)

    def capture_frame(self) -> Any:
        assert self.camera is not None
        frame = self.camera.capture_array("main")
        if frame is None:
            raise RuntimeError("Camera returned empty frame")
        return frame

    def detect_and_read_plate(self, rgb_frame: Any) -> str:
        """Run YOLO plate detection then OCR on best crop using Tesseract."""
        results = self.detector.predict(
            source=rgb_frame,
            conf=self.cfg.yolo_confidence,
            verbose=False,
        )

        plate_crop = None
        best_area = 0

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for x1, y1, x2, y2 in boxes:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(rgb_frame.shape[1], x2)
                y2 = min(rgb_frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    plate_crop = rgb_frame[y1:y2, x1:x2]

        if plate_crop is None:
            plate_crop = rgb_frame

        # OCR pre-processing for plate-like text
        gray = self.cv2.cvtColor(plate_crop, self.cv2.COLOR_RGB2GRAY)
        gray = self.cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = self.cv2.threshold(gray, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)

        text = self.pytesseract.image_to_string(
            thresh,
            config=f"--oem 3 --psm {self.cfg.tesseract_psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return cleaned

    def append_result(self, speed: float, image_path: Path, plate_text: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp},speed_kmph={speed:.2f},plate={plate_text},image={image_path}\n"
        with self.cfg.output_text_file.open("a", encoding="utf-8") as f:
            f.write(line)


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="AGD radar overspeed number plate logger")
    parser.add_argument("--serial-port", default=os.getenv("RADAR_SERIAL_PORT", "/dev/ttyUSB0"))
    parser.add_argument("--baud-rate", type=int, default=int(os.getenv("RADAR_BAUD_RATE", "115200")))
    parser.add_argument("--speed-limit", type=float, default=float(os.getenv("SPEED_LIMIT", "40")))
    parser.add_argument("--trigger-cooldown-s", type=float, default=float(os.getenv("TRIGGER_COOLDOWN_S", "2.0")))
    parser.add_argument("--output-text-file", default=os.getenv("OUTPUT_TEXT_FILE", "data/detected_plates.txt"))
    parser.add_argument("--image-output-dir", default=os.getenv("IMAGE_OUTPUT_DIR", "data/captures"))
    parser.add_argument("--yolo-model-path", default=os.getenv("YOLO_MODEL_PATH", "models/indian_lp_yolov8.pt"))
    parser.add_argument("--yolo-confidence", type=float, default=float(os.getenv("YOLO_CONFIDENCE", "0.25")))
    parser.add_argument("--camera-w", type=int, default=int(os.getenv("CAMERA_W", "1920")))
    parser.add_argument("--camera-h", type=int, default=int(os.getenv("CAMERA_H", "1080")))
    parser.add_argument("--tesseract-psm", type=int, default=int(os.getenv("TESSERACT_PSM", "7")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    return AppConfig(
        serial_port=args.serial_port,
        baud_rate=args.baud_rate,
        speed_limit=args.speed_limit,
        trigger_cooldown_s=args.trigger_cooldown_s,
        output_text_file=Path(args.output_text_file),
        image_output_dir=Path(args.image_output_dir),
        yolo_model_path=Path(args.yolo_model_path),
        yolo_confidence=args.yolo_confidence,
        camera_w=args.camera_w,
        camera_h=args.camera_h,
        tesseract_psm=args.tesseract_psm,
    )


def main() -> int:
    cfg = parse_args()
    app = SpeedCameraApp(cfg)

    def stop_handler(signum, _frame) -> None:
        logging.info("Received signal %s, stopping...", signum)
        app.running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
