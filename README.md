# AGD-307 Radar Overspeed Plate Logger (Raspberry Pi 5)

This project provides a complete Python application for Raspberry Pi 5 that:
1. Reads AGD-307 radar speed values from RS-422 via FTDI USB-serial.
2. Triggers image capture when speed exceeds a configured limit.
3. Runs YOLOv8 number-plate detection.
4. Runs OCR on the plate region.
5. Appends detected plate text to a persistent text file.

## Why this revision is safer (dependency conflict mitigation)
The previous stack mixed packages that commonly conflict on Raspberry Pi (notably venv + apt-only `picamera2`, and OCR stacks that require fragile platform wheels). This version mitigates those issues by:
- Using **Tesseract OCR + pytesseract** (stable on Pi via apt).
- Keeping **YOLOv8** for detection.
- Requiring venv with `--system-site-packages` so apt-installed `picamera2` is visible.
- Listing exact tested versions and install order.

---

## 1) Hardware assumptions
- Raspberry Pi 5 (Raspberry Pi OS Bookworm 64-bit recommended)
- AGD-307 traffic radar (RS-422 output)
- FTDI-based RS-422 to USB converter
- Raspberry Pi Camera Module (libcamera / picamera2 compatible)

---

## 2) OS packages (install first)

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv python3-dev \
  python3-picamera2 tesseract-ocr libtesseract-dev \
  libatlas-base-dev libopenblas-dev libjpeg-dev libtiff5 libpng-dev
```

> We intentionally install `python3-picamera2` from apt (recommended on Pi OS).

---

## 3) Python environment and pinned dependencies

```bash
cd /workspace/AGD_traffic_warning_system
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Supported dependency versions
- Python: **3.11.x**
- numpy: **1.26.4**
- opencv-python: **4.10.0.84**
- pyserial: **3.5**
- ultralytics: **8.3.0**
- pytesseract: **0.3.10**
- picamera2: apt package `python3-picamera2` (system package)
- tesseract binary: apt package `tesseract-ocr`

---

## 4) YOLO model file
Place your Indian number-plate detector model at:

```text
models/indian_lp_yolov8.pt
```

You can change the path via `--yolo-model-path`.

---

## 5) Run the application

```bash
source .venv/bin/activate
python app.py \
  --serial-port /dev/ttyUSB0 \
  --baud-rate 115200 \
  --speed-limit 40 \
  --output-text-file data/detected_plates.txt \
  --image-output-dir data/captures \
  --yolo-model-path models/indian_lp_yolov8.pt \
  --yolo-confidence 0.25 \
  --tesseract-psm 7
```

### Useful environment variables
- `RADAR_SERIAL_PORT`
- `RADAR_BAUD_RATE`
- `SPEED_LIMIT`
- `TRIGGER_COOLDOWN_S`
- `OUTPUT_TEXT_FILE`
- `IMAGE_OUTPUT_DIR`
- `YOLO_MODEL_PATH`
- `YOLO_CONFIDENCE`
- `CAMERA_W`
- `CAMERA_H`
- `TESSERACT_PSM`
- `LOG_LEVEL`

---

## 6) Output format
Each overspeed trigger appends one line to `data/detected_plates.txt`:

```text
2026-01-01T10:20:30,speed_kmph=52.40,plate=MH12AB1234,image=data/captures/overspeed_20260101_102030_123456_52.jpg
```

---

## 7) Validate installation to catch dependency problems early

```bash
source .venv/bin/activate
python -c "import cv2, serial, pytesseract; from ultralytics import YOLO; from picamera2 import Picamera2; print('imports_ok')"
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python -m pip check
```

---

## 8) Optional systemd service
Create `/etc/systemd/system/agd-radar.service`:

```ini
[Unit]
Description=AGD Radar Overspeed Plate Logger
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/workspace/AGD_traffic_warning_system
Environment=RADAR_SERIAL_PORT=/dev/ttyUSB0
Environment=RADAR_BAUD_RATE=115200
Environment=SPEED_LIMIT=40
ExecStart=/workspace/AGD_traffic_warning_system/.venv/bin/python /workspace/AGD_traffic_warning_system/app.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Enable/start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agd-radar
sudo systemctl start agd-radar
sudo systemctl status agd-radar
```

---

## Notes on AGD serial format
The parser currently extracts **the last numeric token** from each serial line as speed.
If your AGD message frame differs, customize `extract_speed()` in `app.py`.
