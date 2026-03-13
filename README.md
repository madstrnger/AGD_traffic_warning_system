# AGD307 Radar Speed Camera

A Raspberry Pi 5 speed enforcement camera that interfaces with the AGD307 K-band Doppler traffic radar over RS-422, captures USB webcam images of speeding vehicles, and detects Indian number plates using YOLOv8 and EasyOCR.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Wiring](#wiring)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Camera](#running-the-camera)
- [Simulator Mode](#simulator-mode)
- [Configuration Reference](#configuration-reference)
- [Output Files](#output-files)
- [Software Architecture](#software-architecture)
- [Detection Pipeline](#detection-pipeline)
- [Troubleshooting](#troubleshooting)

---

## How It Works

```
AGD307 Radar ──RS-422──► FTDI CA-250 ──USB──► Raspberry Pi 5
                                                     │
                              ┌──────────────────────┘
                              │
                    Speed reading arrives
                    (plain integer kph, 10 fps)
                              │
                    Speed > 20 kph ?
                    Cooldown elapsed ?
                              │  YES
                    Capture frame from
                    Lenovo USB webcam
                              │
                    ┌─────────▼──────────┐
                    │  ViolationWorker   │  ← async thread
                    │  (queue maxsize=4) │
                    └─────────┬──────────┘
                              │
                    YOLOv8 plate detection
                    (yolov8_plate.pt, imgsz=640)
                              │
                    EasyOCR text extraction
                    (Indian plate regex validation)
                              │
                    Annotated JPEG saved
                    Log entry appended
```

The radar runs at **10 readings/second** (`*MS=5` format). When a speed exceeds the threshold and the 5-second cooldown has elapsed, the frame is enqueued to an asynchronous worker thread so the serial read loop is never blocked by ML inference.

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | Raspberry Pi OS Bookworm 64-bit |
| AGD307 K-band radar | Rotary switch must be at position **0** (RS-422 mode) |
| FTDI CA-250 adapter | RS-422 to USB — appears as `/dev/ttyUSB0` |
| USB webcam | Lenovo or any V4L2-compatible webcam |
| Power supply | 12 V or 24 V DC for the AGD307 (see radar manual) |

---

## Wiring

Connect the AGD307 10-way RS-422 connector to the FTDI CA-250 adapter as follows:

| AGD307 Wire | Colour | FTDI CA-250 Pin |
|-------------|--------|-----------------|
| RS422 TXZ | Orange | RX− |
| RS422 TXY | Pink | RX+ |
| RS422 RXA | Brown | TX− |
| RS422 RXB | Violet | TX+ |
| Ground | Green | GND |
| 12/24 Vdc | Red | External PSU + |
| 0 V | Black | External PSU − |

> **Important:** The FTDI adapter is powered from the Pi's USB port. The radar itself requires a separate 12 V or 24 V supply connected to the Red and Black wires. Do **not** connect the radar supply to the FTDI adapter.

---

## Project Structure

```
speed_camera/
│
├── speed_camera.py          # Production application
├── speed_camera_sim.py      # Radar simulator (test without hardware)
├── install.sh               # One-shot dependency installer
├── requirements.txt         # Pinned Python dependencies
├── README.md
│
├── models/                  # Downloaded by install.sh
│   ├── yolov8_plate.pt      # ANPR model (primary)
│   └── yolov8n.pt           # Generic YOLOv8n (fallback)
│
├── captures/                # Created at runtime
│   └── violation_YYYYMMDD_HHMMSS_ffffff_XXkph.jpg
│
├── detected_plates.txt      # Plate log — created at runtime
├── app.log                  # Application log — created at runtime
└── easyocr_models/          # EasyOCR model cache — created at runtime
```

---

## Installation

### 1 — Run the installer

```bash
chmod +x install.sh
./install.sh
```

`install.sh` does the following:

- Creates the Python virtualenv `~/speed_trap_env` with `--system-site-packages`
- Installs all pinned dependencies from `requirements.txt`
- Downloads `yolov8_plate.pt` from the Muhammad-Zeerak-Khan GitHub repository into `models/`
- Downloads `yolov8n.pt` (fallback) into `models/`
- Pre-caches EasyOCR English models into `~/.EasyOCR/`
- Installs the FTDI udev rule at `/etc/udev/rules.d/99-ftdi.rules`

### 2 — Log out and back in

The installer adds your user to the `dialout` group for serial port access. This only takes effect after a fresh login (or reboot).

```bash
sudo reboot
```

### 3 — Verify camera is detected

```bash
v4l2-ctl --list-devices
# or
ls /dev/video*
```

### 4 — Verify FTDI adapter is detected

```bash
ls /dev/ttyUSB*
```

---

### Verified dependency versions

| Package | Version |
|---------|---------|
| pyserial | 3.5 |
| opencv-python-headless | 4.10.0.84 |
| ultralytics | 8.2.18 |
| easyocr | 1.7.1 |
| numpy | 1.24.4 |
| Pillow | 10.3.0 |
| torch | 2.3.0 |
| torchvision | 0.18.0 |
| scipy | 1.13.0 |
| scikit-image | 0.22.0 |

> **numpy is pinned at 1.24.4.** Do not upgrade it unless you have retested all dependencies together — ABI incompatibilities with easyocr and torch are known to surface on aarch64 at higher versions.

If `pip install` fails for torch on aarch64, install from the PyTorch CPU wheel index:

```bash
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Camera

```bash
source ~/speed_trap_env/bin/activate
python3 speed_camera.py
```

At startup the script will:

1. List all detected serial ports as a diagnostic
2. Open `/dev/ttyUSB0` and send the `AGD` handshake
3. Configure the radar: `*LOWSPEED=10`, `*MS=5`, `*BIDI=0`, `*SAVE!`
4. Open the USB webcam at 1280×720 with 5 warm-up frames
5. Load the YOLOv8 ANPR model and EasyOCR
6. Begin monitoring — violations are captured, detected, and logged automatically

Stop with **Ctrl+C** — the worker thread drains cleanly before exit.

---

## Simulator Mode

`speed_camera_sim.py` replaces `RadarInterface` with `SimulatedRadar`. Everything else — the webcam, YOLO, EasyOCR, logging, and the worker thread — runs identically to production. Use it to verify the full ML pipeline without the physical radar connected.

### AUTO mode (default)

Generates a continuous speed stream at 10 readings/second. Background traffic is randomised between 5–18 kph. Every 15 seconds it injects one violation at 35 kph, giving you time to walk into frame holding a printed plate.

```bash
python3 speed_camera_sim.py
# or explicitly:
python3 speed_camera_sim.py --auto
```

### MANUAL mode

Blocks at a prompt after each iteration. You control exactly when each capture fires.

```bash
python3 speed_camera_sim.py --manual
```

At the prompt:

| Input | Action |
|-------|--------|
| Enter (blank) | Fire violation at 35 kph |
| `55` + Enter | Fire violation at 55 kph |
| `q` + Enter | Clean shutdown |

### What is different in simulator output

Annotated violation images have an **orange** speed overlay (`[SIM] SPEED: XX kph`) instead of the red overlay used in production, making it easy to distinguish test captures from real ones in the `captures/` folder.

---

## Configuration Reference

All constants are at the top of each script. Edit them before running.

### Production — `speed_camera.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `SERIAL_PORT` | `/dev/ttyUSB0` | FTDI adapter device path |
| `BAUD_RATE` | `9600` | AGD307 fixed baud rate |
| `SPEED_LIMIT_KPH` | `20` | Capture trigger threshold (kph) |
| `RADAR_LOW_SPEED` | `10` | Sent as `*LOWSPEED=` — radar ignores below this |
| `CAMERA_INDEX` | `0` | OpenCV webcam device index |
| `CAMERA_WIDTH` | `1280` | Capture width (pixels) |
| `CAMERA_HEIGHT` | `720` | Capture height (pixels) |
| `COOLDOWN_SEC` | `5.0` | Minimum seconds between successive captures |
| `YOLO_MODEL_PATH` | `models/yolov8_plate.pt` | Primary ANPR model |
| `YOLO_FALLBACK` | `models/yolov8n.pt` | Generic fallback model |
| `OCR_LANGUAGES` | `["en"]` | EasyOCR language list |

### Simulator extras — `speed_camera_sim.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `SIM_MODE` | `"AUTO"` | Default mode if no CLI flag given |
| `AUTO_RADAR_HZ` | `10` | Speed readings per second in AUTO mode |
| `AUTO_VIOLATION_EVERY` | `15.0` | Seconds between injected violations |
| `AUTO_VIOLATION_SPEED` | `35` | kph value of each injected violation |
| `AUTO_SUBLIMIT_RANGE` | `(5, 18)` | kph range for background sub-limit traffic |
| `MANUAL_DEFAULT_SPEED` | `35` | kph used when user just presses Enter |

---

## Output Files

### `detected_plates.txt`

Appended to on every violation. Created automatically on first run.

```
# AGD307 Speed Camera Log
# Format: TIMESTAMP | SPEED_KPH | PLATE_TEXT | CONFIDENCE | IMAGE
#────────────────────────────────────────────────────────────────────────────
2025-04-01 14:32:07.441 |   35.0 kph | MH12AB1234     | conf=0.87 | violation_20250401_143207_441123_35kph.jpg
2025-04-01 14:32:22.019 |   35.0 kph | NO_PLATE       | conf=0.00 | violation_20250401_143222_019847_35kph.jpg
```

`NO_PLATE` is logged when the ML pipeline ran but found no matching plate text. The image is still saved.

### `captures/`

One JPEG per violation, named `violation_YYYYMMDD_HHMMSS_ffffff_XXkph.jpg`. The image is saved as a raw frame first and then overwritten with the annotated version (bounding box + plate text + speed overlay) once detection completes.

### `app.log`

Full structured application log including all radar commands, init responses, speed readings at DEBUG level, and all INFO/WARNING/ERROR events. Mirrors stdout in real time.

---

## Software Architecture

### `speed_camera.py`

| Class | Responsibility |
|-------|---------------|
| `RadarInterface` | Serial port lifecycle, AGD307 handshake, init command sequence, blocking speed stream parser |
| `CameraCapture` | USB webcam open/capture/save via OpenCV V4L2 backend |
| `PlateDetector` | YOLOv8 inference + EasyOCR OCR pipeline with class filtering, preprocessing, and Indian plate regex validation |
| `PlateLogger` | Append-only log file writer |
| `ViolationWorker` | Daemon thread with `queue.Queue(maxsize=4)` — processes captures asynchronously so the serial read loop is never blocked |
| `SpeedCameraApp` | Top-level orchestrator, startup sequence, main loop, graceful shutdown |

### `speed_camera_sim.py`

Identical architecture with `RadarInterface` replaced by `SimulatedRadar`. All other classes are copied verbatim from the production file so the test environment is an exact mirror of production.

| Class | Responsibility |
|-------|---------------|
| `SimulatedRadar` | Exposes the same 4-method API as `RadarInterface`. AUTO mode runs a background generator thread; MANUAL mode reads from stdin. |

---

## Detection Pipeline

### Stage 1 — YOLOv8 plate detection

The primary model (`yolov8_plate.pt`) is a custom ANPR model trained to detect license plates as **class 0**. The code filters strictly for `cls_id == 0` so vehicle bodies and background objects are discarded.

If the ANPR model is unavailable, the fallback is `yolov8n.pt` (COCO-trained, 80 classes). In fallback mode the filter changes to vehicle classes only (`{2, 3, 5, 7}` — car, motorcycle, bus, truck) and only the **bottom 45%** of each vehicle bounding box is cropped, because that is where Indian plates are physically mounted. This avoids sending the entire car to EasyOCR.

All inference is capped at `imgsz=640` regardless of capture resolution, keeping RAM usage and inference time within the Pi 5 CPU budget (~1–2 s per frame).

### Stage 2 — EasyOCR

Each cropped region is preprocessed before OCR:

- Zero-pixel crops are discarded immediately (guards against degenerate YOLO boxes)
- Small crops (< 40 px tall or < 120 px wide) are upscaled 2× with cubic interpolation
- Adaptive Gaussian threshold converts to binary — handles uneven lighting and both white and yellow plate backgrounds
- 2×2 dilation reconnects broken character strokes

EasyOCR runs with a character allowlist of `A–Z 0–9 -` to suppress OCR noise.

### Stage 3 — Validation

The raw OCR text is matched against the Indian plate regex:

```
[A-Z]{2}  \d{2}  [A-Z]{1,3}  \d{4}
  state    dist    series      number
  e.g. MH   12      AB          1234
```

Spaces and hyphens in the raw text are stripped before matching. If the regex matches, the canonical form is returned. If not, any partial result of four or more characters is passed through to the log as-is rather than discarded silently.

---

## Troubleshooting

**`Serial port /dev/ttyUSB0 not found`**
Run `ls /dev/ttyUSB*`. If nothing appears, check the USB cable and FTDI adapter. If the port appears but is a different index, update `SERIAL_PORT` in the config.

**`Handshake FAILED`**
- Confirm the AGD307 rotary switch is at position **0** (RS-422 mode).
- Check wiring — TX+/TX−/RX+/RX− are easy to swap.
- Confirm the radar has power (Red/Black wires to external PSU).

**`Cannot open camera at index 0`**
Run `v4l2-ctl --list-devices`. If the Lenovo webcam is at a different index, update `CAMERA_INDEX`. Try `CAMERA_INDEX = 1` or `2` if index 0 is occupied by another device. The script opens the camera with `cv2.CAP_V4L2` first and silently retries with auto-detection if V4L2 is unavailable — both paths are handled automatically.

**YOLO model not found at startup**
`install.sh` should have downloaded `models/yolov8_plate.pt`. If the file is absent, the script will attempt to re-download it from GitHub. If the Pi has no internet access, copy the file manually and place it at `<project>/models/yolov8_plate.pt`. The script will fall back to `yolov8n.pt` automatically if the download also fails.

**Plates not being detected**
- In simulator manual mode, hold a **clearly printed A4 plate** at arm's length directly in front of the camera at 0.5–1 m distance.
- Check `app.log` for `OCR raw=` debug lines — if OCR is running but the regex is not matching, the plate text may have a character substitution error.
- Ensure adequate lighting. The adaptive threshold helps but fails under extreme low light.
- If using the fallback `yolov8n.pt`, detection accuracy is lower — use the ANPR model where possible.

**High RAM usage / slow inference**
Inference is already capped at `imgsz=640`. If the Pi 5 is still under memory pressure, lower `CAMERA_WIDTH` / `CAMERA_HEIGHT` and reduce `AUTO_VIOLATION_EVERY` in the simulator to give more time between shots. Confirm `torch==2.3.0` is installed from the CPU wheel index and not a GPU build.
