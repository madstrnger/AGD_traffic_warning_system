#!/usr/bin/env bash
# =============================================================================
# install.sh — Speed Trap dependencies for Raspberry Pi 5 (Bookworm 64-bit)
# =============================================================================
# Run as normal user (NOT root).  sudo is invoked internally where needed.
# Usage:  chmod +x install.sh && ./install.sh
#
# Install order is intentional and must not be reordered:
#   1. apt system packages (incl. picamera2)
#   2. Create virtualenv with --system-site-packages (keeps picamera2)
#   3. Upgrade pip
#   4. torch + torchvision  ← MUST come before ultralytics and easyocr
#   5. All remaining pip packages via requirements.txt
#   6. Model weight download
#   7. udev rule for FTDI adapter
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 0. Sanity checks ──────────────────────────────────────────────────────────
[[ "$(uname -m)" == "aarch64" ]] || warn "Expected aarch64 (Pi 5); detected $(uname -m)"
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Detected Python $PY_VER"
[[ "$PY_VER" == "3.11" ]] || warn "Python 3.11 recommended for pinned wheels; found $PY_VER"

info "── Step 1/7  System packages ─────────────────────────────────────────"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-picamera2 \
    libcap-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    socat \
    git \
    wget \
    curl \
    build-essential \
    cmake

info "── Step 2/7  Create virtualenv ───────────────────────────────────────"
VENV_DIR="$HOME/speed_trap_env"
# --system-site-packages is essential: it gives the venv access to
# python3-picamera2 which is apt-installed and has no pip wheel.
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

info "── Step 3/7  Upgrade pip / wheel / setuptools ────────────────────────"
pip install --upgrade pip setuptools wheel

# ── Step 4/7  PyTorch — MUST be installed before ultralytics and easyocr ──────
# Official PyTorch provides manylinux aarch64 wheels for Python 3.11 on PyPI.
# torch 2.3.0 + torchvision 0.18.0 is the only supported pairing for this
# torch major version (see ultralytics/utils/checks.py compatibility table).
info "── Step 4/7  PyTorch + TorchVision (aarch64 CPU wheels) ─────────────"
pip install \
    "torch==2.3.0" \
    "torchvision==0.18.0"
python3 -c "import torch; print(f'   torch {torch.__version__} OK  |  CUDA: {torch.cuda.is_available()}')"

# ── Step 5/7  Install from requirements.txt (single source of truth) ─────────
# torch/torchvision are already installed above; pip will see them satisfied
# and skip re-downloading.  Every other version is read from requirements.txt
# so that the bash script and the requirements file cannot diverge.
info "── Step 5/7  Installing from requirements.txt ────────────────────────"

# numpy must be installed in isolation FIRST with --no-deps so that pip does
# not later override it when resolving the full dependency graph.
# Pin to 1.24.4: matches the C-API ABI of system libcamera/picamera2 bindings
# (compiled against Debian Bookworm's system numpy 1.24.x).  A higher numpy
# in the same venv that has --system-site-packages causes dtype-size mismatch
# warnings or segfaults when picamera2 passes libcamera buffers across the boundary.
pip install --no-deps "numpy==1.24.4"

# Install the rest of requirements.txt; pip will see numpy already satisfied.
pip install -r "$(dirname "$0")/requirements.txt"

# Verify numpy was not silently upgraded by either installer
NUMPY_INSTALLED=$(python3 -c "import numpy; print(numpy.__version__)")
if [[ "$NUMPY_INSTALLED" != 1.24* ]]; then
    error "numpy is $NUMPY_INSTALLED but must be 1.24.4 (ABI match for picamera2 libcamera bindings). Run: pip install --no-deps 'numpy==1.24.4'"
fi
info "   numpy $NUMPY_INSTALLED — OK"

# Pre-fetch YOLOv8n COCO weights into the fixed models/ directory.
# We explicitly pass the destination path so the file is always in
# <project>/models/yolov8n.pt — the same path _YOLO_FALLBACK resolves to
# in main.py.  Without this, YOLO('yolov8n.pt') caches to the CWD at
# install time, which is useless if main.py is later run from a different dir.
python3 -c "
from ultralytics import YOLO
import shutil, pathlib
m = YOLO('yolov8n.pt')   # downloads to ultralytics cache
src = pathlib.Path(m.ckpt_path)
dst = pathlib.Path('models/yolov8n.pt')
dst.parent.mkdir(parents=True, exist_ok=True)
if src.resolve() != dst.resolve():
    shutil.copy2(src, dst)
print(f'   yolov8n.pt → {dst.resolve()}')
" && info "   YOLOv8n weights cached in models/."

info "── Step 6/7  EasyOCR model pre-download ─────────────────────────────"
python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && info "   EasyOCR English models cached in ~/.EasyOCR/" \
    || warn "   EasyOCR model download failed — it will retry on first run."

# ── Step 7/7  ANPR weights + udev ────────────────────────────────────────────
info "── Step 7/7  Downloading ANPR YOLOv8 weights …"
mkdir -p models
ANPR_URL="https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
if wget -q --show-progress -O models/yolov8_plate.pt "$ANPR_URL"; then
    info "✅ ANPR weights saved → models/yolov8_plate.pt"
else
    warn "Could not download ANPR weights (no internet?)."
    warn "Place your weights file at  models/yolov8_plate.pt  manually."
    warn "The app will fall back to YOLOv8n vehicle-crop mode."
fi

info "── Adding udev rule for FTDI USB serial …"
RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE="0666"'
echo "$RULE" | sudo tee /etc/udev/rules.d/99-ftdi.rules > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout "$USER"
info "   udev rule installed. IMPORTANT: log out and back in for group to take effect."

# ── Final version audit ───────────────────────────────────────────────────────
echo ""
info "── Installed version audit ───────────────────────────────────────────"
python3 - <<'PYEOF'
import importlib, sys
checks = [
    ("torch",                    "2.3.0"),
    ("torchvision",              "0.18.0"),
    ("numpy",                    "1.24.4"),
    ("PIL",                      "10.3.0"),   # Pillow
    ("cv2",                      "4.10.0"),
    ("scipy",                    "1.13.0"),
    ("skimage",                  "0.22.0"),
    ("shapely",                  "2.0.5"),
    ("yaml",                     "6.0.1"),
    ("easyocr",                  "1.7.1"),
    ("ultralytics",              "8.2.18"),
    ("serial",                   "3.5"),
]
ok = True
for mod, expected in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        # PIL reports version via PIL.__version__, not cv2
        if mod == "PIL":
            import PIL; ver = PIL.__version__
        status = "✅" if ver.startswith(expected.rstrip("0").rstrip(".")) else "⚠️ "
        if not ver.startswith(expected.rstrip("0").rstrip(".")):
            ok = False
        print(f"  {status}  {mod:<20} {ver}  (expected {expected})")
    except ImportError as e:
        print(f"  ❌  {mod:<20} NOT FOUND — {e}")
        ok = False
sys.exit(0 if ok else 1)
PYEOF

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo ""
echo "  Activate env :  source $VENV_DIR/bin/activate"
echo "  Run app      :  python3 main.py"
echo "  Run tests    :  python3 radar_simulator.py --dryrun"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
