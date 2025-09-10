# SnowLeopard-YOLOv8 🐾

Train, evaluate, and run inference for **snow leopard detection** using **Ultralytics YOLOv8**. Windows and Linux/macOS supported. Minimal, reproducible, and GitHub‑ready.

---

## 📁 Project structure
```
snow_leopard_yolo/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ dataset.yaml                 # edit paths if you move things
├─ dataset/                     # (you provide your own data)
│  ├─ _all_images/              # stage all images here (optional)
│  ├─ _all_labels/              # stage all YOLO txt labels here (optional)
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ labels/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ scripts/
│  ├─ split_dataset.py
│  ├─ train.sh
│  ├─ train.ps1
│  ├─ predict.sh
│  └─ predict.ps1
└─ src/
   ├─ infer.py
   └─ debounce_example.py       # optional: simple “N consecutive frames” rule
```

---

## 🚀 Quickstart

### Windows (PowerShell)
```powershell
python -m venv sl-yolo
sl-yolo\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# If you have an NVIDIA GPU, install a matching CUDA build of torch first from pytorch.org
# then: pip install ultralytics opencv-python

# verify YOLO CLI
yolo help
```

### Linux/macOS (bash)
```bash
python3 -m venv sl-yolo
source sl-yolo/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# verify YOLO CLI
yolo help
```

> **Tip (Windows dataloader)**: if training hangs, set `workers=0` (already in the PowerShell script).

---

## 📦 Requirements

**requirements.txt**
```txt
ultralytics>=8.3.0
opencv-python
numpy
matplotlib
```
> Note: `ultralytics` will install a CPU build of PyTorch if one isn’t present. For CUDA, install the correct `torch`/`torchvision` from https://pytorch.org first, then `pip install ultralytics`.

---

## 🧰 Dataset prep

1) Put all labeled images in YOLO format (one bbox per line):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are **normalized [0,1]**. For single class **snow_leopard**, `class_id=0`.

2) (Optional) Stage everything first:
```
dataset/_all_images/  # IMG_001.jpg, ...
dataset/_all_labels/  # IMG_001.txt, ...
```

3) Split to train/val/test with the helper script:

**scripts/split_dataset.py**
```python
import os, random, shutil, pathlib
random.seed(42)

ROOT = pathlib.Path(__file__).resolve().parents[1] / "dataset"
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
for s in ["train","val","test"]:
    (IMG_DIR/s).mkdir(parents=True, exist_ok=True)
    (LBL_DIR/s).mkdir(parents=True, exist_ok=True)

ALL_IMG = ROOT/"_all_images"
ALL_LBL = ROOT/"_all_labels"
images = [p for p in ALL_IMG.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
random.shuffle(images)

n = len(images)
if n == 0:
    raise SystemExit("No images found in dataset/_all_images")

n_train = max(int(n*0.8), 1)
n_val   = max(int(n*0.1), 1)

splits = [
    ("train", images[:n_train]),
    ("val",   images[n_train:n_train+n_val]),
    ("test",  images[n_train+n_val:])
]

missing = 0

def move_pair(img_path, split):
    global missing
    base = img_path.stem
    lbl_path = ALL_LBL/f"{base}.txt"
    if not lbl_path.exists():
        missing += 1
        return
    shutil.copy2(img_path, IMG_DIR/split/img_path.name)
    shutil.copy2(lbl_path, LBL_DIR/split/lbl_path.name)

for split, items in splits:
    for img in items:
        move_pair(img, split)

print({k: len(v) for k,v in splits})
print("missing_label_files:", missing)
print("Done.")
```
Run:
```bash
# from repo root
python scripts/split_dataset.py
```

4) Create/verify **dataset.yaml**:
```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test   # optional

nc: 1
names: ["snow_leopard"]
```

---

## 🏋️ Train

**scripts/train.sh**
```bash
#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-dataset.yaml}
MODEL=${2:-yolov8n.pt}
EPOCHS=${3:-100}
IMG=${4:-640}
NAME=${5:-v8n_${IMG}_e${EPOCHS}}

# Use a custom project folder so results don’t mix with other YOLO runs
yolo detect train \
  data="$DATA" \
  model="$MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMG" \
  batch=-1 \
  device=0 \
  patience=30 \
  project=runs_snowleopard \
  name="$NAME" \
  cache=True
```

**scripts/train.ps1**
```powershell
param(
  [string]$Data = "dataset.yaml",
  [string]$Model = "yolov8n.pt",
  [int]$Epochs = 100,
  [int]$Img = 640,
  [string]$Name
)
if (-not $Name) { $Name = "v8n_${Img}_e${Epochs}" }

yolo detect train `
  data=$Data `
  model=$Model `
  epochs=$Epochs `
  imgsz=$Img `
  batch=-1 `
  device=0 `
  patience=30 `
  project=runs_snowleopard `
  name=$Name `
  cache=True `
  workers=0   # Windows stability
```

Run:
```bash
# Linux/macOS
bash scripts/train.sh dataset.yaml yolov8n.pt 100 640

# Windows
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Data dataset.yaml -Model yolov8n.pt -Epochs 100 -Img 640
```
Outputs land in `runs_snowleopard/detect/<NAME>/`. Best weights: `weights/best.pt`.

When to adjust:
- Low **recall** (misses): more epochs (e.g., 200), move up to `yolov8s.pt`, add more data.
- Low **precision** (false positives): add hard negatives (rocks, goats), tighten labels, diversify data.

---

## ✅ Evaluate
```bash
yolo detect val \
  model=runs_snowleopard/detect/<NAME>/weights/best.pt \
  data=dataset.yaml
```
Check: `confusion_matrix.png`, `PR_curves.png`, `results.png`, and `val_batch*.jpg`.

---

## 👀 Inference

**scripts/predict.sh**
```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-runs_snowleopard/detect/v8n_640_e100/weights/best.pt}
SRC=${2:-dataset/images/val}

yolo predict model="$MODEL" source="$SRC" conf=0.25 save=True
```

**scripts/predict.ps1**
```powershell
param(
  [string]$Model = "runs_snowleopard/detect/v8n_640_e100/weights/best.pt",
  [string]$Source = "dataset/images/val"
)

yolo predict model=$Model source=$Source conf=0.25 save=True
```

**src/infer.py**
```python
from ultralytics import YOLO
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs_snowleopard/detect/v8n_640_e100/weights/best.pt"
source     = sys.argv[2] if len(sys.argv) > 2 else "dataset/images/val"

model = YOLO(model_path)
res = model.predict(source=source, conf=0.25)
for r in res:
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf   = float(b.conf[0])
        xyxy   = b.xyxy[0].tolist()
        print("det:", cls_id, conf, xyxy)
```

---

## 🧪 (Optional) Debounce rule example
Simple “trigger after N consecutive frames with a detection ≥ conf_thres”. Useful before integrating MAVLink/DroneKit.

**src/debounce_example.py**
```python
from ultralytics import YOLO
import cv2

MODEL = "runs_snowleopard/detect/v8n_640_e100/weights/best.pt"
CONF_THRES = 0.25
FRAMES_NEEDED = 5

m = YOLO(MODEL)
cap = cv2.VideoCapture(0)  # webcam; or path to video
streak = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    res = m.predict(source=frame, conf=CONF_THRES, verbose=False)
    found = False
    for r in res:
        for b in r.boxes:
            if int(b.cls[0]) == 0 and float(b.conf[0]) >= CONF_THRES:
                found = True
                break
    streak = streak + 1 if found else 0
    if streak >= FRAMES_NEEDED:
        print("ACTION: snow leopard confirmed — take action (e.g., log GPS, LOITER, RTL)...")
        streak = 0
    if cv2.waitKey(1) == 27:
        break
```

---

## 📝 .gitignore
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
sl-yolo/

# Editors
.vscode/
.idea/

# Data & runs
runs/
runs_*/
dataset/_all_images/
dataset/_all_labels/
*.jpg
*.jpeg
*.png
*.mp4
*.avi

# OS
.DS_Store
Thumbs.db
```
> Keep the repo small: don’t commit large datasets or `runs_*` outputs. Create a tiny demo set if you want examples.

---

## 📜 License

**LICENSE (MIT)**
```
MIT License

Copyright (c) 2025 Yiran Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🧭 Troubleshooting
- **No detections**: check `dataset.yaml` paths; ensure image/label basenames match; labels are valid YOLO.
- **Windows hang**: set `workers=0` (PowerShell script already does).
- **Overfit**: add more diverse data; try `yolov8n` first; increase augmentation later.
- **False positives (rocks/snow)**: add hard negatives; improve label quality.

---

## ✍️ README.md

```markdown
# SnowLeopard-YOLOv8 🐾

> YOLOv8 training pipeline for **snow leopard** detection. Clean, minimal, Windows & Linux/macOS.

## Features
- Standard YOLO dataset layout
- One‑liner train/eval/predict scripts (bash & PowerShell)
- Single‑class config (easy to extend)
- Optional frame‑debounce example for real‑time use

## Quickstart
See the top‑level instructions in this repo (create venv, install, `yolo help`).

### Train
```bash
bash scripts/train.sh dataset.yaml yolov8n.pt 100 640
# or
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Data dataset.yaml -Model yolov8n.pt -Epochs 100 -Img 640
```

### Evaluate
```bash
yolo detect val model=runs_snowleopard/detect/<NAME>/weights/best.pt data=dataset.yaml
```

### Predict
```bash
bash scripts/predict.sh runs_snowleopard/detect/<NAME>/weights/best.pt dataset/images/val
# or (Windows)
powershell -ExecutionPolicy Bypass -File scripts/predict.ps1 -Model runs_snowleopard/detect/<NAME>/weights/best.pt -Source dataset/images/val
```

## Data layout
```
dataset/
  images/{train,val,test}
  labels/{train,val,test}
```
Use YOLO txt labels with normalized coords. For single class `snow_leopard`, `class_id=0`.

## Tips for this domain
- Add **hard negatives** (snowy rocks, goats, dogs) to reduce false positives
- Consider higher `imgsz` if animals are tiny in aerial footage (e.g., 640→960 if you have GPU)
- Keep labels consistent; include full body even if partially occluded

## License
MIT © 2025 Yiran Zhang
```

---

## 🧩 Notes
- If you later want **DroneKit/MAVLink** actions (e.g., LOITER/RTL on confirmed detection), we can extend `debounce_example.py` or add a small `autonomy.py` with ArduPilot SITL integration.
- For multi‑class (e.g., snow leopard, goat), set `nc: 2` and `names: ["snow_leopard","goat"]`, update labels accordingly.

