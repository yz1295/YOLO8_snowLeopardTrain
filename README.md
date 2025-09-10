# Snow Leopard Training (YOLOv8)

This repo documents a clear, end-to-end recipe to train **Ultralytics YOLOv8** to detect **snow leopards**.

---

## 0) Prereqs (once)

- Python 3.9–3.11 recommended  
- (Optional) NVIDIA GPU + CUDA/cuDNN for speed  
- ~10+ GB free disk space (depends on dataset size)

### Windows PowerShell
```powershell
python -m venv sl-yolo
sl-yolo\Scripts\activate
pip install --upgrade pip
pip install ultralytics opencv-python
yolo help   # should print CLI help
````

---

## 1) Project layout (YOLO standard)

```
snow_leopard_yolo/
 ├─ dataset/
 │   ├─ images/
 │   │   ├─ train/
 │   │   ├─ val/
 │   │   └─ test/           # optional but recommended
 │   └─ labels/
 │       ├─ train/
 │       ├─ val/
 │       └─ test/
 └─ dataset.yaml
```

Place images (`.jpg/.png`) into `images/...` and **matching** YOLO labels (`.txt`) into `labels/...` with identical basenames (e.g., `IMG_001.jpg` ↔ `IMG_001.txt`).

---

## 2) Gather data

Aim for hundreds to a few thousand images across conditions:

* camera trap (day/night/IR), drone aerials, terrains, seasons
* include **negatives** (snowy rocks, goats, shadows) to reduce false positives

---

## 3) Annotate (bounding boxes)

Use LabelImg, Roboflow, CVAT, or Label Studio. Export **YOLO** format (one box per line):

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to `[0,1]` relative to image size.
For a single class **snow\_leopard**, `class_id` is `0`.

**Example** `labels/train/IMG_001.txt`:

```
0 0.512 0.431 0.230 0.180
```

---

## 4) Split train/val/test

Use roughly **80/10/10** or **70/20/10**; try to keep `val ≥ 200` images if possible.

If all labeled images sit in one folder, you can use a tiny helper script (optional):

```python
# save as split_dataset.py (run from repo root after staging _all_images/_all_labels)
import os, random, shutil, pathlib
random.seed(42)

ROOT = pathlib.Path("dataset")
img_dir = ROOT/"images"
lbl_dir = ROOT/"labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)
for s in ["train","val","test"]:
    (img_dir/s).mkdir(parents=True, exist_ok=True)
    (lbl_dir/s).mkdir(parents=True, exist_ok=True)

# Put all images in dataset/_all_images and labels in dataset/_all_labels first
ALL_IMG = ROOT/"_all_images"
ALL_LBL = ROOT/"_all_labels"

images = [p for p in ALL_IMG.glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
random.shuffle(images)

def move_pair(img_path, split):
    base = img_path.stem
    lbl_path = ALL_LBL/f"{base}.txt"
    if not lbl_path.exists(): return
    shutil.copy2(img_path, img_dir/split/img_path.name)
    shutil.copy2(lbl_path, lbl_dir/split/lbl_path.name)

n = len(images)
n_train = int(n*0.8)
n_val   = int(n*0.1)
splits = [("train", images[:n_train]),
          ("val",   images[n_train:n_train+n_val]),
          ("test",  images[n_train+n_val:])]

for split, items in splits:
    for img in items:
        move_pair(img, split)

print("Done:", {k: len(v) for k,v in splits})
```

Run it after you stage `dataset/_all_images` and `dataset/_all_labels`.

---

## 5) `dataset.yaml`

Create `dataset.yaml` in repo root:

```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 1
names: ['snow_leopard']
```

---

## 6) Quick data sanity checks

* Image/label basenames exactly match in each split
* Label files only contain class `0`
* Remove/fix empty or garbage label files

---

## 7) Train

Start light with `yolov8n.pt` (nano). If you have a decent GPU, try `yolov8s.pt`.

> Windows tip: if DataLoader hangs, add `workers=0`.

```bash
# from repo root
yolo detect train \
  data=dataset.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  patience=30 \
  project=runs_snowleopard name=v8n_640_e100 \
  cache=True
```

**Flags**

* `batch=-1`: auto batch size
* `device=0`: first GPU; use `cpu` if no GPU
* `patience=30`: early stop if val mAP stalls
* `cache=True`: speed up if RAM allows
* **Windows only**: if issues → `workers=0`

Outputs: `runs_snowleopard/detect/v8n_640_e100/`
Best checkpoint: `weights/best.pt`

**When to train longer/heavier?**

* Low **recall** (misses): increase epochs (e.g., 200), try `yolov8s.pt`, or collect more data.
* Low **precision** (false positives): add hard negatives, tighten labels, improve sampling.

---

## 8) Evaluate & visualize

```bash
yolo detect val model=runs_snowleopard/detect/v8n_640_e100/weights/best.pt data=dataset.yaml
```

Check:

* `confusion_matrix.png`
* `PR_curves.png`
* `results.png`
* `val_batch*.jpg` (qualitative)

---

## 9) Inference (images/folders/video)

```bash
# Single image or folder of images
yolo predict \
  model=runs_snowleopard/detect/v8n_640_e100/weights/best.pt \
  source=path/to/images_or_video \
  conf=0.25 \
  save=True

# Webcam
yolo predict model=... source=0
```

Outputs: `runs/detect/predict*`.

---

## 10) Export for deployment (optional)

```bash
# ONNX (good for many runtimes)
yolo export model=runs_snowleopard/detect/v8n_640_e100/weights/best.pt format=onnx

# TensorRT (NVIDIA Jetson / GPU, requires TensorRT)
yolo export model=... format=engine

# OpenVINO (Intel CPU/VPUs)
yolo export model=... format=openvino

# TFLite / CoreML also available
```

---

## 11) Quality tips for snow leopards

* **Hard negatives**: rock patterns, goats/dogs, snow shadows
* **Multi-scale**: animals can be tiny → keep `imgsz=640` or test `960` if you have GPU headroom
* **Augmentation**: day/night, snow glare, motion blur
* **Consistent labels**: include entire body; label partial occlusions if obvious

---

## 12) Common pitfalls & fixes

* **No detections at all**: wrong `dataset.yaml` paths, mismatched basenames, or non-YOLO labels
* **Hangs on Windows**: set `workers=0`
* **Severe overfit**: too few images → add data/augment; use smaller model
* **False positives on rocks/snow**: add negatives; don’t just lower conf—improve data

---

## 13) Simple Python inference snippet (optional)

```python
from ultralytics import YOLO

model = YOLO(r"runs_snowleopard/detect/v8n_640_e100/weights/best.pt")
res = model.predict(source="path/to/img_or_video.mp4", conf=0.25)
for r in res:
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf   = float(b.conf[0])
        xyxy   = b.xyxy[0].tolist()
        print("det:", cls_id, conf, xyxy)
tra canvas I created earlier or just keep working strictly within your existing repo, say the word—I’ll stick to pure text snippets you can paste yourself.

