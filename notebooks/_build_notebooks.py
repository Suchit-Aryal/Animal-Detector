"""Builds the four notebooks (plain nbformat-4 JSON, no extra packages). Edit the cell text here, then re-run:
    python notebooks/_build_notebooks.py
Outputs are filled in afterwards by: python notebooks/run_notebook.py notebooks/<name>.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def save(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python (animal-det)", "language": "python", "name": "animal-det"},
                       "language_info": {"name": "python", "version": "3.11"}}}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:02d}"
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", name, len(cells), "cells")


ROOT_CELL = '''
# Find the project root (the folder that contains PROJECT_PLAN.md) and work from there,
# so every path below is relative to the project root no matter where Jupyter was started.
import os, sys
from pathlib import Path

here = Path.cwd().resolve()
for folder in [here, *here.parents]:
    if (folder / "PROJECT_PLAN.md").exists():
        os.chdir(folder)
        break
else:
    raise SystemExit("PROJECT_PLAN.md not found: start Jupyter inside the project folder")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))   # lets us import small helpers from src/ if we need them
print("project root folder:", ROOT.name)
'''

SHOW = '''
from IPython.display import display   # shows tables / images under the cell
'''

# ---- shared pieces (kept as text so each notebook is self-contained) -------------------------------------------
LIST_IMAGES = '''
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
cfg = yaml.safe_load(Path(DATA_YAML).read_text())
NAMES = {int(k): v for k, v in cfg["names"].items()}            # our classes: 0 dog, 1 goat, 2 cow, 3 cat, 4 sheep

def list_images(split):
    """All image files of a split, e.g. data/dataset/images/test/*.jpg (sorted, so runs are repeatable)."""
    folder = Path(cfg["path"]) / cfg[split]
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)

def label_file(img):
    """The YOLO label file of an image: .../images/test/a.jpg -> .../labels/test/a.txt"""
    return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")

images = list_images(SPLIT)[:MAX_IMAGES]                         # [:None] keeps everything
n_boxes = sum(len([l for l in label_file(i).read_text().splitlines() if l.strip()]) for i in images if label_file(i).exists())
print(f"{len(images)} images from split '{SPLIT}', {n_boxes} labelled boxes")
'''

EVAL_FUNCS = '''
import shutil, time
import numpy as np
import pandas as pd
from ultralytics import YOLO

TMP = ROOT / "runs/notebooks/_eval_tmp"            # scratch dataset, rebuilt on every evaluation (runs/ is git-ignored)

def build_tmp_dataset(imgs, cls_map):
    """Copy-free dataset for the official validator: symlink the images and write their labels using the MODEL's class ids.
    cls_map = {model_class_id: our_class_id}. Boxes of classes the model cannot predict are dropped (their AP becomes NaN).
    cls_map=None means the model already uses our ids."""
    inverse = {ours: theirs for theirs, ours in cls_map.items()} if cls_map else None
    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / "images/test").mkdir(parents=True); (TMP / "labels/test").mkdir(parents=True)
    for img in imgs:
        (TMP / "images/test" / img.name).symlink_to(img.resolve())
        lines = []
        if label_file(img).exists():
            for line in label_file(img).read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                ours = int(float(parts[0]))
                if inverse is None:
                    lines.append(line)
                elif ours in inverse:
                    lines.append(" ".join([str(inverse[ours])] + parts[1:]))
        (TMP / "labels/test" / (img.stem + ".txt")).write_text("\\n".join(lines) + ("\\n" if lines else ""))

def evaluate_model(weights, imgs, cls_map=None, plots=False):
    """Official Ultralytics validation (conf 0.001, NMS IoU 0.7, batch 1) + FPS. Returns one row of metrics."""
    model = YOLO(str(weights))
    build_tmp_dataset(imgs, cls_map)
    (TMP / "data.yaml").write_text(yaml.safe_dump({"path": str(TMP), "train": "images/test", "val": "images/test",
                                                   "test": "images/test", "names": {int(k): v for k, v in model.names.items()}}))
    r = model.val(data=str(TMP / "data.yaml"), split="test", imgsz=IMGSZ, batch=1, conf=0.001, iou=0.7, device=DEVICE,
                  plots=plots, verbose=False, project=str(TMP / "val"), name="v", exist_ok=True, save=False)
    per_class = {}
    for k, cls_id in enumerate(r.box.ap_class_index):                      # AP50 of every class that has ground truth
        per_class[cls_map[int(cls_id)] if cls_map else int(cls_id)] = float(r.box.ap50[k])
    model.predict(str(imgs[0]), imgsz=IMGSZ, verbose=False, device=DEVICE)  # warm-up (first call is slow)
    t0 = time.perf_counter()
    for img in imgs:
        model.predict(str(img), imgsz=IMGSZ, conf=0.25, verbose=False, device=DEVICE)
    sec = (time.perf_counter() - t0) / len(imgs)
    row = {"images": len(imgs), "precision": float(r.box.mp), "recall": float(r.box.mr), "mAP50": float(r.box.map50),
           "mAP50-95": float(r.box.map), "FPS": 1 / sec}
    for cls_id, name in NAMES.items():
        row[f"AP50_{name}"] = per_class.get(cls_id, np.nan)                # NaN = no ground truth / not supported
    return row, r
'''

# ======================================================= 01 baseline ============================================
nb1 = [
    md('''
# 01 · Baseline evaluation (pretrained models, no training)

**Question:** how good are YOLO models that were *never trained on our data*? Their score is the bar our fine-tuned model has to beat.

We test two pretrained models on the same test images:
- **COCO `yolov8n`**: general model, has dog / cat / cow / sheep but **no goat**.
- **Open Images V7 `yolov8m-oiv7`**: has all five of our animals. Caution: our test images come from Open Images, so this model may have seen them in training, and its score is probably too optimistic.

Each pretrained model uses its own class numbers, so the notebook translates them to our five classes
(0 dog, 1 goat, 2 cow, 3 cat, 4 sheep) and then runs the standard Ultralytics validation.

By default the notebook runs on **10 images** so it finishes in about a minute. Set `FULL_RUN = True` for all 80 test images.
'''),
    md("## 1. Find the project and set the parameters"),
    code(ROOT_CELL),
    code('''
# ---- parameters: change things here ----
FULL_RUN = False                       # False: 10 images (fast demo). True: whole test split (80 images).
SPLIT = "test"                         # which split of the dataset to evaluate
MAX_IMAGES = None if FULL_RUN else 10
DATA_YAML = "data/dataset/data.yaml"
IMGSZ = 640                            # image size the model sees
COCO_WEIGHTS = "yolov8n.pt"            # COCO baseline (downloaded automatically if missing)
OIV7_WEIGHTS = "yolov8m-oiv7.pt"       # Open Images V7 baseline (already in the project root)
SAVE_CSV = False                       # True: overwrite results/baseline.csv (only sensible together with FULL_RUN=True)

import torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| images:", MAX_IMAGES or "all")
'''),
    md('''
## 2. Translate the baselines' classes into ours
A pretrained model predicts class numbers from *its* list (COCO has 80, Open Images has 601). We keep only the five animals
we care about and rename them to our ids. Everything else the model predicts is ignored.
Goat does not exist in COCO, so for that model goat is marked "not supported" (AP stays empty instead of a misleading 0).
'''),
    code('''
from IPython.display import display
import pandas as pd

COCO_MAP = {16: 0, 19: 2, 15: 3, 18: 4}           # COCO id -> our id : dog, cow, cat, sheep   (no goat)
OIV7_MAP = {160: 0, 219: 1, 99: 2, 96: 3, 452: 4}  # Open Images id -> our id : Dog, Goat, Cattle, Cat, Sheep

mapping = pd.DataFrame({
    "our class": ["0 dog", "1 goat", "2 cow", "3 cat", "4 sheep"],
    "COCO yolov8n": ["16 dog", "NOT SUPPORTED", "19 cow", "15 cat", "18 sheep"],
    "Open Images yolov8m": ["160 Dog", "219 Goat", "99 Cattle", "96 Cat", "452 Sheep"]})
display(mapping)
'''),
    md("## 3. Load the test images and their labels"),
    code(LIST_IMAGES),
    md('''
## 4. Evaluation code
`build_tmp_dataset` writes a throw-away copy of the test set whose label numbers match the model being tested
(images are only linked, nothing is copied). Then the official `model.val` computes precision, recall, mAP and per-class AP.
FPS is measured separately by timing `model.predict` image by image.

- **mAP50**: share of animals found with a box overlapping the true box by at least 50%, averaged over classes.
- **mAP50-95**: the same but with stricter overlap limits (50% to 95%). Always lower; rewards tight boxes.
'''),
    code(EVAL_FUNCS),
    md("## 5. Run both baselines"),
    code('''
rows = []
for name, weights, cls_map in [("baseline_coco_yolov8n", COCO_WEIGHTS, COCO_MAP),
                               ("baseline_oiv7_yolov8m", OIV7_WEIGHTS, OIV7_MAP)]:
    row, _ = evaluate_model(weights, images, cls_map)
    rows.append({"model": name, **row})
    print("done:", name)
baseline = pd.DataFrame(rows)
baseline.round(3)
'''),
    md("## 6. Chart the result"),
    code('''
import matplotlib.pyplot as plt

metrics = ["mAP50", "mAP50-95", "precision", "recall"]
x = range(len(metrics))
fig, ax = plt.subplots(figsize=(7, 3.8))
for k, (_, r) in enumerate(baseline.iterrows()):
    bars = ax.bar([i + (k - 0.5) * 0.38 for i in x], [r[m] for m in metrics], 0.38, label=r["model"])
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xticks(list(x), metrics); ax.set_ylim(0, 1.1); ax.legend(fontsize=8)
ax.set_title(f"Baselines on {len(images)} '{SPLIT}' images" + ("" if FULL_RUN else " (fast demo: numbers are noisy)"))
plt.show()
'''),
    md('''
## 7. Compare with the saved full-run numbers
`results/baseline.csv` holds the official numbers on all 80 test images. The 10-image demo above will not match them exactly
because it sees fewer images; the full run (`FULL_RUN = True`) reproduces them.
'''),
    code('''
saved = pd.read_csv("results/baseline.csv")
display(saved[["model", "images", "mAP50", "mAP50-95", "precision", "recall", "FPS"]].round(3))

if SAVE_CSV:
    baseline.to_csv("results/baseline.csv", index=False)
    print("results/baseline.csv overwritten")
else:
    print("SAVE_CSV is False: results/baseline.csv left untouched")
'''),
    md('''
## What to take away
- The Open Images model scores highest, but its test score is probably inflated (it was trained on Open Images).
- COCO cannot see goats at all, which is exactly why we fine-tune on our own five classes (notebook 02).
'''),
]

# ======================================================= 02 train ===============================================
nb2 = [
    md('''
# 02 · Train the model (fine-tuning with `freeze=10`)

**Idea:** start from a YOLOv8 model that already knows how to see edges, shapes and textures (it was pretrained on COCO),
and teach it our five animals: dog, goat, cow, cat, sheep. This is called *fine-tuning*.

**The recipe that gave our best model** (`results/weights/best.pt`): `yolov8s`, `freeze=10`, 80 epochs, batch 8, seed 0.
`freeze=10` keeps the first 10 layers (the backbone, which finds general features) fixed and only trains the rest.
With a small dataset this stops the model from memorising the training pictures (*overfitting*).

By default this notebook only does a **tiny 1-epoch run on about 2% of the training images** so you can see the whole pipeline
in a few minutes. Set `FULL_RUN = True` for the real recipe (about 30 minutes on a laptop GPU).
'''),
    md("## 1. Find the project and set the parameters"),
    code(ROOT_CELL),
    code('''
# ---- parameters: change things here ----
FULL_RUN = False                 # False: 1 epoch on ~2% of the images (demo). True: the real 80-epoch recipe.
MODEL = "yolov8s.pt"             # starting weights (pretrained on COCO)
DATA_YAML = "data/dataset/data.yaml"
FREEZE = 10                      # number of first layers kept frozen (0 = train everything)
EPOCHS = 80 if FULL_RUN else 1
FRACTION = 1.0 if FULL_RUN else 0.02   # share of the training images used
PATIENCE = 20                    # stop early if validation mAP does not improve for this many epochs
BATCH = 8                        # keep <= 8 (6 GB GPU, memory rules of this project)
WORKERS = 2                      # keep <= 2
IMGSZ = 640
SEED = 0                         # fixed seed so the run can be repeated
RUN_NAME = "nb_full" if FULL_RUN else "nb_fast"
PROJECT = "runs/notebooks"       # runs/ is git-ignored
MEMORY_CAP = "15G"               # hard memory limit for the training process

import torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| epochs:", EPOCHS, "| fraction:", FRACTION)
'''),
    md('''
## 2. Look at the training data
Always check what you are training on. This counts images and labelled boxes per class in each split.
'''),
    code('''
from IPython.display import display
import pandas as pd, yaml

cfg = yaml.safe_load(Path(DATA_YAML).read_text())
NAMES = {int(k): v for k, v in cfg["names"].items()}
rows = []
for split in ["train", "val", "test"]:
    folder = Path(cfg["path"]) / cfg[split]
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    counts = dict.fromkeys(NAMES.values(), 0)
    for img in imgs:
        lab = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if lab.exists():
            for line in lab.read_text().splitlines():
                if line.strip():
                    counts[NAMES[int(float(line.split()[0]))]] += 1
    rows.append({"split": split, "images": len(imgs), **{f"{k} boxes": v for k, v in counts.items()}})
display(pd.DataFrame(rows))
'''),
    md('''
## 3. The training settings in one place
These are the arguments handed to Ultralytics `model.train`. Reading them is the quickest way to understand the recipe.
'''),
    code('''
train_args = dict(
    data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS,
    seed=SEED, deterministic=True,            # same seed -> same result when repeated
    patience=PATIENCE, fraction=FRACTION,
    freeze=FREEZE or None,                    # freeze the first N layers (None = nothing frozen)
    device=DEVICE, project=str((ROOT / PROJECT).resolve()), name=RUN_NAME,
    exist_ok=True, plots=False,
)
display(pd.Series(train_args, name="value").to_frame())
print("The real recipe saved from the v2 training for comparison:")
print(Path("results/weights/v2A_s_freeze10_config.yaml").read_text())
'''),
    md('''
## 4. Train (in a separate, memory-capped process)
Training is the heavy part, so it does not run inside this notebook's own process. The cell below writes the training code
into a small script and starts it with `systemd-run --scope -p MemoryMax=15G -p MemorySwapMax=0`:
if the job ever used more than 15 GB it would be stopped instead of freezing the computer.
'''),
    code('''
import json, shutil, subprocess

TRAIN_SCRIPT = """
import json, sys
from ultralytics import YOLO

args = json.load(open(sys.argv[1]))      # the settings from the cell above
model = YOLO(args.pop("model"))          # load the pretrained starting weights
model.train(**args)                      # fine-tune; writes weights/best.pt and results.csv into runs/notebooks/<name>/
"""

def run_capped(script_text, payload, tag):
    """Run a python script in a subprocess with a hard memory cap; returns the cleaned tail of its output."""
    work = ROOT / PROJECT; work.mkdir(parents=True, exist_ok=True)
    (work / f"{tag}.py").write_text(script_text)
    (work / f"{tag}.json").write_text(json.dumps(payload))
    cmd = [sys.executable, str(work / f"{tag}.py"), str(work / f"{tag}.json")]
    if shutil.which("systemd-run"):
        cmd = ["systemd-run", "--user", "--scope", "-p", f"MemoryMax={MEMORY_CAP}", "-p", "MemorySwapMax=0"] + cmd
    else:
        print("WARNING: systemd-run not found, running WITHOUT a memory cap")
    print(subprocess.run(["free", "-h"], capture_output=True, text=True).stdout)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    text = (p.stdout + p.stderr).replace("\\r", "\\n")
    tail = [l for l in text.splitlines() if l.strip() and "\\u2501" not in l][-25:]
    print("\\n".join(tail))
    if p.returncode != 0:
        raise RuntimeError(f"training process exited with code {p.returncode} (137/-9 = killed by the memory cap)")

run_capped(TRAIN_SCRIPT, {"model": MODEL, **train_args}, "train_job")
'''),
    md("## 5. Learning curves of this run\n`results.csv` has one row per epoch. Falling losses and rising mAP mean the model is learning."),
    code('''
import matplotlib.pyplot as plt

run_dir = ROOT / PROJECT / RUN_NAME
hist = pd.read_csv(run_dir / "results.csv"); hist.columns = [c.strip() for c in hist.columns]
display(hist[["epoch", "train/cls_loss", "val/cls_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]].round(3))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
axs[0].plot(hist["epoch"], hist["train/cls_loss"], "o-", label="train"); axs[0].plot(hist["epoch"], hist["val/cls_loss"], "o-", label="validation")
axs[0].set_title("class loss (lower is better)"); axs[0].set_xlabel("epoch"); axs[0].legend()
axs[1].plot(hist["epoch"], hist["metrics/mAP50(B)"], "o-"); axs[1].set_title("validation mAP50"); axs[1].set_xlabel("epoch"); axs[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()
'''),
    md('''
## 6. Why `freeze=10`? The overfitting picture
Below are the full 80-epoch runs we trained earlier (kept in `runs/detect/`). *Train* loss falls the same way in both, but with all layers trainable
the *validation* loss turns upwards: the model memorises the training images. With `freeze=10` the gap is much smaller.
(This cell needs the local `runs/detect/` folders; it skips politely if they are missing.)
'''),
    code('''
plain, frozen = ROOT / "runs/detect/yolov8s_80ep/results.csv", ROOT / "runs/detect/yolov8s_80ep_freeze10/results.csv"
if plain.exists() and frozen.exists():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, path, title in [(axs[0], plain, "yolov8s, all layers trained"), (axs[1], frozen, "yolov8s, freeze=10")]:
        d = pd.read_csv(path); d.columns = [c.strip() for c in d.columns]
        ax.plot(d["epoch"], d["train/cls_loss"], label="train class loss"); ax.plot(d["epoch"], d["val/cls_loss"], label="validation class loss")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    axs[0].set_ylabel("class loss"); plt.tight_layout(); plt.show()
else:
    print("runs/detect/... not found on this machine; see results/figures/loss_curves.png")
'''),
    md('''
## 7. Keep the weights
`best.pt` is the epoch with the best validation mAP50. A **full run** copies it to `results/weights/<RUN_NAME>.pt`.
It never overwrites `results/weights/best.pt` (the model the website uses); promote a new model by hand only after comparing it on the
validation split.
'''),
    code('''
best = run_dir / "weights/best.pt"
print("best weights of this run:", best, "| exists:", best.exists())
if FULL_RUN and best.exists():
    dst = ROOT / "results/weights" / f"{RUN_NAME}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(best, dst)
    print("copied to", dst)
else:
    print("fast demo run: nothing copied (these weights saw 2% of the data for 1 epoch and are not useful)")
'''),
    md('''
## What to take away
- Same recipe + same seed = same result; everything is in the settings cell.
- `freeze=10` is the cheap fix that made fine-tuning work on this small dataset.
- Next: notebook 03 measures the trained model on the test images.
'''),
]

# ======================================================= 03 evaluate ============================================
nb3 = [
    md('''
# 03 · Evaluate and compare

**Question:** is our fine-tuned model better than the baselines, and where does it still fail?

This notebook (1) measures a model on the test images, (2) compares it with the saved results of the baselines and earlier models,
(3) shows a per-class chart and the confusion matrix, and (4) lists the typical mistakes with pictures.

Default is a **fast 10-image run**; set `FULL_RUN = True` to use all 80 test images. The test set is never used to choose settings:
it is looked at only once, at the end.
'''),
    md("## 1. Find the project and set the parameters"),
    code(ROOT_CELL),
    code('''
# ---- parameters: change things here ----
FULL_RUN = False
WEIGHTS = "results/weights/best.pt"     # model to evaluate (the one the website uses)
SPLIT = "test"
MAX_IMAGES = None if FULL_RUN else 10
DATA_YAML = "data/dataset/data.yaml"
IMGSZ = 640
CONF = 0.25                              # confidence a box needs to be counted in the failure-case section

import torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("weights:", WEIGHTS, "| device:", DEVICE, "| images:", MAX_IMAGES or "all")
'''),
    md("## 2. Load the images and labels"),
    code("from IPython.display import display\n" + LIST_IMAGES),
    md('''
## 3. Measure the model
Same evaluation code as in notebook 01 (the official Ultralytics validation on a throw-away dataset). Our model already uses our class
numbers, so no translation is needed (`cls_map=None`). Precision = how many predictions were right; recall = how many real animals were found.
'''),
    code(EVAL_FUNCS),
    code('''
row, val_result = evaluate_model(WEIGHTS, images, cls_map=None, plots=True)
this_run = pd.DataFrame([{"model": "this notebook run", **row}])
this_run.round(3)
'''),
    md('''
## 4. Compare with the recorded results
`results/comparison.csv` holds the official 80-image test numbers of the two baselines and our fine-tuned models.
Your run is shown separately: with only 10 images it is **not** comparable to those rows (set `FULL_RUN = True` to compare fairly).
'''),
    code('''
comparison = pd.read_csv("results/comparison.csv")
cols = ["model", "images", "mAP50", "mAP50-95", "precision", "recall", "FPS"]
display(comparison[cols].round(3))
display(this_run[cols].round(3))
'''),
    md('''
## 5. Per-class chart
Average numbers can hide a weak class. Bars show AP50 per animal; "n/a" means the model has no such class (COCO has no goat)
or the test set has no examples of it (the frozen test set has no sheep).
'''),
    code('''
import matplotlib.pyplot as plt

classes = [n for n in NAMES.values() if comparison[f"AP50_{n}"].notna().any() or this_run[f"AP50_{n}"].notna().any()]
table = pd.concat([comparison, this_run], ignore_index=True)
width = 0.8 / len(table)
fig, ax = plt.subplots(figsize=(10, 4.2))
for k, (_, r) in enumerate(table.iterrows()):
    vals = [r[f"AP50_{c}"] for c in classes]
    xs = [i + (k - (len(table) - 1) / 2) * width for i in range(len(classes))]
    ax.bar(xs, [0 if v != v else v for v in vals], width, label=r["model"])
    for xi, v in zip(xs, vals):
        ax.text(xi, 0.02 if v != v else v + 0.01, "n/a" if v != v else f"{v:.2f}", ha="center", fontsize=6, rotation=90)
ax.set_xticks(range(len(classes)), classes); ax.set_ylim(0, 1.15); ax.set_ylabel("AP50")
ax.legend(fontsize=7, loc="lower right"); ax.set_title("AP50 per class (recorded 80-image runs + this notebook's run)")
plt.tight_layout(); plt.show()
'''),
    md('''
## 6. Confusion matrix
Rows are the predicted class, columns the true class (each column adds up to 1). A bright diagonal is good. The "background" row/column
shows missed animals (true animal, predicted nothing) and false alarms (predicted animal, nothing there).
This is the matrix of the run in section 3 (so it also uses only the images selected above).
'''),
    code('''
import matplotlib.image as mpimg

cm_png = Path(val_result.save_dir) / "confusion_matrix_normalized.png"
plt.figure(figsize=(6.5, 6)); plt.imshow(mpimg.imread(cm_png)); plt.axis("off"); plt.show()
'''),
    md('''
## 7. Failure cases
We compare each prediction with the labelled boxes (a prediction "matches" a box when the boxes overlap by IoU >= 0.5) and sort the mistakes:
- **wrong class**: right place, wrong animal
- **miss**: a labelled animal nobody predicted
- **false positive**: a prediction where nothing is labelled

Green = label, red = prediction.
'''),
    code('''
import cv2

def iou(a, b):
    """Overlap of two boxes (x1, y1, x2, y2): intersection area / union area."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

model = YOLO(WEIGHTS)
found = {"wrong_class": [], "miss": [], "false_positive": []}
for img_path in images:
    img = cv2.imread(str(img_path)); h, w = img.shape[:2]
    truth = []                                                     # labelled boxes in pixels
    if label_file(img_path).exists():
        for line in label_file(img_path).read_text().splitlines():
            t = line.split()
            if len(t) == 5:
                c, x, y, bw, bh = int(t[0]), *map(float, t[1:])
                truth.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    out = model.predict(img, conf=CONF, imgsz=IMGSZ, verbose=False, device=DEVICE)[0].boxes
    preds = [(int(c), float(s), xy) for c, s, xy in zip(out.cls.tolist(), out.conf.tolist(), out.xyxy.tolist())]
    used = set()
    for c, box in truth:
        best_iou, j = max(((iou(box, p[2]), j) for j, p in enumerate(preds)), default=(0, -1))
        if best_iou >= 0.5:
            used.add(j)
            if preds[j][0] != c:
                found["wrong_class"].append((img_path, truth, preds, f"labelled {NAMES[c]}, predicted {NAMES[preds[j][0]]}"))
        else:
            found["miss"].append((img_path, truth, preds, f"{NAMES[c]} not detected"))
    for j, p in enumerate(preds):
        if j not in used and all(iou(p[2], t[1]) < 0.3 for t in truth):
            found["false_positive"].append((img_path, truth, preds, f"predicted {NAMES[p[0]]} ({p[1]:.2f}) with no labelled animal"))

print({k: len(v) for k, v in found.items()}, f"events in {len(images)} images")

def draw(img_path, truth, preds):
    img = cv2.imread(str(img_path))
    for c, b in truth:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 3)
        cv2.putText(img, NAMES[c], (int(b[0]), int(b[1]) + 18), 0, 0.7, (0, 200, 0), 2)
    for c, s, b in preds:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        cv2.putText(img, f"{NAMES[c]} {s:.2f}", (int(b[0]), int(b[3]) - 6), 0, 0.7, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

shown = [(k, v[0]) for k, v in found.items() if v]
if shown:
    fig, axs = plt.subplots(1, len(shown), figsize=(5 * len(shown), 4), squeeze=False)
    for ax, (kind, (p, truth, preds, why)) in zip(axs[0], shown):
        ax.imshow(draw(p, truth, preds)); ax.axis("off"); ax.set_title(f"{kind}: {why}", fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("no mistakes in this small sample; try FULL_RUN = True")
'''),
    md('''
### Failure examples from the full 80-image test run
Saved by `src/make_plots.py` (explanations: `results/failure_cases.md`). A group scene, a wrong class, a miss and a false positive.
'''),
    code('''
saved = ["failure_group", "failure_wrong_class", "failure_miss", "failure_false_positive"]
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for ax, name in zip(axs.ravel(), saved):
    f = Path(f"results/figures/{name}.jpg")
    if f.exists():
        ax.imshow(mpimg.imread(f))
    ax.set_title(name.replace("failure_", ""), fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    md('''
## What to take away
- Compare models only on the same images (use `FULL_RUN = True`).
- Look at the per-class bars and the confusion matrix, not only at mAP: a good average can hide a weak class.
- Known caveats: the test images are Open Images (not Nepal street photos), and the Open Images baseline may have seen them in training.
'''),
]

# ======================================================= 04 demo ================================================
nb4 = [
    md('''
# 04 · Demo: detect animals in an image and a video

The last step: give the trained model a picture (or a video) and see the boxes it draws. This is what the surveillance website does for every frame.

By default the notebook uses a **test image** and a **10-frame demo clip** assembled from test images, so it runs without any extra files.
Point `IMAGE_PATH` / `VIDEO_PATH` at your own files to try them (a webcam index such as `0` also works for `VIDEO_PATH`).
'''),
    md("## 1. Find the project and set the parameters"),
    code(ROOT_CELL),
    code('''
# ---- parameters: change things here ----
FULL_RUN = False
WEIGHTS = "results/weights/best.pt"
CONF = 0.25                      # minimum confidence for a box to be drawn (higher = fewer, surer boxes)
IMGSZ = 640
IMAGE_PATH = None                # None -> first test image. Or e.g. "my_photo.jpg"
VIDEO_PATH = None                # None -> a demo clip built from test images. Or "my_video.mp4", or 0 for a webcam
MAX_FRAMES = 60 if FULL_RUN else 10
OUT_DIR = "notebooks/out"        # annotated results are saved here

import torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print("weights:", WEIGHTS, "| device:", DEVICE)
'''),
    md("## 2. Load the model"),
    code('''
import cv2, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ultralytics import YOLO

model = YOLO(WEIGHTS)
NAMES = model.names
print("classes the model knows:", NAMES)
'''),
    md('''
## 3. One image
`model.predict` returns, for every box, the class, the confidence (0 to 1) and the corner coordinates. `result.plot()` draws them on the picture.
'''),
    code('''
test_images = sorted((ROOT / "data/dataset/images/test").glob("*.jpg"))
image_path = Path(IMAGE_PATH) if IMAGE_PATH else test_images[0]
result = model.predict(str(image_path), conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]

annotated = result.plot()                                    # BGR picture with boxes drawn
cv2.imwrite(f"{OUT_DIR}/image_annotated.jpg", annotated)
plt.figure(figsize=(7, 5)); plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title(f"{image_path.name}: {len(result.boxes)} detection(s)"); plt.show()
'''),
    md("### The detections as a table"),
    code('''
table = pd.DataFrame({
    "class": [NAMES[int(c)] for c in result.boxes.cls],
    "confidence": [round(float(s), 3) for s in result.boxes.conf],
    "box x1,y1,x2,y2 (pixels)": [[round(v) for v in b] for b in result.boxes.xyxy.tolist()]})
table
'''),
    md('''
## 4. Try other confidence thresholds
A low threshold finds more animals but adds false alarms; a high one is stricter but misses animals. The website uses the same idea.
'''),
    code('''
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, conf in zip(axs, [0.10, 0.25, 0.60]):
    r = model.predict(str(image_path), conf=conf, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    ax.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(f"conf >= {conf}: {len(r.boxes)} box(es)")
plt.tight_layout(); plt.show()
'''),
    md('''
## 5. A video
A video is just a list of frames. For each frame we run the model, draw the boxes and write the frame to a new video file.
Without a video of your own, we first build a short clip from test images so the notebook works out of the box.
'''),
    code('''
if VIDEO_PATH is None:                                     # build a small demo clip from test images
    clip = ROOT / OUT_DIR / "demo_input.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
    for k in range(MAX_FRAMES):
        frame = cv2.resize(cv2.imread(str(test_images[k % len(test_images)])), (640, 480))
        writer.write(frame)
    writer.release()
    source = str(clip)
    print("built demo clip:", clip, f"({MAX_FRAMES} frames)")
else:
    source = VIDEO_PATH
    print("using", source)
'''),
    code('''
cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
assert cap.isOpened(), f"cannot open {source}"
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = f"{OUT_DIR}/video_annotated.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

frames, per_frame, t0 = [], [], time.perf_counter()
while len(per_frame) < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    shown = r.plot()
    writer.write(shown)
    frames.append(shown)
    per_frame.append({"frame": len(per_frame), "detections": len(r.boxes),
                      "classes": ", ".join(sorted({NAMES[int(c)] for c in r.boxes.cls}))})
cap.release(); writer.release()
elapsed = time.perf_counter() - t0
print(f"{len(per_frame)} frames in {elapsed:.1f}s ({len(per_frame) / elapsed:.1f} frames/s including drawing and writing) -> {out_path}")
pd.DataFrame(per_frame)
'''),
    md("### Sample frames from the annotated video"),
    code('''
pick = np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)
fig, axs = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.2), squeeze=False)
for ax, k in zip(axs[0], pick):
    ax.imshow(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(f"frame {k}", fontsize=9)
plt.tight_layout(); plt.show()
'''),
    md('''
## What to take away
- Detection is one call (`model.predict`); the rest is drawing and saving.
- `conf` trades missed animals against false alarms.
- The full command-line version is `python src/demo.py --source <image|video|0>`; the website runs the same model on live frames.
'''),
]

if __name__ == "__main__":
    save("01_baseline_evaluation.ipynb", nb1)
    save("02_train_model.ipynb", nb2)
    save("03_evaluate_and_compare.ipynb", nb3)
    save("04_demo_inference.ipynb", nb4)
