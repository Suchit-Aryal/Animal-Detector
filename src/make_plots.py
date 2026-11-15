"""C2: plots + failure cases from real results only.
Outputs in results/figures/: comparison_bar.png, per_class_ap.png, confusion_matrix_test.png, loss_curves.png,
failure_*.jpg, plus results/failure_cases.md.
Usage: python src/make_plots.py [--weights results/weights/best.pt]
"""
import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from evaluate import ROOT, label_path, split_files

FIG = ROOT / "results/figures"
NAMES = ["dog", "goat", "cow", "cat", "sheep"]
RUN_FROZEN, RUN_PLAIN = "yolov8s_80ep_freeze10", "yolov8s_80ep"


def short(m):
    return {"baseline_coco_yolov8n": "COCO yolov8n\n(baseline)", "baseline_oiv7_yolov8m": "OIV7 yolov8m\n(baseline)"}.get(m, "yolov8s freeze=10\n(ours)")


def bar_charts():
    df = pd.read_csv(ROOT / "results/comparison.csv")
    labels = [short(m) for m in df["model"]]
    cols = ["mAP50", "mAP50-95", "precision", "recall"]
    x = np.arange(len(cols)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, r) in enumerate(df.iterrows()):
        b = ax.bar(x + (i - 1) * w, [r[c] for c in cols], w, label=labels[i].replace("\n", " "))
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, cols); ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Test set (80 Open Images imgs): baselines vs final\n(OIV7 was pretrained on Open Images: likely inflated)", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "comparison_bar.png", dpi=150); plt.close(fig)

    cls = ["dog", "goat", "cow", "cat"]
    x = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, r) in enumerate(df.iterrows()):
        vals = [r[f"AP50_{c}"] for c in cls]
        b = ax.bar(x + (i - 1) * w, np.nan_to_num(vals), w, label=labels[i].replace("\n", " "))
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, 0.02 if np.isnan(v) else v + 0.01, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, cls); ax.set_ylim(0, 1.1); ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class AP50 on test (sheep not evaluated: 0 test boxes; COCO has no goat; goat test = 22 boxes)", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "per_class_ap.png", dpi=150); plt.close(fig)


def loss_curves():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, run, title in [(axs[0], RUN_PLAIN, "yolov8s, all layers trained"), (axs[1], RUN_FROZEN, "yolov8s, freeze=10")]:
        d = pd.read_csv(ROOT / f"runs/detect/{run}/results.csv"); d.columns = [c.strip() for c in d.columns]
        ax.plot(d["epoch"], d["train/cls_loss"], label="train cls loss"); ax.plot(d["epoch"], d["val/cls_loss"], label="val cls loss")
        ax.set_title(title, fontsize=10); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    axs[0].set_ylabel("class loss"); fig.suptitle("Overfitting gap (train vs val) shrinks with freeze=10")
    fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=150); plt.close(fig)


def confusion(weights):
    with tempfile.TemporaryDirectory() as td:
        r = YOLO(weights).val(data=str(ROOT / "data/dataset/data.yaml"), split="test", batch=4, imgsz=640, plots=True,
                              project=td, name="cm", exist_ok=True, verbose=False, device=0)
        shutil.copy(Path(r.save_dir) / "confusion_matrix_normalized.png", FIG / "confusion_matrix_test.png")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def failures(weights, conf=0.25):
    imgs, _ = split_files(ROOT / "data/dataset/data.yaml", "test")
    model = YOLO(weights)
    found = {"wrong_class": [], "miss": [], "false_positive": [], "group": []}
    for im in imgs:
        img = cv2.imread(str(im)); h, w = img.shape[:2]
        gt = []
        lp = label_path(im)
        for ln in (lp.read_text().splitlines() if lp.exists() else []):
            t = ln.split()
            if len(t) == 5:
                c, x, y, bw, bh = int(t[0]), *map(float, t[1:])
                gt.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
        b = model.predict(img, conf=conf, imgsz=640, verbose=False, device=0)[0].boxes
        det = [(int(c), float(s), xy) for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        used = set(); ev = []
        for gc, gb in gt:
            best = max(((iou(gb, d[2]), j) for j, d in enumerate(det)), default=(0, -1))
            if best[0] >= 0.5:
                d = det[best[1]]; used.add(best[1])
                if d[0] != gc:
                    ev.append(("wrong_class", f"{NAMES[gc]} labelled, predicted {NAMES[d[0]]} ({d[1]:.2f})", gb, d))
            else:
                ev.append(("miss", f"{NAMES[gc]} not detected (conf>={conf})", gb, None))
        for j, d in enumerate(det):
            if j not in used and all(iou(d[2], g[1]) < 0.3 for g in gt):
                ev.append(("false_positive", f"predicted {NAMES[d[0]]} ({d[1]:.2f}) where there is no labelled animal", None, d))
        for kind, why, gb, d in ev:
            found[kind].append((len(gt), d[1] if d else 0, im, why, gt, det))
        if len(gt) >= 4 and ev:
            found["group"].append((len(gt), 0, im, f"crowded scene: {len(gt)} labelled animals, {len(ev)} error(s): " + "; ".join(e[1] for e in ev), gt, det))
    lines = ["# Failure cases (test set, conf >= 0.25, IoU 0.5)",
             f"Model: {weights}. Green = ground truth, red = prediction. Counts of events over all 80 test images: " +
             ", ".join(f"{k} {len(v)}" for k, v in found.items()) + ".\nExplanations list what the data shows; causes are not verified.\n"]
    used_imgs = set()
    for kind in ["group", "wrong_class", "miss", "false_positive"]:
        cand = sorted(found[kind], key=lambda e: -e[1] if kind in ("wrong_class", "false_positive") else -e[0])
        pick = next((e for e in cand if e[2] not in used_imgs), cand[0] if cand else None)
        if not pick:
            lines.append(f"## {kind}\nNone found in the test set.\n"); continue
        _, _, im, why, gt, det = pick; used_imgs.add(im)
        img = cv2.imread(str(im))
        for gc, gb in gt:
            cv2.rectangle(img, tuple(map(int, gb[:2])), tuple(map(int, gb[2:])), (0, 200, 0), 2)
            cv2.putText(img, NAMES[gc], (int(gb[0]), int(gb[1]) + 14), 0, 0.5, (0, 200, 0), 2)
        for c, s, xy in det:
            cv2.rectangle(img, tuple(map(int, xy[:2])), tuple(map(int, xy[2:])), (0, 0, 255), 2)
            cv2.putText(img, f"{NAMES[c]} {s:.2f}", (int(xy[0]), int(xy[3]) - 4), 0, 0.5, (0, 0, 255), 2)
        out = f"failure_{kind}.jpg"; cv2.imwrite(str(FIG / out), img)
        lines.append(f"## {kind}\n![{kind}](figures/{out})\n{im.name}: {why}.\n")
    (ROOT / "results/failure_cases.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=str(ROOT / "results/weights/best.pt"))
    a = ap.parse_args(); FIG.mkdir(parents=True, exist_ok=True)
    bar_charts(); loss_curves(); confusion(a.weights); failures(a.weights)


if __name__ == "__main__":
    main()
