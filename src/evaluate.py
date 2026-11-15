"""B3: evaluate a detector on the test split (mAP, precision, recall, FPS, per-class AP).

The official Ultralytics val is used (on a temp copy of the test set with labels remapped to the model's class ids),
so numbers match a manual `yolo val`. Same code path is used for the baselines (B1) and the fine-tuned model, so the numbers are comparable.
Predictions are mapped onto our classes {0 dog, 1 goat, 2 cow, 3 cat, 4 sheep}; other classes are dropped.

Usage:
  python src/evaluate.py --weights results/weights/best.pt --data data/dataset/data.yaml
  -> results/finetuned.csv and results/comparison.csv (needs results/baseline.csv from B1)
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUR = {0: "dog", 1: "goat", 2: "cow", 3: "cat", 4: "sheep"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_files(data_yaml, split):
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(cfg.get("path", Path(data_yaml).parent))
    if not base.is_absolute():  # try next to the yaml, then the project root, then the cwd
        cands = [(Path(data_yaml).resolve().parent / base).resolve(), (ROOT / base).resolve(), (Path.cwd() / base).resolve()]
        base = next((c for c in cands if (c / cfg[split]).exists()), cands[0])
    d = base / cfg[split]
    imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
    return imgs, cfg


def label_path(img):
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def build_tmp_dataset(imgs, cls_map, tmp):
    """Symlink the test images and write their labels remapped to the MODEL's class ids (ours -> model).
    Labels of classes the model cannot predict are dropped. Lets us use the official Ultralytics val on any model."""
    inv = {v: k for k, v in cls_map.items()} if cls_map else None
    (tmp / "images/test").mkdir(parents=True); (tmp / "labels/test").mkdir(parents=True)
    for im in imgs:
        (tmp / "images/test" / im.name).symlink_to(im.resolve())
        lines = []
        lp = label_path(im)
        if lp.exists():
            for ln in lp.read_text().splitlines():
                t = ln.split()
                if not t:
                    continue
                c = int(float(t[0]))
                if inv is None:
                    lines.append(ln)
                elif c in inv:
                    lines.append(" ".join([str(inv[c])] + t[1:]))
        (tmp / "labels/test" / (im.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))


def evaluate(weights, data_yaml, split="test", cls_map=None, imgsz=640, device=None, max_images=None):
    """Official Ultralytics val (conf 0.001, NMS IoU 0.7, batch 1 so padding is deterministic).
    cls_map: {model_class_id: our_class_id}; None = model already uses our ids.
    GT of classes the model cannot predict is dropped, so their AP is NaN and they are not in the means."""
    imgs, cfg = split_files(data_yaml, split)
    if max_images:
        imgs = imgs[:max_images]
    if not imgs:
        raise SystemExit(f"No images found for split '{split}' in {data_yaml}")
    names = {int(k): v for k, v in cfg["names"].items()} if isinstance(cfg["names"], dict) else dict(enumerate(cfg["names"]))
    model = YOLO(str(weights))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        build_tmp_dataset(imgs, cls_map, tmp)
        (tmp / "data.yaml").write_text(yaml.safe_dump({"path": str(tmp), "train": "images/test", "val": "images/test",
                                                        "test": "images/test", "names": {int(k): v for k, v in model.names.items()}}))
        r = model.val(data=str(tmp / "data.yaml"), split="test", imgsz=imgsz, batch=1, conf=0.001, iou=0.7,
                      device=device, plots=False, verbose=False, project=str(tmp / "v"), name="v", save=False)
    box = r.box
    per = {}
    for i, c in enumerate(box.ap_class_index):
        per[cls_map[int(c)] if cls_map else int(c)] = float(box.ap50[i])
    # FPS: model.predict end to end (read + preprocess + inference + NMS), one image at a time
    model.predict(str(imgs[0]), imgsz=imgsz, verbose=False, device=device)  # warm-up
    t0 = time.perf_counter()
    for im in imgs:
        model.predict(str(im), imgsz=imgsz, conf=0.25, verbose=False, device=device)
    dt = (time.perf_counter() - t0) / len(imgs)
    row = {"images": len(imgs), "precision": float(box.mp), "recall": float(box.mr),
           "mAP50": float(box.map50), "mAP50-95": float(box.map), "FPS": 1.0 / dt, "ms_per_image": dt * 1000}
    for c, n in names.items():
        row[f"AP50_{n}"] = per.get(c, np.nan)  # NaN = no test ground truth for this class, or model does not support it
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "results/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/dataset/data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results/finetuned.csv"))
    ap.add_argument("--baseline", default=str(ROOT / "results/baseline.csv"))
    ap.add_argument("--comparison", default=str(ROOT / "results/comparison.csv"))
    a = ap.parse_args()
    row = {"model": "finetuned", **evaluate(a.weights, a.data, a.split, None, a.imgsz, device=a.device, max_images=a.max_images)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ft = pd.DataFrame([row]); ft.to_csv(a.out, index=False)
    print(ft.T.to_string())
    if Path(a.baseline).exists():
        comp = pd.concat([pd.read_csv(a.baseline), ft], ignore_index=True)
        comp.to_csv(a.comparison, index=False)
        print(f"\nwrote {a.comparison}\n", comp[["model", "mAP50", "mAP50-95", "precision", "recall", "FPS"]].to_string(index=False))
    else:
        print(f"(no {a.baseline}; run src/baseline_eval.py first for comparison.csv)")


if __name__ == "__main__":
    main()
