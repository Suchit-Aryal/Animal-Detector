"""B1: evaluate pretrained baselines on our test split -> results/baseline.csv + results/baseline_mapping.md.

Models: yolov8n.pt (COCO) and yolov8m-oiv7.pt (Open Images V7, project root).
Their classes are mapped onto ours (dog 0, goat 1, cow 2, cat 3, sheep 4); all other classes are ignored.
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluate import ROOT, evaluate

# COCO has no goat -> goat AP stays NaN ("not supported").
COCO_MAP = {16: 0, 19: 2, 15: 3, 18: 4}          # dog, cow, cat, sheep
OIV7_MAP = {160: 0, 219: 1, 99: 2, 96: 3, 452: 4}  # Dog, Goat, Cattle, Cat, Sheep

NOTE = """# Baseline class mapping
Predictions of each pretrained model are mapped to our classes; all other predicted classes are dropped.
Metrics: official Ultralytics val via src/evaluate.py (conf 0.001, NMS IoU 0.7, imgsz 640, batch 1) on a temp copy of the
test set whose labels are remapped to the model's class ids. Classes with no test ground truth give NaN AP and are
excluded from the means.
Ground-truth boxes of classes a model cannot predict (goat for COCO) are left out of its metrics, so its goat AP is NaN
("not supported") instead of a 0 that would drag the mean down.

| Our class | COCO (yolov8n.pt) | Open Images V7 (yolov8m-oiv7.pt) |
|---|---|---|
| 0 dog | 16 dog | 160 Dog |
| 1 goat | NOT SUPPORTED (no goat class; AP = NaN) | 219 Goat |
| 2 cow | 19 cow | 99 Cattle |
| 3 cat | 15 cat | 96 Cat |
| 4 sheep | 18 sheep | 452 Sheep |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/dataset/data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--coco", default="yolov8n.pt")
    ap.add_argument("--oiv7", default=str(ROOT / "yolov8m-oiv7.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results/baseline.csv"))
    a = ap.parse_args()
    rows = []
    for name, w, m in [("baseline_coco_yolov8n", a.coco, COCO_MAP), ("baseline_oiv7_yolov8m", a.oiv7, OIV7_MAP)]:
        rows.append({"model": name, **evaluate(w, a.data, a.split, m, a.imgsz, device=a.device, max_images=a.max_images)})
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_name("baseline_mapping.md").write_text(NOTE)
    print(df.T.to_string())
    print(f"\nwrote {out} and baseline_mapping.md")


if __name__ == "__main__":
    main()
