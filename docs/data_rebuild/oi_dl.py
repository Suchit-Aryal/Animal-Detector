# Reference script used to build data/raw/openimages_goat_sheep (needs fiftyone + MongoDB; run from the project root). Not part of the quick start.
import os, sys, shutil
from pathlib import Path
import fiftyone as fo, fiftyone.zoo as foz
fo.config.dataset_zoo_dir = "/tmp/fozoo"
fo.config.default_dataset_dir = "/tmp/fozoo"
existing = {p.stem for p in Path("data/raw/openimages_yolo/images").glob("*.jpg")}
out = Path("data/raw/openimages_goat_sheep")
(out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
ds = foz.load_zoo_dataset("open-images-v7", split="train", label_types=["detections"], classes=["Goat", "Sheep"],
                          max_samples=int(sys.argv[1]), seed=7, shuffle=True, dataset_name="oi_gs", only_matching=True)
names = {"Goat": 1, "Sheep": 4}
cnt = {"Goat": 0, "Sheep": 0}; n = 0; skipped = 0
for s in ds:
    stem = Path(s.filepath).stem
    if stem in existing: skipped += 1; continue
    lines = []
    for d in s.ground_truth.detections:
        if d.label in names:
            x, y, w, h = d.bounding_box
            lines.append(f"{names[d.label]} {x+w/2:.6f} {y+h/2:.6f} {w:.6f} {h:.6f}"); cnt[d.label] += 1
    if not lines: continue
    shutil.copy(s.filepath, out / "images" / (stem + ".jpg")); (out / "labels" / (stem + ".txt")).write_text("\n".join(lines) + "\n"); n += 1
print("images", n, "boxes", cnt, "skipped_existing", skipped)
