# Reference script used to build data/raw/openimages_goat_sheep (needs fiftyone + MongoDB; run from the project root). Not part of the quick start.
import csv, shutil
from pathlib import Path
from collections import Counter
R = Path("/tmp/fozoo/open-images-v7/train")
cls = {r[1]: r[0] for r in csv.reader(open(R / "metadata/classes.csv"))}
want = {"Dog": 0, "Goat": 1, "Cattle": 2, "Cat": 3, "Sheep": 4}
mid = {cls[k]: v for k, v in want.items()}
imgs = {p.stem for p in (R / "data").glob("*.jpg")}
existing = {p.stem for p in Path("data/raw/openimages_yolo/images").glob("*.jpg")}
lab = {}
for r in csv.DictReader(open(R / "labels/detections.csv")):
    if r["ImageID"] in imgs and r["LabelName"] in mid:
        x1, x2, y1, y2 = (float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
        lab.setdefault(r["ImageID"], []).append((mid[r["LabelName"]], (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, r["IsGroupOf"]))
out = Path("data/raw/openimages_goat_sheep"); (out/"images").mkdir(parents=True, exist_ok=True); (out/"labels").mkdir(exist_ok=True)
c = Counter(); n = skipped = 0
for i, bs in lab.items():
    if i in existing: skipped += 1; continue
    shutil.copy(R/"data"/f"{i}.jpg", out/"images"/f"{i}.jpg")
    (out/"labels"/f"{i}.txt").write_text("".join(f"{a} {b:.6f} {cc:.6f} {d:.6f} {e:.6f}\n" for a,b,cc,d,e,g in bs))
    for b in bs: c[b[0]] += 1
    n += 1
print("images", n, "skipped_existing", skipped, "boxes per class id", dict(sorted(c.items())), "groupof", sum(b[5]=="1" for v in lab.values() for b in v))
(out/"README.txt").write_text("Open Images V7 train subset (600 images selected by FiftyOne for Goat/Sheep, seed 7). YOLO labels use project ids 0 dog,1 goat,2 cow(Cattle),3 cat,4 sheep. Includes all 5 classes present in each image. IsGroupOf boxes included (one box around a group). Images CC BY 2.0 / annotations CC BY 4.0 (per Open Images; check each image licence in OI metadata).\n")
