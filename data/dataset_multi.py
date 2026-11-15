"""
Open Images V7 subset (Dog, Goat, Cattle, Cat) -> YOLO format, WITHOUT FiftyOne.

Our class ids:  0 dog, 1 goat, 2 cow (Open Images "Cattle"), 3 cat

It reuses the metadata FiftyOne already downloaded for you:
    ~/fiftyone/open-images-v7/train/labels/detections.csv
    ~/fiftyone/open-images-v7/train/metadata/classes.csv
and the dog images that are already in ~/fiftyone/open-images-v7/train/data/.
Other images are downloaded from the public Open Images S3 bucket.

Run from your PROJECT ROOT (the folder that contains data/):
    conda activate animal-det
    python download_openimages_nofo.py              # 400 images per class
    python download_openimages_nofo.py --n 200      # smaller / faster test run

Output:
    data/raw/openimages_yolo/images/<ImageID>.jpg
    data/raw/openimages_yolo/labels/<ImageID>.txt

Choices made (so you can explain them in Q&A):
  * Images that contain "group" boxes (one box around a herd) or drawings/cartoons
    are skipped, so the labels stay one-box-per-animal and photographic.
  * Every box of our 4 classes in a chosen image is written, not only the class
    that caused the image to be picked.

NOTE: untested on your machine. If something fails, paste the last lines of the error.
"""

import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

# (Open Images display name, our class id)
CLASSES = [("Dog", 0), ("Goat", 1), ("Cattle", 2), ("Cat", 3)]

OI_DIR = Path.home() / "fiftyone" / "open-images-v7" / "train"
DET_CSV = OI_DIR / "labels" / "detections.csv"
CLASSES_CSV = OI_DIR / "metadata" / "classes.csv"
LOCAL_IMG_DIR = OI_DIR / "data"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="images per class")
    ap.add_argument("--out", default="data/raw/openimages_yolo")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for f in (DET_CSV, CLASSES_CSV):
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run the FiftyOne download once so these files exist.")

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1) class names -> Open Images label ids (MIDs)
    cls_df = pd.read_csv(CLASSES_CSV, header=None, names=["mid", "name"])
    mid_to_id = {}
    for name, cid in CLASSES:
        row = cls_df[cls_df["name"] == name]
        if row.empty:
            raise SystemExit(f"Class '{name}' not found in {CLASSES_CSV}")
        mid_to_id[row.iloc[0]["mid"]] = cid
    print("Class ids:", {m: c for m, c in mid_to_id.items()})

    # 2) read the big detections file in chunks, keep only our 4 classes
    print("Reading detections.csv (large file, this can take a minute)...")
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]
    frames = []
    for chunk in pd.read_csv(DET_CSV, usecols=cols, chunksize=2_000_000):
        chunk = chunk[chunk["LabelName"].isin(mid_to_id)]
        if len(chunk):
            frames.append(chunk)
    det = pd.concat(frames, ignore_index=True)
    det["cls"] = det["LabelName"].map(mid_to_id)

    # drop images with group boxes or drawings/cartoons
    bad = set(det.loc[(det["IsGroupOf"] == 1) | (det["IsDepiction"] == 1), "ImageID"])
    det = det[~det["ImageID"].isin(bad)]
    print(f"Kept {len(det):,} boxes in {det['ImageID'].nunique():,} images after filtering")

    # 3) choose images per class (prefer ones already on disk)
    local_ids = {p.stem for p in LOCAL_IMG_DIR.glob("*.jpg")} if LOCAL_IMG_DIR.exists() else set()
    rng = random.Random(args.seed)
    selected = set()
    for name, cid in CLASSES:
        ids = det.loc[det["cls"] == cid, "ImageID"].unique().tolist()
        local = [i for i in ids if i in local_ids]
        rng.shuffle(local)
        pick = local[: args.n]
        if len(pick) < args.n:
            picked = set(pick)
            rest = [i for i in ids if i not in picked]
            rng.shuffle(rest)
            pick += rest[: args.n - len(pick)]
        print(f"{name}: {len(pick)} images chosen ({sum(i in local_ids for i in pick)} already local)")
        selected.update(pick)

    # 4) get the images (copy from FiftyOne folder or download)
    def fetch(image_id):
        dst = img_dir / f"{image_id}.jpg"
        if dst.exists():
            return image_id, True
        src = LOCAL_IMG_DIR / f"{image_id}.jpg"
        if src.exists():
            shutil.copy2(src, dst)
            return image_id, True
        for _ in range(3):
            try:
                r = requests.get(IMG_URL.format(image_id), timeout=30)
                if r.status_code == 200 and r.content:
                    dst.write_bytes(r.content)
                    return image_id, True
            except requests.RequestException:
                pass
        return image_id, False

    print(f"Fetching {len(selected)} images...")
    ok_ids, failed = set(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (image_id, ok) in enumerate(ex.map(fetch, sorted(selected)), 1):
            if ok:
                ok_ids.add(image_id)
            else:
                failed += 1
            if k % 100 == 0:
                print(f"  {k}/{len(selected)} done ({failed} failed)")

    # 5) write YOLO label files: class cx cy w h (all normalized 0..1)
    counts = {cid: 0 for _, cid in CLASSES}
    sub = det[det["ImageID"].isin(ok_ids)]
    for image_id, g in sub.groupby("ImageID"):
        lines = []
        for r in g.itertuples():
            w, h = r.XMax - r.XMin, r.YMax - r.YMin
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{r.cls} {(r.XMin + r.XMax) / 2:.6f} {(r.YMin + r.YMax) / 2:.6f} {w:.6f} {h:.6f}")
            counts[r.cls] += 1
        if lines:
            (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n")

    print("\nDone.")
    print(f"Images saved: {len(ok_ids)}  (failed downloads: {failed})")
    print("Boxes per class:", {name: counts[cid] for name, cid in CLASSES})
    print(f"Output folder: {out.resolve()}")
    print("Next: open ~30 random images with their boxes drawn and check them by eye.")


if __name__ == "__main__":
    main()