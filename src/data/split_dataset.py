"""
Task A3: split a YOLO dataset into train/val/test by video or scene, never by frame.

Input : one or more YOLO folders, each with images/ and labels/ (same file stems).
Output: <out>/images/{train,val,test}/, <out>/labels/{train,val,test}/,
        <out>/split_manifest.csv (file, group, split), <out>/data.yaml

Grouping ("video or scene"):
- an image nested in a sub-folder (e.g. frames/<video_name>/frame_000012.jpg)
  belongs to the group <video_name>; all its frames go to the same split.
- an image directly in images/ is its own group (no video info available).

Splits 70/20/10 by default, assigned by image count. Copies files (never moves
or edits the sources). Output names are flattened to avoid collisions:
<source_name>__<sub-folders>__<stem>.jpg

Checks at the end: no group appears in more than one split (fails with exit 1 if so).

Run from the project root:
    python src/data/split_dataset.py --src data/raw/openimages_yolo --out data/dataset
    python src/data/split_dataset.py --src data/raw/frames_src --src data/raw/mendeley_yolo --clean
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ["dog", "goat", "cow", "cat", "sheep"]


def collect(src: Path) -> list:
    """Return [(source_path, group, flat_name)] for every image in src/images/."""
    images_dir = src / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"ERROR: {images_dir} not found")
    items = []
    for img in sorted(images_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        rel = img.relative_to(images_dir)
        parts = rel.parts[:-1]  # sub-folders
        if parts:
            group = f"{src.name}/{'/'.join(parts)}"
        else:
            group = f"{src.name}/{img.stem}"
        flat = "__".join([src.name, *parts, img.stem]) + img.suffix.lower()
        items.append((img, group, flat))
    return items


def assign_splits(groups: dict, ratios: dict, seed: int) -> dict:
    """Give each group a split so image counts approach the ratios. Returns {group: split}."""
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    target = {s: ratios[s] * total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    result = {}
    # largest groups first so small remainders can balance the totals
    for key in sorted(keys, key=lambda k: -len(groups[k])):
        split = max(SPLITS, key=lambda s: target[s] - current[s])
        result[key] = split
        current[split] += len(groups[key])
    return result


def write_yaml(out: Path, data_path: str) -> None:
    lines = [
        f"path: {data_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", action="append", default=[], type=Path,
                        help="YOLO source folder split 70/20/10 (repeat for several)")
    parser.add_argument("--train-only-src", action="append", default=[], type=Path,
                        help="YOLO source that goes ONLY into train (e.g. model-labelled data); never val/test")
    parser.add_argument("--out", default="data/dataset", type=Path)
    parser.add_argument("--ratios", default="70,20,10", help="train,val,test percentages")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--clean", action="store_true", help="delete the existing output images/labels first")
    parser.add_argument("--exclude-file", type=Path, default=None,
                        help="text file: one image stem per line to leave out (spot-check rejects)")
    args = parser.parse_args()

    try:
        parts = [float(x) for x in args.ratios.split(",")]
        assert len(parts) == 3 and abs(sum(parts) - 100) < 1e-6
    except (ValueError, AssertionError):
        print("ERROR: --ratios must be three numbers summing to 100, e.g. 70,20,10", file=sys.stderr)
        return 2
    ratios = dict(zip(SPLITS, [p / 100 for p in parts]))

    out = args.out
    if (out / "images").exists() or (out / "labels").exists():
        if not args.clean:
            print(f"ERROR: {out} already has images/ or labels/. Re-run with --clean to replace them.", file=sys.stderr)
            return 2
        shutil.rmtree(out / "images", ignore_errors=True)
        shutil.rmtree(out / "labels", ignore_errors=True)

    excluded = set()
    if args.exclude_file:
        excluded = {ln.strip() for ln in args.exclude_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        print(f"excluding {len(excluded)} image(s) listed in {args.exclude_file}")
    items = []
    train_only_items = []
    for src in args.src + args.train_only_src:
        found = [f for f in collect(src) if f[0].stem not in excluded]
        tag = "train-only" if src in args.train_only_src else "split 70/20/10"
        print(f"source {src} ({tag}): {len(found)} images")
        target = train_only_items if src in args.train_only_src else items
        target.extend((src, *it) for it in found)
    if not items and not train_only_items:
        print("ERROR: no images found in the source folders", file=sys.stderr)
        return 2

    groups: dict = defaultdict(list)
    for item in items:
        groups[item[2]].append(item)
    assignment = assign_splits(groups, ratios, args.seed) if groups else {}
    # train-only sources: every image goes to train, each image is its own group
    for item in train_only_items:
        groups[item[2]].append(item)
        assignment[item[2]] = "train"

    manifest = []
    missing_labels = 0
    counts: dict = defaultdict(int)
    for group, members in groups.items():
        split = assignment[group]
        for src, img, _, flat in members:
            label = src / "labels" / img.relative_to(src / "images").with_suffix(".txt")
            if not label.exists():
                missing_labels += 1
                label = None
            img_out = out / "images" / split / flat
            lbl_out = out / "labels" / split / (Path(flat).stem + ".txt")
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_out)
            if label is not None:
                shutil.copy2(label, lbl_out)
            else:
                lbl_out.write_text("")  # background image, empty label
            manifest.append((flat, group, split))
            counts[split] += 1

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "group", "split"])
        writer.writerows(manifest)
    write_yaml(out, str(out).replace("\\", "/"))

    # check: no group in two splits
    group_splits: dict = defaultdict(set)
    for _, group, split in manifest:
        group_splits[group].add(split)
    shared = {g: s for g, s in group_splits.items() if len(s) > 1}

    print("Images per split:", ", ".join(f"{s}={counts[s]}" for s in SPLITS))
    print(f"Groups (videos/scenes): {len(groups)}")
    print(f"Images without a label file (written as background): {missing_labels}")
    print(f"Wrote {out / 'data.yaml'} and {out / 'split_manifest.csv'}")
    if shared:
        for g, s in list(shared.items())[:10]:
            print(f"ERROR group {g} is in splits {sorted(s)}")
        print("FAILED: some groups are split across splits")
        return 1
    print("CHECK PASSED: no group appears in more than one split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
