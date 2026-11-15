"""
Task A2: check a YOLO-format dataset before training.

Checks
- every image has a label .txt with the same name (an empty .txt is allowed = background image)
- every label line is "class x_center y_center width height"
- class ids are only 0..4 (dog, goat, cow, cat, sheep)
- box values are between 0 and 1 and width/height are > 0
- every image opens (not corrupt); every label has an image
- optional: duplicate images by perceptual hash (--dedupe, needs `imagehash`)

Prints per-class box counts and per-split image counts.
Exit code 0 = clean, 1 = problems found, 2 = bad arguments.

Accepted layouts
- data/dataset/images/{train,val,test} + data/dataset/labels/{train,val,test}
- a flat folder: <root>/images + <root>/labels (treated as one split named "all")

Run from the project root:
    python src/data/validate_dataset.py --root data/dataset
    python src/data/validate_dataset.py --root data/raw/openimages_yolo
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 5
CLASS_NAMES = ["dog", "goat", "cow", "cat", "sheep"]


def find_splits(root: Path) -> dict:
    """Return {split_name: (images_dir, labels_dir)}."""
    images = root / "images"
    labels = root / "labels"
    if not images.is_dir():
        return {}
    subdirs = sorted(p for p in images.iterdir() if p.is_dir())
    if subdirs:
        return {p.name: (p, labels / p.name) for p in subdirs}
    return {"all": (images, labels)}


def list_images(folder: Path) -> list:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def check_label_file(path: Path, errors: list, counts: Counter) -> None:
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path}:{lineno}"
        if len(parts) != 5:
            errors.append(f"{where}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{where}: non-numeric value in '{line.strip()}'")
            continue
        if not 0 <= cls < NUM_CLASSES:
            errors.append(f"{where}: class id {cls} not in 0..{NUM_CLASSES - 1}")
            continue
        xc, yc, w, h = vals
        if not all(0.0 <= v <= 1.0 for v in vals):
            errors.append(f"{where}: box value outside 0..1 in '{line.strip()}'")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{where}: width/height must be > 0 in '{line.strip()}'")
            continue
        counts[cls] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/dataset", type=Path, help="dataset folder")
    parser.add_argument("--dedupe", action="store_true", help="also report near-duplicate images (needs imagehash)")
    args = parser.parse_args()

    splits = find_splits(args.root)
    if not splits:
        print(f"ERROR: no images/ folder under {args.root}", file=sys.stderr)
        return 2

    errors: list = []
    warnings: list = []
    box_counts: Counter = Counter()
    split_counts: dict = {}
    all_hashes: dict = {}

    for split, (img_dir, lbl_dir) in splits.items():
        images = list_images(img_dir)
        split_counts[split] = len(images)
        image_stems = {p.stem for p in images}

        for img in images:
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                errors.append(f"{img}: missing label file {label}")
            else:
                check_label_file(label, errors, box_counts)
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as exc:  # corrupt or unreadable image
                errors.append(f"{img}: unreadable image ({exc})")
            if args.dedupe:
                try:
                    import imagehash
                    with Image.open(img) as im:
                        all_hashes.setdefault(str(imagehash.phash(im)), []).append(str(img))
                except ImportError:
                    warnings.append("--dedupe skipped: imagehash is not installed")
                    args.dedupe = False

        if lbl_dir.is_dir():
            for lbl in sorted(lbl_dir.glob("*.txt")):
                if lbl.stem not in image_stems:
                    errors.append(f"{lbl}: label has no matching image")

    if args.dedupe:
        for h, files in all_hashes.items():
            if len(files) > 1:
                warnings.append(f"near-duplicates (same phash): {', '.join(files)}")

    print("Per-split image counts:")
    for split, n in split_counts.items():
        print(f"  {split:<8} {n}")
    print("Per-class box counts:")
    for cls in range(NUM_CLASSES):
        print(f"  {cls} {CLASS_NAMES[cls]:<6} {box_counts.get(cls, 0)}")
    print(f"Total boxes: {sum(box_counts.values())}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        return 1
    print("PASSED: no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
