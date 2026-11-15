"""Build the no-whole-frame train list (data-A5).

Drops every train image from Mendeley sources (mendeley5, mendeley_yolo) that has a box covering >=98% of
the frame in both width and height. The whole image is dropped, never just
the box, so an unlabeled animal cannot become background.
Writes data/dataset/train_nowf.txt (entries "./images/train/<name>.jpg").
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "dataset"
THRESH = 0.98
NAMES = ["dog", "goat", "cow", "cat", "sheep"]


def is_whole(label_path):
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if float(parts[3]) >= THRESH and float(parts[4]) >= THRESH:
            return True
    return False


def class_counts(label_paths):
    c = Counter()
    for lp in label_paths:
        for line in lp.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


def main(dry_run=False):
    imgs = sorted((DS / "images" / "train").glob("*.jpg"))
    kept, dropped = [], []
    for img in imgs:
        lp = DS / "labels" / "train" / (img.stem + ".txt")
        if img.name.startswith(("mendeley5__", "mendeley_yolo__")) and lp.exists() and is_whole(lp):
            dropped.append(img)
        else:
            kept.append(img)
    print(f"train images: {len(imgs)}  dropped (whole-frame Mendeley): {len(dropped)}  kept: {len(kept)}")
    print("dropped by source:", Counter(p.name.split("__")[0] for p in dropped))
    kc_all = class_counts([DS / "labels" / "train" / (p.stem + ".txt") for p in imgs])
    kc_kept = class_counts([DS / "labels" / "train" / (p.stem + ".txt") for p in kept])
    print(f"{'class':<8}{'v2 train boxes':>16}{'nowf train boxes':>18}")
    for i, n in enumerate(NAMES):
        print(f"{n:<8}{kc_all[i]:>16}{kc_kept[i]:>18}")
    if dry_run:
        return
    out = DS / "train_nowf.txt"
    out.write_text("".join(f"./images/train/{p.name}\n" for p in kept))
    print("wrote", out)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
