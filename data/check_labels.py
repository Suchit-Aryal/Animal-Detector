"""
Draw YOLO boxes on random images so you can check labels by eye.

Class ids: 0 dog, 1 goat, 2 cow, 3 cat, 4 sheep

Usage (from the project root, inside the conda env):
    python check_labels.py --root data/raw/openimages_yolo --n 30

Expects:  <root>/images/*.jpg   and   <root>/labels/*.txt
Writes annotated copies to:  results/label_check/<root name>/
Then open that folder and look at the pictures. Note down any bad ones.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = {0: "dog", 1: "goat", 2: "cow", 3: "cat", 4: "sheep"}
COLORS = {0: "red", 1: "blue", 2: "lime", 3: "orange", 4: "magenta"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    imgs = sorted((root / "images").glob("*.*"))
    random.Random(args.seed).shuffle(imgs)

    out = Path("results/label_check") / root.name
    out.mkdir(parents=True, exist_ok=True)

    done = 0
    for img_path in imgs:
        lbl = root / "labels" / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                print(f"Bad label line in {lbl.name}: {line!r}")
                continue
            c = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            col = COLORS.get(c, "white")
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
            d.text((x1 + 4, y1 + 2), NAMES.get(c, str(c)), fill=col)
        im.save(out / f"{img_path.stem}.jpg")
        done += 1
        if done >= args.n:
            break

    print(f"Saved {done} annotated images to {out.resolve()}")


if __name__ == "__main__":
    main()