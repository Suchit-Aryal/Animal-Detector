"""
Ingest Roboflow YOLOv8 exports into our 5-class format.

Input : data/raw/roboflow/<owner>_<project>/ (each a Roboflow export, zip already
        extracted). Must contain data.yaml plus train/valid/test folders with
        images/ and labels/ (Roboflow's standard YOLOv8 layout).
Mapping: src/data/roboflow_mapping.json, keyed by folder name, e.g.
        {"fyp-gztsq_stray-dogs-2wfjc": {"stray dog": "dog", "other": "drop"}}
        Values are one of: dog, goat, cow, cat, sheep, drop.
        A folder with no mapping entry is SKIPPED (we never guess a mapping).
Output: data/raw/roboflow_yolo/images/{train,val}/ and labels/{train,val}/
        Roboflow "valid" and "test" splits both go to our "val". Public data never goes to test.
        Dedupe: perceptual difference hash (dHash, 64 bits) across all output images;
        a later duplicate is removed (train is kept in preference to val).

Images with no boxes left after remapping are dropped (not background examples).
Source files are only read; nothing under data/raw/ is changed except new files in data/raw/roboflow_yolo/.

Run from the project root:
    python src/data/roboflow_remap.py --list                # show each folder's class names and mapping status
    python src/data/roboflow_remap.py --test-fixture        # build + check on a tiny fake set in a temp dir
    python src/data/roboflow_remap.py                       # process all mapped folders, then dedupe
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

OUR_NAMES = ["dog", "goat", "cow", "cat", "sheep"]
OUR_IDS = {n: i for i, n in enumerate(OUR_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "val"}  # public data never goes to test
MAPPING_FILE = Path(__file__).with_name("roboflow_mapping.json")


def read_names(data_yaml: Path) -> list:
    """Read class names from a Roboflow data.yaml without PyYAML (not installed here)."""
    text = data_yaml.read_text()
    inline = re.search(r"^names:\s*\[(.*?)\]", text, re.M)
    if inline:
        return [s.strip().strip("'\"") for s in inline.group(1).split(",") if s.strip()]
    block = re.search(r"^names:\s*\n((?:\s+-\s*.*\n?)+)", text, re.M)
    if block:
        return [re.sub(r"^\s+-\s*", "", line).strip().strip("'\"") for line in block.group(1).splitlines() if line.strip()]
    dict_form = re.findall(r"^\s+(\d+):\s*(.+)$", text, re.M)
    if dict_form:
        return [v.strip().strip("'\"") for _, v in sorted(dict_form, key=lambda x: int(x[0]))]
    raise ValueError(f"no 'names' found in {data_yaml}")


def load_mapping() -> dict:
    if MAPPING_FILE.exists():
        return json.loads(MAPPING_FILE.read_text())
    return {}


def dhash(path: Path) -> int:
    """64-bit difference hash: shrink to 9x8 grayscale, compare neighbours."""
    with Image.open(path) as im:
        small = im.convert("L").resize((9, 8), Image.LANCZOS)
        px = list(small.get_flattened_data()) if hasattr(small, 'get_flattened_data') else list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_split_dirs(src: Path) -> dict:
    """Return {roboflow_split: (images_dir, labels_dir)} for the splits that exist."""
    found = {}
    for split in ("train", "valid", "val", "test"):
        img, lbl = src / split / "images", src / split / "labels"
        if img.is_dir():
            found[split] = (img, lbl)
    return found


def remap_source(src: Path, mapping: dict, out: Path, stats: Counter, report: list) -> None:
    names = read_names(src / "data.yaml")
    id_for = {}
    for idx, name in enumerate(names):
        target = mapping.get(name)
        if target is None:
            raise ValueError(f"{src.name}: class '{name}' has no mapping entry (add it to roboflow_mapping.json)")
        if target == "drop":
            id_for[idx] = None
        elif target in OUR_IDS:
            id_for[idx] = OUR_IDS[target]
        else:
            raise ValueError(f"{src.name}: class '{name}' maps to unknown target '{target}'")
    tag = src.name
    for rf_split, (img_dir, lbl_dir) in find_split_dirs(src).items():
        our_split = SPLIT_MAP[rf_split]
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            label = lbl_dir / f"{img.stem}.txt"
            kept = []
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    new_id = id_for.get(int(parts[0]))
                    if new_id is not None:
                        kept.append(f"{new_id} {' '.join(parts[1:])}")
            stats[("seen", tag)] += 1
            if not kept:
                stats[("no_boxes_after_remap", tag)] += 1
                continue
            flat = f"{tag}__{rf_split}__{img.stem}"
            (out / "images" / our_split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / our_split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / our_split / f"{flat}{img.suffix.lower()}")
            (out / "labels" / our_split / f"{flat}.txt").write_text("\n".join(kept) + "\n")
            stats[("kept", tag, our_split)] += 1
            for line in kept:
                stats[("box", OUR_NAMES[int(line.split()[0])])] += 1
    report.append(f"{tag}: classes {names} -> mapped")


def dedupe(out: Path, max_dist: int, stats: Counter) -> None:
    """Remove near-duplicate images across all splits. train is processed first so it wins over val."""
    seen = []  # (hash, path)
    for split in ("train", "val"):
        img_dir = out / "images" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            h = dhash(img)
            dup = next((p for ph, p in seen if hamming(ph, h) <= max_dist), None)
            if dup is not None:
                img.unlink()
                (out / "labels" / split / f"{img.stem}.txt").unlink(missing_ok=True)
                stats["duplicates_removed"] += 1
            else:
                seen.append((h, img))


def count_boxes(out: Path) -> Counter:
    counts = Counter()
    for lbl in (out / "labels").rglob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                counts[OUR_NAMES[int(line.split()[0])]] += 1
    return counts


def run(src_root: Path, out: Path, max_dist: int, mapping: dict) -> int:
    stats: Counter = Counter()
    report: list = []
    folders = sorted(p for p in src_root.iterdir() if (p / "data.yaml").exists())
    if not folders:
        print(f"No Roboflow exports (folders with data.yaml) under {src_root}")
        return 1
    if out.exists():
        shutil.rmtree(out)
    for folder in folders:
        if folder.name not in mapping:
            print(f"SKIP {folder.name}: no mapping in roboflow_mapping.json (read its class names first)")
            continue
        remap_source(folder, mapping[folder.name], out, stats, report)
    dedupe(out, max_dist, stats)
    for line in report:
        print(line)
    print("Images kept per source/split:")
    for key, n in sorted(((k, v) for k, v in stats.items() if k[0] == "kept"), key=str):
        print(f"  {key[1]} -> {key[2]}: {n}")
    print("Seen per source:", {k[1]: v for k, v in stats.items() if k[0] == "seen"})
    print("Dropped (no boxes after remap):", {k[1]: v for k, v in stats.items() if k[0] == "no_boxes_after_remap"})
    print(f"Duplicates removed (dHash distance <= {max_dist}): {stats['duplicates_removed']}")
    print("Boxes per class after dedupe:", dict(count_boxes(out)))
    return 0


def list_folders(src_root: Path) -> None:
    mapping = load_mapping()
    for folder in sorted(p for p in src_root.iterdir() if (p / "data.yaml").exists()):
        names = read_names(folder / "data.yaml")
        status = "mapped" if folder.name in mapping else "NO MAPPING (skipped)"
        print(f"{folder.name}: {names} -> {status}")


def test_fixture() -> int:
    """Build a tiny fake export (2 splits, 2 classes incl. one to drop) and run the whole pipeline."""
    tmp = Path(tempfile.mkdtemp(prefix="rf_fixture_"))
    try:
        src_root = tmp / "roboflow"
        folder = src_root / "fake_owner_fake-dogs"
        for split in ("train", "valid", "test"):
            (folder / split / "images").mkdir(parents=True)
            (folder / split / "labels").mkdir(parents=True)
        (folder / "data.yaml").write_text("nc: 2\nnames: ['stray dog', 'other']\n")
        from PIL import ImageDraw
        for s_idx, (split, n) in enumerate((("train", 3), ("valid", 2), ("test", 2))):
            for i in range(n):
                # each image gets its own layout so only the planted duplicate is a duplicate
                im = Image.new("RGB", (64, 64), (i * 40, 100, 200 - i * 30))
                d = ImageDraw.Draw(im)
                x0 = 3 + s_idx * 9 + i * 13
                d.rectangle([(x0, 5 + s_idx * 4), (x0 + 15, 30 + i * 9)], fill=(255, 255, 255))
                im.save(folder / split / "images" / f"img{i}.jpg")
                lines = ["0 0.5 0.5 0.3 0.3"] + (["1 0.2 0.2 0.1 0.1"] if i == 0 else [])
                (folder / split / "labels" / f"img{i}.txt").write_text("\n".join(lines) + "\n")
        # an exact duplicate of train/img1 in valid, to check dedupe
        shutil.copy2(folder / "train" / "images" / "img1.jpg", folder / "valid" / "images" / "dup.jpg")
        (folder / "valid" / "labels" / "dup.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        mapping = {"fake_owner_fake-dogs": {"stray dog": "dog", "other": "drop"}}
        out = tmp / "out"
        rc = run(src_root, out, 4, mapping)
        test_dir = out / "images" / "test"
        # expected: train 3 imgs (3 dog boxes, 'other' dropped), valid 2 + test 2 -> val (2 + 2 dog boxes),
        # the planted copy of train/img1 removed => 7 dog boxes total; no test folder
        ok = rc == 0 and not test_dir.exists() and dict(count_boxes(out)) == {"dog": 7}
        print("FIXTURE", "PASSED" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-root", default="data/raw/roboflow", type=Path)
    parser.add_argument("--out", default="data/raw/roboflow_yolo", type=Path)
    parser.add_argument("--phash-dist", default=4, type=int, help="max dHash Hamming distance counted as duplicate")
    parser.add_argument("--list", action="store_true", help="list class names and mapping status, then exit")
    parser.add_argument("--test-fixture", action="store_true", help="test on a tiny fake export, then exit")
    args = parser.parse_args()

    if args.test_fixture:
        return test_fixture()
    if not args.src_root.is_dir():
        print(f"ERROR: {args.src_root} not found", file=sys.stderr)
        return 2
    if args.list:
        list_folders(args.src_root)
        return 0
    return run(args.src_root, args.out, args.phash_dist, load_mapping())


if __name__ == "__main__":
    sys.exit(main())
