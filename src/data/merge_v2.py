"""Merge the new data into data/dataset (v2) without changing the existing test split.

Inputs
  data/raw/openimages_goat_sheep   524 Open Images images, labels already in our ids.
                                   Group-of boxes are removed using the Open Images
                                   annotation file (IsGroupOf=1). Images go to train/val,
                                   and about 40 sheep + 40 goat images go to test_extra.
  data/raw/mendeley_herbivores5    Roboflow export. ONE image per source (the 3 augmented
                                   copies are dropped). Ids remapped: 0 Buffalo->2 cow,
                                   2 Cow->2, 3 Goat->1, 4 Sheep->4; 1 Camel drops the image.
                                   Train goes to train, its valid goes to val, its test is
                                   not used. dHash dedupe, then at most 600 images per class.

Existing files under data/dataset are not changed: the new images are added, the old
split_manifest.csv stays, and data/dataset/backup_v1 (hard links) is the rollback point.

Run from the project root:
    python src/data/merge_v2.py --write            # build (after the backup exists)
    python src/data/merge_v2.py --dry-run          # counts only
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from roboflow_remap import dhash, hamming  # noqa: E402

ROOT = Path(".")
DST = ROOT / "data/dataset"
OI_RAW = ROOT / "data/raw/openimages_goat_sheep"
OI_ANN = Path.home() / "fiftyone/open-images-v7/train/labels/detections.csv"
OI_OLD = ROOT / "data/raw/openimages_yolo"
MEN_RAW = ROOT / "data/raw/mendeley_herbivores5"
MEN_MAP = {0: 2, 2: 2, 3: 1, 4: 4}   # Buffalo, Cow, Goat, Sheep -> our ids; Camel (1) drops the image
CAP = 600
DHASH_DIST = 4
GROUP_TOL = 0.002
TEST_EXTRA_PER_CLASS = 40
OUR = {0: "dog", 1: "goat", 2: "cow", 3: "cat", 4: "sheep"}


def read_labels(path: Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), [float(v) for v in p[1:]]))
    return out


def fmt(cls: int, box: list) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in box)


def group_of_boxes(ids: set) -> dict:
    """{image_id: [(cx, cy, w, h), ...]} for IsGroupOf=1 boxes of the given images."""
    import csv as _csv
    out = defaultdict(list)
    with OI_ANN.open() as f:
        r = _csv.reader(f)
        head = next(r)
        i_id, i_g = head.index("ImageID"), head.index("IsGroupOf")
        i_x1, i_x2, i_y1, i_y2 = (head.index(k) for k in ("XMin", "XMax", "YMin", "YMax"))
        for row in r:
            if row[i_g] != "1" or row[i_id] not in ids:
                continue
            x1, x2, y1, y2 = float(row[i_x1]), float(row[i_x2]), float(row[i_y1]), float(row[i_y2])
            out[row[i_id]].append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


def is_group(box, groups) -> bool:
    return any(all(abs(a - b) <= GROUP_TOL for a, b in zip(box, g)) for g in groups)


def load_openimages() -> list:
    """Return [{id, boxes:[(cls, box)], src}] with group-of boxes removed."""
    ids = [p.stem for p in (OI_RAW / "images").glob("*.jpg")]
    old = {p.stem for p in (OI_OLD / "images").glob("*.*")}
    ids = [i for i in ids if i not in old]  # 76 candidates were already in openimages_yolo
    groups = group_of_boxes(set(ids))
    items, removed = [], 0
    for i in sorted(ids):
        boxes = []
        for cls, box in read_labels(OI_RAW / "labels" / f"{i}.txt"):
            if is_group(box, groups.get(i, [])):
                removed += 1
                continue
            boxes.append((cls, box))
        if boxes:
            items.append({"id": i, "boxes": boxes, "src": OI_RAW / "images" / f"{i}.jpg"})
    print(f"openimages_goat_sheep: {len(items)} images kept, {removed} group-of boxes removed")
    return items


def load_mendeley() -> list:
    """One image per source. Returns [{split, name, boxes, src}]; split is train or val."""
    items, seen_src = [], set()
    for split_dir, split in (("train", "train"), ("valid", "val")):
        imgs = sorted((MEN_RAW / split_dir / "images").glob("*.*"))
        for img in imgs:
            source = img.stem.split(".rf.")[0]
            if split == "train":
                if source in seen_src:
                    continue  # augmented copy of a source already kept
                seen_src.add(source)
            labels = read_labels(MEN_RAW / split_dir / "labels" / f"{img.stem}.txt")
            if any(c == 1 for c, _ in labels):
                continue  # camel: dropped
            boxes = [(MEN_MAP[c], b) for c, b in labels if c in MEN_MAP]
            if not boxes:
                continue
            items.append({"split": split, "name": img.name, "boxes": boxes, "src": img, "source": source})
    print(f"mendeley_herbivores5: {len(items)} images before dedupe and cap")
    return items


def dedupe(items: list) -> list:
    """Drop near-duplicates (dHash within DHASH_DIST). Train items are seen first, so a val copy of a
    train image is the one dropped."""
    order = sorted(items, key=lambda it: 0 if it["split"] == "train" else 1)
    kept, hashes = [], []
    for it in order:
        h = dhash(it["src"])
        if any(hamming(h, o) <= DHASH_DIST for o in hashes):
            continue
        hashes.append(h)
        kept.append(it)
    print(f"mendeley_herbivores5: {len(kept)} after dHash dedupe (dropped {len(items) - len(kept)})")
    return kept


def dominant(item) -> int:
    return Counter(c for c, _ in item["boxes"]).most_common(1)[0][0]


def cap_per_class(items: list, rng: random.Random) -> list:
    by = defaultdict(list)
    for it in items:
        by[dominant(it)].append(it)
    out = []
    for c, lst in by.items():
        rng.shuffle(lst)
        out.extend(lst[:CAP])
        print(f"  cap {OUR[c]}: {len(lst)} -> {min(len(lst), CAP)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--spotcheck", type=int, default=30)
    ap.add_argument("--spot-only", action="store_true", help="copy the spot-check sample to results/label_check_src")
    args = ap.parse_args()
    rejects = set()
    rej = HERE / "mendeley5_rejects.txt"
    if rej.exists():
        rejects = {l.strip() for l in rej.read_text().splitlines() if l.strip() and not l.startswith("#")}
    rng = random.Random(0)

    oi = load_openimages()
    # OI goat/sheep split: test_extra gets ~40 sheep and ~40 goat images, rest train/val 80/20
    rng.shuffle(oi)
    extra_ids = set()
    for cls in (4, 1):
        pool = [it for it in oi if any(c == cls for c, _ in it["boxes"]) and it["id"] not in extra_ids]
        for it in pool[:TEST_EXTRA_PER_CLASS]:
            extra_ids.add(it["id"])
    for it in oi:
        if it["id"] in extra_ids:
            it["split"] = "test_extra"
        else:
            it["split"] = "train" if rng.random() < 0.8 else "val"
    print("  test_extra images:", len(extra_ids))

    men = dedupe(load_mendeley())
    men = cap_per_class(men, rng)
    men = [it for it in men if it["name"] not in rejects]  # hand spot-check rejects

    new_items = [{"group": f"openimages_goat_sheep/{it['id']}", "name": f"openimages_goat_sheep__{it['id']}.jpg",
                  "split": it["split"], "boxes": it["boxes"], "src": it["src"]} for it in oi]
    new_items += [{"group": f"mendeley_herbivores5/{it['source']}", "name": f"mendeley5__{it['name']}",
                   "split": it["split"], "boxes": it["boxes"], "src": it["src"]} for it in men]

    # Remap-only copy for the spot-check: boxes already in our ids.
    chk = [it for it in new_items if it["group"].startswith("mendeley")]
    rng.shuffle(chk)
    sample = chk[:args.spotcheck]
    print("spot-check sample:", [s["name"] for s in sample])
    if args.spot_only:
        root = ROOT / "results/label_check_src/merge_v2_spot"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        for it in sample:
            shutil.copy2(it["src"], root / "images" / it["name"])
            (root / "labels" / (Path(it["name"]).stem + ".txt")).write_text(
                "\n".join(fmt(c, b) for c, b in it["boxes"]) + "\n")
        print("wrote", root)
        return 0
    if args.dry_run or not args.write:
        cnt = Counter((it["split"], c) for it in new_items for c, _ in it["boxes"])
        for (sp, c), n in sorted(cnt.items()):
            print(f"  {sp:<10} {OUR[c]:<6} {n}")
        return 0

    # write
    existing = {r["file"] for r in csv.DictReader((DST / "split_manifest.csv").open())}
    manifest_rows = list((csv.DictReader((DST / "split_manifest.csv").open())))
    for it in new_items:
        assert it["name"] not in existing, it["name"]
        img_dst = DST / "images" / it["split"] / it["name"]
        lbl_dst = DST / "labels" / it["split"] / (Path(it["name"]).stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["src"], img_dst)
        lbl_dst.write_text("\n".join(fmt(c, b) for c, b in it["boxes"]) + "\n")
        manifest_rows.append({"file": it["name"], "group": it["group"], "split": it["split"]})
    tmp = DST / "split_manifest_v2.csv.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "group", "split"])
        w.writeheader()
        w.writerows(manifest_rows)
    tmp.replace(DST / "split_manifest.csv")  # new file; the old one stays in data/dataset/backup_v1
    # group leak check
    where = defaultdict(set)
    for r in manifest_rows:
        where[r["group"]].add(r["split"])
    leaks = [g for g, s in where.items() if len(s) > 1]
    print("groups in two splits:", len(leaks))
    if leaks:
        return 1
    (ROOT / "results").mkdir(exist_ok=True)
    print(json.dumps({"new_images": len(new_items)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
