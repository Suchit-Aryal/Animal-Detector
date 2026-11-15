"""
Turn the Mendeley street-animals images (sorted into class folders, NO boxes)
into a YOLO detection dataset by AUTO-LABELING:

  * the FOLDER name tells us the class (DOG, CAT, COW, GOAT, SHEEP)
  * a pretrained Open Images YOLO model finds WHERE the animal is (the box)

Our class ids:  0 dog, 1 goat, 2 cow, 3 cat, 4 sheep
Ignored folders: DEER, RABBIT, HEN (and anything else not listed above)
Flat NightVision/ images have no class in the path, so the detector picks the class.

Run from the PROJECT ROOT (the folder that contains data/), env activated:
    python data/mendeley_autolabel.py --n 300
    python data/mendeley_autolabel.py --n 300 --no-sheep      # drop sheep (4 classes only)
    python data/mendeley_autolabel.py --model yolov8n-oiv7.pt  # smaller/faster, less accurate
    python data/mendeley_autolabel.py --n 300 --device 0       # use the GPU (default is CPU)

Output:
    data/raw/mendeley_yolo/images/*.jpg
    data/raw/mendeley_yolo/labels/*.txt

IMPORTANT (say this in your report / Q&A):
  * These boxes are MODEL-GENERATED, not hand-drawn. Use this data for TRAINING only,
    never as your test set. Your hand-checked Nepali photos are the test set.
  * Always run check_labels.py on the result and look at the pictures.

Quality filters:
  * image kept only if the model finds at least one box of the expected animal
    (conf >= --conf)
  * image skipped if the model also sees a DIFFERENT animal strongly
    (conf >= --conflict-conf), e.g. a dog in the CAT folder
  * goat and sheep are allowed to substitute for each other (they look alike)

NOTE: copied to src/data/ and changed: chunked detector calls (predict_chunks) for memory. Paste the last lines of any error.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

# folder name (upper case) -> our class id
OUR = {"DOG": 0, "GOAT": 1, "COW": 2, "CAT": 3, "SHEEP": 4}
OUR_NAMES = ["dog", "goat", "cow", "cat", "sheep"]

# which detector class names (Open Images names, lower case) count as "the right animal"
ALLOWED = {
    "DOG": {"dog"},
    "CAT": {"cat"},
    "COW": {"cattle"},
    "GOAT": {"goat", "sheep"},
    "SHEEP": {"sheep", "goat"},
}
# animals that, if seen strongly and NOT allowed, make the image ambiguous
ANIMALS = {"dog", "cat", "cattle", "goat", "sheep"}
IMG_EXT = {".jpg", ".jpeg", ".png"}

def predict_chunks(model, paths, chunk=8, **kw):
    """Run the detector in small chunks. One big list of paths grew memory past 10 GB; chunks stay ~1.3 GB."""
    for k in range(0, len(paths), chunk):
        yield from model.predict(source=paths[k:k + chunk], **kw)


# flat NightVision folder: no class in the path, so the model picks the class
NIGHT_MAP = {"dog": "DOG", "cat": "CAT", "cattle": "COW", "goat": "GOAT", "sheep": "SHEEP"}


def night_class(name, no_sheep):
    c = NIGHT_MAP.get(name)
    if no_sheep and c == "SHEEP":
        return "GOAT"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/mendeley/extracted/DATASETT")
    ap.add_argument("--out", default="data/raw/mendeley_yolo")
    ap.add_argument("--n", type=int, default=300, help="images kept per class")
    ap.add_argument("--model", default="yolov8m-oiv7.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--conflict-conf", type=float, default=0.6)
    ap.add_argument("--no-sheep", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (safe) or 0 for the first GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root.resolve()}  (run from the project root)")

    classes = [c for c in OUR if not (args.no_sheep and c == "SHEEP")]

    # 1) find images and group by class folder (works for NightVision/<CLASS>/ too)
    pool = defaultdict(list)
    night = set()
    night_flat = []
    ignored = Counter()
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        parts = [x.upper() for x in p.relative_to(root).parts[:-1]]
        cls = next((x for x in parts if x in classes), None)
        if cls is None:
            if "NIGHTVISION" in parts:
                night_flat.append(p)
            else:
                ignored[parts[0] if parts else "(root)"] += 1
            continue
        pool[cls].append(p)
        if "NIGHTVISION" in parts:
            night.add(p)

    print("Images found per class:", {c: len(pool[c]) for c in classes})
    print("  of which night-vision (inside a class folder):", len(night))
    print("  flat NightVision images (model decides the class):", len(night_flat))
    print("Ignored (not in our classes):", dict(ignored))

    # 2) load the detector and check its class names
    model = YOLO(args.model)
    names = {i: str(n).lower() for i, n in model.names.items()}
    missing = [n for n in ANIMALS if n not in names.values()]
    if missing:
        raise SystemExit(f"Model has no class(es) {missing}. Use an Open Images V7 model (…-oiv7.pt).")

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    summary = {}

    # 3) auto-label class by class
    for cname in classes:
        paths = pool[cname]
        rng.shuffle(paths)
        paths = paths[: args.n * 3]  # look at up to 3x candidates, stop at n accepted
        allowed = ALLOWED[cname]
        stats = Counter()
        accepted = 0
        print(f"\n=== {cname}: checking up to {len(paths)} images, keeping {args.n} ===")

        for r in predict_chunks(model, [str(p) for p in paths], conf=args.conf,
                                device=args.device, verbose=False):
            stats["checked"] += 1
            src = Path(r.path)
            if r.boxes is None or len(r.boxes) == 0:
                stats["no_detection"] += 1
                continue

            good, conflict = [], False
            for ci, cf, bb in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                nm = names[ci]
                if nm in allowed:
                    good.append(bb)
                elif nm in ANIMALS and cf >= args.conflict_conf:
                    conflict = True
            if conflict:
                stats["conflict"] += 1
                continue
            if not good:
                stats["no_match"] += 1
                continue

            tag = "_night" if src in night else ""
            stem = f"mdy_{cname.lower()}_{accepted:05d}{tag}"
            shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")
            lines = [f"{OUR[cname]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in good]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            accepted += 1
            stats["accepted"] += 1
            if accepted >= args.n:
                break

        summary[cname] = dict(stats)
        print(f"{cname}: {dict(stats)}")

    # 4) flat NightVision folder: the detector decides which animal class each image is
    night_added = Counter()
    if night_flat and all(summary[c].get("accepted", 0) >= args.n for c in classes):
        print("\n=== NIGHTVISION (flat): every class is already full, skipping ===")
    elif night_flat:
        rng.shuffle(night_flat)
        print(f"\n=== NIGHTVISION (flat): checking {len(night_flat)} images, model decides the class ===")
        stats = Counter()
        total = {c: summary[c].get("accepted", 0) for c in classes}
        for r in predict_chunks(model, [str(p) for p in night_flat], conf=args.conf,
                                device=args.device, verbose=False):
            stats["checked"] += 1
            src = Path(r.path)
            if r.boxes is None or len(r.boxes) == 0:
                stats["no_detection"] += 1
                continue

            by_cls = defaultdict(list)
            for ci, cf, bb in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                c = night_class(names[ci], args.no_sheep)
                if c in classes:
                    by_cls[c].append((cf, bb))
            if not by_cls:
                stats["no_match"] += 1
                continue

            # dominant animal = highest-confidence box; skip if another animal is also strong
            dom = max(by_cls, key=lambda c: max(cf for cf, _ in by_cls[c]))
            conflict = any(c != dom and max(cf for cf, _ in v) >= args.conflict_conf
                           for c, v in by_cls.items())
            if conflict:
                stats["conflict"] += 1
                continue
            if total[dom] >= args.n:
                stats["class_full"] += 1
                continue

            stem = f"mdy_{dom.lower()}_night_{total[dom]:05d}"
            shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")
            lines = [f"{OUR[dom]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for _, (x, y, w, h) in by_cls[dom]]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            total[dom] += 1
            night_added[dom] += 1
            stats["accepted"] += 1
            if all(total[c] >= args.n for c in classes):
                break

        print(f"NIGHTVISION: {dict(stats)}")
        print("  night images added per class:", {c: night_added[c] for c in classes})
        for c in classes:
            summary[c]["accepted"] = total[c]
            summary[c]["night"] = night_added[c]

    print("\n=== SUMMARY ===")
    for cname, s in summary.items():
        print(f"{cname:6s} accepted {s.get('accepted', 0):4d} / checked {s.get('checked', 0):4d}  "
              f"(no detection {s.get('no_detection', 0)}, conflict {s.get('conflict', 0)}, "
              f"wrong class only {s.get('no_match', 0)}, night {s.get('night', 0)})")
    print(f"Output folder: {out.resolve()}")
    print("Next: python data/check_labels.py --root data/raw/mendeley_yolo --n 40  and LOOK at the images.")


if __name__ == "__main__":
    main()
