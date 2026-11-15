"""C1: run a detector on an image, a video file, or a webcam; save annotated output to disk (no display needed).

Usage:
  python src/demo.py --source street.jpg   --weights results/weights/best.pt
  python src/demo.py --source street.mp4   --weights results/weights/best.pt
  python src/demo.py --source 0 --max-frames 300   # webcam index (saves mp4; stop with Ctrl+C)
Output: results/figures/<name>_annotated.(jpg|mp4) unless --out is given.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image path, video path, or webcam index (e.g. 0)")
    ap.add_argument("--weights", default=str(ROOT / "results/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model = YOLO(a.weights)
    kw = dict(conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)
    outdir = ROOT / "results/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(a.source)

    if src.suffix.lower() in IMG_EXT:
        out = Path(a.out) if a.out else outdir / f"{src.stem}_annotated.jpg"
        r = model.predict(str(src), **kw)[0]
        cv2.imwrite(str(out), r.plot())
        print(f"{len(r.boxes)} detections -> {out}")
        return

    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(a.out) if a.out else outdir / f"{'webcam' if is_cam else src.stem}_annotated.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = dets = 0
    try:
        while a.max_frames is None or n < a.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, **kw)[0]
            dets += len(r.boxes)
            vw.write(r.plot())
            n += 1
    except KeyboardInterrupt:
        pass
    cap.release(); vw.release()
    print(f"{n} frames, {dets} detections -> {out}")


if __name__ == "__main__":
    main()
