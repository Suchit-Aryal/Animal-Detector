"""Detection pipeline: animal model + COCO person model, overlap filter, de-duplicated logging.

Rules
- People come from the COCO model (class 'person' only). They are drawn and counted
  separately and are logged with kind='person', never as animals.
- An animal box whose IoU with ANY person box is above config suppress_animal_iou is dropped.
- Each (source, kind, class) is logged at most once per dedupe_seconds. For uploads the
  clock is the clip time; for live cameras it is wall-clock time.
"""

import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_config() -> dict:
    return json.loads((WEB / "config.json").read_text())


def resolve(path_str: str) -> str:
    """Relative paths are taken from the project root (so the config can name results/weights/best.pt)."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    cand = ROOT / p
    return str(cand) if cand.exists() else path_str


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class Det:
    kind: str          # 'animal' or 'person'
    name: str
    conf: float
    box: tuple         # x1, y1, x2, y2 in pixels


class Detector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()  # one predict call at a time across threads
        self.animal_model = YOLO(resolve(cfg["animal_weights"]))
        self.person_model = YOLO(resolve(cfg["person_weights"]))
        self.animal_names = {int(i): str(n) for i, n in self.animal_model.names.items()}
        self.person_id = next((int(i) for i, n in self.person_model.names.items() if str(n) == "person"), 0)
        self.animal_weights = resolve(cfg["animal_weights"])

    def detect(self, frame) -> tuple:
        """Return (animals, persons, suppressed_count) for one BGR frame.

        Animal boxes are dropped when a person is found and the animal box
        (a) has its centre inside a person box, or
        (b) overlaps a person box with IoU above suppress_animal_iou, or
        (c) has more than suppress_containment of its area inside a person box.
        A box with confidence >= person_override_conf is kept even then (a confident
        cow beside a person is still a cow). Thresholds are read from the config on
        every call, so the Live page can change them while the app runs."""
        c = self.cfg
        per = c.get("animal_conf_per_class") or {}
        floor = min([float(c["animal_conf"])] + [float(v) for v in per.values()])
        kw = dict(imgsz=c["imgsz"], device=c["device"], verbose=False)
        with self.lock:
            ra = self.animal_model.predict(frame, conf=floor, **kw)[0]
            rp = self.person_model.predict(frame, conf=float(c["person_conf"]), classes=[self.person_id], **kw)[0]
        persons = [Det("person", "person", float(cf), tuple(float(v) for v in b))
                   for cf, b in zip(rp.boxes.conf.tolist(), rp.boxes.xyxy.tolist())] if rp.boxes is not None else []
        animals, suppressed = [], 0
        raw = []
        if ra.boxes is not None:
            raw = zip(ra.boxes.cls.int().tolist(), ra.boxes.conf.tolist(), ra.boxes.xyxy.tolist())
        for cls, cf, b in raw:
            name = self.animal_names.get(cls, str(cls))
            if cf < float(per.get(name, c["animal_conf"])):
                continue  # below this class's threshold: not a detection at all
            box = tuple(float(v) for v in b)
            strong = cf >= float(c.get("person_override_conf", 1.1))  # a very confident animal box is kept
            if (not strong and persons
                    and person_overlap(box, persons, c["suppress_animal_iou"], c["suppress_containment"])):
                suppressed += 1
                continue
            animals.append(Det("animal", name, float(cf), box))
        return animals, persons, suppressed


def person_overlap(box, persons, iou_thr: float, contain_thr: float) -> bool:
    """True if an animal box belongs to a person: centre inside, IoU above iou_thr, or
    more than contain_thr of the animal box inside a person box."""
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for p in persons:
        px1, py1, px2, py2 = p.box
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
        if iou(box, p.box) > iou_thr:
            return True
        iw = max(0.0, min(x2, px2) - max(x1, px1))
        ih = max(0.0, min(y2, py2) - max(y1, py1))
        if iw * ih / area > contain_thr:
            return True
    return False


def annotate(frame, dets: list):
    """Draw boxes on a copy: animals green, people orange (separate colour)."""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = (0, 200, 0) if d.kind == "animal" else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.name} {d.conf:.2f}" if d.kind == "animal" else f"person {d.conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


class EventLogger:
    """Writes de-duplicated rows to the DB and one snapshot per logged frame."""

    def __init__(self, db, snapshot_dir: Path, dedupe_seconds: float):
        self.db = db
        self.snap_dir = snapshot_dir
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.dedupe = dedupe_seconds
        self.last = {}  # (source_id, kind, name) -> last logged clock value
        self._lock = threading.Lock()

    def log(self, source_id: str, source_label: str, clock: float, dets: list, annotated_frame,
            clip_time=None, ts_wall=None) -> int:
        """clock: seconds used for de-duplication (clip time or time.time()). Returns rows written."""
        with self._lock:
            due = []
            for d in dets:
                key = (source_id, d.kind, d.name)
                prev = self.last.get(key)
                if prev is None or clock - prev >= self.dedupe:
                    self.last[key] = clock
                    due.append(d)
        if not due:
            return 0
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        snap = f"{stamp}_{source_id}.jpg".replace("/", "_")
        cv2.imwrite(str(self.snap_dir / snap), annotated_frame)
        ts = (ts_wall or datetime.now().astimezone()).isoformat(timespec="seconds")
        rows = [(ts, d.kind, d.name, d.conf, source_label, clip_time,
                 d.box[0], d.box[1], d.box[2], d.box[3], snap) for d in due]
        self.db.insert_many(rows)
        return len(rows)


def process_media(path: Path, label: str, job: dict, detector: Detector, logger: EventLogger,
                  out_dir: Path, cfg: dict) -> None:
    """Annotate an uploaded image or video, log detections, write output for the browser.
    job is a dict updated in place: status, progress, outputs, counts."""
    job_id = job["id"]
    source_id = f"upload-{job_id}"
    if path.suffix.lower() in IMG_EXT:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("could not read image")
        animals, persons, sup = detector.detect(frame)
        ann = annotate(frame, animals + persons)
        out = out_dir / f"{job_id}_annotated.jpg"
        cv2.imwrite(str(out), ann)
        n = logger.log(source_id, f"upload:{label}", 0.0, animals + persons, ann, clip_time=None)
        job["outputs"].append({"kind": "image", "url": out.name})
        job["counts"] = {"animal": len(animals), "person": len(persons), "suppressed": sup, "rows_logged": n}
        job["progress"] = 100
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_out = out_dir / f"{job_id}_raw.mp4"
    vw = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    skip = max(1, int(cfg["upload_frame_skip"]))
    idx = 0
    last_dets = []
    counts = {"animal": 0, "person": 0, "suppressed": 0, "rows_logged": 0}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if idx % skip == 0:
                animals, persons, sup = detector.detect(frame)
                last_dets = animals + persons
                counts["suppressed"] += sup
                ann = annotate(frame, last_dets)
                counts["rows_logged"] += logger.log(source_id, f"upload:{label}", t, last_dets, ann, clip_time=round(t, 2))
                counts["animal"] += len(animals)
                counts["person"] += len(persons)
            else:
                ann = annotate(frame, last_dets)  # keep boxes on in-between frames
            vw.write(ann)
            idx += 1
            if total:
                job["progress"] = round(100 * idx / total, 1)
    finally:
        cap.release()
        vw.release()
    final = out_dir / f"{job_id}_annotated.mp4"
    _to_browser_mp4(raw_out, final)
    raw_out.unlink(missing_ok=True)
    job["outputs"].append({"kind": "video", "url": final.name})
    job["counts"] = counts
    job["progress"] = 100
    job["frames"] = idx


def _to_browser_mp4(src: Path, dst: Path) -> None:
    """Re-encode to H.264 so browsers can play it. Falls back to the mp4v file if ffmpeg is missing."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    shutil.move(str(src), str(dst))


class CameraWorker:
    """Reads a webcam (integer index) or IP camera URL in a thread. Keeps the latest annotated JPEG."""

    def __init__(self, cam_id: str, source: str, detector: Detector, logger: EventLogger, cfg: dict):
        self.cam_id = cam_id
        self.source = source
        self.detector = detector
        self.logger = logger
        self.skip = max(1, int(cfg["live_frame_skip"]))
        self.latest = None
        self.error = None
        self.running = False
        self.started = time.time()
        self.counts = {"animal": 0, "person": 0, "rows_logged": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.running = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.running = False

    def _run(self):
        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error = f"cannot open camera {self.source}"
            self.running = False
            return
        idx, last = 0, []
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "camera read failed"
                    break
                if idx % self.skip == 0:
                    animals, persons, _ = self.detector.detect(frame)
                    last = animals + persons
                    ann = annotate(frame, last)
                    self.counts["animal"] += len(animals)
                    self.counts["person"] += len(persons)
                    self.counts["rows_logged"] += self.logger.log(
                        f"camera-{self.cam_id}", f"camera:{self.source}", time.time(), last, ann)
                else:
                    ann = annotate(frame, last)
                ok2, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    self.latest = jpg.tobytes()
                idx += 1
        finally:
            cap.release()
            self.running = False
