"""Detection pipeline: animal model + COCO person model, overlap filter, de-duplicated logging.

Rules
- People come from the COCO model (class 'person' only). They are drawn and counted
  separately and are logged with kind='person', never as animals.
- An animal box whose IoU with ANY person box is above config suppress_animal_iou is dropped.
- Animals get a track ID from ByteTrack (ultralytics model.track). Each source (one camera
  start or one upload) has its own tracker, so IDs restart in every session.
- Each (source, kind, class) is logged at most once per dedupe_seconds, and every new track
  is logged once when it first appears. For uploads the clock is the clip time; for live
  cameras it is wall-clock time.
"""

import json
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

TRACKER = "bytetrack.yaml"  # ships with ultralytics

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
    track_id: Optional[int] = None  # animals only, when tracked


class Detector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()  # one predict call at a time across threads
        self.animal_model = YOLO(resolve(cfg["animal_weights"]))
        self.person_model = YOLO(resolve(cfg["person_weights"]))
        self.animal_names = {int(i): str(n) for i, n in self.animal_model.names.items()}
        self.person_id = next((int(i) for i, n in self.person_model.names.items() if str(n) == "person"), 0)
        self.animal_weights = resolve(cfg["animal_weights"])

    def new_tracker(self):
        """A separate copy of the animal model for one source (camera or upload). Its ByteTrack
        state is kept inside it, so tracks from different sources never mix."""
        return YOLO(self.animal_weights)

    def detect(self, frame, tracker=None) -> tuple:
        """Return (animals, persons, suppressed_count) for one BGR frame.

        tracker: a model from new_tracker(); when given, animals get track_id values.
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
            ra = self.animal_model.predict(frame, conf=floor, **kw)[0]  # every plain detection is kept
            tracked = []
            if tracker is not None:
                # ByteTrack only reports a new track from its second frame, so boxes it drops are
                # taken from the plain run above and get track_id None (see README limits).
                rt = tracker.track(frame, conf=floor, persist=True, tracker=TRACKER, **kw)[0]
                if rt.boxes is not None and rt.boxes.id is not None:
                    tracked = [(tuple(float(v) for v in b), int(t))
                               for b, t in zip(rt.boxes.xyxy.tolist(), rt.boxes.id.int().tolist())]
            rp = self.person_model.predict(frame, conf=float(c["person_conf"]), classes=[self.person_id], **kw)[0]
        persons = [Det("person", "person", float(cf), tuple(float(v) for v in b))
                   for cf, b in zip(rp.boxes.conf.tolist(), rp.boxes.xyxy.tolist())] if rp.boxes is not None else []
        animals, suppressed = [], 0
        raw = []
        if ra.boxes is not None:
            raw = zip(ra.boxes.cls.int().tolist(), ra.boxes.conf.tolist(), ra.boxes.xyxy.tolist())
        for cls, cf, b in raw:
            tid = _track_for(tuple(float(v) for v in b), tracked)
            name = self.animal_names.get(cls, str(cls))
            if cf < float(per.get(name, c["animal_conf"])):
                continue  # below this class's threshold: not a detection at all
            box = tuple(float(v) for v in b)
            strong = cf >= float(c.get("person_override_conf", 1.1))  # a very confident animal box is kept
            if (not strong and persons
                    and person_overlap(box, persons, c["suppress_animal_iou"], c["suppress_containment"])):
                suppressed += 1
                continue
            animals.append(Det("animal", name, float(cf), box, None if tid is None else int(tid)))
        return animals, persons, suppressed


def _track_for(box, tracked, min_iou: float = 0.5):
    """The track ID of the tracked box that best matches this plain box, or None."""
    best, best_iou = None, min_iou
    for tbox, tid in tracked:
        v = iou(box, tbox)
        if v > best_iou:
            best, best_iou = tid, v
    return best


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray casting. poly is a list of [x, y] pairs; the shape closes itself."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


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
        if d.kind == "animal":
            tag = f" #{d.track_id}" if d.track_id is not None else ""
            label = f"{d.name}{tag} {d.conf:.2f}"
        else:
            label = f"person {d.conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


class EventLogger:
    """Writes de-duplicated rows to the DB and one snapshot per logged frame."""

    def __init__(self, db, snapshot_dir: Path, dedupe_seconds: float):
        self.db = db
        self.snap_dir = snapshot_dir
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.dedupe = dedupe_seconds
        self.last = {}          # (source_id, kind, name) -> last logged clock value
        self.known_tracks = set()  # (source_id, track_id) already logged once
        self._lock = threading.Lock()
        self.zones_fn = None       # set by the app: returns the zones to watch (camera sources only)
        self.in_zone = set()       # (source_id, track_id, zone_id) while the track is inside
        self.zone_seen = {}        # (source_id, class, zone_id) -> last untracked zone alert clock
        self.zone_cooldown = 60.0

    def log(self, source_id: str, source_label: str, clock: float, dets: list, annotated_frame,
            clip_time=None, ts_wall=None, is_camera=False) -> int:
        """clock: seconds used for de-duplication and tracking (clip time or time.time()).
        Every call updates the source clock and the tracks; rows are written only when due:
        a new track, or a class not logged within dedupe_seconds. Returns rows written."""
        ts = (ts_wall or datetime.now().astimezone()).isoformat(timespec="seconds")
        with self._lock:
            due, new_tracks = [], set()
            for d in dets:
                key = (source_id, d.kind, d.name)
                prev = self.last.get(key)
                is_new = d.track_id is not None and (source_id, d.track_id) not in self.known_tracks
                if is_new or prev is None or clock - prev >= self.dedupe:
                    self.last[key] = clock
                    due.append(d)
                if is_new:
                    new_tracks.add((source_id, d.track_id))
                    self.known_tracks.add((source_id, d.track_id))
        snap = None
        if due:
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            snap = f"{stamp}_{source_id}.jpg".replace("/", "_")
            cv2.imwrite(str(self.snap_dir / snap), annotated_frame)
        det_rows = [(ts, d.kind, d.name, d.conf, source_label, clip_time,
                     d.box[0], d.box[1], d.box[2], d.box[3], snap, d.track_id) for d in due]
        track_rows = [{"source_id": source_id, "track_id": d.track_id, "class_name": d.name,
                       "clock": clock, "ts": ts,
                       "snapshot": snap if (source_id, d.track_id) in new_tracks else None}
                      for d in dets if d.track_id is not None]
        if is_camera and self.zones_fn is not None:
            self._zone_alerts(source_id, source_label, clock, ts, dets, annotated_frame)
        return self.db.record_frame(source_id, source_label, is_camera, clock, ts, det_rows, track_rows)

    def _zone_alerts(self, source_id, source_label, clock, ts, dets, frame):
        """Alert when an animal's box centre enters a zone. A tracked animal alerts once per entry;
        an untracked one alerts at most once per zone_cooldown seconds."""
        zones = self.zones_fn()
        if not zones:
            return
        h, w = frame.shape[:2]
        snap = None
        for d in dets:
            if d.kind != "animal":
                continue
            cx = ((d.box[0] + d.box[2]) / 2) / w
            cy = ((d.box[1] + d.box[3]) / 2) / h
            for z in zones:
                inside = point_in_polygon(cx, cy, z["points"])
                if d.track_id is not None:
                    key = (source_id, d.track_id, z["id"])
                    if not inside:
                        self.in_zone.discard(key)
                        continue
                    if key in self.in_zone:
                        continue
                    self.in_zone.add(key)
                    who, tid = f"{d.name} #{d.track_id}", d.track_id
                else:
                    if not inside:
                        continue
                    k = (source_id, d.name, z["id"])
                    prev = self.zone_seen.get(k)
                    if prev is not None and clock - prev < self.zone_cooldown:
                        continue
                    self.zone_seen[k] = clock
                    who, tid = d.name, 0
                if snap is None:
                    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
                    snap = f"{stamp}_{source_id}_zone.jpg".replace("/", "_")
                    cv2.imwrite(str(self.snap_dir / snap), frame)
                self.db.add_alert(ts, source_id, tid, d.name, source_label, ts,
                                  f"{who} entered zone '{z['name']}' ({source_label})", snap, "zone")


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
        n = logger.log(source_id, f"upload:{label}", 0.0, animals + persons, ann, clip_time=None)  # no tracks on a still
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
    tracker = detector.new_tracker()  # this clip's own track IDs
    job["markers"] = []  # {t, class} for the timeline on the job page
    job["duration"] = round(total / fps, 2) if total else None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if idx % skip == 0:
                animals, persons, sup = detector.detect(frame, tracker)
                last_dets = animals + persons
                counts["suppressed"] += sup
                ann = annotate(frame, last_dets)
                counts["rows_logged"] += logger.log(source_id, f"upload:{label}", t, last_dets, ann, clip_time=round(t, 2))
                counts["animal"] += len(animals)
                counts["person"] += len(persons)
                for a in animals:  # one mark per class per half second, so the bar stays readable
                    if not any(m["class"] == a.name and abs(m["t"] - t) < 0.5 for m in job["markers"][-20:]):
                        job["markers"].append({"t": round(t, 2), "class": a.name})
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
        self.source_id = f"camera-{cam_id}"
        self._frame_times = deque(maxlen=30)  # times of recent processed frames, for the FPS overlay
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def fps(self) -> float:
        t = list(self._frame_times)
        if len(t) < 2 or t[-1] - t[0] <= 0:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def _stamp(self, img, n_animal: int, n_person: int):
        """Overlay in the corner: processing speed and what the last processed frame had."""
        text = f"{self.fps():.1f} fps  |  animals {n_animal}  people {n_person}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        out = img.copy()
        cv2.rectangle(out, (0, 0), (w + 16, h + 14), (20, 24, 32), -1)
        cv2.putText(out, text, (8, h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 250), 2)
        return out

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
        n_animal = n_person = 0
        tracker = self.detector.new_tracker()  # this camera session's own track IDs
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "camera read failed"
                    break
                if idx % self.skip == 0:
                    animals, persons, _ = self.detector.detect(frame, tracker)
                    last = animals + persons
                    n_animal, n_person = len(animals), len(persons)
                    self._frame_times.append(time.time())
                    ann = annotate(frame, last)
                    self.counts["animal"] += len(animals)
                    self.counts["person"] += len(persons)
                    self.counts["rows_logged"] += self.logger.log(
                        self.source_id, f"camera:{self.source}", time.time(), last, ann, is_camera=True)
                else:
                    ann = annotate(frame, last)
                ann = self._stamp(ann, n_animal, n_person)
                ok2, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    self.latest = jpg.tobytes()
                idx += 1
        finally:
            cap.release()
            self.running = False
