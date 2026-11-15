"""Animal surveillance web app (Flask). Binds to 127.0.0.1 only.

Run from the project root:
    ~/.conda/envs/animal-det/bin/python web/app.py
Then open http://127.0.0.1:5055/
"""

import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import csv
import io
import json
from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from db import DetectionDB
from detector import (IMG_EXT, VID_EXT, CameraWorker, Detector, EventLogger, load_config,
                      process_media)

WEB = Path(__file__).resolve().parent
CFG = load_config()
INSTANCE = (WEB.parent / CFG["instance_dir"]).resolve()
UPLOADS = INSTANCE / "uploads"
OUTPUTS = INSTANCE / "outputs"
SNAPS = INSTANCE / "snapshots"
for folder in (UPLOADS, OUTPUTS, SNAPS):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(WEB / "templates"), static_folder=str(WEB / "static"))
app.config["MAX_CONTENT_LENGTH"] = CFG["max_upload_mb"] * 1024 * 1024

SETTINGS_FILE = INSTANCE / "settings.json"
TUNABLE = {"animal_conf", "person_conf", "person_override_conf", "suppress_animal_iou",
           "suppress_containment", "animal_conf_per_class", "presence_seconds", "missing_alert_minutes",
           "dedupe_seconds"}
if SETTINGS_FILE.exists():  # thresholds saved from the Live page survive a restart
    try:
        CFG.update({k: v for k, v in json.loads(SETTINGS_FILE.read_text()).items() if k in TUNABLE})
    except (ValueError, OSError):
        pass

db = DetectionDB(INSTANCE / "detections.sqlite")
detector = Detector(CFG)
logger = EventLogger(db, SNAPS, CFG["dedupe_seconds"])

JOBS = {}          # job_id -> dict (in memory; the log itself is in SQLite)
JOBS_LOCK = threading.Lock()
CAMERAS = {}       # cam_id -> CameraWorker


def allowed(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXT | VID_EXT


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$")


def int_arg(name: str, default: int, lo: int, hi: int) -> int:
    """Read an integer query argument; bad or out-of-range values fall back to the default."""
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if lo <= v <= hi else default


def date_arg(name: str):
    """Return (value, error). value is the date/time prefix with 'T' between date and time."""
    raw = (request.args.get(name) or "").strip()
    if not raw:
        return None, None
    if not DATE_RE.match(raw):
        return None, f"'{name}' must look like 2026-11-15 or 2026-11-15T14:30 (got '{raw}'), so it was ignored"
    return raw.replace(" ", "T"), None


def alert_loop():
    """Every 5 s: raise a missing-animal alert for each tracked animal on a running camera that has
    not been seen for missing_alert_minutes. Off when that setting is 0."""
    while True:
        time.sleep(5)
        try:
            minutes = float(CFG.get("missing_alert_minutes") or 0)
            if minutes <= 0:
                continue
            running = [c.source_id for c in list(CAMERAS.values()) if c.running]
            fmt = "{cls} #{tid} not seen for " + f"{minutes:g}" + " min ({src})"
            now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
            db.raise_missing_alerts(time.time() - minutes * 60, running, fmt, now_iso)
        except Exception:
            app.logger.exception("missing-animal check failed")


def fmt_clock(source_is_camera: bool, clock: float, ts: str) -> str:
    return ts[11:19] if source_is_camera else f"{clock:.1f} s into clip"


def run_job(job: dict, path: Path, label: str):
    job["status"] = "running"
    try:
        process_media(path, label, job, detector, logger, OUTPUTS, CFG)
        job["status"] = "done"
    except Exception as exc:  # shown in the job page
        job["status"] = "error"
        job["error"] = str(exc)


@app.route("/")
def index():
    return redirect(url_for("live"))


@app.errorhandler(404)
def not_found(exc):
    return render_template("not_found.html", page="", message="That page or job does not exist."), 404


@app.route("/live")
def live():
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)[:20]
    cams = [{"id": cid, "source": c.source, "running": c.running, "error": c.error}
            for cid, c in CAMERAS.items()]
    return render_template("live.html", page="live", jobs=jobs, cams=cams,
                           max_mb=CFG["max_upload_mb"], error=request.args.get("error"),
                           cam_error=request.args.get("cam_error"),
                           demo_loaded=db.demo_loaded(), empty=(db.summary()["animal"]["count"] == 0
                                                               and db.summary()["person"]["count"] == 0),
                           demo_msg=request.args.get("demo"),
                           cfg=CFG,
                           per_text=", ".join(f"{k}={v:g}" for k, v in (CFG.get("animal_conf_per_class") or {}).items()))


def epoch_of(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def group_events(rows: list, gap: float) -> list:
    """Merge animal detections of the same class in the same source into one event when they are
    no more than gap seconds apart. Uploads use clip time; cameras use wall-clock time."""
    def clock(r):
        return r["clip_time"] if r["clip_time"] is not None else epoch_of(r["ts"])
    events = []
    for r in sorted(rows, key=lambda r: (r["source"], r["class_name"], clock(r))):
        c = clock(r)
        last = events[-1] if events else None
        if last and last["source"] == r["source"] and last["class"] == r["class_name"] and c - last["_end"] <= gap:
            last["_end"] = c
            last["end_ts"] = r["ts"]
            last["count"] += 1
            last["tracks"].add(r["track_id"])
            if r["snapshot"] and r["confidence"] > last["best_conf"]:
                last.update(snapshot=r["snapshot"], best_conf=r["confidence"])
            last["max_conf"] = max(last["max_conf"], r["confidence"])
        else:
            events.append({"source": r["source"], "class": r["class_name"], "_start": c, "_end": c,
                           "start_ts": r["ts"], "end_ts": r["ts"], "count": 1, "tracks": {r["track_id"]},
                           "max_conf": r["confidence"], "best_conf": r["confidence"],
                           "snapshot": r["snapshot"], "clip": r["clip_time"] is not None})
    for e in events:
        e["duration"] = round(e["_end"] - e["_start"], 1)
        e["tracks"] = len({t for t in e["tracks"] if t is not None})
    return sorted(events, key=lambda e: e["start_ts"], reverse=True)


@app.route("/events")
def events_page():
    rng = request.args.get("range", "24h")
    if rng not in RANGES and rng != "all":
        rng = "24h"
    gap = float(int_arg("gap", 30, 1, 3600))
    since = None
    if rng != "all":
        since = (datetime.now().astimezone() - timedelta(hours=RANGES[rng])).isoformat(timespec="seconds")
    rows = db.search(kind="animal", since=since, limit=5000)
    events = group_events(rows, gap)
    return render_template("events.html", page="events", events=events[:200], rng=rng, gap=int(gap),
                           ranges=list(RANGES) + ["all"], total_rows=len(rows))


@app.route("/animals")
def animals_page():
    status = request.args.get("status", "all")
    if status not in ("all", "present", "missing"):
        status = "all"
    presence = float(CFG["presence_seconds"])
    everything = db.tracks_list(presence, "all", limit=300)
    n_present = sum(1 for r in everything if r["status"] == "present")
    rows = [r for r in everything if status == "all" or r["status"] == status]
    for r in rows:
        r["first_show"] = fmt_clock(bool(r["is_camera"]), r["first_clock"], r["first_ts"])
        r["last_show"] = fmt_clock(bool(r["is_camera"]), r["last_clock"], r["last_ts"])
        d = max(0.0, r["last_clock"] - r["first_clock"])
        r["dur_show"] = f"{int(d // 60)}:{int(d % 60):02d}"
    return render_template("animals.html", page="animals", rows=rows, status=status,
                           n_all=len(everything), n_present=n_present,
                           n_missing=len(everything) - n_present, presence=presence,
                           minutes=float(CFG.get("missing_alert_minutes") or 0),
                           alerts=db.alerts_list(30))


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("media")
    if f is None or not f.filename:
        return redirect(url_for("live", error="Choose a file first."))
    if not allowed(f.filename):
        return redirect(url_for("live", error="Unsupported file type. Use a video (mp4, avi, mov, mkv, webm) or an image (jpg, png, bmp, webp)."))
    job_id = uuid.uuid4().hex[:10]
    name = secure_filename(f.filename) or "upload"
    path = UPLOADS / f"{job_id}_{name}"
    f.save(path)
    job = {"id": job_id, "name": name, "status": "queued", "progress": 0, "outputs": [],
           "counts": None, "error": None, "started": time.time()}
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job, path, name), daemon=True).start()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    return render_template("job.html", page="live", job=job)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    return jsonify({k: job.get(k) for k in ("status", "progress", "outputs", "counts", "error", "markers", "duration")})


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUTS, name)


@app.route("/snapshots/<path:name>")
def snapshots(name):
    return send_from_directory(SNAPS, name)


@app.route("/camera/start", methods=["POST"])
def camera_start():
    source = (request.form.get("source") or "").strip()
    if not source:
        return redirect(url_for("live", cam_error="Enter a webcam index (0, 1, ...) or an IP camera URL."))
    cam_id = uuid.uuid4().hex[:8]
    worker = CameraWorker(cam_id, source, detector, logger, CFG)
    CAMERAS[cam_id] = worker
    worker.start()
    return redirect(url_for("live"))


@app.route("/camera/<cam_id>/stop", methods=["POST"])
def camera_stop(cam_id):
    worker = CAMERAS.pop(cam_id, None)
    if worker is None:
        abort(404)
    worker.stop()
    return redirect(url_for("live"))


@app.route("/camera/<cam_id>/stream")
def camera_stream(cam_id):
    worker = CAMERAS.get(cam_id)
    if worker is None:
        abort(404)

    def gen():
        while worker.running:
            frame = worker.latest
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/log")
def log_page():
    kind = request.args.get("kind", "animal")
    if kind not in ("animal", "person", "all"):
        kind = "animal"
    cls = (request.args.get("class") or "").strip() or None
    source = (request.args.get("source") or "").strip() or None
    since, err_since = date_arg("since")
    until, err_until = date_arg("until")
    errors = [e for e in (err_since, err_until) if e]
    page = int_arg("page", 1, 1, 100000)
    per_page = 100
    kind_arg = None if kind == "all" else kind
    rows = db.search(kind=kind_arg, class_name=cls, source=source, since=since, until=until,
                     limit=per_page, offset=(page - 1) * per_page)
    total = db.count_search(kind=kind_arg, class_name=cls, source=source, since=since, until=until)
    if kind == "animal":
        classes = db.classes_seen("animal")
    elif kind == "person":
        classes = db.classes_seen("person")
    else:
        classes = sorted(set(db.classes_seen("animal")) | set(db.classes_seen("person")))
    filters = {"kind": kind, "class": cls or "", "source": source or "",
               "since": request.args.get("since", "") if since else "",
               "until": request.args.get("until", "") if until else ""}
    base_qs = urlencode(filters)
    return render_template("log.html", page="log", rows=rows, total=total, pg=page,
                           per_page=per_page, kind=kind, cls=cls or "", source=source or "",
                           since=filters["since"], until=filters["until"], classes=classes,
                           base_qs=base_qs, errors=errors)


PALETTE = ["#4ade80", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa", "#34d399", "#f87171", "#22d3ee"]
RANGES = {"1h": 1, "24h": 24, "7d": 24 * 7}


def chart_svg(rows: list, hours: int) -> str:
    """Stacked bar chart of animal detections, drawn on the server as inline SVG (no JavaScript library).
    Hourly bars for windows up to 48 h, daily bars for longer windows."""
    now = datetime.now().astimezone()
    per_day = hours > 48
    if per_day:
        keys = [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(hours // 24 - 1, -1, -1)]
    else:
        keys = [(now - timedelta(hours=i)).strftime("%Y-%m-%dT%H") for i in range(hours - 1, -1, -1)]
    classes = sorted({r["class_name"] for r in rows})
    totals = {k: {c: 0 for c in classes} for k in keys}
    for r in rows:
        key = r["hour"][:10] if per_day else r["hour"]
        if key in totals:
            totals[key][r["class_name"]] += r["n"]
    peak = max((sum(v.values()) for v in totals.values()), default=0)
    if peak == 0:
        return '<p class="muted">No animal detections in this window.</p>'
    W, H, pad = 900, 220, 26
    n = len(keys)
    bw = (W - pad) / n
    parts = [f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H + 30}" role="img" '
             f'aria-label="Animal detections over time" style="min-width:560px;width:100%;height:auto">']
    for i, k in enumerate(keys):
        x = pad + i * bw
        y = H
        for j, c in enumerate(classes):
            v = totals[k][c]
            if not v:
                continue
            h = (H - 10) * v / peak
            y -= h
            color = PALETTE[classes.index(c) % len(PALETTE)]
            parts.append(f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{max(bw - 4, 1):.1f}" height="{h:.1f}" '
                         f'fill="{color}"><title>{k} {c}: {v}</title></rect>')
        if i % max(1, n // 8) == 0:
            label = k[11:] + ":00" if not per_day else k[5:]
            parts.append(f'<text x="{x + 2:.1f}" y="{H + 18}" font-size="11" fill="currentColor">{label}</text>')
    parts.append("</svg></div>")
    legend = " ".join(f'<span class="chip-dot" style="background:{PALETTE[i % len(PALETTE)]}"></span>{c}'
                      for i, c in enumerate(classes))
    return "".join(parts) + f'<p class="legend">{legend}</p>'


def heatmap_svg(rows: list, classes: list) -> str:
    """Hour of day (columns) by class (rows), drawn as inline SVG. Darker cell = more detections."""
    if not rows or not classes:
        return '<p class="muted">No animal detections in this window.</p>'
    grid = {(r["class_name"], r["hod"]): r["n"] for r in rows}
    peak = max(grid.values())
    cell, left, top = 26, 84, 22
    W, H = left + 24 * cell + 6, top + len(classes) * cell + 20
    parts = [f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Detections by hour of day and class" '
             f'style="min-width:560px;width:100%;height:auto">']
    for h in range(24):
        if h % 3 == 0:
            parts.append(f'<text x="{left + h * cell + 2}" y="{top - 6}" font-size="11">{h:02d}</text>')
    for j, c in enumerate(classes):
        y = top + j * cell
        parts.append(f'<text x="4" y="{y + cell * 0.65}" font-size="12" fill="currentColor">{c}</text>')
        for h in range(24):
            n = grid.get((c, h), 0)
            a = 0.08 + 0.92 * n / peak if n else 0.0
            fill = f'rgba(91,157,255,{a:.2f})' if n else "transparent"
            parts.append(f'<rect x="{left + h * cell + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" '
                         f'fill="{fill}" stroke="var(--line)"><title>{c} at {h:02d}:00: {n}</title></rect>')
    parts.append("</svg></div>")
    return "".join(parts)


def hod_svg(rows: list) -> str:
    """Detections per hour of day, all classes, as bars (inline SVG)."""
    totals = [0] * 24
    for r in rows:
        totals[r["hod"]] += r["n"]
    peak = max(totals)
    if peak == 0:
        return '<p class="muted">No animal detections in this window.</p>'
    W, H, pad = 900, 180, 26
    bw = (W - pad) / 24
    parts = [f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H + 26}" role="img" aria-label="Detections by hour of day" '
             f'style="min-width:560px;width:100%;height:auto">']
    for h, v in enumerate(totals):
        bh = (H - 10) * v / peak
        x = pad + h * bw
        parts.append(f'<rect x="{x + 2:.1f}" y="{H - bh:.1f}" width="{max(bw - 4, 1):.1f}" height="{bh:.1f}" rx="3" '
                     f'fill="#2dd4bf"><title>{h:02d}:00: {v}</title></rect>')
        if h % 3 == 0:
            parts.append(f'<text x="{x + 2:.1f}" y="{H + 18}" font-size="11">{h:02d}:00</text>')
    parts.append("</svg></div>")
    return "".join(parts)


@app.route("/stats")
def stats_page():
    rng = request.args.get("range", "24h")
    if rng not in RANGES:
        rng = "24h"
    hours = RANGES[rng]
    since = (datetime.now().astimezone() - timedelta(hours=hours)).isoformat(timespec="seconds")
    animal_hours = db.hourly_counts("animal", since=since)
    animal_totals = db.class_totals("animal", since=since)
    animal_last = db.class_totals("animal")          # last seen, all time
    person_totals = db.class_totals("person", since=since)
    summary = db.summary(since=since)
    hod_rows = db.hour_of_day_counts("animal", since=since)
    heat_classes = sorted({r["class_name"] for r in hod_rows})
    return render_template("stats.html", page="stats", rng=rng, ranges=list(RANGES),
                           heat=heatmap_svg(hod_rows, heat_classes), hod=hod_svg(hod_rows),
                           hours=hours, animal_hours=animal_hours,
                           animal_totals=animal_totals, animal_last=animal_last,
                           person_totals=person_totals, summary=summary,
                           chart=chart_svg(animal_hours, hours))


@app.route("/demo/load", methods=["POST"])
def demo_load():
    names = [p.name for p in SNAPS.glob("*.jpg")][:200]
    db.load_demo(names, datetime.now().astimezone())
    return redirect(url_for("live", demo=1))


@app.route("/demo/clear", methods=["POST"])
def demo_clear():
    db.clear_demo()
    return redirect(url_for("live"))


@app.route("/report")
def report_page():
    """A printable one-page summary for the chosen window. 'Print / save as PDF' uses the browser."""
    rng = request.args.get("range", "24h")
    if rng not in RANGES:
        rng = "24h"
    hours = RANGES[rng]
    since = (datetime.now().astimezone() - timedelta(hours=hours)).isoformat(timespec="seconds")
    animal_hours = db.hourly_counts("animal", since=since)
    totals = db.class_totals("animal", since=since)
    person_totals = db.class_totals("person", since=since)
    tiles = db.gallery(since=since, limit=6)
    alerts = db.alerts_list(10)
    return render_template("report.html", page="report", rng=rng, hours=hours,
                           generated=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                           totals=totals, person_total=sum(r["n"] for r in person_totals),
                           animal_total=sum(r["n"] for r in totals), latest=db.summary(since=since)["latest_animal"],
                           chart=chart_svg(animal_hours, hours), tiles=tiles, alerts=alerts)


@app.route("/gallery")
def gallery_page():
    cls = (request.args.get("class") or "").strip() or None
    since, err_since = date_arg("since")
    until, err_until = date_arg("until")
    page = int_arg("page", 1, 1, 100000)
    per_page = 48
    tiles = db.gallery(class_name=cls, since=since, until=until, limit=per_page,
                       offset=(page - 1) * per_page)
    classes = db.classes_seen("animal")
    filters = {"class": cls or "", "since": request.args.get("since", "") if since else "",
               "until": request.args.get("until", "") if until else ""}
    return render_template("gallery.html", page="gallery", tiles=tiles, classes=classes, cls=cls or "",
                           since=filters["since"], until=filters["until"], pg=page, per_page=per_page,
                           base_qs=urlencode(filters), errors=[e for e in (err_since, err_until) if e])


@app.route("/log.csv")
def log_csv():
    kind = request.args.get("kind", "animal")
    if kind not in ("animal", "person", "all"):
        kind = "animal"
    since, _ = date_arg("since")
    until, _ = date_arg("until")
    rows = db.search(kind=None if kind == "all" else kind,
                     class_name=(request.args.get("class") or "").strip() or None,
                     source=(request.args.get("source") or "").strip() or None,
                     since=since, until=until, limit=1000000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "kind", "class", "confidence", "source", "clip_time_s", "x1", "y1", "x2", "y2", "snapshot"])
    for r in rows:
        w.writerow([r["ts"], r["kind"], r["class_name"], f"{r['confidence']:.4f}", r["source"],
                    "" if r["clip_time"] is None else r["clip_time"], r["x1"], r["y1"], r["x2"], r["y2"],
                    r["snapshot"] or ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="detections_{kind}.csv"'})


def _unit(name: str, raw: str, lo: float = 0.0, hi: float = 1.0) -> float:
    v = float(raw)  # ValueError -> caller shows a message
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}")
    return v


SETTING_FIELDS = [
    # (key, label, lo, hi, help)
    ("animal_conf", "Animal confidence", 0, 1, "Lower finds more animals and more mistakes."),
    ("person_conf", "Person detector threshold", 0, 1, "People found by the COCO model."),
    ("person_override_conf", "Keep an animal anyway at confidence", 0, 1, "A very confident animal beside a person is kept."),
    ("suppress_animal_iou", "Overlap with a person (IoU) that drops an animal", 0, 1, ""),
    ("suppress_containment", "Share of an animal box inside a person box that drops it", 0, 1, ""),
    ("presence_seconds", "Animal is present if seen within (seconds)", 1, 3600, "Used on the Animals page."),
    ("missing_alert_minutes", "Missing alert after (minutes, 0 = off)", 0, 1440, "Raised for running cameras only."),
    ("dedupe_seconds", "Log each class at most once per (seconds)", 1, 3600, "New tracks are always logged."),
]


def animal_class_names() -> list:
    return sorted(set(detector.animal_names.values()))


@app.route("/settings")
def settings_page():
    per = CFG.get("animal_conf_per_class") or {}
    return render_template("settings.html", page="settings", fields=SETTING_FIELDS, cfg=CFG,
                           classes=[(n, per.get(n, "")) for n in animal_class_names()],
                           error=request.args.get("error"), saved=request.args.get("saved"),
                           demo_loaded=db.demo_loaded())


@app.route("/settings", methods=["POST"])
def settings_save():
    """Change the thresholds while the app runs. Takes effect on the next frame."""
    try:
        new = {k: _unit(label, request.form.get(k, ""), lo, hi) for k, label, lo, hi, _ in SETTING_FIELDS}
        per = {}
        for name in animal_class_names():
            raw = (request.form.get(f"cls_{name}") or "").strip()
            if raw:
                per[name] = _unit(f"Threshold for {name}", raw)
        new["animal_conf_per_class"] = per
    except ValueError as exc:
        return redirect(url_for("settings_page", error=str(exc)))
    CFG.update(new)
    logger.dedupe = float(CFG["dedupe_seconds"])
    SETTINGS_FILE.write_text(json.dumps({k: CFG[k] for k in TUNABLE}, indent=2))
    return redirect(url_for("settings_page", saved=1))


@app.route("/settings/reset", methods=["POST"])
def settings_reset():
    """Forget the saved values and go back to config.json."""
    base = load_config()
    CFG.update({k: base[k] for k in TUNABLE if k in base})
    logger.dedupe = float(CFG["dedupe_seconds"])
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
    return redirect(url_for("settings_page", saved=1))


@app.route("/api/summary")
def api_summary():
    data = db.summary()
    data["alert"] = db.latest_alert()
    if data["alert"]:
        data["alert"]["title"] = "Missing animal"
    return jsonify(data)


if __name__ == "__main__":
    host = CFG.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise SystemExit("refusing to bind anything but 127.0.0.1")
    threading.Thread(target=alert_loop, daemon=True).start()
    app.run(host=host, port=CFG["port"], threaded=True, debug=False)
