"""Animal surveillance web app (Flask). Binds to 127.0.0.1 only.

Run from the project root:
    ~/.conda/envs/animal-det/bin/python web/app.py
Then open http://127.0.0.1:5055/
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

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

db = DetectionDB(INSTANCE / "detections.sqlite")
detector = Detector(CFG)
logger = EventLogger(db, SNAPS, CFG["dedupe_seconds"])

JOBS = {}          # job_id -> dict (in memory; the log itself is in SQLite)
JOBS_LOCK = threading.Lock()
CAMERAS = {}       # cam_id -> CameraWorker


def allowed(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXT | VID_EXT


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


@app.route("/live")
def live():
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)[:20]
    cams = [{"id": cid, "source": c.source, "running": c.running, "error": c.error}
            for cid, c in CAMERAS.items()]
    return render_template("live.html", page="live", jobs=jobs, cams=cams,
                           max_mb=CFG["max_upload_mb"])


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("media")
    if f is None or not f.filename:
        abort(400, "no file")
    if not allowed(f.filename):
        abort(400, "unsupported file type")
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
    return jsonify({k: job[k] for k in ("status", "progress", "outputs", "counts", "error")})


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
        abort(400, "camera source required (webcam index like 0, or an IP camera URL)")
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
    cls = request.args.get("class") or None
    source = request.args.get("source") or None
    since = request.args.get("since") or None
    until = request.args.get("until") or None
    page = max(1, int(request.args.get("page", 1) or 1))
    per_page = 100
    kind_arg = None if kind == "all" else kind
    rows = db.search(kind=kind_arg, class_name=cls, source=source, since=since, until=until,
                     limit=per_page, offset=(page - 1) * per_page)
    total = db.count_search(kind=kind_arg, class_name=cls, source=source, since=since, until=until)
    classes = sorted(set(db.classes_seen("animal")) | set(db.classes_seen("person")))
    return render_template("log.html", page="log", rows=rows, total=total, pg=page,
                           per_page=per_page, kind=kind, cls=cls or "", source=source or "",
                           since=since or "", until=until or "", classes=classes)


@app.route("/stats")
def stats_page():
    hours = int(request.args.get("hours", 24) or 24)
    since = (datetime.now().astimezone() - timedelta(hours=hours)).isoformat(timespec="seconds")
    animal_hours = db.hourly_counts("animal", since=since)
    person_hours = db.hourly_counts("person", since=since)
    animal_totals = db.class_totals("animal", since=since)
    animal_last = db.class_totals("animal")          # last seen, all time
    person_totals = db.class_totals("person", since=since)
    return render_template("stats.html", page="stats", hours=hours,
                           animal_hours=animal_hours, person_hours=person_hours,
                           animal_totals=animal_totals, animal_last=animal_last,
                           person_totals=person_totals)


if __name__ == "__main__":
    host = CFG.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise SystemExit("refusing to bind anything but 127.0.0.1")
    app.run(host=host, port=CFG["port"], threaded=True, debug=False)
