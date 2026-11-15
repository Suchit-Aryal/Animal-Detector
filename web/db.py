"""SQLite detection log, track table and alerts.

detections: one row per logged detection. kind is 'animal' or 'person'. People are stored
with kind='person' so they are counted separately and never appear as animals.
track_id is the tracker's ID for an animal (NULL for people). IDs are per source session
(one camera start or one upload), not true re-identification.

tracks: one row per (source, track_id) with first/last time seen. Used for the Animals page
and the missing-animal alert. sources: last time each source produced a frame, which is the
reference clock for "present" versus "missing". alerts: missing-animal alerts.
"""

import json
import random
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,          -- local time, ISO 8601 with offset, when logged
    kind        TEXT NOT NULL CHECK (kind IN ('animal', 'person')),
    class_name  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    source      TEXT NOT NULL,          -- e.g. 'upload:clip.mp4' or 'camera:0'
    clip_time   REAL,                   -- seconds into an uploaded clip (NULL for live)
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, -- box in pixels of the source frame
    snapshot    TEXT                    -- file name in the snapshots folder
);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_det_class ON detections(kind, class_name, ts);
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    is_camera   INTEGER NOT NULL DEFAULT 0,
    last_clock  REAL NOT NULL,          -- source clock of the last frame (clip seconds or epoch)
    last_ts     TEXT NOT NULL           -- wall time of the last frame
);
CREATE TABLE IF NOT EXISTS tracks (
    source_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    source_label TEXT NOT NULL,
    is_camera   INTEGER NOT NULL DEFAULT 0,
    first_clock REAL NOT NULL,
    last_clock  REAL NOT NULL,
    first_ts    TEXT NOT NULL,
    last_ts     TEXT NOT NULL,
    frames      INTEGER NOT NULL DEFAULT 1,
    snapshot    TEXT,
    alerted     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source_id, track_id)
);
CREATE TABLE IF NOT EXISTS zones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    points      TEXT NOT NULL,          -- JSON list of [x, y], normalised 0..1 of the frame
    created     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    source      TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    message     TEXT NOT NULL,
    snapshot    TEXT,
    kind        TEXT NOT NULL DEFAULT 'missing'   -- 'missing' or 'zone'
);
"""

DET_COLUMNS = ("ts, kind, class_name, confidence, source, clip_time, x1, y1, x2, y2, snapshot, track_id")


def status_of(ref_clock: float, last_clock: float, presence_s: float) -> str:
    """'present' if the track was seen within presence_s of the source's latest frame, else 'missing'."""
    return "present" if ref_clock - last_clock <= presence_s else "missing"


class DetectionDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that older databases do not have. Existing rows are kept."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(detections)")}
        if "track_id" not in cols:
            with self._conn:
                self._conn.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER")
        acols = {r["name"] for r in self._conn.execute("PRAGMA table_info(alerts)")}
        if "kind" not in acols:
            with self._conn:
                self._conn.execute("ALTER TABLE alerts ADD COLUMN kind TEXT NOT NULL DEFAULT 'missing'")

    def insert_many(self, rows: list) -> None:
        if not rows:
            return
        sql = f"INSERT INTO detections ({DET_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        with self._lock, self._conn:
            self._conn.executemany(sql, rows)

    def record_frame(self, source_id: str, label: str, is_camera: bool, clock: float, ts: str,
                     det_rows: list, track_rows: list) -> int:
        """One transaction per processed frame: detections, track upserts and the source clock.
        track_rows: dicts with source_id, track_id, class_name, is_camera, clock, ts, snapshot
        (snapshot only for a track's first logged frame; kept after that)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sources (source_id, label, is_camera, last_clock, last_ts) VALUES (?,?,?,?,?) "
                "ON CONFLICT(source_id) DO UPDATE SET label=excluded.label, last_clock=excluded.last_clock, "
                "last_ts=excluded.last_ts",
                (source_id, label, int(is_camera), clock, ts))
            if det_rows:
                self._conn.executemany(
                    f"INSERT INTO detections ({DET_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", det_rows)
            for t in track_rows:
                self._conn.execute(
                    "INSERT INTO tracks (source_id, track_id, class_name, source_label, is_camera, first_clock, "
                    "last_clock, first_ts, last_ts, frames, snapshot) VALUES (?,?,?,?,?,?,?,?,?,1,?) "
                    "ON CONFLICT(source_id, track_id) DO UPDATE SET last_clock=excluded.last_clock, "
                    "last_ts=excluded.last_ts, frames=tracks.frames+1, alerted=0, "
                    "snapshot=COALESCE(tracks.snapshot, excluded.snapshot)",
                    (t["source_id"], t["track_id"], t["class_name"], label, int(is_camera),
                     t["clock"], t["clock"], t["ts"], t["ts"], t.get("snapshot")))
        return len(det_rows)

    def tracks_list(self, presence_s: float, status: str = "all", limit: int = 300) -> list:
        """Tracks newest first, each with its status. The reference clock is the source's last frame."""
        sql = ("SELECT t.*, s.last_clock AS ref_clock FROM tracks t JOIN sources s USING (source_id) "
               "ORDER BY s.last_ts DESC, t.last_clock DESC LIMIT ?")
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(sql, (limit * 3,)).fetchall()]
        out = []
        for r in rows:
            r["status"] = status_of(r["ref_clock"], r["last_clock"], presence_s)
            if status in ("present", "missing") and r["status"] != status:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def raise_missing_alerts(self, cutoff: float, running_sources: list, message_fmt: str, now_iso: str) -> list:
        """Alert once per track of a running camera that has not been seen since cutoff (epoch seconds).
        Returns the alerts written. Sets alerted=1 so the same gap does not alert twice."""
        if not running_sources:
            return []
        marks = ",".join("?" for _ in running_sources)
        with self._lock, self._conn:
            rows = [dict(r) for r in self._conn.execute(
                f"SELECT * FROM tracks WHERE is_camera = 1 AND alerted = 0 AND last_clock <= ? "
                f"AND source_id IN ({marks})", (cutoff, *running_sources)).fetchall()]
            out = []
            for r in rows:
                msg = message_fmt.format(cls=r["class_name"], tid=r["track_id"], src=r["source_label"])
                self._conn.execute(
                    "INSERT INTO alerts (ts, source_id, track_id, class_name, source, last_seen, message, snapshot) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (now_iso, r["source_id"], r["track_id"], r["class_name"], r["source_label"],
                     r["last_ts"], msg, r["snapshot"]))
                self._conn.execute("UPDATE tracks SET alerted = 1 WHERE source_id=? AND track_id=?",
                                   (r["source_id"], r["track_id"]))
                out.append({"class": r["class_name"], "track_id": r["track_id"], "message": msg})
            return out

    def add_alert(self, ts, source_id, track_id, class_name, source, last_seen, message, snapshot, kind) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO alerts (ts, source_id, track_id, class_name, source, last_seen, message, snapshot, kind) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, source_id, track_id, class_name, source, last_seen, message, snapshot, kind))

    def list_zones(self) -> list:
        with self._lock:
            rows = self._conn.execute("SELECT id, name, points, created FROM zones ORDER BY id").fetchall()
        return [{"id": r["id"], "name": r["name"], "points": json.loads(r["points"]), "created": r["created"]}
                for r in rows]

    def add_zone(self, name: str, points: list, created: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("INSERT INTO zones (name, points, created) VALUES (?,?,?)",
                                     (name, json.dumps(points), created))
            return cur.lastrowid

    def delete_zone(self, zone_id: int) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,)).rowcount > 0

    def alerts_list(self, limit: int = 50) -> list:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def latest_alert(self):
        with self._lock:
            r = self._conn.execute("SELECT id, ts, message, kind FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
            return dict(r) if r else None

    @staticmethod
    def _where(kind=None, class_name=None, source=None, since=None, until=None):
        """Build the WHERE clause. since/until are prefixes of the local time such as
        '2026-11-15' or '2026-11-15T14:30', compared on the same number of characters,
        so a date-only 'until' includes the whole day."""
        where, args = [], []
        if kind in ("animal", "person"):
            where.append("kind = ?")
            args.append(kind)
        if class_name:
            where.append("class_name = ?")
            args.append(class_name)
        if source:
            where.append("source LIKE ?")
            args.append(f"%{source}%")
        if since:
            where.append("substr(ts, 1, ?) >= ?")
            args.extend([len(since), since])
        if until:
            where.append("substr(ts, 1, ?) <= ?")
            args.extend([len(until), until])
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        return clause, args

    def search(self, kind=None, class_name=None, source=None, since=None, until=None,
               limit=200, offset=0) -> list:
        clause, args = self._where(kind, class_name, source, since, until)
        sql = f"SELECT * FROM detections {clause} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, (*args, limit, offset)).fetchall()]

    def count_search(self, kind=None, class_name=None, source=None, since=None, until=None) -> int:
        clause, args = self._where(kind, class_name, source, since, until)
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM detections {clause}", args).fetchone()[0]

    def hourly_counts(self, kind: str, since=None) -> list:
        """Rows of (hour, class_name, n) for one kind, hour like '2026-11-15T16'."""
        sql = ("SELECT substr(ts,1,13) AS hour, class_name, COUNT(*) AS n FROM detections "
               "WHERE kind = ?")
        args = [kind]
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        sql += " GROUP BY hour, class_name ORDER BY hour DESC, class_name"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    DEMO_SOURCE = "demo:sample"
    DEMO_SOURCE_ID = "demo-sample"

    def demo_loaded(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM detections WHERE source = ? LIMIT 1",
                                      (self.DEMO_SOURCE,)).fetchone() is not None

    def clear_demo(self) -> None:
        """Remove only the demo rows (source 'demo:sample'). Real rows are never touched."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM detections WHERE source = ?", (self.DEMO_SOURCE,))
            self._conn.execute("DELETE FROM alerts WHERE source = ?", (self.DEMO_SOURCE,))
            self._conn.execute("DELETE FROM tracks WHERE source_id = ?", (self.DEMO_SOURCE_ID,))
            self._conn.execute("DELETE FROM sources WHERE source_id = ?", (self.DEMO_SOURCE_ID,))

    def load_demo(self, snapshots: list, now) -> dict:
        """Fill the log with about a week of invented detections so the pages look alive.
        Clears earlier demo rows first. Snapshots are picked from real snapshot files (may be empty)."""
        self.clear_demo()
        rnd = random.Random(7)
        classes = ["cow", "dog", "goat", "cat", "sheep"]
        weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        day_hours = [6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21]
        det, trk, alerts = [], [], []
        for tid in range(1, 41):
            cls = rnd.choices(classes, weights)[0]
            day = now - timedelta(days=rnd.randint(0, 6))
            start = day.replace(hour=rnd.choice(day_hours), minute=rnd.randint(0, 59), second=0, microsecond=0)
            if start > now:
                start -= timedelta(days=1)
            end = start + timedelta(seconds=rnd.randint(8, 240))
            snap = rnd.choice(snapshots) if snapshots and rnd.random() < 0.6 else None
            frames = 0
            t = start
            while t <= end:
                conf = round(rnd.uniform(0.52, 0.97), 2)
                x1, y1 = rnd.uniform(0, 400), rnd.uniform(0, 260)
                det.append((t.isoformat(timespec="seconds"), "animal", cls, conf, self.DEMO_SOURCE, None,
                            x1, y1, x1 + rnd.uniform(80, 220), y1 + rnd.uniform(60, 180),
                            snap if frames == 0 else None, tid))
                frames += 1
                t += timedelta(seconds=20)
            trk.append((self.DEMO_SOURCE_ID, tid, cls, self.DEMO_SOURCE, 0, start.timestamp(), end.timestamp(),
                        start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"), frames, snap))
        # two animals seen just now, so the Animals page shows some 'present' rows
        for tid, cls in ((41, "cow"), (42, "goat")):
            t0 = now - timedelta(seconds=rnd.randint(5, 20))
            det.append((t0.isoformat(timespec="seconds"), "animal", cls, 0.9, self.DEMO_SOURCE, None,
                        100.0, 100.0, 260.0, 240.0, None, tid))
            trk.append((self.DEMO_SOURCE_ID, tid, cls, self.DEMO_SOURCE, 0, t0.timestamp(), t0.timestamp(),
                        t0.isoformat(timespec="seconds"), t0.isoformat(timespec="seconds"), 1, None))
        for _ in range(90):
            t = now - timedelta(days=rnd.randint(0, 6), minutes=rnd.randint(0, 1400))
            det.append((t.isoformat(timespec="seconds"), "person", "person", round(rnd.uniform(0.4, 0.95), 2),
                        self.DEMO_SOURCE, None, 50.0, 40.0, 200.0, 300.0, None, None))
        for tid in (3, 9):
            t = now - timedelta(hours=rnd.randint(1, 60))
            alerts.append((t.isoformat(timespec="seconds"), self.DEMO_SOURCE_ID, tid, "cow", self.DEMO_SOURCE,
                           t.isoformat(timespec="seconds"),
                           f"cow #{tid} not seen for 10 min ({self.DEMO_SOURCE})", None))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sources (source_id, label, is_camera, last_clock, last_ts) VALUES (?,?,?,?,?)",
                (self.DEMO_SOURCE_ID, self.DEMO_SOURCE, 0, now.timestamp(), now.isoformat(timespec="seconds")))
            self._conn.executemany(f"INSERT INTO detections ({DET_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", det)
            self._conn.executemany(
                "INSERT INTO tracks (source_id, track_id, class_name, source_label, is_camera, first_clock, last_clock, "
                "first_ts, last_ts, frames, snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?)", trk)
            self._conn.executemany(
                "INSERT INTO alerts (ts, source_id, track_id, class_name, source, last_seen, message, snapshot) "
                "VALUES (?,?,?,?,?,?,?,?)", alerts)
        return {"detections": len(det), "tracks": len(trk), "alerts": len(alerts)}

    def hour_of_day_counts(self, kind: str = "animal", since=None) -> list:
        """Rows of (hod, class_name, n): when of the day detections happen (0-23, local time)."""
        sql = ("SELECT CAST(substr(ts,12,2) AS INTEGER) AS hod, class_name, COUNT(*) AS n "
               "FROM detections WHERE kind = ?")
        args = [kind]
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        sql += " GROUP BY hod, class_name"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def class_totals(self, kind: str, since=None) -> list:
        sql = "SELECT class_name, COUNT(*) AS n, MAX(ts) AS last_seen FROM detections WHERE kind = ?"
        args = [kind]
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        sql += " GROUP BY class_name ORDER BY class_name"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def classes_seen(self, kind: str) -> list:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT DISTINCT class_name FROM detections WHERE kind = ? ORDER BY class_name", (kind,))]

    def gallery(self, class_name=None, source=None, since=None, until=None, limit=60, offset=0) -> list:
        """One tile per snapshot (animal rows only), newest first, with the classes in that frame."""
        clause, args = self._where("animal", class_name, source, since, until)
        sql = (f"SELECT snapshot, MAX(ts) AS ts, GROUP_CONCAT(DISTINCT class_name) AS classes, "
               f"MAX(source) AS source FROM detections {clause} AND snapshot IS NOT NULL "
               f"GROUP BY snapshot ORDER BY ts DESC LIMIT ? OFFSET ?")
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, (*args, limit, offset)).fetchall()]

    def summary(self, since=None) -> dict:
        """Counts and last-seen values for the dashboard counters and the live alert."""
        with self._lock:
            out = {}
            for kind in ("animal", "person"):
                sql = "SELECT COUNT(*), MAX(ts) FROM detections WHERE kind = ?"
                args = [kind]
                if since:
                    sql += " AND ts >= ?"
                    args.append(since)
                n, last = self._conn.execute(sql, args).fetchone()
                out[kind] = {"count": n, "last_seen": last}
            out["animal"]["classes"] = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT class_name FROM detections WHERE kind='animal' ORDER BY class_name")]
            r = self._conn.execute(
                "SELECT class_name, ts FROM detections WHERE kind='animal' ORDER BY ts DESC, id DESC LIMIT 1").fetchone()
            out["latest_animal"] = {"class": r[0], "ts": r[1]} if r else None
            return out
