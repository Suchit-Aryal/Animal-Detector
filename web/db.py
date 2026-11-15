"""SQLite detection log.

One row per logged detection. kind is 'animal' or 'person'. People are stored
with kind='person' so they are counted separately and never appear as animals.
"""

import sqlite3
import threading
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
"""


class DetectionDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def insert_many(self, rows: list) -> None:
        if not rows:
            return
        sql = ("INSERT INTO detections (ts, kind, class_name, confidence, source, clip_time, "
               "x1, y1, x2, y2, snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
        with self._lock, self._conn:
            self._conn.executemany(sql, rows)

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
