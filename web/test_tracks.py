"""Checks for tracking, the Animals status, the track table migration and missing-animal alerts.
Run from the project root (no model needed):
    ~/.conda/envs/animal-det/bin/python -m unittest web/test_tracks.py
"""
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from db import DetectionDB, status_of  # noqa: E402
from detector import Det, EventLogger  # noqa: E402

FMT = "{cls} #{tid} not seen for 1 min ({src})"
NOW = "2026-11-15T12:00:00+00:00"


def track(source, tid, clock, cls="cow", camera=False, snap=None):
    return {"source_id": source, "track_id": tid, "class_name": cls, "clock": clock,
            "ts": "2026-11-15T12:00:00+00:00", "snapshot": snap, "_camera": camera}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DetectionDB(Path(self.tmp.name) / "d.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def frame(self, source, clock, tracks, camera=False):
        rows = [dict(t) for t in tracks]
        for r in rows:
            r.pop("_camera", None)
        return self.db.record_frame(source, "label", camera, clock, NOW, [], rows)


class Migration(unittest.TestCase):
    def test_old_database_gains_track_id_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.sqlite"
            con = sqlite3.connect(str(path))
            con.executescript("""CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                kind TEXT NOT NULL, class_name TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL,
                clip_time REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, snapshot TEXT);
                INSERT INTO detections (ts, kind, class_name, confidence, source) VALUES ('2026-11-01T10:00:00+00:00','animal','cow',0.9,'upload:x.mp4');""")
            con.commit()
            con.close()
            db = DetectionDB(path)
            cols = {r["name"] for r in db._conn.execute("PRAGMA table_info(detections)")}
            self.assertIn("track_id", cols)
            rows = db.search()
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["track_id"])
            DetectionDB(path)  # opening again must not fail


class Status(Base):
    def test_present_and_missing_against_source_clock(self):
        self.frame("camera-a", 100.0, [track("camera-a", 1, 95.0, camera=True), track("camera-a", 2, 60.0, camera=True)], camera=True)
        by_id = {r["track_id"]: r["status"] for r in self.db.tracks_list(30.0)}
        self.assertEqual(by_id, {1: "present", 2: "missing"})

    def test_status_of_boundary(self):
        self.assertEqual(status_of(100.0, 70.0, 30.0), "present")
        self.assertEqual(status_of(100.0, 69.9, 30.0), "missing")

    def test_upsert_keeps_first_snapshot_and_counts_frames(self):
        self.frame("upload-j", 1.0, [track("upload-j", 7, 1.0, snap="first.jpg")])
        self.frame("upload-j", 2.0, [track("upload-j", 7, 2.0, snap="second.jpg")])
        (r,) = self.db.tracks_list(30.0)
        self.assertEqual((r["first_clock"], r["last_clock"], r["frames"]), (1.0, 2.0, 2))
        self.assertEqual(r["snapshot"], "first.jpg")


class Alerts(Base):
    def test_camera_track_alerts_once_when_running_and_gone(self):
        now = time.time()
        self.frame("camera-a", now - 600, [track("camera-a", 1, now - 600, camera=True)], camera=True)
        self.frame("camera-a", now, [], camera=True)  # the camera is still producing frames
        out = self.db.raise_missing_alerts(now - 60, ["camera-a"], FMT, NOW)
        self.assertEqual([a["track_id"] for a in out], [1])
        self.assertEqual(self.db.raise_missing_alerts(now - 60, ["camera-a"], FMT, NOW), [])
        self.assertEqual(len(self.db.alerts_list()), 1)
        self.assertIn("not seen for 1 min", self.db.latest_alert()["message"])

    def test_stopped_camera_and_uploads_do_not_alert(self):
        now = time.time()
        self.frame("camera-b", now - 600, [track("camera-b", 1, now - 600, camera=True)], camera=True)
        self.frame("upload-j", 5.0, [track("upload-j", 1, 1.0)])
        self.assertEqual(self.db.raise_missing_alerts(now - 60, [], FMT, NOW), [])
        self.assertEqual(self.db.raise_missing_alerts(now - 60, ["camera-a"], FMT, NOW), [])
        self.assertEqual(self.db.alerts_list(), [])

    def test_track_seen_again_can_alert_again(self):
        now = time.time()
        self.frame("camera-a", now - 600, [track("camera-a", 1, now - 600, camera=True)], camera=True)
        self.db.raise_missing_alerts(now - 60, ["camera-a"], FMT, NOW)
        self.frame("camera-a", now - 5, [track("camera-a", 1, now - 5, camera=True)], camera=True)
        self.assertEqual(self.db.raise_missing_alerts(now - 60, ["camera-a"], FMT, NOW), [])  # seen again: no alert
        self.assertEqual(len(self.db.raise_missing_alerts(now + 60, ["camera-a"], FMT, NOW)), 1)  # gone again later


class Logging(Base):
    def test_new_track_logged_once_then_dedupe(self):
        logger = EventLogger(self.db, Path(self.tmp.name) / "snaps", dedupe_seconds=10)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cow = lambda tid: Det("animal", "cow", 0.9, (1.0, 1.0, 10.0, 10.0), tid)
        self.assertEqual(logger.log("upload-x", "upload:x", 0.0, [cow(1)], img), 1)   # new track
        self.assertEqual(logger.log("upload-x", "upload:x", 2.0, [cow(1)], img), 0)   # same track, within 10 s
        self.assertEqual(logger.log("upload-x", "upload:x", 3.0, [cow(2)], img), 1)   # new track id
        self.assertEqual(logger.log("upload-x", "upload:x", 20.0, [cow(1)], img), 1)  # 10 s window passed
        self.assertEqual(len(self.db.search(kind="animal")), 3)
        self.assertEqual({r["track_id"] for r in self.db.search(kind="animal")}, {1, 2})
        tracks = {r["track_id"]: r for r in self.db.tracks_list(30.0)}
        self.assertEqual(tracks[1]["frames"], 3)
        self.assertIsNotNone(tracks[1]["snapshot"])

    def test_person_rows_have_no_track(self):
        logger = EventLogger(self.db, Path(self.tmp.name) / "snaps", dedupe_seconds=10)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        p = Det("person", "person", 0.9, (1.0, 1.0, 10.0, 10.0))
        logger.log("camera-z", "camera:0", time.time(), [p], img, is_camera=True)
        (r,) = self.db.search(kind="person")
        self.assertIsNone(r["track_id"])
        self.assertEqual(self.db.tracks_list(30.0), [])


if __name__ == "__main__":
    unittest.main()
