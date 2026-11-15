"""Checks for the W5 features: events, zones, zone alerts, demo data, alert kinds, hour-of-day counts.
Run from the project root (no model, no Flask needed):
    ~/.conda/envs/animal-det/bin/python -m unittest web/test_features.py
"""
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from db import DetectionDB  # noqa: E402
from detector import Det, EventLogger, point_in_polygon  # noqa: E402
from events import group_events  # noqa: E402

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
ROAD = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]  # top-left quarter of a 200x100 frame
INSIDE = (40.0, 10.0, 60.0, 30.0)     # centre (0.25, 0.2)
OUTSIDE = (150.0, 60.0, 170.0, 80.0)  # centre (0.8, 0.7)


class Polygon(unittest.TestCase):
    def test_inside_outside(self):
        self.assertTrue(point_in_polygon(0.5, 0.5, SQUARE))
        self.assertFalse(point_in_polygon(1.5, 0.5, SQUARE))
        self.assertFalse(point_in_polygon(0.5, -0.1, SQUARE))

    def test_concave_shape(self):
        # an L shape: the notch at the top right is outside
        l_shape = [[0, 0], [1, 0], [1, 0.4], [0.4, 0.4], [0.4, 1], [0, 1]]
        self.assertTrue(point_in_polygon(0.2, 0.8, l_shape))
        self.assertFalse(point_in_polygon(0.8, 0.8, l_shape))


def row(ts, cls="cow", source="upload:a.mp4", clip=None, track=None, conf=0.8, snap=None):
    return {"ts": ts, "class_name": cls, "source": source, "clip_time": clip, "track_id": track,
            "confidence": conf, "snapshot": snap}


class Events(unittest.TestCase):
    def test_gap_merges_and_splits(self):
        rows = [row("", clip=t) for t in (1.0, 5.0, 9.0, 60.0, 62.0)]
        ev = group_events(rows, gap=30)
        self.assertEqual(len(ev), 2)
        counts = sorted(e["count"] for e in ev)
        self.assertEqual(counts, [2, 3])

    def test_class_and_source_split_events(self):
        rows = [row("", cls="cow", clip=1.0), row("", cls="dog", clip=2.0),
                row("", cls="cow", source="upload:b.mp4", clip=3.0), row("", cls="cow", clip=4.0)]
        ev = group_events(rows, gap=30)
        self.assertEqual(len(ev), 3)

    def test_best_snapshot_and_tracks(self):
        rows = [row("", clip=1.0, track=1, conf=0.6, snap=None),
                row("", clip=2.0, track=1, conf=0.9, snap="best.jpg"),
                row("", clip=3.0, track=2, conf=0.7, snap="other.jpg")]
        (e,) = group_events(rows, gap=30)
        self.assertEqual(e["snapshot"], "best.jpg")
        self.assertEqual(e["tracks"], 2)
        self.assertAlmostEqual(e["max_conf"], 0.9)
        self.assertEqual(e["duration"], 2.0)

    def test_camera_rows_use_wall_clock(self):
        t0 = datetime(2026, 11, 15, 12, 0, 0).astimezone()
        rows = [row((t0 + timedelta(seconds=s)).isoformat(timespec="seconds"), source="camera:0") for s in (0, 10, 50)]
        ev = group_events(rows, gap=30)
        self.assertEqual(sorted(e["count"] for e in ev), [1, 2])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "d.sqlite"
        self.db = DetectionDB(self.path)

    def tearDown(self):
        self.tmp.cleanup()


class Zones(Base):
    def make_logger(self):
        logger = EventLogger(self.db, Path(self.tmp.name) / "snaps", dedupe_seconds=10)
        logger.zones_fn = self.db.list_zones
        return logger

    def frame(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def cow(self, tid, box=INSIDE):
        return Det("animal", "cow", 0.9, box, tid)

    def test_zone_crud(self):
        zid = self.db.add_zone("Road", SQUARE, "2026-11-15T12:00:00+00:00")
        self.assertEqual(self.db.list_zones()[0]["points"], SQUARE)
        self.assertTrue(self.db.delete_zone(zid))
        self.assertEqual(self.db.list_zones(), [])

    def test_tracked_animal_alerts_once_per_entry(self):
        self.db.add_zone("Road", ROAD, "t")
        logger = self.make_logger()
        cam = "camera-z"
        logger.log(cam, "camera:0", 1.0, [self.cow(7)], self.frame(), is_camera=True)
        logger.log(cam, "camera:0", 2.0, [self.cow(7)], self.frame(), is_camera=True)  # still inside: no new alert
        alerts = [a for a in self.db.alerts_list() if a["kind"] == "zone"]
        self.assertEqual(len(alerts), 1)
        self.assertIn("cow #7 entered zone 'Road'", alerts[0]["message"])

    def test_leave_and_re_enter_alerts_again(self):
        self.db.add_zone("Road", ROAD, "t")
        logger = self.make_logger()
        cam = "camera-z"
        logger.log(cam, "camera:0", 1.0, [self.cow(7)], self.frame(), is_camera=True)
        logger.log(cam, "camera:0", 2.0, [self.cow(7, box=OUTSIDE)], self.frame(), is_camera=True)
        logger.log(cam, "camera:0", 3.0, [self.cow(7)], self.frame(), is_camera=True)
        self.assertEqual(len([a for a in self.db.alerts_list() if a["kind"] == "zone"]), 2)

    def test_outside_zone_no_alert(self):
        self.db.add_zone("Corner", [[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]], "t")
        logger = self.make_logger()
        logger.log("camera-z", "camera:0", 1.0, [self.cow(7)], self.frame(), is_camera=True)
        self.assertEqual(self.db.alerts_list(), [])

    def test_untracked_alerts_are_rate_limited(self):
        self.db.add_zone("Road", ROAD, "t")
        logger = self.make_logger()
        plain = Det("animal", "dog", 0.8, INSIDE, None)
        logger.log("camera-z", "camera:0", 100.0, [plain], self.frame(), is_camera=True)
        logger.log("camera-z", "camera:0", 130.0, [plain], self.frame(), is_camera=True)  # within 60 s
        logger.log("camera-z", "camera:0", 170.0, [plain], self.frame(), is_camera=True)  # after 60 s
        self.assertEqual(len([a for a in self.db.alerts_list() if a["kind"] == "zone"]), 2)

    def test_uploads_do_not_raise_zone_alerts(self):
        self.db.add_zone("Road", ROAD, "t")
        logger = self.make_logger()
        logger.log("upload-x", "upload:x.mp4", 1.0, [self.cow(7)], self.frame(), clip_time=1.0)
        self.assertEqual(self.db.alerts_list(), [])


class Demo(Base):
    def test_load_and_clear_keep_real_rows(self):
        real = [("2026-11-15T10:00:00+00:00", "animal", "cow", 0.9, "camera:0", None, 1, 1, 2, 2, None, 1)]
        self.db.insert_many(real)
        now = datetime(2026, 11, 15, 12, 0, 0).astimezone()
        counts = self.db.load_demo([], now)
        self.assertGreater(counts["detections"], 100)
        self.assertTrue(self.db.demo_loaded())
        again = self.db.load_demo([], now)  # loading again replaces the demo rows, it does not add to them
        self.assertEqual(again["detections"], counts["detections"])
        self.assertEqual(self.db.count_search(source="demo:sample"), counts["detections"])
        self.assertEqual(self.db.count_search(kind="person", source="demo:sample"), 90)
        self.db.clear_demo()
        self.assertFalse(self.db.demo_loaded())
        self.assertEqual([r["source"] for r in self.db.search()], ["camera:0"])

    def test_hour_of_day_counts(self):
        self.db.insert_many([
            ("2026-11-15T07:10:00+00:00", "animal", "cow", 0.9, "camera:0", None, 1, 1, 2, 2, None, 1),
            ("2026-11-15T07:40:00+00:00", "animal", "cow", 0.9, "camera:0", None, 1, 1, 2, 2, None, 1),
            ("2026-11-15T19:00:00+00:00", "animal", "dog", 0.9, "camera:0", None, 1, 1, 2, 2, None, 2),
        ])
        rows = {(r["hod"], r["class_name"]): r["n"] for r in self.db.hour_of_day_counts("animal")}
        self.assertEqual(rows, {(7, "cow"): 2, (19, "dog"): 1})


class AlertKindMigration(unittest.TestCase):
    def test_old_alerts_table_gains_kind(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.sqlite"
            con = sqlite3.connect(str(path))
            con.executescript("""CREATE TABLE alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                source_id TEXT NOT NULL, track_id INTEGER NOT NULL, class_name TEXT NOT NULL, source TEXT NOT NULL,
                last_seen TEXT NOT NULL, message TEXT NOT NULL, snapshot TEXT);
                INSERT INTO alerts (ts, source_id, track_id, class_name, source, last_seen, message)
                VALUES ('t','camera-a',3,'cow','camera:0','t','cow #3 not seen');""")
            con.commit()
            con.close()
            db = DetectionDB(path)
            self.assertEqual(db.alerts_list()[0]["kind"], "missing")
            self.assertEqual(db.latest_alert()["kind"], "missing")


if __name__ == "__main__":
    unittest.main()
