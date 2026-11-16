"""Checks for the hosted-site helpers: rate limit, content sniffing, filenames, cleanup, purge.
Run from the project root (no model, no Flask):
    ~/.conda/envs/animal-det/bin/python -m unittest web/test_hardening.py
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import DetectionDB  # noqa: E402
from hardening import (RateLimiter, purge_older_than, safe_filename, sniff_kind,  # noqa: E402
                       trim_to_cap)


class Limiter(unittest.TestCase):
    def test_window(self):
        rl = RateLimiter(2, 10)
        self.assertTrue(rl.allow("ip", 0))
        self.assertTrue(rl.allow("ip", 1))
        self.assertFalse(rl.allow("ip", 2))          # third hit inside 10 s
        self.assertTrue(rl.allow("ip", 11.5))        # both old hits have expired

    def test_keys_are_separate(self):
        rl = RateLimiter(1, 60)
        self.assertTrue(rl.allow("a", 0))
        self.assertFalse(rl.allow("a", 1))
        self.assertTrue(rl.allow("b", 1))


class Sniff(unittest.TestCase):
    def test_real_types(self):
        self.assertEqual(sniff_kind(b"\xff\xd8\xff\xe0" + b"\0" * 12), "image")
        self.assertEqual(sniff_kind(b"\x89PNG\r\n\x1a\n" + b"\0" * 8), "image")
        self.assertEqual(sniff_kind(b"BM" + b"\0" * 14), "image")
        self.assertEqual(sniff_kind(b"RIFF\0\0\0\0WEBPVP8 "), "image")
        self.assertEqual(sniff_kind(b"\0\0\0\x18ftypmp42" + b"\0" * 4), "video")
        self.assertEqual(sniff_kind(b"\0\0\0\x14ftypqt  "), "video")
        self.assertEqual(sniff_kind(b"\x1a\x45\xdf\xa3" + b"\0" * 12), "video")
        self.assertEqual(sniff_kind(b"RIFF\0\0\0\0AVI LIST"), "video")

    def test_text_and_scripts_refused(self):
        self.assertIsNone(sniff_kind(b"not a picture at all"))
        self.assertIsNone(sniff_kind(b"<script>alert(1)</script>"))
        self.assertIsNone(sniff_kind(b""))


class Names(unittest.TestCase):
    def test_sanitised(self):
        self.assertEqual(safe_filename("My clip!.mp4", ".mp4"), "My_clip.mp4")
        self.assertEqual(safe_filename("../../x.mp4", ".mp4"), "x.mp4")
        self.assertEqual(safe_filename("ü.mp4", ".mp4"), "upload.mp4")
        self.assertEqual(len(safe_filename("a" * 200 + ".jpg", ".jpg")), 60 + 4)


class Cleanup(unittest.TestCase):
    def test_purge_by_age(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d, "old.jpg"), Path(d, "new.jpg")
            old.write_bytes(b"x")
            new.write_bytes(b"y")
            now = datetime.now().timestamp()
            os.utime(old, (now - 25 * 3600, now - 25 * 3600))
            self.assertEqual(purge_older_than([d], 24 * 3600, now), 1)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_trim_oldest_first(self):
        with tempfile.TemporaryDirectory() as d:
            names = ["a.jpg", "b.jpg", "c.jpg"]
            for i, n in enumerate(names):
                p = Path(d, n)
                p.write_bytes(b"z" * 100)
                os.utime(p, (1000 + i, 1000 + i))
            self.assertEqual(trim_to_cap([d], 150), 2)   # 300 bytes -> 100 bytes
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["c.jpg"])

    def test_trim_keeps_named_files(self):
        with tempfile.TemporaryDirectory() as d:
            for i, n in enumerate(["keep.db", "x.jpg"]):
                p = Path(d, n)
                p.write_bytes(b"z" * 100)
                os.utime(p, (1000 + i, 1000 + i))
            trim_to_cap([d], 100, keep={"keep.db"})
            self.assertTrue(Path(d, "keep.db").exists())
            self.assertFalse(Path(d, "x.jpg").exists())

    def test_db_purge_keeps_demo_rows(self):
        with tempfile.TemporaryDirectory() as d:
            db = DetectionDB(Path(d) / "t.sqlite")
            old = (datetime.now().astimezone() - timedelta(hours=30)).isoformat(timespec="seconds")
            new = datetime.now().astimezone().isoformat(timespec="seconds")
            with db._conn:
                for ts, src in ((old, "camera:0"), (new, "camera:0"), (old, DetectionDB.DEMO_SOURCE)):
                    db._conn.execute("INSERT INTO detections (ts, kind, class_name, confidence, source) "
                                     "VALUES (?, 'animal', 'cat', 0.9, ?)", (ts, src))
            cutoff = (datetime.now().astimezone() - timedelta(hours=24)).isoformat(timespec="seconds")
            removed = db.purge_before(cutoff)
            self.assertEqual(removed["detections"], 1)
            left = [r["source"] for r in db._conn.execute("SELECT source FROM detections")]
            self.assertEqual(sorted(left), sorted(["camera:0", DetectionDB.DEMO_SOURCE]))


if __name__ == "__main__":
    unittest.main()
