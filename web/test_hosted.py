"""Checks for the public (HOSTED=1) site: read-only 403s, safe pages, uploads, limits, headers.
Starts the real Flask app in hosted mode with a temporary instance folder (localhost only, no network).
Run from the project root:
    ~/.conda/envs/animal-det/bin/python -m unittest web/test_hosted.py
"""
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import unquote_plus

WEB = Path(__file__).resolve().parent
os.environ["HOSTED"] = "1"
os.environ["INSTANCE_DIR"] = tempfile.mkdtemp(prefix="animal-watch-test-")
sys.path.insert(0, str(WEB))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import app as webapp  # noqa: E402
from hardening import RateLimiter  # noqa: E402

SESSION = "/api/frame?session=abcdefgh1234"


def jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((96, 128, 3), 120, np.uint8))
    assert ok
    return buf.tobytes()


def error_of(resp) -> str:
    return unquote_plus(resp.headers.get("Location", ""))


class ReadOnly(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_admin_posts_refused_with_403(self):
        cases = [
            ("post", "/settings", {"data": {"animal_conf": "0.5"}}),
            ("post", "/settings/reset", {}),
            ("post", "/zones", {"json": {"name": "Road", "points": [[0, 0], [1, 0], [1, 1]]}}),
            ("post", "/zones/1/delete", {}),
            ("post", "/demo/load", {}),
            ("post", "/demo/clear", {}),
            ("post", "/camera/start", {"data": {"source": "0"}}),
            ("post", "/camera/abc12345/stop", {}),
        ]
        for _method, path, kw in cases:
            with self.subTest(path=path):
                r = self.c.post(path, **kw)
                self.assertEqual(r.status_code, 403, path)
                self.assertIn(b"Read-only public demo", r.data)

    def test_public_pages_200(self):
        for path in ["/live", "/events", "/animals", "/log", "/log.csv", "/stats", "/report",
                     "/gallery", "/settings", "/api/summary", "/api/zones"]:
            with self.subTest(path=path):
                self.assertEqual(self.c.get(path).status_code, 200, path)

    def test_settings_hides_controls(self):
        body = self.c.get("/settings").data
        self.assertIn(b"read-only on the public demo", body.lower())
        self.assertNotIn(b'action="/settings"', body)
        self.assertNotIn(b'action="/settings/reset"', body)
        self.assertNotIn(b'action="/demo/load"', body)
        self.assertNotIn(b'action="/demo/clear"', body)

    def test_live_hides_admin_controls(self):
        body = self.c.get("/live").data
        self.assertIn(b"Read-only on the public demo", body)
        self.assertNotIn(b'action="/demo/load"', body)
        self.assertNotIn(b'action="/demo/clear"', body)
        self.assertNotIn(b"data-del=", body)
        self.assertNotIn(b'id="z-save"', body)
        self.assertNotIn(b'action="/camera/start"', body)
        self.assertIn(b'id="bc-start"', body)           # the browser camera stays
        self.assertIn(b"Privacy.", body)                # privacy note is shown

    def test_security_headers(self):
        h = self.c.get("/live").headers
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        self.assertEqual(h["X-Frame-Options"], "DENY")
        csp = h["Content-Security-Policy"]
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("script-src 'self' 'unsafe-inline'", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_no_directory_listing(self):
        self.assertEqual(self.c.get("/static/").status_code, 404)
        self.assertEqual(self.c.get("/outputs/").status_code, 404)
        self.assertEqual(self.c.get("/snapshots/").status_code, 404)

    def test_client_ip_from_cloudflare(self):
        with webapp.app.test_request_context("/", headers={"CF-Connecting-IP": "203.0.113.9"}):
            self.assertEqual(webapp.client_ip(), "203.0.113.9")


class Uploads(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_fake_image_refused(self):
        r = self.c.post("/upload", data={"media": (io.BytesIO(b"not a picture at all"), "x.jpg")},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        self.assertIn("not a real image", error_of(r))

    def test_real_image_accepted_and_analysed(self):
        r = self.c.post("/upload", data={"media": (io.BytesIO(jpeg_bytes()), "../../My photo!.jpg")},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/job/", r.headers["Location"])
        job_id = r.headers["Location"].rsplit("/", 1)[-1]
        for _ in range(240):
            status = webapp.JOBS[job_id]["status"]
            if status in ("done", "error"):
                break
            time.sleep(0.5)
        self.assertEqual(webapp.JOBS[job_id]["status"], "done", webapp.JOBS[job_id].get("error"))
        self.assertEqual(webapp.JOBS[job_id]["name"], "My_photo.jpg")  # name sanitised

    def test_upload_rate_limit(self):
        saved = webapp.UPLOAD_LIMIT
        webapp.UPLOAD_LIMIT = RateLimiter(1, 3600)
        try:
            first = self.c.post("/upload", data={"media": (io.BytesIO(b"nope"), "a.jpg")},
                                content_type="multipart/form-data")
            self.assertIn("not a real image", error_of(first))   # counted, then refused on content
            second = self.c.post("/upload", data={"media": (io.BytesIO(jpeg_bytes()), "b.jpg")},
                                 content_type="multipart/form-data")
            self.assertIn("Too many uploads", error_of(second))
        finally:
            webapp.UPLOAD_LIMIT = saved

    def test_concurrent_job_limit(self):
        saved = webapp.CFG["max_upload_jobs"]
        webapp.CFG["max_upload_jobs"] = 0
        try:
            r = self.c.post("/upload", data={"media": (io.BytesIO(jpeg_bytes()), "c.jpg")},
                            content_type="multipart/form-data")
            self.assertIn("Other analyses are running", error_of(r))
        finally:
            webapp.CFG["max_upload_jobs"] = saved

    def test_frame_rate_limit(self):
        saved = webapp.FRAME_LIMIT
        webapp.FRAME_LIMIT = RateLimiter(2, 60)
        try:
            codes = [self.c.post(SESSION, data=b"not a jpeg", content_type="image/jpeg").status_code
                     for _ in range(3)]
            self.assertEqual(codes, [400, 400, 429])
        finally:
            webapp.FRAME_LIMIT = saved

    def test_frame_good_jpeg_still_works(self):
        r = self.c.post(SESSION, data=jpeg_bytes(), content_type="image/jpeg")
        self.assertEqual(r.status_code, 200, r.data[:200])
        self.assertIn("animals", r.get_json())


if __name__ == "__main__":
    unittest.main()
