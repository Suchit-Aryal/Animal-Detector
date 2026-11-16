"""Abuse limits and safety helpers for the hosted (public) site. Used only when HOSTED=1.

Nothing here touches Flask, so the checks can be unit tested directly (see test_hardening.py).
"""

import os
import re
import threading
import time
from collections import deque
from pathlib import Path

# Security headers sent on every hosted response. The CSP is conservative: scripts and images from this
# site only; inline styles are allowed (the templates use style="..." attributes); no frames, no plugins.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), payment=(), usb=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"),
}

READ_ONLY_MESSAGE = ("Read-only public demo: this change is turned off on the hosted site. "
                     "Run the app on your own computer to change it.")


class RateLimiter:
    """Sliding window per key (for example a client IP). allow() records the hit if it is within the limit."""

    def __init__(self, limit: int, window_s: float, max_keys: int = 5000):
        self.limit, self.window_s, self.max_keys = int(limit), float(window_s), max_keys
        self._hits = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > self.max_keys:  # forget the quietest keys so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window_s][:1000]:
                    self._hits.pop(k, None)
            return True


# First bytes of each accepted type. Checked against the file's content, not only its name.
_IMAGE_SIGNS = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM")
_VIDEO_BOXES = {b"ftyp", b"moov", b"mdat", b"wide", b"free"}  # MP4 / MOV ('ftyp' is at byte 4)


def sniff_kind(head: bytes) -> str | None:
    """'image', 'video' or None, from the first 16 bytes of a file."""
    if head.startswith(_IMAGE_SIGNS):
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "video"
    if head[4:8] in _VIDEO_BOXES:
        return "video"
    if head[:4] == b"\x1a\x45\xdf\xa3":  # Matroska / WebM
        return "video"
    return None


def safe_filename(name: str, ext: str) -> str:
    """Keep letters, digits, '-' and '_' from the name; the extension is the one already validated."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")[:60]
    return (stem or "upload") + ext


def tree_bytes(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def purge_older_than(folders, max_age_s: float, now: float | None = None) -> int:
    """Delete regular files whose modification time is older than max_age_s. Returns the count removed."""
    now = time.time() if now is None else now
    removed = 0
    for folder in folders:
        for p in Path(folder).glob("*"):
            try:
                if p.is_file() and now - p.stat().st_mtime > max_age_s:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def trim_to_cap(folders, cap_bytes: int, keep: set = frozenset()) -> int:
    """If the folders hold more than cap_bytes, delete the oldest files first until they fit.
    Files named in keep are never deleted. Returns the count removed."""
    files = []
    for folder in folders:
        for p in Path(folder).glob("*"):
            if p.is_file() and p.name not in keep:
                try:
                    st = p.stat()
                    files.append((st.st_mtime, st.st_size, p))
                except OSError:
                    pass
    used = sum(size for _m, size, _p in files) + sum(
        p.stat().st_size for folder in folders for p in Path(folder).glob("*")
        if p.is_file() and p.name in keep)
    removed = 0
    for _m, size, p in sorted(files, key=lambda t: t[0]):
        if used <= cap_bytes:
            break
        try:
            p.unlink()
            used -= size
            removed += 1
        except OSError:
            pass
    return removed
