"""
Task A1: extract every Nth frame from each video as JPG.

Input : videos in data/raw/videos/ (mp4, avi, mov, mkv, webm)
Output: data/raw/frames/<video_name>/frame_000001.jpg, ...

Uses the ffmpeg command-line tool (no Python video library needed).
Existing frame folders are never overwritten: a video whose output
folder already holds JPGs is skipped with a message.

Run from the project root:
    python src/data/extract_frames.py --every 12
    python src/data/extract_frames.py --videos data/raw/videos --out data/raw/frames --every 12
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_one(video: Path, out_dir: Path, every: int) -> int:
    """Save every `every`-th frame of `video` into `out_dir`. Returns frames saved."""
    if out_dir.exists() and any(out_dir.glob("*.jpg")):
        print(f"SKIP {video.name}: {out_dir} already has JPGs (not overwritten)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video),
        "-vf", f"select=not(mod(n\\,{every}))",
        "-fps_mode", "vfr",
        "-q:v", "2",
        "-n",  # never overwrite an existing file
        str(out_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {result.stderr.strip()}")
    saved = len(list(out_dir.glob("*.jpg")))
    print(f"OK   {video.name}: {saved} frames -> {out_dir}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default="data/raw/videos", type=Path)
    parser.add_argument("--out", default="data/raw/frames", type=Path)
    parser.add_argument("--every", default=12, type=int, help="keep every Nth frame (default 12)")
    args = parser.parse_args()

    if args.every < 1:
        print("ERROR: --every must be >= 1", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH", file=sys.stderr)
        return 2
    if not args.videos.is_dir():
        print(f"ERROR: video folder not found: {args.videos}", file=sys.stderr)
        return 2

    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {args.videos}")
        return 0

    total = 0
    for video in videos:
        total += extract_one(video, args.out / video.stem, args.every)
    print(f"Done: {total} frames saved from {len(videos)} video(s), every {args.every} frame(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
