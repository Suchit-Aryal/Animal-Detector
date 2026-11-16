#!/usr/bin/env bash
# Build deploy/hf_space/: the folder that is pushed to the Hugging Face Space repo.
# Everything is copied from this repo, never typed by hand. Run it again after any change.
# Needs results/weights/best.pt and yolov8n.pt in the repo. Nothing is downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/deploy/hf_space"

for f in results/weights/best.pt yolov8n.pt web/app.py web/config.json deploy/Dockerfile deploy/start.sh \
         deploy/requirements-container.txt deploy/space_README.md deploy/dockerignore deploy/gitattributes; do
  [ -e "$ROOT/$f" ] || { echo "missing: $f" >&2; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT/web" "$OUT/results/weights"

# Website code and its templates and static files. Not the instance folder, caches or tests.
cp -r "$ROOT/web/." "$OUT/web/"
rm -rf "$OUT/web/instance"
find "$OUT/web" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f "$OUT/web/"test_*.py

cp "$ROOT/results/weights/best.pt" "$OUT/results/weights/best.pt"
cp "$ROOT/yolov8n.pt" "$OUT/yolov8n.pt"

cp "$ROOT/deploy/Dockerfile" "$ROOT/deploy/start.sh" "$ROOT/deploy/requirements-container.txt" "$OUT/"
cp "$ROOT/deploy/dockerignore" "$OUT/.dockerignore"
cp "$ROOT/deploy/gitattributes" "$OUT/.gitattributes"
cp "$ROOT/deploy/space_README.md" "$OUT/README.md"

# Checksums of the two model files, so the pushed copy can be checked against the repo.
(cd "$OUT" && sha256sum results/weights/best.pt yolov8n.pt > MODELS.sha256)

echo "built $OUT"
du -sh "$OUT"
find "$OUT" -type f | sort | sed "s#^$OUT/#  #"
