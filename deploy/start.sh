#!/bin/sh
# Start the public website (Hugging Face Space or a laptop tunnel): one worker process, 4 threads, port 7860.
# One worker keeps one copy of the models in memory and one set of job and rate-limit records. Logs go to stdout.
set -e
cd /app/web
exec gunicorn --workers 1 --worker-class gthread --threads 4 --timeout 120 \
  --access-logfile - --error-logfile - --bind 0.0.0.0:7860 app:app
