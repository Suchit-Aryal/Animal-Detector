#!/bin/sh
# Start the website for the Hugging Face Space: one worker process, 8 threads, port 7860.
# One worker keeps one copy of the models in memory. Logs go to the Space's Logs tab.
set -e
cd /app/web
exec gunicorn --workers 1 --worker-class gthread --threads 8 --timeout 120 \
  --access-logfile - --error-logfile - --bind 0.0.0.0:7860 app:app
