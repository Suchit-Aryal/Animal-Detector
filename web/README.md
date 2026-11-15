# Animal surveillance web app

Detects animals in uploaded videos/images and in live webcam or IP-camera feeds.
People are detected with the COCO `person` class, shown separately, and never logged as animals.
Every logged detection goes to a SQLite log with time, class, confidence, source and a snapshot.

## Run

From the project root (needs Flask in the `animal-det` conda env):

    ~/.conda/envs/animal-det/bin/python web/app.py

Open http://127.0.0.1:5055/ . The server binds to 127.0.0.1 only.

## Pages

- **Live / Upload**: upload a video or image (processed in the background, annotated output plays in the browser); or start a webcam (index such as `0`) or IP-camera URL and watch the annotated MJPEG feed.
- **Log**: searchable, filterable table (kind, class, source, time range), newest first, with snapshot links.
- **Stats**: animals per class in a time window, animals per hour, last seen per class, and people counted separately.

## Settings: `web/config.json`

- `animal_weights`: the animal model (default `results/weights/best.pt`). Replace the file or change the path here.
- `person_weights`: COCO model for people (default `yolov8n.pt`).
- `animal_conf`, `person_conf`, `imgsz`, `device` (default `cpu`).
- `suppress_animal_iou`: an animal box overlapping any person box with IoU above this is dropped (default 0.6).
- `dedupe_seconds`: log each class at most once per this many seconds per source (default 10). Uploads use clip time; live cameras use wall-clock time.
- `upload_frame_skip`, `live_frame_skip`: run detection every N frames (keeps CPU calm).

## Data

- `web/instance/detections.sqlite`: the detection log (table `detections`).
- `web/instance/snapshots/`: annotated JPEG per logged frame.
- `web/instance/outputs/`: annotated videos and images from uploads.
- `web/instance/uploads/`: original uploads.

`web/instance/` is git-ignored.

## Notes

- Uploaded videos: a clip-time column records where in the clip each detection was.
- Ultralytics runs one predict call at a time (a lock), so a live feed and an upload share the model safely.
- Nothing is sent to external services.
