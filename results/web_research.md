# Web UI research: similar projects (2026-11-15)
Features only. Do NOT copy code. Stars/licences from the GitHub API today; feature notes from READMEs (what a README did not mention is marked "not stated").
Honest finding: no repo combines animal/stray detection + Flask + SQLite + snapshot gallery. Closest are listed below.

## Repos
| # | Repo | Stars | Licence | What it does | Worth copying (as an idea) |
|---|---|---|---|---|---|
| 1 | [blakeblackshear/frigate](https://github.com/blakeblackshear/frigate) | 37.5k | MIT | Self-hosted NVR with local object detection for IP cameras | Live dashboard, multi-camera scrubbing, mask/zone editor, retention by detected object, MQTT hook |
| 2 | [roflcoopter/viseron](https://github.com/roflcoopter/viseron) | 3.7k | MIT | Self-hosted NVR + object/face detection (YOLO among backends) | Web UI, component/config approach; events/timeline are in its docs site (not checked) |
| 3 | [ZoneMinder/zoneminder](https://github.com/ZoneMinder/zoneminder) | 6.0k | GPL-2.0 | Classic CCTV system; AI via add-on | Event list per camera; GPL so ideas only |
| 4 | [FlyingFathead/dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) | 28 | NOASSERTION (unclear; treat as no reuse) | YOLOv8 human/animal DVR, Flask web UI, Telegram alerts | Image carousel of captured stills, live preview (MJPEG/HLS), detection graphs over several time ranges, per-zone min confidence, cooldown grouping (30 s), save crop vs full frame |
| 5 | [gotogrub/OpenCV-Animal-Detector](https://github.com/gotogrub/OpenCV-Animal-Detector) | 3 | none stated (no licence = all rights reserved) | YOLOv8 wildlife zones, SQLite log, alerts | Polygon zones filtered by species, loitering detection, SQLite with indexes (species/zone/time), alert dedup cooldown (60 s), pre/post-event clip recording, YAML config |
| 6 | [Gocodein/spidy](https://github.com/Gocodein/spidy) | 0 | Apache-2.0 | YOLOv8 wildlife detector + tracking, SQLite, Streamlit dashboard (per search summary, not opened) | Behaviour/disturbance labels, tracking IDs |
| 7 | [CodeforNepal/wild-watch](https://github.com/CodeforNepal/wild-watch) | 0 | MIT | Nepal wildlife-near-settlement YOLOv8; sends detections to an ESP32 siren/endpoint | Local-context idea: physical alert output (siren/webhook) |
| 8 | [shreehari-revankar/WildEye](https://github.com/shreehari-revankar/WildEye) | 31 | MIT | Animal detection + classification with YOLO | Upload-and-annotate flow (not opened in detail) |
| 9 | [ldebele/Real-Time-Animal-Species-Detection](https://github.com/ldebele/Real-Time-Animal-Species-Detection) | 14 | none stated | YOLOv8 + Streamlit species demo, hosted on Hugging Face | Simple demo page with model metrics shown (mAP, precision, recall) |

## What our site already has (web/README.md, templates)
Upload (video/image, background job), live webcam/IP camera MJPEG, Log (filter by kind/class/source/time, snapshot links), Stats (per class, per hour, last seen, people separate), person suppression, dedupe_seconds, SQLite, config.json.

## Ranked top-8 features for OUR site
| Rank | Feature | Why | Effort |
|---|---|---|---|
| 1 | Snapshot gallery page (thumbnail grid, filter by class/date, click for full frame + metadata) | Snapshots exist already; best demo value; seen in dvr-yolov8 carousel | S |
| 2 | Live "Alerts" banner + sound/toast when a chosen class (e.g. cow on road) is detected, with cooldown | Core of the "stray animals on streets" story; cooldown pattern seen in 2 repos | S |
| 3 | Per-clip timeline: upload result page shows detection markers on a time bar; click jumps the video to that second (clip-time column exists) | Strong for presentation; Frigate-style scrubbing idea | M |
| 4 | Stats upgrade: time-range selector (1h/24h/7d), stacked per-class chart, heatmap hour x class | dvr-yolov8 has graphs over several ranges | S |
| 5 | Per-class confidence thresholds + settings page (edit conf, dedupe, frame-skip without editing config.json) | Goat/dog are weaker than cat; per-zone/class thresholds are common | M |
| 6 | Polygon zones ("road area"): only log/alert animals whose box centre is inside a drawn zone | Frigate/OpenCV-Animal-Detector both do this; fits "on the street" | L |
| 7 | Export: CSV download of the log + one-click PDF/HTML summary report | Cheap, useful for slides | S |
| 8 | Event grouping: merge detections of same class within N seconds into one "event" with first/last seen and best snapshot (optionally simple IoU tracking IDs) | Reduces log noise; seen as cooldown grouping in dvr-yolov8, tracking in spidy | M |

Out of scope / skip: MQTT, Home Assistant, pre/post-event clip recording (disk heavy), face recognition, sound deterrents.
Licence note: only MIT/Apache repos could ever be reused with attribution; we are taking ideas only, no code.
