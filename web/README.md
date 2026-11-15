# Animal surveillance web app

Finds animals in uploaded videos or images and in live webcam or IP-camera feeds. Logs every
detection with time, class, confidence, source and a snapshot. Tracks each animal so you can see
when one has not been seen for a while. People are detected with the COCO `person` class and are
never logged as animals.

## User guide

1. **Start the app** (needs Flask in the `animal-det` conda env), from the project root:

       ~/.conda/envs/animal-det/bin/python web/app.py

   Open http://127.0.0.1:5055/ . The server only listens on 127.0.0.1.

2. **Check a video or photo**: on *Live / Upload*, choose a file and click Analyse. When it is
   done the annotated result plays in the browser and the detections go to the log.

3. **Watch a camera**: on *Live / Upload*, type a webcam index (`0`) or an IP-camera URL and click
   Start. Click Stop to end it.

4. **See which animals were seen**: the *Animals* page lists one row per tracked animal, with its
   class, track number, source, first and last sighting and duration. *present* means seen within
   the last few seconds; *missing* means not seen for longer.

5. **Get told when an animal is missing**: in *Settings* on the Live page, set *Alert when an
   animal has not been seen for (minutes)* to, for example, 10. While a camera runs, a banner
   appears on the Live page and a row goes to the alerts list on the Animals page. The default is 0,
   which turns alerts off.

6. **Look back**: *Log* (filter by kind, class, source, time), *Gallery* (one tile per snapshot),
   *Stats* (counts per class and hour). *Export CSV* on the Log page downloads the same rows.

## How it works and its limits

- Each animal in a video or camera feed gets a track number from ByteTrack, the tracker built into
  Ultralytics. Each camera start and each upload has its own tracker, so numbers start again at
  each start.
- A track number is **not** identification of a specific animal. If an animal leaves the frame
  for long enough, or the tracker loses it, it comes back with a new number. Two animals can never
  share a number within one session, but one animal can have several numbers.
- The tracker confirms a new track only on its second processed frame. Detection still logs the
  first frame, but without a track number, so a brief glimpse may have no track.
- *present* / *missing* compares each track with the newest frame of its source. For a camera that
  is the current time while the camera runs. For an upload it is the end of the clip, so with the
  default 30 s window a 14 s clip shows every animal as present.
- Missing alerts are raised only for cameras that are running. Stopping a camera does not raise
  alerts. Each gap alerts once; the same track can alert again after it is seen and goes missing
  again.
- Animal boxes that overlap a person are dropped (see the thresholds on the Live page), so people
  are not logged as animals.

## Settings: `web/config.json`

- `animal_weights`: the animal model (default `results/weights/best.pt`).
- `person_weights`: COCO model for people (default `yolov8n.pt`).
- `animal_conf` (0.5), `animal_conf_per_class` (e.g. `cow=0.6`), `person_conf` (0.25),
  `person_override_conf` (0.8), `suppress_animal_iou` (0.3), `suppress_containment` (0.7).
- `presence_seconds` (30): seen within this many seconds counts as present.
- `missing_alert_minutes` (0 = off): alert when a tracked animal on a running camera is not seen for
  this long.
- `dedupe_seconds` (10): a class is logged at most once per this many seconds per source (a new
  track is always logged once).
- `upload_frame_skip`, `live_frame_skip`: run detection every N frames.
- `imgsz`, `device` (`cpu`).

The thresholds, presence window and alert minutes can also be changed on the Live page. Those
changes are saved to `web/instance/settings.json` and win over `config.json` at the next start.

## Data

- `web/instance/detections.sqlite`: tables `detections` (log, with `track_id`), `tracks`,
  `sources` and `alerts`. An older database is upgraded when the app starts; no rows are lost.
- `web/instance/snapshots/`: annotated JPEG per logged frame.
- `web/instance/outputs/`: annotated videos and images from uploads.
- `web/instance/uploads/`: original uploads.

`web/instance/` is git-ignored.

## Tests

    ~/.conda/envs/animal-det/bin/python -m unittest web/test_rules.py web/test_tracks.py

Neither test needs the model or Flask. See `web/QA.md` for the browser tests.
