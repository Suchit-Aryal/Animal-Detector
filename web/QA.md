# Web QA log

Tested in the Munder Difflin Browser (md-browser) at http://127.0.0.1:5055/ on 15 Nov 2026. Only localhost; no sign-in, no downloads.
Test data: Open Images frames and the two Open Images test clips (people only, mixed), plus a webcam (index 0).

Tooling note (not an app bug): md-browser `click` on a `<select>` or after a page reload did not always submit a form. JavaScript `element.click()` and `requestSubmit()` did. A real Start click on an empty camera field is blocked by the browser's required-field check, which is correct.

## Phase 1: bugs found and fixed

| # | Bug | Cause | Fix | How verified |
|---|---|---|---|---|
| 1 | A date-only "To" (or "From") dropped every row of that day. Filter `since=2026-11-15&until=2026-11-15` showed 0 rows. | Dates were compared as full timestamps, so `2026-11-15T21:50:00+05:45` is greater than `2026-11-15`. | `db.py` `_where()`: compare `substr(ts,1,len)` with the typed prefix, so a date means the whole day and `T14:30` means that minute. | Browser: the same filter now shows 24 rows. |
| 2 | Pager and links dropped part of a source filter with `&` or spaces. | Template built the query string by hand, without URL encoding. | `app.py` builds `base_qs` with `urlencode`; template uses it. | Browser: 105 temp rows with source `qa&pager test`; "older" link keeps the source; page 2 shows 5 rows. Temp rows deleted afterwards. |
| 3 | `/log?page=abc` returned 500 Internal Server Error. | `int()` on a raw query value without checks. | `int_arg()` falls back to the default for bad or out-of-range values. | Browser: 200, page 1. |
| 4 | `/stats?hours=abc` returned 500; `hours=-5` was accepted silently. | Same as 3; no range check. | `int_arg("hours", 24, 1, 8760)`. | Browser: `abc` shows the 24 h default; `-5` also falls back. |
| 5 | A malformed date (e.g. `garbage`) was ignored with no message. | No validation. | `date_arg()` checks the format and shows a warning on the Log page. | Browser: "'since' must look like 2026-11-15 or 2026-11-15T14:30 (got 'garbage'), so it was ignored". |
| 6 | With Kind = animal, the Class list still offered "person", which always returns 0 rows. | Class list was the union of all kinds. | Class list follows the Kind choice. | HTML check: animal list has no person option. |
| 7 | An unsupported upload (e.g. .txt) showed a bare "Bad Request" page. | `abort(400)`. | Redirects to Live with a clear message. | Browser: "Unsupported file type. Use a video (...) or an image (...)." |
| 8 | Starting a camera with an empty field gave a bare 400 page. | `abort(400)`. | Redirects to Live with "Enter a webcam index ... or an IP camera URL." (The browser also blocks an empty field before any request.) | HTTP check: redirect to `/live?cam_error=...` and the message is shown on the page. |
| 9 | An unknown job or page showed Flask's plain default 404 page. | No custom 404 handler. | `not_found.html` with a link back. | Browser: "Not found. That page or job does not exist." |
| 10 | An image upload said "Animal detections (sampled frames)". | Same wording for images and videos. | Wording depends on output type ("in this image" for images). | Browser: job page says "Animal detections in this image: 1 ...". |

## Checked and working (no change)

- Kind, class and source filters: browser test, e.g. class cow + source mixed returned the 2 expected rows.
- Image upload: annotated image shown; log row written; screenshot checked.
- Video upload: annotated H.264 video loads (readyState 4, duration 8 s, no error).
- Corrupt video: job shows status "error" and "could not open video".
- Live webcam: start shows a running feed (640 px MJPEG), stop returns the page to "No camera running".
- Stats: last-seen table, per-class bars, per-hour table, people count all render; screenshot checked.
- Empty states: Log shows "No rows." and Stats shows "No animal detections in this window." (window code path).
- Refresh and back: the Log filter is kept in the URL; the job page reloads from the server.

## Known limits (not bugs; see Phase 2 for what was added)

- Uploads and job status are kept in memory: after a server restart, "Recent uploads" is empty. The detections stay in SQLite.
- A single image yields one log row per class (dedupe window), even if it shows two animals of one class. This is by design (dedupe_seconds).

## Phase 2: UI upgrade (from Captain's note results/web_research.md)

Ideas taken (features only, no code copied; no CDN, no new libraries):

| Idea (research rank) | What was built | Where |
|---|---|---|
| Snapshot gallery (1) | `/gallery`: one tile per snapshot, class chips, date filter, click opens the full frame | `app.py` `gallery_page`, `templates/gallery.html`, `db.gallery()` |
| Live counters + alert banner (2) | Live page counters (animals, people, last animal seen) and a banner when a new animal is logged; polls `/api/summary` every 5 s; the first poll shows no banner for old detections | `app.py` `api_summary`, `static/app.js`, `templates/live.html` |
| Stats upgrade (4) | Range buttons 1h / 24h / 7d, summary cards, stacked bar chart drawn as inline SVG on the server (no JS library); hourly bars up to 48 h, daily bars beyond | `app.py` `chart_svg`, `templates/stats.html` |
| CSV export (7) | `/log.csv` with the same filters as the Log page ("Export CSV" link) | `app.py` `log_csv` |
| Per-class chips + responsive layout | Class chips on Log and Gallery; grid layout for cards and gallery; chart scrolls sideways on phones | `templates/`, `static/style.css` |

Not built (too big or out of scope for this pass): timeline scrubber on uploads (3), per-class settings page (5), polygon zones (6), event grouping (8). Skipped by the note: MQTT, face recognition, clip recording.

Verification (browser and headless screenshots):

| Check | Result |
|---|---|
| Gallery: 16 snapshot tiles, class chip "cow" gives 5 tiles, all labelled cow | pass |
| Gallery screenshot (desktop) | thumbnails, chips and captions render |
| Log: class chips and "Export CSV"; `/log.csv?kind=animal&class=cow` returns text/csv | pass |
| Stats 7 d: 4 cards; chart stacked by class; totals match the class table (6+5+5+1 = 17) | pass (screenshot checked) |
| Stats on a 390 px phone: cards stack; chart scrolls sideways with readable labels | pass (screenshot checked) |
| Live: counter 16 -> 17 and banner "Animal detected: cow ..." after a new upload, with no reload | pass (browser) |
| Live: camera card and stop | pass (Phase 1) |

Not verified: the dark theme. The headless browser did not apply the dark colour scheme, so the dark CSS is untested visually (the colours come from the same CSS variables).

## Findings for god (not web bugs)

- Dark webcam frames: the animal model labels people's faces as "dog" or "cat" (gallery frames from the webcam test). The COCO person model did not detect those faces, so the overlap rule (IoU > 0.6) had nothing to compare against. This is the animal model, not the page. Fix options: a higher animal confidence for low-light scenes, or a person-confidence threshold change.

## Phase 3 (web-W3): dark-frame false alarms, thresholds, Settings

Problem: the dark webcam view produced "dog" and "cat" logs on a person's face.

Changes (all in web/):
- Animal confidence default raised from 0.35 to 0.5. Per-class override in config (`animal_conf_per_class`, e.g. cow=0.6).
- Person detector threshold lowered from 0.4 to 0.25.
- Animal box dropped when a person is found and any of these hold: centre inside a person box; IoU > 0.3 with a person box; more than 70% of the animal box inside a person box.
- New: a confident animal (confidence >= 0.8, `person_override_conf`) is kept even beside a person. Without this, a real cow at 0.94 next to a person was dropped (found in the mixed clip).
- Settings section on the Live page: change the thresholds while the app runs; applies to the next frame; saved to `web/instance/settings.json`; validated (0 to 1; per-class format checked).

Important: during this test the animal model file `results/weights/best.pt` was replaced (modified 22:16:15, while testing). Earlier numbers and the new numbers come from different weights. The table below uses a frozen copy of the new weights for both settings.

Test clips: the raw dark webcam clip (15 s, mean brightness 31/255), `people_only.mp4`, `mixed.mp4`, and `cows_only.mp4` (9 real cow photos from the checked Mendeley set, no people). Counts are per sampled frame (every 3rd frame).

| Clip | Old model, old settings (before the swap) | Frozen new weights, OLD settings | Frozen new weights, NEW settings |
|---|---|---|---|
| dark webcam | dog 2 (false) | 0 | 0 |
| people only (mascots, see note) | cow 6 | cow 7 | cow 6 |
| mixed (cows + people) | goat 7, cow 13 | goat 3, cow 6 (21 suppressed) | goat 3, cow 13 (0 suppressed) |
| cows only (real cows) | cow 40 | cow 36 | cow 36 |

Notes:
- The new weights already stop the dark-webcam false alarms. The settings still matter for the rest.
- People-only "cow" hits are on inflatable cow mascots in the picture (`0032cf08baca880c.jpg`), not real cows. No threshold setting removes these; they are a limit of the model and should be reported as such.
- Lowering the animal threshold to 0.5 removes some weak real-cow hits (40 to 36 on the cows clip). This is the trade-off god asked for.
- Live check: with the animal threshold at 0.95, a cow photo gave 0 animal hits; at 0.5 it gave 1. The Settings change took effect on the next upload. A 20-second run of the webcam in the dark room logged no new rows.
- Settings form checked in the browser: change to 0.6, "Saved." shown, value kept.

Rule checks: `web/test_rules.py` (6 tests, all pass): centre inside, IoU rule (checked on and off), containment, cow beside a person kept, and small edge touch kept.

Dark theme (visual): the browser tool's dark-scheme setting did not change the screenshot (the page reported dark mode but the screenshot timed out). Visual check done instead on a temporary copy of the Stats page with the dark palette forced on (the real stylesheet with its dark rule enabled). It renders with dark cards, readable text and chart labels.

## Phase 4 (web-W4): tracking, Animals page, missing-animal alert

What was built:
- Track IDs per animal (ByteTrack via Ultralytics `model.track`, `persist=True`). Each camera start and each upload has its own tracker. Stored in `detections.track_id` (the table was upgraded in place; old rows kept, `track_id` NULL).
- Tables `tracks` (first/last sighting, frames, snapshot), `sources` (last frame clock per source), `alerts`.
- `/animals` page: status (present or missing), class, track, source, first/last seen, duration, snapshot; filter all/present/missing; alert list.
- Missing-animal alert: raised once per gap for a tracked animal on a running camera, when not seen for `missing_alert_minutes` (default 0 = off). Shown as a banner on Live and a row on Animals.
- Settings on the Live page: `presence_seconds` (default 30) and `missing_alert_minutes`.

Test clip `leave.mp4` (scratchpad `leave/`): 6 s of cow photos from `cows_only.mp4`, then 8 s of a plain grey frame (the animal leaves). Test stream: the same clip looped by ffmpeg over HTTP at 127.0.0.1:8099, used as an IP camera.

Browser checks (md-browser, localhost only):

| Check | Result |
|---|---|
| Upload `leave.mp4` on Live, Animals page | 5 cow tracks listed with clip times (e.g. "5.2 s into clip"). |
| Settings saved: presence 3 s, alert 0.5 min | "Saved", values kept after reload. |
| Upload status after the clip (3 s window) | all 5 "missing": the cows left the frame before the clip ended. |
| IP camera (loop stream) for about 90 s, alerts 0.5 min | Animals page: 30 tracks, 29 missing, 1 present; alert rows such as "cow #20 not seen for 0.5 min (camera:http://127.0.0.1:8099/)". |
| Live banner | "Missing animal: cow #29 not seen for 0.5 min ... open Animals" shown on the Live page. |
| Stop camera | Page shows "No camera running". No new alerts after the stop (rule tested in `test_tracks.py`). |
| Server log | No tracebacks. |

Re-run after the final code (curl, same settings): 30 alerts, the latest "cow #8 not seen for 0.5 min", no errors. Screenshots: the md-browser screenshot timed out again (CDP `Page.captureScreenshot`), so no image of the Animals page was saved. The page was checked through the page text instead.

Unit tests: `web/test_tracks.py` (migration of an old database, status rule, upsert keeps the first snapshot, alert once per gap, no alert for stopped cameras or uploads, a track seen again can alert again, new track logged once then dedupe, person rows have no track) and `web/test_rules.py`. 15 tests, all pass.

Detection regression (mixed.mp4, every 5th frame, 16 samples, same weights and thresholds):

| Run | Animal boxes | With a track ID | Without |
|---|---|---|---|
| Plain detection (before tracking) | 10 | – | – |
| Tracker output only (first version) | 7 | – | – |
| Final: plain detection + tracker IDs | 10 | 5 | 5 |

The first version used only the tracker output. ByteTrack reports a new track only from its second processed frame, and it dropped boxes at the start of each track, so 3 of 10 animal boxes were lost. The final version keeps every plain detection and copies the track ID onto a box when the tracker has a box with IoU above 0.5. The cost is a second run of the animal model per processed frame (about 0.37 s per sample pair on this CPU, measured on the mixed clip).

Limits found in testing:
- The test clip is a slideshow of photos, not motion, so track numbers change every few seconds (up to 30 in 90 s). Real video should give fewer, longer tracks; not measured yet.
- Half of the animal boxes on the mixed clip have no track ID (the first frame of each track). They are logged, but cannot trigger a missing alert on their own.
- Upload status uses the clip end as "now". With the default 30 s window, a clip shorter than 30 s always shows "present".
- Missing alerts are judged on wall time for cameras; a camera that stops reading frames keeps its last status.

## Phase 5 (web-W5): features from web_research.md and polish

Built, in order of value for effort:

| Feature | Where | Checked how |
|---|---|---|
| Design pass: logo in the header, one colour set (tokens), theme button (light/dark, saved in the browser, set before the page paints) | `base.html`, `static/style.css`, `static/logo.svg`, `static/app.js` | Browser: toggle to light, reload, stays light; computed background dark `rgb(12,17,24)` when dark. |
| Snapshot lightbox with prev/next, keys (arrows, Esc), counter | Gallery and Events tiles (`a.lb-link`) | Browser: 48 tiles; opens at 1/48; next shows 2/48; close works. Events page: opens 1/18. |
| Live FPS and count overlay on the camera feed | `detector.py` `CameraWorker._stamp` | Frame saved from the stream and viewed: "4.6 fps · animals 1 people 0", track label "cow #2 0.84". |
| Sound and desktop alert options (remembered) | Live page, `static/app.js` | Options save and reload. Sound is a short beep (Web Audio). Desktop notification needs the browser permission; not granted in the test, so not seen working. |
| Stats: activity by hour of day (bars) and heatmap hour x class | `app.py` `hod_svg`, `heatmap_svg`, `stats.html` | Browser: 3 charts; heatmap 129 cells. |
| Printable report | `/report?range=`, `report.html`, print CSS | Browser: 3 cards, 6 snapshot tiles. "Print / save as PDF" uses the browser print dialog (not run). |
| Demo mode: invented week of detections, labelled "demo", clearable; real rows kept apart | `db.py` `load_demo`, `clear_demo`; Live and Settings buttons | Browser: demo note shown, 386 animals (real + demo), Animals 130 rows. Clear: 386 -> 110 (real rows kept). Test: loading twice does not double rows. |
| Event grouping: same class in the same source, merged when no more than N s apart (N in the page, default 30) | `events.py`, `/events` | Browser: 16 event tiles. Unit tests: gap, class/source split, best snapshot, tracks, wall-clock cameras. |
| Clip timeline on the job page: one mark per animal per half second; click seeks the video | `detector.py` markers, `job.html`, `static/app.js` | Browser: 10 marks on the leave clip; click sets the video time. |
| Settings page: all thresholds, per-class confidence table (blank = default), dedupe seconds, reset to config.json, sample-data buttons | `/settings`, `templates/settings.html` | Browser: per-class 0.6 and dedupe 12 saved, shown after reload, `settings.json` written. Server rejects 1.5 ("must be between 0 and 1"). Reset removes `settings.json`. |
| Zones: draw a rectangle (2 clicks) or polygon (click corners, Close) on a camera feed, name it, save; delete from the list | Live page zone card, `/zones`, `zones` table | Browser: rectangle saved (4 corners). Test camera with zone "Road" (whole frame) and zone "Corner" (top-left 5%): "Road" 13 entry alerts; "Corner" 0. Banner shows "Zone entry: cow #17 entered zone 'Browser test'". |

Zone alert rules: a tracked animal alerts once each time its box centre enters the zone (leaving and re-entering alerts again). An untracked animal (no track ID yet) alerts at most once a minute per zone. Uploads never raise zone alerts. Zones apply to every camera, not one camera.

Tests: `web/test_features.py` (30 tests in total with `test_tracks.py` and `test_rules.py`, all pass).

Not done, or limited:
- Zones are the same for every camera (no per-camera zones); rectangles are drawn with two clicks, not by dragging.
- Event grouping works on the detection log by time gap; it does not merge tracks across gaps.
- Timeline is per clip (on the job page), not one timeline across all clips.
- "Hourly activity" is hour-of-day bars over the chosen window, not a time series; the existing "Animals over time" chart is unchanged.
- Screenshots: the md-browser screenshot timed out again and headless Chrome is not installed here, so no page images were saved. Checks were DOM checks, computed colours and one saved frame.
- The dark theme colours were checked through computed styles, not by eye. Phone widths were not re-checked in this phase.
- Desktop notifications were not tested with a granted permission.
