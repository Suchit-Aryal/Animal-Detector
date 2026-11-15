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
