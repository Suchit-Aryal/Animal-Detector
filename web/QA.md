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
| 8 | Starting a camera with an empty field gave a bare 400 page. | `abort(400)`. | Redirects to Live with "Enter a webcam index ... or an IP camera URL." (The browser also blocks an empty field before any request.) | Code path checked; empty field is blocked by the browser first. |
| 9 | An unknown job or page showed Flask's plain default 404 page. | No custom 404 handler. | `not_found.html` with a link back. | Browser: "Not found. That page or job does not exist." |
| 10 | An image upload said "Animal detections (sampled frames)". | Same wording for images and videos. | Wording depends on output type ("in this image" for images). | See Phase 2 check. |

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
