---
title: Animal Surveillance
emoji: 🐄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Find and log animals in videos, images and your browser camera
---

# Animal Surveillance

Upload a video or an image and the site finds cows, goats, sheep, dogs and cats, logs each one with a time,
a confidence and a snapshot, and tracks each animal so it is logged once. People are detected too, but are never
logged as animals.

- **Upload** a video or image (up to 50 MB). On this site only the first 60 seconds of a video are analysed.
- **Your camera**: press *Start camera* on the Live page. Your camera stays in your browser. One frame a second is
  sent to this site for detection.
- **Log, Stats, Animals, Gallery, Settings, Zones** work as in the full version.

This is a public demo on free CPU hardware. Data resets when the Space restarts (demo data is loaded again),
anyone who visits can change settings and zones, and analysis is slower than on a PC with a graphics card.
Not for real monitoring.
