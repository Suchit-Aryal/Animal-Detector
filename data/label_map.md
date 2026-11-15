# Label map

Our ids: `0 dog, 1 goat, 2 cow, 3 cat, 4 sheep`. Rules:
- Bovines (cow, cattle, ox, buffalo, bull) are all `cow`.
- Any class not in our five (deer, rabbit, hen, vehicle, person, ...) is `drop`: its boxes are removed.
- The mapping for each source is in `src/data/roboflow_mapping.json`. A source with no entry is skipped, never guessed.

Status: no Roboflow source mapped yet (all BLOCKED on download). Each source's class names must be read from its `data.yaml` before its entry is added.

| Source folder | Source class name | Our id | Notes |
|---|---|---|---|
| (none yet) | | | |

Open Images (`data/raw/openimages_yolo`), already in our ids: 0 Dog, 1 Goat, 2 Cattle, 3 Cat.
