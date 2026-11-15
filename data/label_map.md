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

## Dataset v2 mapping (merge_v2.py)

| Source | Source class (id) | Our id | Notes |
|---|---|---|---|
| Open Images goat/sheep (`openimages_goat_sheep`) | already our ids (0 dog, 1 goat, 2 cow/Cattle, 3 cat, 4 sheep) | same | Group-of boxes removed. |
| Mendeley herbivores5 (`mendeley_herbivores5`) | 0 Buffalo | 2 cow | Bovine rule. |
| | 1 Camel | drop the image | Not one of our classes. |
| | 2 Cow | 2 cow | |
| | 3 Goat | 1 goat | |
| | 4 Sheep | 4 sheep | |
