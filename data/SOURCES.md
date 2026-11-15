# Data sources

Status key: **BLOCKED** = not downloaded yet. Roboflow Universe pages return 403 to anonymous requests and export needs a login/API key. The human must download each set (YOLOv8 export, **no augmentation**) and unzip it into `data/raw/roboflow/<owner>_<project>/`.

Only public data goes to train/val. Never test. Licence and image counts are filled in after download (check the licence on each page).

| Folder (`<owner>_<project>`) | Source URL | Licence | Images used | Status | Notes |
|---|---|---|---|---|---|
| fyp-gztsq_stray-dogs-2wfjc | https://universe.roboflow.com/fyp-gztsq/stray-dogs-2wfjc | CC BY 4.0 (per plan) | - | BLOCKED | Augmented export: re-export without augmentation |
| cv-nwjrg_stray-dogs-hfuft | https://universe.roboflow.com/cv-nwjrg/stray-dogs-hfuft | check page | - | BLOCKED | |
| bothina_stray-dogs-ikjii | https://universe.roboflow.com/bothina/stray-dogs-ikjii | check page | - | BLOCKED | |
| lr3_stray-dogs-dsns7 | https://universe.roboflow.com/lr3/stray-dogs-dsns7 | CC BY 4.0 (per plan) | - | BLOCKED | **3 classes**: read names before mapping |
| z-firewood-house_goat-5eecg | https://universe.roboflow.com/z-firewood-house/goat-5eecg | check page | - | BLOCKED | |
| au-90a9a_goat-u87lc | https://universe.roboflow.com/au-90a9a/goat-u87lc | check page | - | BLOCKED | |
| jush_goat-unuwu | https://universe.roboflow.com/jush/goat-unuwu | CC BY 4.0 (per plan) | - | BLOCKED | |
| deer-tvf5a_goat-brt8c | https://universe.roboflow.com/deer-tvf5a/goat-brt8c | CC BY 4.0 (per plan) | - | BLOCKED | |
| amna-yps68_goats-ln02f | https://universe.roboflow.com/amna-yps68/goats-ln02f | CC BY 4.0 (per plan) | - | BLOCKED | |
| reshma-silp9_goat-detection-m0vb6-ge3d2 | https://universe.roboflow.com/reshma-silp9/goat-detection-m0vb6-ge3d2 | check page | - | BLOCKED | **Augmented** (~5097 images): re-export without augmentation or flag |
| object-detection-naauz_cowsonroad | https://universe.roboflow.com/object-detection-naauz/cowsonroad | check page | - | BLOCKED | |
| loliktry_cattle_detection-060yo | https://universe.roboflow.com/loliktry/cattle_detection-060yo | check page | - | BLOCKED | |
| lin-vc6m8_cattle915 | https://universe.roboflow.com/lin-vc6m8/cattle915 | check page | - | BLOCKED | |
| harsh-rsnmo_cattle_ahead | https://universe.roboflow.com/harsh-rsnmo/cattle_ahead | check page | - | BLOCKED | |
| ash-pkvgo_cattle-detection-rv8bv | https://universe.roboflow.com/ash-pkvgo/cattle-detection-rv8bv | check page | - | BLOCKED | |
| stray-cats_stray-cat-ph | https://universe.roboflow.com/stray-cats/stray-cat-ph | check page | - | BLOCKED | |
| detection-of-stray-cats (project id unknown) | project page not yet found | check page | - | BLOCKED | Find the project id before downloading |
| masterdrawup_stray-cat | https://universe.roboflow.com/masterdrawup/stray-cat | check page | - | BLOCKED | Classes: dog + straycat: read names before mapping |

Already on disk (not Roboflow): `data/raw/openimages_yolo` (Open Images V7, 798 images), `data/raw/mendeley` (street animals, no boxes), `data/raw/mendeley_yolo` (empty).
