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

## Interim dataset (built 15 Nov 2026, data/dataset/)

**The interim test set is NOT Nepal data.** It is 80 held-out Open Images frames with real human-drawn boxes. It only shows the pipeline works. The honest Nepal test set is still to be made from our own labelled photos.

| Source | Used for | Images | Labels | Notes |
|---|---|---|---|---|
| openimages_yolo (Open Images V7, Dog/Goat/Cattle/Cat) | train 70% / val 20% / test 10% by image | 798 | real human boxes | Split by image (each image is its own group) |
| mendeley_yolo (Mendeley street animals, folders as class) | TRAIN ONLY | 689 (after 2 rejects) | MODEL-GENERATED boxes (yolov8m-oiv7 detector) | Never in val or test |

Mendeley boxes come from a pretrained detector, not people. They are for training only.
Spot-check: 30 Mendeley images viewed (contact sheet); 2 rejected (`src/data/mendeley_rejects.txt`): mdy_cow_00017 (box on a tower), mdy_cat_00113 (no visible box).

Final counts (boxes per class, from data/dataset/labels/):

| Split | images | dog | goat | cow | cat | sheep |
|---|---|---|---|---|---|---|
| train | 1247 | 332 | 313 | 1083 | 467 | 19 |
| val | 160 | 51 | 90 | 128 | 55 | 0 |
| test | 80 | 33 | 22 | 41 | 22 | 0 |

Known gaps: sheep has only 19 boxes (all train, none in val or test). Mendeley goat (20 images) and sheep (16 images) were far below the ~300 target: the detector found few goats or sheep in those folders.

## Dataset v2 (merged 15 Nov 2026, data/dataset/)

Built by `src/data/merge_v2.py`. Rollback: `data/dataset/backup_v1/` (hard links to the v1 files; v1 manifest and data.yaml are there).

| Source | Licence | Used for | Images | Notes |
|---|---|---|---|---|
| Open Images V7, goat/sheep subset (`data/raw/openimages_goat_sheep`) | annotations CC BY 4.0; images CC BY 2.0 (per-image licence in OI metadata) | train 80% / val 20%; 80 images to `test_extra` (40 sheep + 40 goat picks; some overlap) | 503 kept | Human boxes. 21 of the 524 images were already in `openimages_yolo` and are left out (no leak). 114 group-of boxes removed (from the OI annotation file, IsGroupOf=1). |
| Mendeley "Annotated Object Detection Dataset of Five Herbivorous Animal Classes" (`data/raw/mendeley_herbivores5`, DOI 10.17632/fwm6z9tv3f.1) | CC BY 4.0 | TRAIN ONLY (its valid split goes to val; its test split is not used) | 2,127 kept of 3,062 (after the 600-per-class cap) | Roboflow export, boxes as published (not eyeballed). ONE image per source (augmented copies dropped), dHash dedupe (distance 4), camel images dropped, cap 600 per class. Buffalo mapped to cow. |

Mendeley remap: 0 Buffalo -> 2 cow, 1 Camel -> dropped image, 2 Cow -> 2 cow, 3 Goat -> 1 goat, 4 Sheep -> 4 sheep.

Held-out "extra" split (`test_extra`): only goat and sheep images from Open Images. The original 80-image `test` is unchanged (checked byte for byte against v1). Use `data/dataset/data_extra.yaml` to evaluate on test_extra.

Quality notes (read before trusting numbers):
- Spot-check: 30 Mendeley images drawn with `data/check_labels.py` and looked at on one contact sheet. 3 rejected (listed in `src/data/mendeley5_rejects.txt`): Goat-640 (one box covers the whole frame), Goat-233 (sheep box on a goat image), Cow-345 (box fits, animal looks like a goat). The sample used for the final build was drawn again after the rejects were set, so the last 30 are not the same 30 that were looked at. Roughly 1 in 10 looked wrong.
- Whole-frame boxes: 2,761 of 8,598 Mendeley boxes cover at least 98% of the frame in both width and height. Many are close-ups where the animal fills the picture, but some are wrong. Not removed automatically; worth checking before a final model.
- Mendeley images are mostly farm and natural scenes with one animal each, not streets.

Counts per split (boxes per class; images in brackets):

| Split | dog | goat | cow | cat | sheep | images |
|---|---|---|---|---|---|---|
| train | 337 | 1109 | 1808 | 467 | 1066 | 2993 |
| val | 52 | 300 | 318 | 55 | 222 | 634 |
| test (frozen, v1) | 33 | 22 | 41 | 22 | 0 | 80 |
| test_extra (new) | 4 | 108 | 70 | 0 | 116 | 80 |

Total boxes 6,150 across train/val/test/test_extra. Dog has no test_extra data and no test sheep: sheep is evaluated only on test_extra (116 boxes). `validate_dataset.py --root data/dataset` PASSED. No source is in two splits (group check in the merge script).

## Variant: no whole-frame Mendeley images (data-A5, 15 Nov 2026)

- Config: `data/dataset/data_nowf.yaml`. Same val, test and test_extra as v2. Train points to `data/dataset/train_nowf.txt`.
- Rule: a train image from a Mendeley source (`mendeley5__`, `mendeley_yolo__`) is dropped whole if any of its boxes covers at least 98% of the frame in both width and height. Whole images are dropped, never just the box, so an unlabeled animal does not become background.
- Built by `src/data/make_nowf_list.py` (run it to regenerate the list; `--dry-run` prints counts only).
- Result: 661 of 2,993 train images dropped (572 mendeley5, 89 mendeley_yolo). 2,332 train images remain.
- Train boxes per class, v2 → no-whole-frame: dog 337 → 331, goat 1109 → 981, cow 1808 → 1669, cat 467 → 386, sheep 1066 → 751.
- 17 whole-frame boxes remain in the variant (15 from openimages_yolo, 2 from openimages_goat_sheep). The rule covers Mendeley only, so they are kept.
- `validate_dataset.py` on `data/dataset`: PASSED.
