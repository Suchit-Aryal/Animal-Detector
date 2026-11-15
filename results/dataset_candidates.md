# Dataset candidates (2026-11-15). Disk: 58 GB free on /home (above the 20 GB warning level).
"Verified" = I read the API/page or opened the files; "search only" = seen in a search result, not opened.

## Downloaded (no login)
| Name | URL | Classes / boxes | Size | Licence | Login? | How |
|---|---|---|---|---|---|---|
| Mendeley "Annotated Object Detection Dataset of Five Herbivorous Animal Classes" (verified) | https://data.mendeley.com/datasets/fwm6z9tv3f | Buffalo, Camel, Cow, Goat, Sheep; real boxes (Roboflow export, ids 0 Buffalo,1 Camel,2 Cow,3 Goat,4 Sheep); one animal class per image | 10,756 imgs (train 9,411 = 3 augmented copies of 3,137 sources; valid 897; test 448); 560 MB zip | CC BY 4.0 | no | public file URL from Mendeley public API -> `data/raw/mendeley_herbivores5/` |
| Open Images V7 train subset, Goat+Sheep images (verified) | https://storage.googleapis.com/openimages/web/index.html (via FiftyOne zoo, seed 7) | Goat, Sheep (+Dog, Cattle, Cat where present); human boxes | 524 new imgs, 184 MB | annotations CC BY 4.0; images CC BY 2.0 (per-image licence in OI metadata) | no | FiftyOne downloader + own conversion -> `data/raw/openimages_goat_sheep/` |

## Not downloaded
| Name | URL | What | Size | Licence | Login? | Why skipped / note |
|---|---|---|---|---|---|---|
| IndiaScene365 (verified files) | https://data.mendeley.com/datasets/pwffhg6nhz | Indian traffic scenes, ~3k imgs, 34 classes claimed | 1.7 GB | CC BY 4.0 | no | In the 40 XML files I read, class names are "-" or Roboflow boilerplate (names stripped), so classes unusable without the authors |
| Mendeley street animals (have it) | DOI 10.17632/fk29shm2kn.1 | already in data/raw/mendeley (no boxes) | | CC BY 4.0 | no | already used |
| DriveIndia (search only) | https://arxiv.org/abs/2507.19912 | 66,986 Indian road imgs, 24 classes, YOLO | large | not checked | not checked | cattle/dog classes unconfirmed; hosted by TiHAN-IIT Hyderabad repository |
| IDD Detection (search only) | https://idd.insaan.iiit.ac.in | 46k Indian road imgs incl. animal class | large | check terms | registration usually required | not species-specific; skip |
| Zenodo livestock YOLOv8 labels (search only) | https://zenodo.org/records/15600650 | pasture cows (grazing/lying), YOLOv8 | ? | not seen | no | cows only, pasture not roads |
| HF keremberke/aerial-sheep-object-detection (verified listing) | https://huggingface.co/datasets/keremberke/aerial-sheep-object-detection | aerial sheep boxes (Roboflow origin) | small zips | not stated on card | no | aerial view, wrong domain for street cameras |
| Goat Image Dataset (search only) | https://datasetninja.com/goat-image-dataset | goat face/ear/eye/mouth boxes | ? | CC BY 4.0 | no | parts not whole body |
| Cherry Chevre (search only) | https://datasetninja.com/cherry-chevre | 6,159 goat imgs | ? | Etalab 2.0 | no | licence differs; maybe later |
| Cows2021 (search only) | https://datasetninja.com/cows2021 | cattle torso | ? | NCGL v2.0 | no | not whole animal |
| Roboflow Universe sets in PROJECT_PLAN 4b (goat/cow/stray dog/cat, CowsOnRoad, Indian roads) | universe.roboflow.com | many | | mostly CC BY 4.0 | yes (login/API key to download) | blocked: needs human login; this is where "animals on road" and more goat data live |
| COCO 2017 sheep/cow/dog/cat (known) | https://cocodataset.org | boxes, no goat | 18 GB full | images CC BY 2.0 | no | would need subset download; not done; COCO-pretrained model has seen it |
| LILA camera traps / iNaturalist | lila.science, inaturalist.org | wild animals, mostly not our classes | huge | varies | no | not relevant to street livestock |
| Kaggle sets | kaggle.com | various | | varies | yes | login required |

## Counts actually obtained
**openimages_goat_sheep** (524 images; labels already in PROJECT ids 0 dog,1 goat,2 cow,3 cat,4 sheep): dog 11, goat 546, cow 267, cat 0, sheep 859 boxes. 119 boxes are "group of" boxes (one box around many animals). 76 candidate images skipped because they are already in data/raw/openimages_yolo.
**mendeley_herbivores5** (its own ids, see NOTE_FOR_MERGE.txt): boxes in all splits: Buffalo 1,926, Camel 3,550, Cow 2,605, Goat 2,105, Sheep 2,348. Train with one copy per source image: Buffalo 555, Camel 1,051, Cow 756, Goat 614, Sheep 682; valid 897 imgs, test 448 imgs (single copies; per-class counts valid: Buffalo 179, Camel 236, Cow 219, Goat 176, Sheep 203; test: 82/161/118/87/99).
Caveats: Mendeley5 images show one animal per image in mostly natural/farm settings (not streets), train has 3x augmented copies (dedupe before merge), boxes not eyeballed by us; Open Images images may overlap what the OIV7 pretrained model saw. Neither set contains road/street-specific scenes; "animals on road" data needs Roboflow login or IDD registration.
