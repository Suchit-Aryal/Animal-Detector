# Animal Detection on Streets: Project Plan

> Put this file in the project root. When you reopen the project with Claude in the terminal, say:
> "Read PROJECT_PLAN.md and give me a summary of where we left off."
> Keep the **Progress Log** at the bottom up to date. It is the project's memory.

---

## 1. What we are building (focus)

A small computer vision system that **detects animals on streets** (stray dogs, cows, goats, etc.) in images and video, using a fine-tuned YOLO object detector.

**Core story for the presentation:**
> A pretrained YOLO model misses or confuses street animals. We fine-tuned it on a street-animal dataset and improved accuracy from **X to Y mAP**, running at **Z FPS**.

**Scope (keep it small):**
- **Context: Nepal.** Four main classes, chosen because they are what is actually seen on Nepali streets (cows and goats are common on roads in the Terai region): **dog, goat, cow, cat**. **Sheep** is kept as an optional 5th class (decided 15 Nov 2026, mainly so the model learns to tell goats from sheep; drop it with `--no-sheep` if not wanted). Don't add more classes than that; more means more labeling and more bugs.
- Class ids (keep fixed everywhere): `0: dog, 1: goat, 2: cow, 3: cat, 4: sheep`
- One task: object detection (boxes + class + confidence). No segmentation, no lane detection.
- Offline evaluation on a held-out test set, plus a short video demo.

**Out of scope (only if time remains):** tracking animals across frames, alerts, deployment on a Raspberry Pi/phone.

---

## 2. Final aim (what "done" means)

- [ ] Baseline: pretrained YOLO evaluated on our test set (mAP, precision, recall, FPS)
- [ ] Fine-tuned model evaluated on the same test set
- [ ] Comparison table baseline vs fine-tuned
- [ ] 3 to 4 failure-case images (night, occlusion, groups, blur) with explanation
- [ ] Short demo video or live webcam/image demo
- [ ] 5-slide presentation with real numbers
- [ ] Q&A prep done (section 9)

---

## 3. Tools

| Purpose | Tool |
|---|---|
| Language | Python 3.11 (conda env `animal-det`) |
| Detection model + training | Ultralytics YOLO (`pip install ultralytics`) |
| Deep learning backend | PyTorch (installed with ultralytics; match CUDA version if using GPU) |
| Image/video I/O | OpenCV (`opencv-python`) |
| Labeling and dataset hosting | Roboflow (free tier) or Label Studio |
| Training without a GPU | Google Colab (free GPU) |
| Plots and results tables | matplotlib, pandas |
| Dataset inspection (optional) | FiftyOne |
| Version control | Git (worktrees if running several agents) |
| Orchestrator (GOD) | Claude Code (see section 10) |
| Worker agents | OpenCode (open source, many model providers, some free models) |
| Agent harness | Munder Difflin (open source desktop app that runs the agents together) |

**Model choice:** start with the smallest Ultralytics detection model (e.g. `yolov8n.pt`; newer versions such as YOLO26 exist, so check https://docs.ultralytics.com for current names). Small model = fast training, runs on a laptop, good enough for a small project. Move to the `s` size only if accuracy is too low.

---

## 4. Repos and resources to use

**Main (the one we build on):**
- Ultralytics: https://github.com/ultralytics/ultralytics (training, validation, prediction, export; use as an installed package, don't copy its code)

**Datasets to start from:**
- Roboflow cattle detection: https://universe.roboflow.com/ash-pkvgo/cattle-detection-rv8bv
- Search Roboflow Universe for: "stray dog", "dog detection street", "goat detection", "animal road". Always open a few images and check that the boxes are correct before using a set.
- Our own street photos (labeled in Roboflow). Even 100 to 200 of our own images makes the project original.

**Reference repos (read for ideas, don't depend on them):**
- https://github.com/maheshpaulj/lane_detection : real-time YOLOv8 + OpenCV video pipeline with distance estimation (good reference for video loop code)
- https://github.com/Khaliladib11/MSc_Thesis : compares detection models on speed vs accuracy (good reference for how to structure a comparison)
- https://github.com/Nawaf-Rayhan585/YOLO_Projects : ready-to-run YOLO projects including ByteTrack (only if we add tracking)
- Street Scene Vision Toolkit demo: https://mingze-streetscenesegmentation.hf.space/ (YOLO + segmentation demo for presentation inspiration)

**Papers for the methods/Q&A slide:**
- Evaluating YOLO architectures for detecting road-killed endangered Brazilian animals (Scientific Reports, 2024): compares YOLO versions, augmentation, transfer learning. Dataset: BRA-Dataset, 1,458 images, 5 species.
- Cow crossing detection alert system (IJTech): YOLO with a custom cow dataset, reports very high mAP@0.5 on their own data. Don't expect the same number on ours.

**Nepal-specific notes (important):**
- Most public datasets come from other countries, so **our own Nepali street photos are the most valuable data**. Aim for 150 to 300 photos per class, taken in different places, times of day and distances. Phone photos or short video clips (extract every 10th to 15th frame) both work.
- **Goat is the weak spot for the COCO baseline.** The pretrained COCO model has dog, cat and cow, but no goat class (the closest is sheep). So that baseline will likely miss or mislabel goats. A second baseline, the Open Images V7 model (`yolov8n-oiv7.pt`), does include Goat and Cattle, so evaluate both: it keeps the comparison fair and gives you a stronger "before vs after" story.
- Expect class imbalance: cats are usually fewer and smaller in street scenes than dogs and cows. Check per-class counts early and collect more cat images if needed.
- Expect hard cases: herds of goats (overlapping boxes), cows partly hidden behind vehicles, dogs lying down, dusty/low-light evening scenes.
- Photograph people's property and faces carefully: avoid clear faces and licence plates in images you plan to show on slides.

## 4b. Dataset catalog (researched 15 Nov 2026; verify each before use)

**Status key:** Seen = I read the listing page. Unchecked = only seen as a name in search results; open it and check classes, label quality and licence before using.

### Per-class sources
| Class | Source | What the listing shows | Licence | Notes |
|---|---|---|---|---|
| dog | Roboflow "Stray Dogs" (fyp-gztsq/stray-dogs-2wfjc) | 554 images (YOLO formats available) | CC BY 4.0 | 1 class; version has augmentation, so re-export without augmentation |
| dog | Roboflow "stray dogs" (cv-nwjrg/stray-dogs-hfuft) | 395 images, no augmentation | check page | |
| dog | Roboflow "stray dogs" (bothina/stray-dogs-ikjii) | 464 images (v1), 266 (v2) | check page | |
| dog | Roboflow "stray-dogs" (lr3/stray-dogs-dsns7) | model page says 3,469 images (likely includes augmentation); has 3 classes, so check what they mean | CC BY 4.0 | check class names before merging |
| dog | Roboflow user "Street Dog Detection" (universe.roboflow.com/street-dog-detection) | one project listed at 2.8k images | check | Unchecked |
| goat | Roboflow "goat" (jush/goat-unuwu) | object detection | CC BY 4.0 | Unchecked size |
| goat | Roboflow "Goat" (deer-tvf5a/goat-brt8c) | object detection | CC BY 4.0 | Unchecked size |
| goat | Roboflow "Goats" (amna-yps68/goats-ln02f) | object detection | CC BY 4.0 | Unchecked size |
| goat | Roboflow "goat" (au-90a9a/goat-u87lc) | 1,692 images | check | |
| goat | Roboflow "Goat-Detection" (reshma-silp9/goat-detection-m0vb6-ge3d2) | 5,097 images (probably includes augmentation) | CC BY 4.0 | check real image count |
| goat | Roboflow "Goat Dataset" (z-firewood-house/goat-5eecg) | 500 images, 400/50/50 split | check | smaller but clean split |
| cow | Roboflow "CowsOnRoad" (object-detection-naauz/cowsonroad) | on-road cows; image count not confirmed | check | closest to our use case. Unchecked |
| cow | Roboflow "cattle_detection" (loliktry/cattle_detection-060yo) | 1,457 images | CC BY 4.0 | |
| cow | Roboflow "cattle915" (lin-vc6m8/cattle915) | cow and vehicle classes | CC BY 4.0 | Unchecked size |
| cow | Roboflow "CATTLE_AHEAD" (harsh-rsnmo/cattle_ahead) | cattle on roads, small | check | Unchecked |
| cow | Roboflow "cattle detection" (ash-pkvgo/cattle-detection-rv8bv) | listed earlier | check | |
| cat | Roboflow "stray cat" (stray-cats/stray-cat-ph) | version 8: 1,909 images (72/14/14 split) | check | includes augmented versions; use the no-augmentation export |
| cat | Roboflow "Detection of stray cats" (universe.roboflow.com/detection-of-stray-cats) | 1,589 images | check | Unchecked |
| cat | Roboflow "stray cat" (masterdrawup/stray-cat) | 2 classes: dog, straycat | CC BY 4.0 | small, also has dogs |

### Multi-animal / road-scene sources
| Source | What it gives | Licence | Notes |
|---|---|---|---|
| Mendeley Data: "Labelled dataset of animal images for detecting their presence on streets" (DOI 10.17632/fk29shm2kn.1, 2024) | street animals, focused on night-vision images; categories include Cat, Dog, Goat, Sheep, Deer, Rabbit | CC BY 4.0 | best match for our topic and gives night data. Image count and label format NOT confirmed: download and inspect |
| Open Images V7 (via FiftyOne subset download) | has Dog, Cat, Cattle, Goat as box-labeled classes | check (annotations and images carry separate licences) | do NOT download the full set (about 561 GB); pull only these 4 classes, e.g. 300 to 500 images per class |
| COCO | dog, cat, cow boxes (no goat; sheep is the closest) | check | already inside the pretrained COCO model, so use mainly for baseline, not as new data |
| Roboflow "Indian Road Dataset" family (e.g. car-bike-pothole-dog-cattle, 6,532 images; cow-VECHILES, 7,119 images) | Indian road scenes with dog and cattle labels | check each | Unchecked: class lists and quality unknown. Good for road context if labels are clean |
| India Driving Dataset (IDD) | about 46k images of Indian roads with an animal class | listed as "unknown" on one aggregator; check the official terms | large and complex; only use if time allows |
| Kaggle/other "Dataset on Road and Traffic Obstacles" (Mendeley, DOI 10.17632/jr3yv3wfhx) | 38,295 images, has an "Animal on road" class (not split by species) | check | not species-specific, so low priority |

### Pretrained baselines (to compare against)
- **COCO model** (e.g. `yolov8n.pt`): knows dog, cat, cow; no goat.
- **Open Images V7 model** (`yolov8n-oiv7.pt`): trained on a 601-class set that includes Cat, Cattle, Dog and Goat. This gives a fairer goat baseline. Evaluate **both** baselines and report both.

### Target data mix and rules
1. **Test set = our own Nepali photos only** (about 40 to 60 images per class, from scenes not used for training). This is the honest number for "does it work in Nepal".
2. Training set = public data (about 300 to 500 images per class after cleaning) + the rest of our own photos (about 100 to 150 per class). Validation set = a mix.
3. Use the **no-augmentation** export of each Roboflow set, then let the training script do augmentation. Augmented copies inflate counts and can leak near-duplicates.
4. Dedupe across sources (the same web images appear in many Roboflow projects). Use a perceptual-hash check (e.g. `imagehash`) in the dataset validator.
5. **Remap every source's labels** to our ids: `0 dog, 1 goat, 2 cow, 3 cat, 4 sheep`. Delete other classes (sheep, deer, vehicle) from the label files, or the model will learn them as background. Keep a `data/label_map.md` noting what each source's class became.
6. Open 30 random images per source and check the boxes by eye before merging. Drop any source with bad labels.
7. Keep a `data/SOURCES.md` listing every dataset (name, URL, licence, number of images used). CC BY 4.0 requires attribution, and the examiners may ask where data came from.
8. **Decision needed: buffalo and oxen.** Nepal has many buffalo and oxen. Simplest rule: label all bovines (cow, ox, buffalo) as `cow` and say so on the slide. Write the rule down and apply it to our own labels consistently.

### Open Images subset (FiftyOne), verify the syntax in the FiftyOne docs
```python
import fiftyone as fo
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Dog", "Cat", "Cattle", "Goat"],
    max_samples=1500,
    dataset_name="oiv7-animals",
)
# then export to YOLO format and remap: Dog->0, Goat->1, Cattle->2, Cat->3
```

### Quick data-check commands for the workers
- Count images per class and per split (validator script, task A2).
- List any label file whose class id is not in 0..4.
- Print 20 random images with boxes drawn, and look at them yourself.

### 4c. Data status and scripts (updated 15 Nov 2026)

**Project layout in use:** `~/comp_vision/` (project root). Scripts currently live in `data/`. **Always run scripts from the project root** (e.g. `python data/check_labels.py ...`), otherwise outputs land in the wrong folder.

**Environment:** conda env `animal-det`, Python 3.11, packages installed with pip; `requirements.txt` exists in the project root.

**Done so far**
- Open Images V7 subset (Dog, Goat, Cattle, Cat) built WITHOUT FiftyOne (FiftyOne cannot start its MongoDB on this Linux machine). Script: `data/dataset_multi.py` (the pandas-based version). Result: 798 images in `data/raw/openimages_yolo/{images,labels}`; boxes per class: dog 269, goat 403, cow(Cattle) 783, cat 236. Groups-of boxes and drawings were excluded. Label spot-check: `python data/check_labels.py --root data/raw/openimages_yolo --n 30` (annotated pictures go to `results/label_check/`).
- Mendeley "street animals" set downloaded and unzipped to `data/raw/mendeley/extracted/DATASETT/`. **It has NO bounding boxes**: 18,248 jpg files sorted into folders CAT, COW, DEER, DOG, GOAT, HEN, NightVision, RABBIT, SHEEP. Decision: ignore DEER, RABBIT (and HEN); keep SHEEP as class 4; keep copies/duplicates (they only make validation look slightly better; the Nepal test set is what counts).
- Script written for it: `data/mendeley_autolabel.py` (folder name = class; a pretrained Open Images YOLO supplies the box). Output goes to `data/raw/mendeley_yolo/`. **Boxes are model-generated: use for TRAINING ONLY, never for testing.** Must be spot-checked with `check_labels.py`. (Status: script given to the user; run result still to be reported.)

**Still to do (data)**
- Run `mendeley_autolabel.py --n 300`, check labels by eye.
- Download 2 to 3 Roboflow sets per class (no-augmentation, YOLOv8 export) and remap their class names to our ids.
- Collect own Nepali photos/videos (test set, about 40 to 60 per class) and label them by hand.
- Merge, dedupe across sources, split (public data to train/val; own photos to test only).

---

## 5. Safe workflow (so the project doesn't break)

1. **Virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install ultralytics opencv-python pandas matplotlib
   pip freeze > requirements.txt    # pin versions once it works
   ```
2. **Never edit third-party repos in place.** Clone references into `third_party/` and copy only the few lines you need into our own `src/` files.
3. **Git from day one.** Commit after every working step. Use a branch (`git checkout -b experiment-x`) for risky changes.
4. **Test on a tiny subset first** (10 images, 1 epoch) before any long training run.
5. **Back up `runs/` and weights** (`best.pt`) outside the project folder after every good run.
6. **Save the baseline results immediately** (a CSV), so we never lose the "before" number.
7. **Change one thing per experiment** (model size, epochs, augmentation) and log it in the Progress Log.

**Suggested folder structure**
```
project/
├── PROJECT_PLAN.md
├── requirements.txt
├── data/
│   ├── raw/                # original downloaded/collected images
│   └── dataset/            # YOLO format: images/{train,val,test}, labels/{train,val,test}
│       └── data.yaml
├── src/
│   ├── baseline_eval.py    # pretrained model on test set
│   ├── train.py            # fine-tuning
│   ├── evaluate.py         # mAP / precision / recall / FPS
│   └── demo.py             # image/video/webcam inference
├── results/                # CSVs, plots, failure-case images
├── runs/                   # ultralytics output (weights, curves)
└── third_party/            # cloned reference repos (read-only)
```

---

## 6. Step-by-step plan

1. **Setup:** venv, install, run `yolo predict` on one image to confirm everything works.
2. **Data:** download Roboflow sets, add our own photos, label them, export in **YOLOv8 format**.
3. **Split:** about 70% train / 20% val / 10% test. Split by scene or video so near-identical frames don't land in both train and test (data leakage).
4. **Baseline:** evaluate the pretrained model on the test set. Save mAP@0.5, mAP@0.5:0.95, precision, recall, FPS.
5. **Train:** fine-tune from pretrained weights (transfer learning), about 50 epochs, image size 640, augmentation on.
6. **Evaluate:** same test set, same metrics.
7. **Analyze:** confusion matrix, PR curve, failure cases.
8. **Demo:** run on a short street video and save the annotated output.
9. **Presentation + Q&A prep.**

Example commands (verify current syntax in the Ultralytics docs):
```bash
yolo detect val   model=yolov8n.pt          data=data/dataset/data.yaml split=test
yolo detect train model=yolov8n.pt          data=data/dataset/data.yaml epochs=50 imgsz=640
yolo detect val   model=runs/detect/train/weights/best.pt data=data/dataset/data.yaml split=test
yolo detect predict model=runs/detect/train/weights/best.pt source=video.mp4 save=True
```

---

## 7. Bugs we might hit and fixes

| Problem | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch or image size too large | Lower `batch` (e.g. 8 or 4), lower `imgsz` (e.g. 512), use a smaller model |
| Training is extremely slow | No GPU / running on CPU | Use Google Colab GPU; reduce epochs for tests |
| `torch.cuda.is_available()` is False | CUDA/torch version mismatch | Reinstall torch with the right CUDA build from pytorch.org, or use Colab |
| "No labels found" / 0 images | Wrong paths in `data.yaml`, or labels folder not parallel to images | Use absolute paths in `data.yaml`; check `images/` and `labels/` have the same structure and filenames |
| Many "corrupt label" warnings | Wrong label format (not YOLO txt), class ids outside range | Re-export from Roboflow as **YOLOv8** format; check class ids are 0..N-1 |
| mAP is 0 or very low | Class names/ids mismatch between sets, empty labels | Print `names:` in `data.yaml`; remap class ids when merging datasets |
| Suspiciously high mAP (~99%) | Data leakage (similar frames in train and test), tiny test set | Re-split by scene/video; enlarge the test set; add our own unseen photos |
| Overfitting (train good, val bad) | Too little data / too many epochs | More data, stronger augmentation, early stopping (`patience`), smaller model |
| Rare class never detected | Class imbalance | Add more images for that class, or drop it |
| False positives (bags, statues as dogs) | Not enough background examples | Add "negative" images with no animals (empty label files) |
| Misses at night / in blur | Training data is mostly daytime | Add brightness/blur augmentation; include some night images; report it as a limitation |
| `cv2.imshow` crashes or window missing | Headless OpenCV / no display (Colab, servers) | Save output to a file instead (`save=True`), or use `opencv-python` (not `-headless`) locally |
| Video won't open or output is empty | Missing codec, wrong path | Convert to mp4 with ffmpeg; test with `cv2.VideoCapture(path).isOpened()` |
| Windows path errors | Backslashes, spaces in paths | Use forward slashes or raw strings; avoid spaces in folder names |
| Code from a reference repo fails | Old library versions | Don't install into our env; copy the logic only, or run the repo in a separate venv |
| "It worked yesterday" breakage | Auto-upgraded packages | Use `requirements.txt` with pinned versions; Git history to roll back |

**Fallback plan if fine-tuning underperforms:**
1. Check data quality first (wrong boxes cause most problems).
2. Use more data or reduce the number of classes.
3. Try the `s` model size or train longer.
4. Freeze early layers (`freeze=10`) when the dataset is very small.
5. Worst case: present the baseline plus our analysis of why it fails, with a smaller fine-tuned result. A modest honest improvement is better than an unexplained one.

---

## 8. Numbers to collect for the presentation

| Metric | Baseline (pretrained) | Fine-tuned |
|---|---|---|
| mAP@0.5 | | |
| mAP@0.5:0.95 | | |
| Precision | | |
| Recall | | |
| FPS (hardware: ____) | | |
| Model size / parameters | | |

Also record: number of images (train/val/test), classes and per-class counts, image size, epochs, training time, augmentation used.

**Slide outline (about 5 minutes):**
1. Problem + goal (30s)
2. Pipeline diagram: image → YOLO → boxes/classes (60s)
3. Why this approach: transfer learning with a small YOLO, speed vs accuracy (60s)
4. Dataset + training setup (45s)
5. Results table + one chart (90s)
6. Failure cases + limits (30s)

---

## 9. Q&A prep

- **Why YOLO?** Single-stage, real-time, strong pretrained weights, simple training API. Alternatives: Faster R-CNN (more accurate on small objects but slower), SSD, RT-DETR.
- **What is mAP?** Mean of average precision across classes. mAP@0.5 counts a detection correct if IoU ≥ 0.5; mAP@0.5:0.95 averages over stricter thresholds.
- **What is IoU?** Overlap area divided by union area of predicted and true boxes.
- **Precision vs recall?** Precision = correct detections / all detections. Recall = correct detections / all real animals.
- **What is transfer learning and why use it?** Start from weights trained on COCO and adapt to our classes, which needs far less data and time.
- **How did you avoid data leakage?** Split by scene/video, never by random frame.
- **What does NMS do?** Removes duplicate overlapping boxes, keeping the highest-confidence one.
- **What augmentation did you use and why?** Flip, brightness/HSV shifts, mosaic, to improve robustness to lighting and position.
- **What are the limitations?** Small dataset, night/occlusion, similar-looking animals, a limited number of classes.
- **Is it real-time?** Report FPS and the hardware.
- **How would you improve it?** More diverse data, tracking across frames, a larger model, deployment on an edge device.
- **Which part is yours vs borrowed?** Be specific: Ultralytics for the model/training; we did the dataset, split, evaluation, analysis, demo.
- **License?** Ultralytics is AGPL-3.0; datasets have their own licenses (cite them on the slide).

---

## 10. Multi-agent workflow (Munder Difflin + OpenCode)

**Idea:** Claude Code is the **GOD** (orchestrator). OpenCode instances are the **workers**. Munder Difflin is the harness that runs them side by side. This is an experiment: the project must still be finishable without it (see "Fallback").

### 10.1 What these tools are
- **Munder Difflin**: free, open-source (MIT) desktop app that spawns real terminal coding agents (Claude Code, Codex, OpenCode and others) and adds a shared "hive": per-agent memory, mailboxes, a shared blackboard, an event log, and a GOD agent you talk to who routes the work. It describes itself as a working prototype. Features useful for us: optional per-agent **git worktrees** (so agents don't collide on files) and per-agent **token budgets**.
- **OpenCode**: free, open-source (MIT) terminal coding agent that works with 75+ model providers (OpenRouter, Ollama local models, OpenAI, Anthropic and others). It has a Build agent (edits files, runs commands) and a read-only Plan-style agent, and some free models are available.
- Download Munder Difflin only from its official site (munderdiffl.in) or the release page linked from there. Several unrelated projects share the name (for example a Udacity course project about a paper company), so check the link before installing.

### 10.2 Honest expectations
- Most of this project is **sequential and human-driven** (taking photos, labeling, checking labels, GPU training on Colab). Agents can write and fix scripts, but they cannot photograph streets or judge whether a label box is right.
- Multi-agent setups add coordination overhead. For a small project, **2 workers is plenty**. Add a third only if the first two are idle.
- Agents can be confidently wrong. Every result number in the presentation must come from a command **you** ran and a CSV you looked at.
- Free models are weaker than top paid ones: expect to review and sometimes redo their code. Use the stronger model for the GOD.
- Hosted models send prompts and code to the provider. Do **not** let agents upload your photos or dataset anywhere.

### 10.3 Roles
| Role | Tool | Owns (only edits these) | Does NOT do |
|---|---|---|---|
| GOD | Claude Code | `PROJECT_PLAN.md`, task cards, reviews, merges | Bulk coding, long training |
| Worker A: data | OpenCode | `src/data/`, `data/dataset/data.yaml` | Touch `data/raw/` originals, edit training code |
| Worker B: train/eval | OpenCode | `src/train.py`, `src/baseline_eval.py`, `src/evaluate.py`, `results/` | Change the dataset or split |
| Worker C (optional): demo/report | OpenCode | `src/demo.py`, `src/make_plots.py`, `results/figures/` | Change metrics code |
| You | Human | Photos, labeling, Colab runs, final numbers, presentation | n/a |

### 10.4 Task cards (GOD hands these out; each has a clear "done" test)
**A1: Frame extractor**
- Input: videos in `data/raw/videos/`. Output: every Nth frame as JPG into `data/raw/frames/<video_name>/`, N configurable (default 12).
- Done when: running it on one test video prints the number of frames saved and the folder contains them; no frame is overwritten.

**A2: Dataset validator**
- Checks a YOLO-format dataset: every image has a label file (empty file allowed for background images), class ids are only 0 to 4, box values are between 0 and 1, no corrupt/unreadable images.
- Prints per-class box counts and per-split image counts. Exits with an error if anything is wrong.
- Done when: it passes on a clean tiny dataset and fails correctly on a deliberately broken one.

**A3: Split by video/scene**
- Splits into train/val/test (70/20/10) **by video or scene folder, never by individual frame**, copies images and labels into `data/dataset/{images,labels}/{train,val,test}`, writes `data.yaml` with `names: {0: dog, 1: goat, 2: cow, 3: cat, 4: sheep}`.
- Done when: a check script confirms no source video appears in more than one split.

**B1: Baseline evaluation**
- Runs the pretrained model on the test split and saves `results/baseline.csv` with mAP@0.5, mAP@0.5:0.95, precision, recall, FPS, plus per-class AP.
- Run **two** baselines: the COCO model (no goat class, so map its classes onto ours and record goat AP as "not supported") and the Open Images V7 model (has Goat and Cattle; map Dog, Goat, Cattle, Cat onto ours). Document each mapping in a note next to the CSV.
- Done when: the CSV exists and the numbers match a manual `yolo val` run.

**B2: Training script**
- Fine-tunes from pretrained weights using config values (epochs, imgsz, batch, model size) from one config file; sets a fixed random seed; saves the run under `runs/` and copies `best.pt` to `results/weights/`.
- Done when: a 1-epoch run on 10 images completes without errors.

**B3: Final evaluation**
- Same metrics and same test set as B1, saved to `results/finetuned.csv`, and a combined `results/comparison.csv` (baseline vs fine-tuned).

**C1: Demo script**
- Runs the model on an image, a video file, or a webcam; draws boxes with class name and confidence; saves annotated output to disk (no dependency on a display window).
- Done when: it produces an annotated mp4 from a test video.

**C2: Plots and failure cases**
- Makes the comparison bar chart, the confusion matrix copy, and saves the 10 lowest-confidence or wrong detections for manual review.

### 10.5 Rules for every worker (paste into AGENTS.md and into each worker's first prompt)
1. Only edit files you own (section 10.3). Ask the GOD before touching anything else.
2. Never delete or modify anything in `data/raw/`.
3. Do not install or upgrade packages without asking; use the project's `.venv` and `requirements.txt`.
4. Do not upload images, labels or code to any external service.
5. Work on your own git worktree/branch; make small commits with clear messages.
6. Test on a tiny subset (10 images, 1 epoch) before any long run.
7. When finished, write a short report: what changed, the exact command to run it, the output you saw, and anything you are unsure about.
8. If something fails twice, stop and report instead of guessing.

### 10.6 GOD review checklist (before merging any worker's branch)
- Does it do exactly what the task card says, and nothing extra?
- Did you (the human) run the command and see the real output?
- No hardcoded absolute paths; class ids match `0 dog, 1 goat, 2 cow, 3 cat, 4 sheep`.
- Numbers saved to CSV, not copied from terminal text.
- `requirements.txt` still installs cleanly in a fresh venv.
- Merge to `main` only after this passes; tag good states (`git tag v0.1-baseline`).

### 10.7 Setup order (try this the first time)
1. Make sure the project works **without** agents first (section 6, steps 1 to 2) so you know what "working" looks like.
2. Install OpenCode alone; run `opencode models` and pick a model; do one tiny task (for example "write `extract_frames.py`") to learn how it behaves.
3. Install Munder Difflin from the official site; add one GOD (Claude Code) and one OpenCode worker; turn on **git isolation** (per-agent worktrees) and set a **token budget** per agent.
4. Give the GOD this plan and the task cards; start with worker A only. Add worker B once A1 to A3 are merged.
5. Keep the progress log (section 11) updated after every merge.

### 10.8 Fallback (if the harness gets in the way)
- Run Claude Code in one terminal and OpenCode in another, in separate git worktrees, and copy the task cards by hand. Same workflow, no Munder Difflin.
- If a free OpenCode model keeps producing broken code, switch models (`/models` inside OpenCode) or have Claude Code write that script directly.
- If everything agent-related misbehaves, follow section 6 manually. The scripts are small enough to write by hand.
- Never run two agents on the same files at the same time without worktrees.

### 10.9 Models and settings (decided 15 Nov 2026)
- **GOD:** Claude Code on the Claude Pro account, Sonnet (current Sonnet 5.5). Log in once with `claude` in a terminal using the Pro account before starting Munder Difflin (it drives the existing `claude` command). Set a token budget on the GOD; avoid Opus on a Pro plan, it uses the limit quickly.
- **If the GOD is stuck on OpenCode** (it was set up that way earlier): first look for the per-agent model/provider selector (Command Center, Floor tab, the GOD's card) and Settings, AI Engines. If it cannot be switched, keep the OpenCode GOD with the best free model, and add one Claude Code agent as a **reviewer** that checks each worker's diff before merging. Do not reinstall or delete app data without checking first.
- **Worker A (data) and Worker B (train/eval):** both OpenCode. Open `/models` inside OpenCode and pick from the ones marked `-free` (free lists change). Give Worker B the strongest free coding model available, Worker A a fast one. Free models have IP-level rate limits, so run one worker at a time at first.
- **Approvals/autonomy:** set every worker to ask before deleting files, running `pip install`, running `git push`, or touching `data/raw/`.
- **Git before agents:** run `git init` in the project root, add the `.gitignore` below, make a first commit, and only then turn on Git isolation (worktrees) in Munder Difflin.

`.gitignore`:
```
.venv/
__pycache__/
data/raw/
data/dataset/
runs/
*.pt
results/label_check/
.ipynb_checkpoints/
```

### 10.10 Kickoff prompts (paste these)

**GOD (once, at start):**
> You are the GOD (orchestrator) for a small computer vision course project. Read PROJECT_PLAN.md fully, especially sections 1, 4b, 4c, 10 and 11. First, summarise the project back to me in 10 lines so I can confirm you understood. Do not write bulk code yourself. Your job: (1) turn the task cards in 10.4, plus the updates in 10.10, into a short ordered task list; (2) give tasks to the workers one at a time; (3) review every worker report and diff against the checklist in 10.6 before merging; (4) after each merge, update section 11 of PROJECT_PLAN.md including "Where we left off". Never run long training (I train on Colab). Never edit or delete anything in data/raw. Ask me before installing packages, deleting files, or changing class ids (0 dog, 1 goat, 2 cow, 3 cat, 4 sheep). Do not claim a result unless you saw the command output.

**Worker A (data):**
> You are Worker A (data) on a small computer vision project. Read PROJECT_PLAN.md sections 1, 4b, 4c, 10.3, 10.4 and 10.5 and follow the rules in 10.5. You may only edit files in src/data/ and data/dataset/data.yaml; the project root is the working directory and scripts are run from it. Do the tasks in this order and report after each: (A2) validate_dataset.py; (A3) roboflow_remap.py and merge_and_split.py as described in the task updates below. Test on a tiny subset first. When you finish a task, report: files changed, the exact command to run, the output you saw, and anything you are unsure about. If something fails twice, stop and report.

**Worker B (train/eval):**
> You are Worker B (train/eval) on a small computer vision project. Read PROJECT_PLAN.md sections 1, 3, 6, 8, 10.3, 10.4 and 10.5 and follow the rules in 10.5. You may only edit src/train.py, src/baseline_eval.py, src/evaluate.py and files in results/. Use Ultralytics YOLO from the installed package; do not copy its code. Do the tasks B1, B2, B3 in order, as described in the task updates below. Do NOT run full training: test only with 10 images and 1 epoch on CPU; I run real training on Colab. Report after each task: files changed, exact command, output you saw, doubts.

**Task updates since 10.4 (give these to the workers):**
- **A2 validator:** classes are 0 to 4 (dog, goat, cow, cat, sheep). Also print per-source image counts when given several folders.
- **A3 roboflow_remap.py:** takes a Roboflow YOLOv8 export folder and a mapping `{their class name: our id or "drop"}`; writes a new folder in our class ids; boxes of dropped classes are removed; images left with no boxes are kept only if they are meant as background examples.
- **A3 merge_and_split.py:** reads `data/sources.yaml` (list of folders with images/ and labels/); merges them; splits **public** data 85% train / 15% val (by source folder where possible); puts images from `data/raw/own_photos/` into the **test** split only (skip with a warning if that folder does not exist yet); writes `data/dataset/data.yaml`. Never put model-auto-labeled data (`mendeley_yolo`) into val or test.
- **B1 baseline_eval.py:** two baselines. COCO model (`yolov8n.pt`: dog, cat, cow only; goat and sheep reported as "not supported" or via sheep mapping, stated clearly) and Open Images V7 model (`yolov8n-oiv7.pt`: Dog, Cat, Cattle, Goat, Sheep). Map their class names onto our ids with a visible mapping table.
- **B2 train.py:** reads one config file; fine-tunes from pretrained weights; fixed seed; model size configurable.
- **B3 evaluate.py:** mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP, FPS (state the hardware) into `results/*.csv`.

---

## 11. Progress Log (update after every session)

**Current status:** data collection in progress. Environment ready; Open Images subset built and spot-checked; Mendeley set unzipped (no boxes) and an auto-label script written; Roboflow sets and own Nepal photos still to collect.

**Decisions made:**
- Project: animal detection on streets (friend's idea; chosen over street/lane detection because it is simpler)
- Model: small Ultralytics YOLO, fine-tuned from pretrained weights
- Workflow: Munder Difflin harness; Claude Code = GOD, OpenCode instances = workers (section 10); fallback is two terminals by hand
- Classes (final): dog, goat, cow, cat (Nepal streets; cows and goats common in Terai) + sheep as optional 5th class (id 4)
- Data plan: public Roboflow sets + our own Nepali street photos (own photos are the priority)

**Done:**
- [x] Environment set up (conda env animal-det, Python 3.11, requirements.txt)
- [ ] OpenCode tried alone on one small task
- [ ] Munder Difflin installed (official site) with GOD + 1 worker, git isolation + token budget on
- [ ] Dataset collected and labeled
- [ ] Train/val/test split
- [ ] Baseline evaluated
- [ ] Model fine-tuned
- [ ] Final evaluation
- [ ] Demo video
- [ ] Slides
- [ ] Q&A rehearsed

**Where we left off:** Open Images subset done (798 images). Mendeley autolabel script ready to run (`python data/mendeley_autolabel.py --n 300`). Next: run it, spot-check labels, download Roboflow sets, collect own photos; then set up Munder Difflin (GOD on Claude Pro, 2 OpenCode workers) per section 10.9.

**Open problems / bugs:** _(none yet)_

**Experiment log:**
| Date | Change | Result (mAP / FPS) | Notes |
|---|---|---|---|
| | | | |
