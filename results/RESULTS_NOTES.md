# Results notes (run 2026-11-15, RTX 4050, yolov8n, 50 epochs, imgsz 640, batch 8, seed 0)
Caveats:
- Test set = 80 held-out Open Images images (118 boxes), NOT Nepal street data.
- Sheep: 19 train boxes, 0 val/test -> sheep NOT evaluated (NaN). Goat test = 22 boxes only; per-class AP is noisy.
- The OIV7 baseline was pretrained on Open Images V7, so the test images may have been in its training data (possible leakage, inflates its score).
- The COCO baseline has no goat class (NaN); its means cover dog/cow/cat only; the others cover the classes with test boxes.
- Training: 50 epochs = 368 s (6.1 min). Validation mAP50 0.596 (val set, 160 imgs), test mAP50 0.747.

## Experiments (results/experiments.csv; one change per run; chosen by VAL mAP50, not test)
| run | val mAP50 | test mAP50 | test mAP50-95 | train cls loss | val cls loss | time |
|---|---|---|---|---|---|---|
| yolov8n 50ep | 0.603 | 0.747 | 0.509 | 0.52 | 1.53 | 368 s |
| yolov8s 80ep | 0.663 | 0.716 | 0.512 | 0.36 | 1.50 | 1094 s |
| yolov8m 80ep | 0.635 | 0.671 | 0.487 | 0.39 | 1.43 | 2512 s |
| yolov8s 80ep freeze=10 | 0.792 | 0.854 | 0.577 | 0.30 | 1.09 | 750 s |
Overfitting: train-vs-val class loss gap is large for all unfrozen runs (0.4-0.5 vs 1.4-1.5); freezing the backbone narrows it and gives the best val and test. Val is much lower than test for unfrozen runs (val has 160 imgs, 324 boxes; test 80 imgs, 118 boxes), so differences of a few points are noisy.
Chosen (best val mAP50): yolov8s freeze=10 -> copied to results/weights/best.pt.
Caveat: the OIV7 baseline (test mAP50 0.906) was pretrained on Open Images and the test set is Open Images, so its score is likely inflated; the interim comparison favours it.
Sheep not evaluated (no val/test boxes).

## Dataset v2 retrain (2026-11-16): yolov8s freeze=10, 80 ep max, patience 20, batch 8, seed 0 (results/v2/, experiments.csv)
v2 data: train 2993 / val 634 / test 80 (frozen) / test_extra 80 (116 sheep, 108 goat boxes). Variant A = data.yaml; variant B = data_nowf.yaml (train without whole-frame Mendeley boxes, 2332 images).
| run | epochs (best) | val mAP50 (best epoch) | train/val cls loss | time |
|---|---|---|---|---|
| v2A (all) | 75 (55) | 0.773 | 0.38 / 1.02 | 1754 s |
| v2B (no whole-frame) | 73 (60) | 0.767 | 0.43 / 1.05 | 1388 s |
Chosen by val: A (difference 0.006 = a tie; B has higher precision, A higher recall). Val set was re-evaluated with evaluate.py: A 0.772, B 0.762; old v1 model 0.357 on the new val (it never saw the Mendeley-style data, so this is not a fair like-for-like).
Frozen test (80 imgs): v2A mAP50 0.842 / mAP50-95 0.589 / P 0.803 / R 0.763 vs v1 0.854 / 0.577 / 0.851 / 0.808 (v1 slightly higher mAP50, v2 higher mAP50-95; within noise).
test_extra (80 imgs; dog 4, goat 108, cow 70, sheep 116 boxes): mAP50 v2A 0.354, v1 0.218, COCO 0.258, OIV7 0.415. Sheep AP50: v2A 0.711, v1 0.268, COCO 0.717, OIV7 0.767. Goat AP50: v2A 0.674, v1 0.561, OIV7 0.726. Dog (4 boxes) and cow AP on test_extra are not meaningful (cow 0.033 for v2A).
Same caveats: Open Images test data (OIV7 baseline likely inflated), not Nepal street data.
