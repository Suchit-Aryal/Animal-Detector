# Results notes (run 2026-11-15, RTX 4050, yolov8n, 50 epochs, imgsz 640, batch 8, seed 0)
Caveats:
- Test set = 80 held-out Open Images images (118 boxes), NOT Nepal street data.
- Sheep: 19 train boxes, 0 val/test -> sheep NOT evaluated (NaN). Goat test = 22 boxes only; per-class AP is noisy.
- The OIV7 baseline was pretrained on Open Images V7, so the test images may have been in its training data (possible leakage, inflates its score).
- The COCO baseline has no goat class (NaN); its means cover dog/cow/cat only; the others cover the classes with test boxes.
- Training: 50 epochs = 368 s (6.1 min). Validation mAP50 0.596 (val set, 160 imgs), test mAP50 0.747.
