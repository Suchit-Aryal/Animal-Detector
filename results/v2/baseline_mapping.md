# Baseline class mapping
Predictions of each pretrained model are mapped to our classes; all other predicted classes are dropped.
Metrics: official Ultralytics val via src/evaluate.py (conf 0.001, NMS IoU 0.7, imgsz 640, batch 1) on a temp copy of the
test set whose labels are remapped to the model's class ids. Classes with no test ground truth give NaN AP and are
excluded from the means.
Ground-truth boxes of classes a model cannot predict (goat for COCO) are left out of its metrics, so its goat AP is NaN
("not supported") instead of a 0 that would drag the mean down.

| Our class | COCO (yolov8n.pt) | Open Images V7 (yolov8m-oiv7.pt) |
|---|---|---|
| 0 dog | 16 dog | 160 Dog |
| 1 goat | NOT SUPPORTED (no goat class; AP = NaN) | 219 Goat |
| 2 cow | 19 cow | 99 Cattle |
| 3 cat | 15 cat | 96 Cat |
| 4 sheep | 18 sheep | 452 Sheep |
