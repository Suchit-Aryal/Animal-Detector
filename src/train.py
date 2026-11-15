"""B2: config-driven fine-tune. Fixed seed; run saved under runs/; best.pt copied to results/weights/.

Usage: python src/train.py [--config results/train_config.yaml] [--epochs 1 --fraction ...]  (CLI overrides config)
Config keys (all optional, defaults below): model, data, epochs, imgsz, batch, seed, device, patience, name, fraction.
Full training is meant for Colab; locally only run the tiny smoke test.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"model": "yolov8n.pt", "data": "data/dataset/data.yaml", "epochs": 50, "imgsz": 640, "batch": 16,
            "seed": 0, "device": None, "patience": 20, "name": "animal_ft", "project": "runs/detect",
            "workers": 2, "fraction": 1.0, "freeze": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "results/train_config.yaml"))
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if Path(a.config).exists():
        cfg.update(yaml.safe_load(Path(a.config).read_text()) or {})
    cfg.update({k: v for k, v in vars(a).items() if k != "config" and v is not None})
    cfg["project"] = str((ROOT / cfg["project"]).resolve()) if not Path(cfg["project"]).is_absolute() else cfg["project"]
    print("config:", cfg)
    model = YOLO(cfg["model"])
    model.train(data=cfg["data"], epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                seed=int(cfg["seed"]), deterministic=True, device=cfg["device"], patience=int(cfg["patience"]),
                project=cfg["project"], name=cfg["name"], workers=int(cfg["workers"]),
                fraction=float(cfg["fraction"]), freeze=int(cfg["freeze"]) or None, exist_ok=False)
    best = Path(model.trainer.best)
    dst = ROOT / "results/weights" / f"{cfg['name']}.pt"  # one file per run; copy the chosen one to best.pt by hand
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    (ROOT / "results/weights" / f"{cfg['name']}_config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"best.pt copied: {best} -> {dst}")


if __name__ == "__main__":
    main()
