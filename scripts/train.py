#!/usr/bin/env python3
"""Fine-tune YOLOv8n on VisDrone persons (small-object aerial settings)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/yolo/visdrone_person.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["train"]
    dcfg = cfg["detector"]
    weights = dcfg["base_weights"]

    model = YOLO(weights)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=tcfg["epochs"],
        imgsz=dcfg["imgsz"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        close_mosaic=tcfg["close_mosaic"],
        optimizer=tcfg["optimizer"],
        lr0=tcfg["lr0"],
        device=tcfg["device"],
        project=str(ROOT / "outputs/models"),
        name="yolov8n_visdrone_person",
        exist_ok=True,
        resume=args.resume,
        # Drone-oriented augmentation emphasis
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        single_cls=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "outputs/models/best.pt"
    if best.exists():
        import shutil

        shutil.copy2(best, dest)
        print(f"Copied best weights to {dest}")
    print("Training complete:", results.save_dir)


if __name__ == "__main__":
    main()
