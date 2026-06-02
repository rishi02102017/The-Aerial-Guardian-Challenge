#!/usr/bin/env python3
"""Benchmark end-to-end pipeline FPS on a VisDrone sequence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_guardian.visdrone import find_sequence_dirs


def gpu_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        )
        info["gpus"] = [line.strip() for line in out.strip().split("\n")]
    except Exception as e:
        info["gpu_error"] = str(e)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/models/best.pt")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "VisDrone2019-MOT-val")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights = args.weights if args.weights.exists() else Path(cfg["detector"]["base_weights"])
    seqs = find_sequence_dirs(args.raw_dir) or find_sequence_dirs(ROOT / "data/raw")
    seq_dir = seqs[0]
    img_dir = seq_dir / "img" if (seq_dir / "img").is_dir() else seq_dir
    frames = sorted(img_dir.glob("*.jpg"))[: cfg["benchmark"]["measure_frames"] + cfg["benchmark"]["warmup_frames"]]

    model = YOLO(str(weights))
    det = cfg["detector"]
    trk = cfg["tracker"]
    tiled_cfg = det.get("tiled_inference", {})
    infer_sz = det.get("inference_imgsz") or det["imgsz"]

    if args.tiled:
        from aerial_guardian.tiled_detect import tiled_predict

    track_kw = dict(
        persist=True,
        tracker=str(ROOT / "configs/bytetrack_aerial.yaml"),
        conf=det["conf"],
        iou=det["iou"],
        imgsz=infer_sz,
        max_det=det["max_det"],
        verbose=False,
        classes=[0],
        device=0,
        half=True,
    )

    times = []
    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        t0 = time.perf_counter()
        if args.tiled and tiled_cfg.get("enabled"):
            tiled_predict(
                model,
                frame,
                tile_size=tiled_cfg["tile_size"],
                overlap=tiled_cfg["overlap"],
                conf=det["conf"],
                iou=det["iou"],
                max_det=det["max_det"],
            )
        else:
            model.track(frame, **track_kw)
        dt = time.perf_counter() - t0
        if i >= cfg["benchmark"]["warmup_frames"]:
            times.append(dt)

    avg = float(np.mean(times)) if times else 0
    fps = 1.0 / avg if avg > 0 else 0
    report = {
        "hardware": gpu_info(),
        "weights": str(weights),
        "sequence": seq_dir.name,
        "frames_measured": len(times),
        "imgsz": infer_sz,
        "train_imgsz": det["imgsz"],
        "tiled": args.tiled,
        "avg_latency_ms": avg * 1000,
        "fps": fps,
    }
    out = ROOT / "outputs/benchmarks/fps_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
