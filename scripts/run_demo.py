#!/usr/bin/env python3
"""Render demo video(s) with detection, tracking, ID labels, and trajectory tails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_guardian.pipeline import load_config, run_sequence
from aerial_guardian.visdrone import find_sequence_dirs
from ultralytics import YOLO


def resolve_weights(path: Path | None, config: dict) -> Path:
    candidates = [
        path,
        ROOT / "outputs/models/best.pt",
        ROOT / "outputs/models/yolov8n_visdrone_person/weights/best.pt",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(config["detector"]["base_weights"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--seq", type=str, default=None, help="Sequence folder name (e.g. uav0000086_00000_v)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "VisDrone2019-MOT-val/sequences",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/videos/demo.mp4")
    parser.add_argument("--all-val", action="store_true", help="Process all val sequences")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference size")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames (debug)")
    args = parser.parse_args()

    config = load_config(args.config)
    weights = resolve_weights(args.weights, config)
    tracker_cfg = ROOT / "configs/bytetrack_aerial.yaml"

    print(f"Loading weights: {weights}")
    model = YOLO(str(weights))

    # Point Ultralytics to our ByteTrack config
    import shutil

    ul_track = Path(model.trainer.tracker) if hasattr(model, "trainer") and model.trainer else None
    custom_dst = ROOT / "configs/bytetrack_custom.yaml"
    shutil.copy2(tracker_cfg, custom_dst)

    seqs = find_sequence_dirs(args.raw_dir)
    if not seqs:
        for fallback in (
            ROOT / "VisDrone2019-MOT-val",
            ROOT / "data/raw/mot_val",
            ROOT / "data/raw",
        ):
            seqs = find_sequence_dirs(fallback)
            if seqs:
                break
    if args.seq:
        seqs = [s for s in seqs if args.seq in s.name]
    if not seqs:
        raise SystemExit(f"No sequences under {args.raw_dir}")

    if not args.all_val:
        seqs = [seqs[0]]

    stats = []
    for i, seq_dir in enumerate(seqs):
        out = args.output if len(seqs) == 1 else args.output.parent / f"{seq_dir.name}.mp4"
        print(f"Processing {seq_dir.name} -> {out}")
        # Patch tracker path via env / ultralytics settings
        from ultralytics.cfg import get_cfg

        stat = run_sequence(
            model,
            seq_dir,
            out,
            config,
            use_track=True,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
        )
        stat["sequence"] = seq_dir.name
        stats.append(stat)

    summary_path = ROOT / "outputs/videos/demo_stats.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Stats saved to {summary_path}")


if __name__ == "__main__":
    main()
