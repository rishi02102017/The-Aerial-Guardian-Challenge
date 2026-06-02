#!/usr/bin/env python3
"""Discover VisDrone MOT sequences and build YOLO person-only dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_guardian.visdrone import (
    build_yolo_dataset,
    find_sequence_dirs,
    load_config,
    write_data_yaml,
)


def assign_splits(raw_root: Path) -> dict[str, list[Path]]:
    """Assign sequences to train/val based on directory layout."""
    project_root = Path(__file__).resolve().parents[1]
    mot_val = project_root / "VisDrone2019-MOT-val"

    train_seqs = find_sequence_dirs(raw_root / "mot_train") if (raw_root / "mot_train").exists() else []
    val_seqs = find_sequence_dirs(raw_root / "mot_val") if (raw_root / "mot_val").exists() else []

    all_seqs: list[Path] = []
    if mot_val.exists() and not val_seqs and not train_seqs:
        all_seqs = find_sequence_dirs(mot_val)
    elif not train_seqs and not val_seqs:
        all_seqs = find_sequence_dirs(raw_root)
        if not all_seqs and mot_val.exists():
            all_seqs = find_sequence_dirs(mot_val)
    elif not train_seqs and val_seqs:
        all_seqs = val_seqs
        val_seqs = []

    # Only val MOT release: hold out 2 sequences for metrics, rest for fine-tune
    if all_seqs and not train_seqs:
        all_seqs = sorted(all_seqs, key=lambda p: p.name)
        if len(all_seqs) >= 3:
            split = max(1, len(all_seqs) - 2)
            train_seqs = all_seqs[:split]
            val_seqs = all_seqs[split:]
        else:
            train_seqs = all_seqs
            val_seqs = all_seqs[-1:]

    if not val_seqs and train_seqs:
        val_seqs = [train_seqs[-1]]
        train_seqs = train_seqs[:-1]

    return {"train": train_seqs, "val": val_seqs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--yolo-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_root = (args.raw_dir or ROOT / cfg["data"]["raw_dir"]).resolve()
    yolo_root = (args.yolo_dir or ROOT / cfg["data"]["yolo_dir"]).resolve()

    print(f"Raw root: {raw_root}")
    splits = assign_splits(raw_root)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} sequences — {[p.name for p in v[:5]]}{'...' if len(v) > 5 else ''}")

    person_ids = cfg["classes"]["visdrone_person_ids"]
    stats = build_yolo_dataset(raw_root, yolo_root, splits, person_ids)
    yaml_path = write_data_yaml(yolo_root, cfg)
    print("YOLO dataset stats:", stats)
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
