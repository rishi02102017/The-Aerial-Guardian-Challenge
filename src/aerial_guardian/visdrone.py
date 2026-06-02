"""VisDrone MOT annotation parsing and YOLO conversion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import yaml

# VisDrone MOT line format (10 fields):
# frame, id, bb_left, bb_top, bb_width, bb_height, score, category, truncation, occlusion
PERSON_CATEGORIES = {1, 2}  # pedestrian, person


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_mot_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) < 8:
        return None
    frame = int(parts[0])
    category = int(parts[7])
    score = int(parts[6])
    if score == 0:
        return None
    return {
        "frame": frame,
        "target_id": int(parts[1]),
        "x": float(parts[2]),
        "y": float(parts[3]),
        "w": float(parts[4]),
        "h": float(parts[5]),
        "category": category,
    }


def bbox_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def get_mot_dataset_root(root: Path) -> Path:
    """Resolve VisDrone MOT root (supports .../VisDrone2019-MOT-val or data/raw/mot_val)."""
    root = root.resolve()
    if (root / "sequences").is_dir():
        return root
    for name in ("VisDrone2019-MOT-val", "VisDrone2019-MOT-train", "mot_val", "mot_train"):
        candidate = root / name
        if (candidate / "sequences").is_dir():
            return candidate
    return root


def find_sequence_dirs(root: Path) -> list[Path]:
    """Find sequence directories with frames (and optional MOT annotations)."""
    root = get_mot_dataset_root(root)
    sequences: list[Path] = []

    seq_root = root / "sequences"
    if seq_root.is_dir():
        for seq_dir in sorted(seq_root.iterdir()):
            if seq_dir.is_dir() and (
                any(seq_dir.glob("*.jpg"))
                or any((seq_dir / "img").glob("*.jpg"))
            ):
                sequences.append(seq_dir)

    for ann in root.rglob("*.txt"):
        if ann.name in ("gt.txt", "annotations.txt") or ann.parent.name == "annotations":
            seq_dir = ann.parent.parent if ann.parent.name == "annotations" else ann.parent
            if seq_dir.name == "sequences":
                continue
            if any(seq_dir.glob("*.jpg")) or any((seq_dir / "img").glob("*.jpg")):
                sequences.append(seq_dir)
                continue
        parent = ann.parent
        if ann.name == "gt.txt" and parent.name == "gt":
            sequences.append(parent.parent)
        elif list(parent.glob("*.jpg")):
            sequences.append(parent)

    for gt in root.rglob("gt/gt.txt"):
        sequences.append(gt.parent.parent)

    seen = set()
    unique = []
    for s in sequences:
        key = str(s.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return sorted(unique, key=lambda p: p.name)


def get_image_dir(seq_dir: Path) -> Path:
    if (seq_dir / "img").is_dir():
        return seq_dir / "img"
    return seq_dir


def get_annotation_file(seq_dir: Path) -> Path | None:
    """MOT labels: per-seq gt.txt or VisDrone2019-MOT-val/annotations/<seq>.txt."""
    candidates = [
        seq_dir / "gt" / "gt.txt",
        seq_dir / "annotations.txt",
        seq_dir / "gt.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    mot_root = get_mot_dataset_root(seq_dir.parent.parent if seq_dir.parent.name == "sequences" else seq_dir.parent)
    ext_ann = mot_root / "annotations" / f"{seq_dir.name}.txt"
    if ext_ann.is_file():
        return ext_ann
    if seq_dir.parent.parent.name != "sequences":
        ext_ann = seq_dir.parent.parent / "annotations" / f"{seq_dir.name}.txt"
        if ext_ann.is_file():
            return ext_ann
    return None


def convert_sequence_to_yolo(
    seq_dir: Path,
    out_images: Path,
    out_labels: Path,
    person_categories: Iterable[int] = PERSON_CATEGORIES,
    split_prefix: str = "",
) -> int:
    """Convert one MOT sequence to YOLO images + per-frame labels. Returns image count."""
    import cv2

    ann_file = get_annotation_file(seq_dir)
    img_dir = get_image_dir(seq_dir)
    if ann_file is None or not img_dir.is_dir():
        return 0

    person_categories = set(person_categories)
    frame_boxes: dict[int, list[str]] = {}

    for line in ann_file.read_text().splitlines():
        rec = parse_mot_line(line)
        if rec is None or rec["category"] not in person_categories:
            continue
        frame_boxes.setdefault(rec["frame"], []).append(rec)

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        images = sorted(img_dir.glob("*.png"))
    count = 0
    seq_name = seq_dir.name

    for img_path in images:
        # VisDrone: img00000001.jpg -> frame 1
        stem = img_path.stem
        try:
            frame_idx = int("".join(c for c in stem if c.isdigit()) or "0")
        except ValueError:
            frame_idx = count + 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        out_name = f"{split_prefix}{seq_name}_{img_path.name}"
        shutil.copy2(img_path, out_images / out_name)

        label_lines = []
        for rec in frame_boxes.get(frame_idx, []):
            cx, cy, nw, nh = bbox_to_yolo(rec["x"], rec["y"], rec["w"], rec["h"], w, h)
            if nw < 0.001 or nh < 0.001:
                continue
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_labels / out_name.replace(img_path.suffix, ".txt")).write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )
        count += 1

    return count


def build_yolo_dataset(
    raw_root: Path,
    yolo_root: Path,
    splits: dict[str, list[Path]],
    person_categories: Iterable[int] = PERSON_CATEGORIES,
) -> dict:
    """Build YOLO train/val directory structure from sequence paths per split."""
    stats = {}
    for split, seq_dirs in splits.items():
        img_out = yolo_root / split / "images"
        lbl_out = yolo_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total = 0
        for seq_dir in seq_dirs:
            total += convert_sequence_to_yolo(seq_dir, img_out, lbl_out, person_categories)
        stats[split] = {"images": total, "sequences": len(seq_dirs)}
    return stats


def write_data_yaml(yolo_root: Path, config: dict) -> Path:
    data_yaml = yolo_root / "visdrone_person.yaml"
    content = {
        "path": str(yolo_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
        "nc": 1,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    return data_yaml
