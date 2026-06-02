"""End-to-end detection + ByteTrack + trajectory visualization."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from aerial_guardian.tiled_detect import tiled_predict


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TrackTrailVisualizer:
    """Draw boxes, IDs, and motion trails per track."""

    def __init__(self, trail_length: int = 25, palette: int = 12):
        self.trail_length = trail_length
        self.histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))
        self.palette = palette

    def _color(self, track_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(track_id % 9973)
        return tuple(int(c) for c in rng.integers(64, 255, size=3))

    def update(self, frame: np.ndarray, boxes, track_ids, confs) -> np.ndarray:
        out = frame.copy()
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.asarray(boxes)
        for i, tid in enumerate(track_ids):
            if tid is None or int(tid) < 0:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = map(int, xyxy[i])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.histories[tid].append((cx, cy))
            color = self._color(tid)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            conf = float(confs[i]) if confs is not None else 0.0
            label = f"ID:{tid} {conf:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            pts = list(self.histories[tid])
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, pts[j - 1], pts[j], color, thickness)
        return out


def run_sequence(
    model: YOLO,
    seq_dir: Path,
    out_video: Path,
    config: dict,
    use_track: bool = True,
    imgsz: int | None = None,
    max_frames: int | None = None,
) -> dict:
    """Process one VisDrone sequence folder; write annotated video; return stats."""
    img_dir = seq_dir / "img" if (seq_dir / "img").is_dir() else seq_dir
    frames = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {seq_dir}")
    if max_frames:
        frames = frames[:max_frames]

    det_cfg = config["detector"]
    infer_sz = imgsz or det_cfg.get("inference_imgsz") or det_cfg["imgsz"]
    trk_cfg = config["tracker"]
    vis_cfg = config["visualization"]
    tiled = det_cfg.get("tiled_inference", {})
    visualizer = TrackTrailVisualizer(trail_length=vis_cfg.get("trail_length", 25))

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), fourcc, trk_cfg.get("frame_rate", 30), (w, h))

    times = []
    tracker_yaml = str(Path(__file__).resolve().parents[2] / "configs/bytetrack_aerial.yaml")
    track_kw = dict(
        persist=True,
        tracker=tracker_yaml,
        conf=det_cfg["conf"],
        iou=det_cfg["iou"],
        imgsz=infer_sz,
        max_det=det_cfg["max_det"],
        verbose=False,
        classes=[0],
        device=0,
        half=True,
    )

    total = len(frames)
    for i, frame_path in enumerate(frames):
        if i % 50 == 0:
            print(f"  frame {i}/{total}", flush=True)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        t0 = time.perf_counter()

        if use_track:
            # Native ByteTrack + GMC for drone ego-motion (tiled off in demo for stable IDs/FPS)
            results = model.track(frame, **track_kw)
        elif tiled.get("enabled", False):
            results = tiled_predict(
                model,
                frame,
                tile_size=tiled.get("tile_size", 640),
                overlap=tiled.get("overlap", 0.25),
                conf=det_cfg["conf"],
                iou=det_cfg["iou"],
                max_det=det_cfg["max_det"],
            )
        else:
            results = model.predict(
                frame,
                conf=det_cfg["conf"],
                iou=det_cfg["iou"],
                imgsz=det_cfg["imgsz"],
                max_det=det_cfg["max_det"],
                verbose=False,
            )

        times.append(time.perf_counter() - t0)
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes):
            tids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(boxes)
            confs = boxes.conf.cpu().numpy()
        else:
            tids, confs = [], []

        vis = visualizer.update(frame, boxes, tids, confs)
        writer.write(vis)

    writer.release()
    avg_ms = np.mean(times) * 1000 if times else 0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    return {
        "frames": len(times),
        "avg_ms": avg_ms,
        "fps": fps,
        "imgsz": infer_sz,
        "output": str(out_video),
    }
