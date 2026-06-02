"""SAHI-style tiled inference for small aerial persons."""

from __future__ import annotations

import numpy as np


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes. Returns indices to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def tiled_predict(model, frame: np.ndarray, tile_size: int, overlap: float, conf: float, iou: float, max_det: int):
    """
    Run detector on overlapping tiles and merge with NMS.
    Returns ultralytics-style Results list (single synthetic result).
    """
    from ultralytics.engine.results import Boxes, Results

    h, w = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))
    all_boxes = []
    all_scores = []
    all_cls = []

    for y0 in range(0, max(h - tile_size, 0) + 1, max(stride, 1)):
        for x0 in range(0, max(w - tile_size, 0) + 1, max(stride, 1)):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            preds = model.predict(crop, conf=conf, iou=iou, verbose=False, max_det=max_det)
            if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
                continue
            boxes = preds[0].boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            scores = preds[0].boxes.conf.cpu().numpy()
            cls = preds[0].boxes.cls.cpu().numpy()
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls)

    if not all_boxes:
        return model.predict(frame, conf=conf, iou=iou, verbose=False, max_det=max_det)

    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    cls = np.concatenate(all_cls)
    keep = _nms_boxes(boxes, scores, iou)
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    import torch

    res = Results(orig_img=frame, path="", names=model.names)
    xywh = np.zeros((len(boxes), 4))
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    data = torch.cat(
        [
            torch.tensor(xywh, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32).unsqueeze(1),
            torch.tensor(cls, dtype=torch.float32).unsqueeze(1),
        ],
        dim=1,
    )
    res.boxes = Boxes(data, orig_shape=frame.shape[:2])
    return [res]
