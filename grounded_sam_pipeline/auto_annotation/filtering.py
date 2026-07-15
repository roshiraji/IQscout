"""Detection filtering: clip, area, NMS."""

from __future__ import annotations

import cv2
import numpy as np

from auto_annotation.utils import polygon_area


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area_fraction(box: list[float], width: int, height: int) -> float:
    return box_area(box) / max(width * height, 1)


def is_valid_box(box: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    return box_area(box) > 0


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
    iou_threshold: float,
) -> list[int]:
    """Return indices to keep after class-aware NMS."""
    keep: list[int] = []
    by_class: dict[int, list[int]] = {}
    for i, cid in enumerate(class_ids):
        by_class.setdefault(cid, []).append(i)

    for indices in by_class.values():
        order = sorted(indices, key=lambda i: scores[i], reverse=True)
        while order:
            best = order.pop(0)
            keep.append(best)
            order = [i for i in order if iou(boxes[best], boxes[i]) < iou_threshold]
    keep.sort()
    return keep


def filter_detections(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
    class_names: list[str],
    width: int,
    height: int,
    *,
    min_area_px: int,
    min_area_frac: float,
    max_area_frac: float,
    box_threshold: float,
    nms_iou: float,
) -> tuple[list[list[float]], list[float], list[int], list[str]]:
    filtered_boxes: list[list[float]] = []
    filtered_scores: list[float] = []
    filtered_class_ids: list[int] = []
    filtered_names: list[str] = []

    for box, score, cid, cname in zip(boxes, scores, class_ids, class_names):
        if score < box_threshold:
            continue
        if cid is None or cname is None:
            continue
        clipped = clip_box(box, width, height)
        if not is_valid_box(clipped, width, height):
            continue
        area = box_area(clipped)
        if area < min_area_px:
            continue
        frac = box_area_fraction(clipped, width, height)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        filtered_boxes.append(clipped)
        filtered_scores.append(score)
        filtered_class_ids.append(int(cid))
        filtered_names.append(cname)

    if not filtered_boxes:
        return [], [], [], []

    keep = nms_per_class(filtered_boxes, filtered_scores, filtered_class_ids, nms_iou)
    return (
        [filtered_boxes[i] for i in keep],
        [filtered_scores[i] for i in keep],
        [filtered_class_ids[i] for i in keep],
        [filtered_names[i] for i in keep],
    )
