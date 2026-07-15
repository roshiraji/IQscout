"""Court ROI polygon — filters out crowd/sideline detections."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default ROI margins as fractions of frame size.
# Inset ~12% from left/right and ~8% from top, ~5% from bottom.
_DEFAULT_MARGINS = dict(left=0.12, right=0.12, top=0.08, bottom=0.05)


def default_roi_polygon(width: int, height: int) -> np.ndarray:
    left = int(width * _DEFAULT_MARGINS["left"])
    right = int(width * (1 - _DEFAULT_MARGINS["right"]))
    top = int(height * _DEFAULT_MARGINS["top"])
    bottom = int(height * (1 - _DEFAULT_MARGINS["bottom"]))
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.int32)


def load_or_default_roi(roi_path: Path, width: int, height: int) -> np.ndarray:
    if roi_path.is_file():
        payload = json.loads(roi_path.read_text(encoding="utf-8"))
        polygon = payload.get("polygon")
        if polygon and len(polygon) >= 3:
            return np.array(polygon, dtype=np.int32)
    logger.info("No court_roi.json found — using default court margins.")
    return default_roi_polygon(width, height)


def point_in_roi(x: float, y: float, roi: np.ndarray) -> bool:
    return cv2.pointPolygonTest(roi.astype(np.float32), (float(x), float(y)), False) >= 0


def box_center_in_roi(bbox_xyxy: list[int] | tuple, roi: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return point_in_roi(cx, cy, roi)


def filter_detections_by_roi(
    detections: list[dict],
    roi: np.ndarray,
    *,
    strict_classes: tuple[str, ...] = ("player_red", "player_black_blue", "referee"),
    lenient_classes: tuple[str, ...] = ("ball",),
) -> list[dict]:
    """Keep detections whose centers are inside the court ROI.

    Ball is kept even if barely outside (small object, allow 20px margin).
    """
    result: list[dict] = []
    for det in detections:
        name = det.get("class_name", "")
        box = det.get("bbox_xyxy", [])
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if name in lenient_classes:
            # Expand ROI test by 20 pixels for small objects.
            expanded = roi.copy()
            in_roi = cv2.pointPolygonTest(roi.astype(np.float32), (cx, cy), True) >= -20.0
        else:
            in_roi = point_in_roi(cx, cy, roi)

        if in_roi:
            result.append(det)
    return result


def apply_nms(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """Per-class NMS to remove duplicate boxes."""
    if not detections:
        return []
    from collections import defaultdict

    by_class: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_class[det.get("class_name", "unknown")].append(det)

    kept: list[dict] = []
    for class_name, class_dets in by_class.items():
        boxes = np.array([d["bbox_xyxy"][:4] for d in class_dets], dtype=np.float32)
        scores = np.array([d.get("confidence", 0.5) for d in class_dets], dtype=np.float32)
        # Convert to [x1,y1,w,h] for NMS not needed — use xyxy directly.
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes[:, :4].tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        if indices is not None and len(indices) > 0:
            for idx in np.array(indices).flatten():
                kept.append(class_dets[int(idx)])
    return kept
