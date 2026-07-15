"""Run the custom NBA player detector (best_player_detector.pt)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}

CANONICAL_CLASSES: dict[int, str] = {
    0: "player",
    1: "referee",
    2: "hoop",
    3: "ball",
}


def _get_model(weights: str):
    if weights not in _model_cache:
        from ultralytics import YOLO
        logger.info("Loading trained player detector: %s", weights)
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def detect_with_trained_model(
    image_bgr: np.ndarray,
    *,
    frame_id: int,
    timestamp: float,
    image_name: str,
    weights: str | Path,
    conf: float = 0.25,
) -> list[dict]:
    """Detect player/referee/hoop/ball using the NBA-trained YOLO weights."""
    model = _get_model(str(weights))
    h, w = image_bgr.shape[:2]
    results = model(image_bgr, conf=conf, verbose=False)
    if not results:
        return []

    res = results[0]
    detections: list[dict] = []
    if res.boxes is None:
        return detections

    names = res.names or CANONICAL_CLASSES
    for box in res.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        class_name = names.get(cls_id, CANONICAL_CLASSES.get(cls_id, f"class_{cls_id}"))
        detections.append({
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "image": image_name,
            "width": w,
            "height": h,
            "class_id": cls_id,
            "class_name": class_name,
            "canonical_class": class_name,
            "prompt": f"trained detector {class_name}",
            "confidence": round(score, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "source": "trained_detector",
        })
    return detections
