"""Live ball detector (dedicated YOLO weights) with HSV fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.ball.proposals import merge_proposals, propose_hsv_balls

logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}


def _get_yolo(weights: str | Path):
    key = str(weights)
    if key not in _model_cache:
        from ultralytics import YOLO

        logger.info("Loading ball detector: %s", key)
        _model_cache[key] = YOLO(key)
    return _model_cache[key]


def detect_balls(
    image_bgr: np.ndarray,
    *,
    weights: str | Path | None = None,
    conf: float = 0.25,
    device: str | int = "cpu",
    use_hsv_fallback: bool = True,
) -> list[dict]:
    """Detect basketball boxes. Prefer trained weights; HSV soft-fallback."""
    dets: list[dict] = []
    if weights is not None and Path(weights).is_file():
        try:
            model = _get_yolo(weights)
            results = model.predict(
                source=image_bgr,
                conf=conf,
                device=device,
                verbose=False,
                half=str(device).lower() not in {"cpu", ""},
            )
            for r in results or []:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = [float(v) for v in box.xyxy[0].tolist()[:4]]
                    c = float(box.conf[0]) if box.conf is not None else 0.0
                    dets.append(
                        {
                            "bbox_xyxy": xyxy,
                            "confidence": round(c, 4),
                            "class_id": 0,
                            "class_name": "ball",
                            "source": "ball_yolo",
                        }
                    )
        except Exception as exc:
            logger.warning("Ball YOLO failed (%s); falling back to HSV", exc)
            dets = []

    if not dets and use_hsv_fallback:
        for p in propose_hsv_balls(image_bgr):
            dets.append(
                {
                    "bbox_xyxy": p["bbox_xyxy"],
                    "confidence": p["confidence"],
                    "class_id": 0,
                    "class_name": "ball",
                    "source": "hsv_fallback",
                }
            )

    dets = merge_proposals(dets)
    return dets[:2]
