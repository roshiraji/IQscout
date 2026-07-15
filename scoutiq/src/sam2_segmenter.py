"""SAM2 segmentation via Ultralytics SAM API.

Uses bounding boxes from YOLO11-seg (or LocateAnything) as prompts.
Falls back gracefully if SAM2 is unavailable — the pipeline continues
with just bounding boxes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SAM_MODEL = "sam2.1_b.pt"
_sam_cache: dict = {}
_sam_device: str | None = None

_sam2_available: bool | None = None


def sam2_available() -> bool:
    global _sam2_available
    if _sam2_available is None:
        try:
            from ultralytics import SAM  # noqa: F401
            _sam2_available = True
        except Exception:
            _sam2_available = False
    return _sam2_available


def set_sam_device(device: str | None) -> None:
    """Prefer a specific CUDA device for SAM2 (e.g. cuda:1)."""
    global _sam_device, _sam_cache
    if device != _sam_device:
        _sam_cache = {}
    _sam_device = device


def _resolve_sam_device() -> str | int:
    import torch
    if _sam_device is not None:
        return _sam_device
    if torch.cuda.is_available():
        # If two GPUs exist, default SAM2 to the second card.
        return 1 if torch.cuda.device_count() >= 2 else 0
    return "cpu"


def _get_sam(weights: str = _SAM_MODEL):
    key = f"{weights}|{_resolve_sam_device()}"
    if key not in _sam_cache:
        from ultralytics import SAM
        device = _resolve_sam_device()
        logger.info("Loading SAM2 model: %s (device=%s)", weights, device)
        model = SAM(weights)
        try:
            model.to(device)
        except Exception:
            pass
        _sam_cache[key] = model
    return _sam_cache[key]


def run_sam2_on_boxes(
    image_bgr: np.ndarray,
    boxes: list[list[int]],
    weights: str = _SAM_MODEL,
) -> list[list[list[float]] | None]:
    """Return a list of mask polygons (or None) for each input box.

    Each polygon is a list of [x, y] float points.
    """
    if not boxes:
        return []
    if not sam2_available():
        logger.warning("SAM2 unavailable — using None masks.")
        return [None] * len(boxes)

    try:
        model = _get_sam(weights)
        device = _resolve_sam_device()
        # Run all boxes at once in one forward pass.
        results = model(image_bgr, bboxes=boxes, verbose=False, device=device)
        polygons: list[list[list[float]] | None] = []

        if results and results[0].masks is not None:
            for pts in results[0].masks.xy:
                if pts is not None and len(pts) > 2:
                    polygons.append([[float(p[0]), float(p[1])] for p in pts])
                else:
                    polygons.append(None)
        else:
            polygons = [None] * len(boxes)

        # Pad if fewer masks than boxes.
        while len(polygons) < len(boxes):
            polygons.append(None)
        return polygons

    except Exception as exc:
        logger.warning("SAM2 inference failed: %s. Using None masks.", exc)
        return [None] * len(boxes)


def refine_detections_with_sam2(
    image_bgr: np.ndarray,
    detections: list[dict],
    weights: str = _SAM_MODEL,
) -> list[dict]:
    """Add/replace mask_polygon in each detection using SAM2."""
    if not detections or not sam2_available():
        return detections

    boxes = [det["bbox_xyxy"][:4] for det in detections]
    polygons = run_sam2_on_boxes(image_bgr, boxes, weights)

    enriched = []
    for det, poly in zip(detections, polygons):
        if poly is not None:
            det = {**det, "mask_polygon": poly, "mask_source": "sam2"}
        enriched.append(det)
    return enriched
