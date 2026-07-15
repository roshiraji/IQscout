"""SAM segmentation from Grounding DINO box prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from auto_annotation.config import AppConfig
from auto_annotation.detector import RawDetection
from auto_annotation.utils import bgr_to_rgb

logger = logging.getLogger(__name__)


@dataclass
class SegmentedInstance:
    bbox_xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    prompt: str
    mask: np.ndarray
    sam_score: float
    mask_area_px: int
    mask_area_fraction: float


def segment_instances(
    sam_predictor,
    image_bgr: np.ndarray,
    detections: list[RawDetection],
    cfg: AppConfig,
) -> list[SegmentedInstance]:
    if not detections:
        return []

    rgb = bgr_to_rgb(image_bgr)
    h, w = image_bgr.shape[:2]
    sam_predictor.set_image(rgb)

    instances: list[SegmentedInstance] = []
    for det in detections:
        try:
            box = np.array(det.bbox_xyxy, dtype=np.float32)
            masks, scores, _ = sam_predictor.predict(
                box=box,
                multimask_output=cfg.multimask_output,
            )
            if cfg.choose_highest_score and len(scores) > 0:
                idx = int(np.argmax(scores))
            else:
                idx = 0
            mask = masks[idx].astype(bool)
            sam_score = float(scores[idx]) if len(scores) else 0.0

            if mask.shape[:2] != (h, w):
                logger.warning("Mask shape mismatch for %s — skipping", det.class_name)
                continue
            area_px = int(mask.sum())
            if area_px <= 0:
                logger.warning("Empty SAM mask for %s — skipping", det.class_name)
                continue

            instances.append(SegmentedInstance(
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                prompt=det.prompt,
                mask=mask,
                sam_score=sam_score,
                mask_area_px=area_px,
                mask_area_fraction=area_px / max(h * w, 1),
            ))
        except Exception as exc:
            logger.warning("SAM failed for %s: %s", det.class_name, exc)
    return instances
