"""Grounding DINO detection wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from auto_annotation.config import AppConfig
from auto_annotation.filtering import filter_detections
from auto_annotation.prompts import ClassPromptMap, match_phrase_to_class

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    bbox_xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    prompt: str


def detect_objects(
    model,
    image_bgr: np.ndarray,
    prompt_map: ClassPromptMap,
    cfg: AppConfig,
) -> tuple[list[RawDetection], int]:
    """Run Grounding DINO and return filtered detections. Returns (detections, raw_count)."""
    import supervision as sv

    height, width = image_bgr.shape[:2]
    enhanced = list(prompt_map.prompts)

    sv_dets: sv.Detections = model.predict_with_classes(
        image=image_bgr,
        classes=enhanced,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    )
    raw_count = len(sv_dets)

    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    class_names: list[str] = []
    prompts_used: list[str] = []

    for i in range(len(sv_dets)):
        cid = sv_dets.class_id[i] if sv_dets.class_id is not None else None
        conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 0.0
        box = sv_dets.xyxy[i].tolist()

        if cid is None:
            continue
        class_name = prompt_map.classes[int(cid)] if int(cid) < len(prompt_map.classes) else None
        if class_name is None:
            continue

        boxes.append(box)
        scores.append(conf)
        class_ids.append(int(cid))
        class_names.append(class_name)
        prompts_used.append(enhanced[int(cid)] if int(cid) < len(enhanced) else class_name)

    fb, fs, fc, fn = filter_detections(
        boxes, scores, class_ids, class_names, width, height,
        min_area_px=cfg.minimum_box_area_pixels,
        min_area_frac=cfg.minimum_image_area_percentage,
        max_area_frac=cfg.maximum_image_area_percentage,
        box_threshold=cfg.box_threshold,
        nms_iou=cfg.nms_iou_threshold,
    )

    results: list[RawDetection] = []
    for box, score, cid, cname in zip(fb, fs, fc, fn):
        results.append(RawDetection(
            bbox_xyxy=box,
            confidence=score,
            class_id=cid,
            class_name=cname,
            prompt=prompt_map.prompts[cid] if cid < len(prompt_map.prompts) else cname,
        ))
    return results, raw_count
