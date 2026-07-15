"""Run SAM on YOLO box prompts and compute simple mask quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.segmentation.sam_service import SamModelType, load_sam_predictor
from src.segmentation_lab.yolo_prompts import BoxPrompt


@dataclass
class MaskResult:
    label: str
    yolo_confidence: float
    box_xyxy: list[int]
    sam_score: float
    mask_area_px: int
    box_area_px: int
    fill_ratio: float
    mask_inside_box_ratio: float
    mask_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _mask_inside_box_ratio(mask: np.ndarray, box: BoxPrompt) -> float:
    if box.area <= 0:
        return 0.0
    region = mask[box.y1 : box.y2, box.x1 : box.x2]
    if region.size == 0:
        return 0.0
    return float(region.sum()) / float(box.area)


def segment_box_prompts(
    frame_bgr: np.ndarray,
    boxes: list[BoxPrompt],
    *,
    model_type: SamModelType,
) -> list[tuple[MaskResult, np.ndarray]]:
    if not boxes:
        return []

    predictor = load_sam_predictor(model_type)
    image_rgb = frame_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    results: list[tuple[MaskResult, np.ndarray]] = []
    for box in boxes:
        prompt = np.array(box.xyxy, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=prompt, multimask_output=False)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
        mask_area = int(mask.sum())
        fill_ratio = mask_area / box.area if box.area > 0 else 0.0

        result = MaskResult(
            label=box.label,
            yolo_confidence=box.confidence,
            box_xyxy=box.xyxy,
            sam_score=round(float(scores[best_idx]), 4),
            mask_area_px=mask_area,
            box_area_px=box.area,
            fill_ratio=round(fill_ratio, 4),
            mask_inside_box_ratio=round(_mask_inside_box_ratio(mask, box), 4),
            mask_path="",
        )
        results.append((result, mask))
    return results
