"""Preview image rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from auto_annotation.segmenter import SegmentedInstance


def _build_supervision_detections(instances: list[SegmentedInstance]) -> sv.Detections:
    if not instances:
        return sv.Detections.empty()
    xyxy = np.array([inst.bbox_xyxy for inst in instances], dtype=np.float32)
    conf = np.array([inst.confidence for inst in instances], dtype=np.float32)
    class_id = np.array([inst.class_id for inst in instances], dtype=int)
    mask = np.stack([inst.mask.astype(bool) for inst in instances], axis=0)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id, mask=mask)


def render_preview(
    image_bgr: np.ndarray,
    instances: list[SegmentedInstance],
) -> np.ndarray:
    if not instances:
        return image_bgr.copy()

    dets = _build_supervision_detections(instances)
    labels = [
        f"{inst.class_name} | DINO {inst.confidence:.2f} | SAM {inst.sam_score:.2f}"
        for inst in instances
    ]

    canvas = image_bgr.copy()
    canvas = sv.MaskAnnotator().annotate(scene=canvas, detections=dets)
    canvas = sv.BoxAnnotator().annotate(scene=canvas, detections=dets)
    canvas = sv.LabelAnnotator().annotate(scene=canvas, detections=dets, labels=labels)
    return canvas


def save_preview(
    image_bgr: np.ndarray,
    instances: list[SegmentedInstance],
    output_path: Path,
    *,
    save_comparison: bool = False,
) -> None:
    annotated = render_preview(image_bgr, instances)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_comparison:
        h, w = image_bgr.shape[:2]
        if annotated.shape[:2] != (h, w):
            annotated = cv2.resize(annotated, (w, h))
        combined = np.hstack([image_bgr, annotated])
        cv2.imwrite(str(output_path), combined)
    else:
        cv2.imwrite(str(output_path), annotated)


def save_individual_masks(
    instances: list[SegmentedInstance],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, inst in enumerate(instances):
        mask_u8 = (inst.mask.astype(np.uint8) * 255)
        path = output_dir / f"{stem}_{inst.class_name}_{i:03d}.png"
        cv2.imwrite(str(path), mask_u8)
        paths.append(path)
    return paths
