"""Fuse bbox detections from multiple sources and boost confidence with SAM2 masks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Map LocateAnything / legacy class names to NBA training schema.
CANONICAL_CLASS: dict[str, str] = {
    "player": "player",
    "player_red": "player",
    "player_black_blue": "player",
    "referee": "referee",
    "hoop": "hoop",
    "ball": "ball",
}

CANONICAL_CLASS_ID: dict[str, int] = {
    "player": 0,
    "referee": 1,
    "hoop": 2,
    "ball": 3,
}


@dataclass
class FusionWeights:
    bbox: float = 0.55
    mask_iou: float = 0.25
    multi_source: float = 0.20


def iou(a: list | tuple, b: list | tuple) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)
    area_b = max(bx2 - bx1, 1) * max(by2 - by1, 1)
    return inter / (area_a + area_b - inter)


def _canonicalize(det: dict) -> dict:
    name = det.get("class_name", "player")
    canonical = CANONICAL_CLASS.get(name, name)
    out = dict(det)
    out["canonical_class"] = canonical
    out["class_id"] = CANONICAL_CLASS_ID.get(canonical, det.get("class_id", 0))
    out["class_name"] = canonical
    return out


def _polygon_bbox(polygon: list[list[float]]) -> list[int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.float32)
    return float(cv2.contourArea(pts))


def mask_consistency_score(det: dict, polygon: list[list[float]] | None) -> float:
    """Score how well a SAM mask agrees with the detection bbox (0-1)."""
    if not polygon or len(polygon) < 3:
        return 0.0
    box = det.get("bbox_xyxy", [])
    if len(box) < 4:
        return 0.0

    mask_box = _polygon_bbox(polygon)
    overlap = iou(box, mask_box)
    mask_area = _polygon_area(polygon)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
    fill_ratio = min(mask_area / box_area, 1.5) / 1.5  # cap overshoot
    return float(min(1.0, 0.6 * overlap + 0.4 * fill_ratio))


def fuse_bbox_detections(
    detections: list[dict],
    *,
    match_iou: float = 0.45,
    source_bonus: float = 0.12,
) -> list[dict]:
    """Cluster overlapping boxes per class and merge confidence across sources."""
    normalized = [_canonicalize(d) for d in detections]
    fused: list[dict] = []

    for canonical in sorted({d["canonical_class"] for d in normalized}):
        class_dets = [d for d in normalized if d["canonical_class"] == canonical]
        used = [False] * len(class_dets)

        for i, anchor in enumerate(class_dets):
            if used[i]:
                continue
            cluster = [anchor]
            sources = {anchor.get("source", "unknown")}
            used[i] = True

            for j in range(i + 1, len(class_dets)):
                if used[j]:
                    continue
                if iou(anchor["bbox_xyxy"], class_dets[j]["bbox_xyxy"]) >= match_iou:
                    cluster.append(class_dets[j])
                    sources.add(class_dets[j].get("source", "unknown"))
                    used[j] = True

            # Weighted bbox average by confidence.
            total_w = sum(float(d.get("confidence", 0.5)) for d in cluster) or 1.0
            box = [0.0, 0.0, 0.0, 0.0]
            for d in cluster:
                w = float(d.get("confidence", 0.5))
                for k in range(4):
                    box[k] += w * float(d["bbox_xyxy"][k])
            box = [int(round(v / total_w)) for v in box]

            base_conf = max(float(d.get("confidence", 0.5)) for d in cluster)
            bonus = source_bonus * max(len(sources) - 1, 0)
            merged_conf = min(base_conf + bonus, 1.0)

            fused.append({
                **cluster[0],
                "bbox_xyxy": box,
                "confidence": round(merged_conf, 4),
                "bbox_confidence": round(base_conf, 4),
                "fusion_sources": sorted(sources),
                "source": "+".join(sorted(sources)),
            })

    return fused


def apply_mask_confidence_boost(
    detections: list[dict],
    *,
    weights: FusionWeights | None = None,
    min_fused_conf: float = 0.20,
) -> list[dict]:
    """Raise or lower final confidence using SAM mask agreement."""
    w = weights or FusionWeights()
    boosted: list[dict] = []

    for det in detections:
        polygon = det.get("mask_polygon")
        mask_score = mask_consistency_score(det, polygon)
        bbox_conf = float(det.get("bbox_confidence", det.get("confidence", 0.5)))
        source_count = len(det.get("fusion_sources", [det.get("source", "")]))
        multi_bonus = w.multi_source * max(source_count - 1, 0)

        fused_conf = (
            w.bbox * bbox_conf
            + w.mask_iou * mask_score
            + multi_bonus
        )
        fused_conf = min(max(fused_conf, 0.0), 1.0)

        out = dict(det)
        out["mask_consistency"] = round(mask_score, 4)
        out["confidence"] = round(fused_conf, 4)
        out["fused_confidence"] = round(fused_conf, 4)
        if fused_conf >= min_fused_conf:
            boosted.append(out)

    return boosted
