"""Geometry helpers for track positions (bbox + segmentation).

Stance points approximate foot/ground contact for pairwise distance metrics.
No court calibration — pixel coordinates only.
"""

from __future__ import annotations

from typing import Any, Sequence


def stance_point_from_mask(
    mask_polygon: Sequence[Sequence[float]] | None,
    bbox_xyxy: Sequence[float] | None,
    *,
    bottom_fraction: float = 0.15,
    min_points: int = 3,
) -> tuple[float, float, str]:
    """Return (x, y, source) for distance / speed metrics.

    Prefers the mean of the lowest ``bottom_fraction`` of mask y-points
    (``mask_bottom``). Falls back to bbox bottom-center (``bbox_bottom``).
    """
    bbox = list(bbox_xyxy or [0.0, 0.0, 0.0, 0.0])[:4]
    while len(bbox) < 4:
        bbox.append(0.0)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bbox_cx = (x1 + x2) / 2.0
    bbox_bottom = (bbox_cx, float(y2), "bbox_bottom")

    if not mask_polygon or len(mask_polygon) < min_points:
        return bbox_bottom

    points: list[tuple[float, float]] = []
    for pt in mask_polygon:
        if pt is None or len(pt) < 2:
            continue
        try:
            points.append((float(pt[0]), float(pt[1])))
        except (TypeError, ValueError):
            continue
    if len(points) < min_points:
        return bbox_bottom

    ys = sorted(p[1] for p in points)
    y_max = ys[-1]
    y_min = ys[0]
    span = max(y_max - y_min, 1.0)
    threshold = y_max - span * float(bottom_fraction)
    bottom = [p for p in points if p[1] >= threshold]
    if len(bottom) < 2:
        # Degenerate thin mask — use lowest few points.
        ordered = sorted(points, key=lambda p: p[1], reverse=True)
        bottom = ordered[: max(2, min(len(ordered), 5))]

    sx = sum(p[0] for p in bottom) / len(bottom)
    sy = sum(p[1] for p in bottom) / len(bottom)
    return (float(sx), float(sy), "mask_bottom")


def stance_from_row_or_det(row_or_det: dict[str, Any]) -> tuple[float, float, str]:
    """Compute stance from a detection dict or CSV-ready row with bbox keys."""
    poly = row_or_det.get("mask_polygon")
    bbox = row_or_det.get("bbox_xyxy")
    if bbox is None and all(k in row_or_det for k in ("x1", "y1", "x2", "y2")):
        bbox = [
            row_or_det["x1"],
            row_or_det["y1"],
            row_or_det["x2"],
            row_or_det["y2"],
        ]
    return stance_point_from_mask(poly, bbox)
