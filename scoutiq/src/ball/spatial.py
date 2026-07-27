"""Court-band gates for ball detections (reject crowd / scorebug FPs)."""

from __future__ import annotations

from typing import Any


# Broadcast defaults: keep mid-frame court, drop upper stands + lower scorebug.
DEFAULT_Y_MIN = 0.18
DEFAULT_Y_MAX = 0.90
DEFAULT_X_MIN = 0.02
DEFAULT_X_MAX = 0.98


def spatial_gates_from_cfg(ball_cfg: dict[str, Any] | None = None) -> dict[str, float | bool]:
    """Read ``ball.spatial_filter`` with safe defaults."""
    cfg = (ball_cfg or {}).get("spatial_filter") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "y_min": float(cfg.get("y_min", DEFAULT_Y_MIN)),
        "y_max": float(cfg.get("y_max", DEFAULT_Y_MAX)),
        "x_min": float(cfg.get("x_min", DEFAULT_X_MIN)),
        "x_max": float(cfg.get("x_max", DEFAULT_X_MAX)),
    }


def bbox_center_norm(bbox_xyxy: Any, *, frame_width: int, frame_height: int) -> tuple[float, float] | None:
    if not bbox_xyxy or len(bbox_xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    except (TypeError, ValueError):
        return None
    w = max(int(frame_width), 1)
    h = max(int(frame_height), 1)
    return (((x1 + x2) * 0.5) / w, ((y1 + y2) * 0.5) / h)


def center_in_court_band(
    x_norm: float,
    y_norm: float,
    *,
    y_min: float = DEFAULT_Y_MIN,
    y_max: float = DEFAULT_Y_MAX,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
) -> bool:
    """True when normalized center is inside the allowed court band."""
    return x_min <= x_norm <= x_max and y_min <= y_norm <= y_max


def filter_balls_by_court_band(
    detections: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    y_min: float = DEFAULT_Y_MIN,
    y_max: float = DEFAULT_Y_MAX,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Drop ball boxes whose centers sit in upper crowd or scorebug strips."""
    if not enabled or not detections:
        return list(detections)
    kept: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        xy = bbox_center_norm(
            det.get("bbox_xyxy"), frame_width=frame_width, frame_height=frame_height
        )
        if xy is None:
            continue
        if center_in_court_band(
            xy[0],
            xy[1],
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
        ):
            kept.append(det)
    return kept
