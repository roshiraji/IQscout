"""Exclusion zones for court ROI, shot clock/scorebug, crowd, and benches.

JSON schema (data/exclusions/<clip_id>.json or a shared template):

{
  "clip_id": "fever_sparks_2026_07_08",
  "width": 1920,
  "height": 1080,
  "court_roi": [[x,y], ...],          # keep detections whose center is inside
  "exclude_rects": [                   # kill HUD / shot clock / scorebug
    {"name": "shot_clock", "xyxy": [x1,y1,x2,y2]},
    {"name": "scorebug", "xyxy": [x1,y1,x2,y2]}
  ],
  "exclude_polygons": [                # optional crowd/bench polygons
    {"name": "crowd_left", "points": [[x,y], ...]}
  ],
  "margins": {"left": 0.12, "right": 0.12, "top": 0.08, "bottom": 0.05}
}

Coordinates may be absolute pixels or normalized 0-1 (auto-detected if all <= 1.0).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.court_roi import default_roi_polygon, point_in_roi

logger = logging.getLogger(__name__)


@dataclass
class ExclusionRect:
    name: str
    xyxy: list[float]  # absolute pixels after resolve


@dataclass
class ExclusionPolygon:
    name: str
    points: np.ndarray  # Nx2 int32 absolute pixels


@dataclass
class ExclusionSet:
    clip_id: str
    width: int
    height: int
    court_roi: np.ndarray
    exclude_rects: list[ExclusionRect] = field(default_factory=list)
    exclude_polygons: list[ExclusionPolygon] = field(default_factory=list)

    def center_allowed(self, cx: float, cy: float) -> bool:
        """True if a detection center should be kept."""
        if not point_in_roi(cx, cy, self.court_roi):
            return False
        for rect in self.exclude_rects:
            x1, y1, x2, y2 = rect.xyxy
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return False
        for poly in self.exclude_polygons:
            if cv2.pointPolygonTest(poly.points.astype(np.float32), (float(cx), float(cy)), False) >= 0:
                return False
        return True

    def box_allowed(self, bbox_xyxy: list[float] | tuple) -> bool:
        x1, y1, x2, y2 = bbox_xyxy[:4]
        return self.center_allowed((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _looks_normalized(values: list[float]) -> bool:
    return bool(values) and max(abs(float(v)) for v in values) <= 1.0 + 1e-6


def _resolve_xyxy(xyxy: list[float], width: int, height: int) -> list[float]:
    vals = [float(v) for v in xyxy[:4]]
    if _looks_normalized(vals):
        return [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    return vals


def _resolve_points(points: list[list[float]], width: int, height: int) -> np.ndarray:
    flat = [float(c) for pt in points for c in pt[:2]]
    if _looks_normalized(flat):
        arr = np.array([[p[0] * width, p[1] * height] for p in points], dtype=np.int32)
    else:
        arr = np.array([[p[0], p[1]] for p in points], dtype=np.int32)
    return arr


def default_exclusion_set(width: int, height: int, clip_id: str = "default") -> ExclusionSet:
    """Court ROI from default margins + top-right shot-clock-ish rect + top scorebug strip."""
    roi = default_roi_polygon(width, height)
    # Typical broadcast: shot clock top-right, scorebug top or bottom banner.
    shot = ExclusionRect(
        name="shot_clock",
        xyxy=[width * 0.88, height * 0.02, width * 0.99, height * 0.18],
    )
    scorebug = ExclusionRect(
        name="scorebug",
        xyxy=[width * 0.30, height * 0.0, width * 0.70, height * 0.10],
    )
    return ExclusionSet(
        clip_id=clip_id,
        width=width,
        height=height,
        court_roi=roi,
        exclude_rects=[shot, scorebug],
        exclude_polygons=[],
    )


def load_exclusion_set(
    path: Path | None,
    *,
    width: int,
    height: int,
    clip_id: str = "default",
) -> ExclusionSet:
    if path is None or not path.is_file():
        logger.info("No exclusion JSON at %s — using defaults.", path)
        return default_exclusion_set(width, height, clip_id=clip_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    w = int(payload.get("width") or width)
    h = int(payload.get("height") or height)
    cid = str(payload.get("clip_id") or clip_id)

    if payload.get("court_roi") and len(payload["court_roi"]) >= 3:
        roi = _resolve_points(payload["court_roi"], w, h)
    elif payload.get("margins"):
        m = payload["margins"]
        left = int(w * float(m.get("left", 0.12)))
        right = int(w * (1.0 - float(m.get("right", 0.12))))
        top = int(h * float(m.get("top", 0.08)))
        bottom = int(h * (1.0 - float(m.get("bottom", 0.05))))
        roi = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.int32)
    else:
        roi = default_roi_polygon(w, h)

    rects: list[ExclusionRect] = []
    for item in payload.get("exclude_rects") or []:
        xyxy = item.get("xyxy") or item.get("box")
        if not xyxy or len(xyxy) < 4:
            continue
        rects.append(ExclusionRect(name=str(item.get("name", "rect")), xyxy=_resolve_xyxy(xyxy, w, h)))

    polys: list[ExclusionPolygon] = []
    for item in payload.get("exclude_polygons") or []:
        pts = item.get("points") or item.get("polygon")
        if not pts or len(pts) < 3:
            continue
        polys.append(
            ExclusionPolygon(name=str(item.get("name", "poly")), points=_resolve_points(pts, w, h))
        )

    return ExclusionSet(
        clip_id=cid,
        width=w,
        height=h,
        court_roi=roi,
        exclude_rects=rects,
        exclude_polygons=polys,
    )


def resolve_exclusion_path(
    exclusions_dir: Path,
    clip_id: str,
    *,
    fallback: Path | None = None,
) -> Path | None:
    candidates = [
        exclusions_dir / f"{clip_id}.json",
        exclusions_dir / "default.json",
    ]
    if fallback is not None:
        candidates.append(fallback)
    for path in candidates:
        if path.is_file():
            return path
    return None


def filter_detections(
    detections: list[dict],
    exclusions: ExclusionSet,
    *,
    class_names: tuple[str, ...] | None = ("player",),
) -> list[dict]:
    """Keep detections that pass exclusion checks (and optional class filter)."""
    kept: list[dict] = []
    for det in detections:
        name = det.get("class_name") or det.get("canonical_class") or ""
        if class_names is not None and name not in class_names:
            continue
        box = det.get("bbox_xyxy") or []
        if len(box) < 4:
            continue
        if exclusions.box_allowed(box):
            kept.append(det)
    return kept


def write_default_exclusion_template(path: Path, *, width: int = 1920, height: int = 1080) -> Path:
    """Write an editable default template (normalized coords for portability)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": "default",
        "width": width,
        "height": height,
        "margins": {"left": 0.12, "right": 0.12, "top": 0.08, "bottom": 0.05},
        "exclude_rects": [
            {"name": "shot_clock", "xyxy": [0.88, 0.02, 0.99, 0.18]},
            {"name": "scorebug", "xyxy": [0.30, 0.00, 0.70, 0.10]},
        ],
        "exclude_polygons": [],
        "_comment": "Edit court_roi (pixel or 0-1) or margins; add benches as exclude_rects/polygons.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
