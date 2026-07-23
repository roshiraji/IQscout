"""Lightweight density grids from analytics-safe trajectory points.

No model inference — bins existing screen-space points only.
Coordinate space remains normalized screen-space (not court units).
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.analytics.trajectories import COORDINATE_SPACE
from src.pipeline.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)

DENSITY_FILENAME = "heatmap_density.json"

DEFAULT_DENSITY_CONFIG: dict[str, Any] = {
    "bins_x": 48,
    "bins_y": 27,
}


def merge_density_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_DENSITY_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_analytics_points(
    trajectories: Iterable[dict[str, Any]],
    *,
    person_group_ids: set[int] | None = None,
    team_ids: set[int] | None = None,
    segment_id: int | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
    analytics_safe_only: bool = True,
) -> list[dict[str, float]]:
    """Collect normalized points with optional player / team / segment / time filters."""
    points: list[dict[str, float]] = []
    for row in trajectories:
        if not isinstance(row, dict):
            continue
        gid = _as_int(row.get("person_group_id"))
        team_id = _as_int(row.get("team_id")) if row.get("team_id") is not None else None
        if person_group_ids is not None and (gid is None or gid not in person_group_ids):
            continue
        if team_ids is not None and (team_id is None or team_id not in team_ids):
            continue

        fragments = row.get("path_fragments")
        source_points: list[dict[str, Any]] = []
        if isinstance(fragments, list) and fragments:
            for frag in fragments:
                if not isinstance(frag, dict):
                    continue
                if analytics_safe_only and not frag.get("usable_for_analytics", True):
                    continue
                source_points.extend(frag.get("points") or [])
        else:
            source_points = list(row.get("points") or [])

        for point in source_points:
            if not isinstance(point, dict):
                continue
            if analytics_safe_only and point.get("usable_for_analytics") is False:
                continue
            if segment_id is not None:
                sid = _as_int(point.get("segment_id"))
                if sid != segment_id:
                    continue
            t = _as_float(point.get("time_seconds"))
            if time_start is not None and t is not None and t < time_start:
                continue
            if time_end is not None and t is not None and t > time_end:
                continue
            xn = _as_float(point.get("x_normalized"))
            yn = _as_float(point.get("y_normalized"))
            if xn is None or yn is None:
                continue
            if not (0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0):
                # Still accept slight out-of-range by clipping later.
                pass
            points.append(
                {
                    "x_normalized": float(xn),
                    "y_normalized": float(yn),
                    "person_group_id": float(gid) if gid is not None else -1.0,
                    "team_id": float(team_id) if team_id is not None else -1.0,
                }
            )
    return points


def build_density_grid(
    points: Sequence[dict[str, Any]],
    *,
    bins_x: int = 48,
    bins_y: int = 27,
) -> dict[str, Any]:
    """2D histogram over normalized screen-space coordinates."""
    bins_x = max(1, int(bins_x))
    bins_y = max(1, int(bins_y))
    grid = [[0 for _ in range(bins_x)] for _ in range(bins_y)]
    max_count = 0
    used = 0

    for point in points:
        xn = _as_float(point.get("x_normalized"))
        yn = _as_float(point.get("y_normalized"))
        if xn is None or yn is None:
            continue
        ix = int(min(bins_x - 1, max(0, xn * bins_x)))
        iy = int(min(bins_y - 1, max(0, yn * bins_y)))
        grid[iy][ix] += 1
        max_count = max(max_count, grid[iy][ix])
        used += 1

    return {
        "coordinate_space": COORDINATE_SPACE,
        "bins_x": bins_x,
        "bins_y": bins_y,
        "point_count": used,
        "max_count": max_count,
        "grid": grid,
    }


def load_trajectory_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def generate_density_for_job(
    job_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    prefer_analytics: bool = True,
) -> dict[str, Any]:
    """Build job-local density from trajectory JSONL (analytics-safe preferred)."""
    job_dir = Path(job_dir).resolve()
    cfg = merge_density_config(config)

    analytics_path = job_dir / "analytics" / "trajectories_analytics.jsonl.gz"
    raw_path = job_dir / "analytics" / "trajectories.jsonl.gz"
    if prefer_analytics and analytics_path.is_file():
        rows = load_trajectory_rows(analytics_path)
        source = "trajectories_analytics.jsonl.gz"
        analytics_safe_only = True
    elif raw_path.is_file():
        rows = load_trajectory_rows(raw_path)
        source = "trajectories.jsonl.gz"
        analytics_safe_only = True  # still drop closeup/transition points
    else:
        raise FileNotFoundError(
            f"Trajectory artifacts missing under {job_dir / 'analytics'}"
        )

    points = iter_analytics_points(rows, analytics_safe_only=analytics_safe_only)
    density = build_density_grid(
        points,
        bins_x=int(cfg["bins_x"]),
        bins_y=int(cfg["bins_y"]),
    )
    payload = {
        "job_id": job_dir.name,
        "source": source,
        "analytics_safe_only": analytics_safe_only,
        "filters": {
            "excluded_closeups": True,
            "excluded_transitions": True,
        },
        **density,
        "note": "Density over normalized screen-space coordinates (not court units).",
    }
    out_path = job_dir / "analytics" / DENSITY_FILENAME
    write_json_atomic(out_path, payload)
    logger.info(
        "Wrote density grid %sx%s (%s points) → %s",
        density["bins_x"],
        density["bins_y"],
        density["point_count"],
        out_path,
    )
    return payload
