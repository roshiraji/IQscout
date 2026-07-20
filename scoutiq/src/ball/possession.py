"""Possession inference → offense / defense team from ball + person tracks."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _person_point(row: dict[str, Any]) -> tuple[float, float] | None:
    if row.get("stance_x") is not None and row.get("stance_y") is not None:
        try:
            return float(row["stance_x"]), float(row["stance_y"])
        except (TypeError, ValueError):
            pass
    box = row.get("bbox_xyxy")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return _center([float(v) for v in box[:4]])
    if row.get("center_x") is not None and row.get("center_y") is not None:
        try:
            return float(row["center_x"]), float(row["center_y"])
        except (TypeError, ValueError):
            pass
    return None


def assign_frame_possession(
    ball_det: dict[str, Any] | None,
    person_rows: list[dict[str, Any]],
    *,
    max_dist_norm: float = 0.12,
    frame_diagonal: float = 1.0,
) -> dict[str, Any]:
    """Nearest non-official person to ball, or unknown/in_flight."""
    if not ball_det or not person_rows:
        return {
            "state": "lost" if not ball_det else "unknown",
            "person_group_id": None,
            "team_id": None,
            "jersey_color_name": None,
            "distance_norm": None,
            "confidence": 0.0,
        }
    box = ball_det.get("bbox_xyxy") or []
    if len(box) < 4:
        return {
            "state": "unknown",
            "person_group_id": None,
            "team_id": None,
            "jersey_color_name": None,
            "distance_norm": None,
            "confidence": 0.0,
        }
    bx, by = _center([float(v) for v in box[:4]])
    diag = max(float(frame_diagonal), 1.0)
    best = None
    best_d = 1e18
    for row in person_rows:
        role = str(row.get("role") or "").lower()
        if role == "official" or int(row.get("team_id") or -1) == 2:
            continue
        pt = _person_point(row)
        if pt is None:
            continue
        d = math.hypot(pt[0] - bx, pt[1] - by) / diag
        if d < best_d:
            best_d = d
            best = row
    if best is None or best_d > max_dist_norm:
        return {
            "state": "in_flight" if best is None or best_d > max_dist_norm * 1.8 else "contested",
            "person_group_id": None,
            "team_id": None,
            "jersey_color_name": None,
            "distance_norm": round(best_d, 4) if best is not None else None,
            "confidence": float(ball_det.get("confidence") or 0.0) * 0.35,
        }
    ball_conf = float(ball_det.get("confidence") or 0.5)
    prox = max(0.0, 1.0 - best_d / max(max_dist_norm, 1e-6))
    return {
        "state": "held",
        "person_group_id": int(best["person_group_id"])
        if best.get("person_group_id") is not None
        else None,
        "team_id": best.get("team_id"),
        "jersey_color_name": best.get("jersey_color_name"),
        "distance_norm": round(best_d, 4),
        "confidence": round(min(1.0, 0.45 * ball_conf + 0.55 * prox), 4),
    }


def summarize_possession(
    frame_possessions: list[dict[str, Any]],
    *,
    min_held_fraction: float = 0.12,
    min_confidence: float = 0.45,
    override_offense_team_id: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-frame possession into clip offense/defense teams."""
    held = [
        p
        for p in frame_possessions
        if p.get("state") == "held" and p.get("team_id") is not None
    ]
    total = max(len(frame_possessions), 1)
    team_counts: Counter = Counter()
    conf_sum: dict[int, float] = {}
    for p in held:
        try:
            tid = int(p["team_id"])
        except (TypeError, ValueError):
            continue
        if tid == 2:
            continue
        team_counts[tid] += 1
        conf_sum[tid] = conf_sum.get(tid, 0.0) + float(p.get("confidence") or 0.0)

    offense_team_id = None
    defense_team_id = None
    source = "unresolved"
    confidence = 0.0
    held_fraction = len(held) / total

    if override_offense_team_id is not None:
        offense_team_id = int(override_offense_team_id)
        source = "manual_override"
        confidence = 1.0
    elif held and held_fraction >= min_held_fraction:
        offense_team_id, n = team_counts.most_common(1)[0]
        confidence = (conf_sum.get(offense_team_id, 0.0) / max(n, 1)) * min(
            1.0, held_fraction / 0.25
        )
        if confidence >= min_confidence:
            source = "ball_possession"
        else:
            offense_team_id = None
            source = "unresolved_low_confidence"

    if offense_team_id is not None:
        other = [t for t in team_counts if t != offense_team_id]
        if other:
            defense_team_id = other[0]
        elif offense_team_id == 0:
            defense_team_id = 1
        elif offense_team_id == 1:
            defense_team_id = 0

    return {
        "offense_team_id": offense_team_id,
        "defense_team_id": defense_team_id,
        "source": source,
        "confidence": round(float(confidence), 4),
        "held_fraction": round(held_fraction, 4),
        "held_frames": len(held),
        "total_frames_scored": len(frame_possessions),
        "team_held_counts": {str(k): int(v) for k, v in team_counts.items()},
        "states": dict(Counter(p.get("state") or "unknown" for p in frame_possessions)),
        "note": (
            "Offense = team with ball possession when confident; "
            "jersey color still defines sides. Manual override allowed."
        ),
    }


def write_possession_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_possession_override(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("source") == "manual_override" and data.get("offense_team_id") is not None:
        try:
            return int(data["offense_team_id"])
        except (TypeError, ValueError):
            return None
    # Separate override file support
    if data.get("offense_team_id") is not None and data.get("manual"):
        try:
            return int(data["offense_team_id"])
        except (TypeError, ValueError):
            return None
    return None
