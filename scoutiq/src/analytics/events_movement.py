"""Movement events from analytics-safe screen-space trajectories (Phase 10).

Speeds/distances are in normalized screen-space units (not feet/meters).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from src.analytics.trajectories import COORDINATE_SPACE

logger = logging.getLogger(__name__)

EVENT_BECAME_STATIONARY = "player_became_stationary"
EVENT_STARTED_MOVING = "player_started_moving"
EVENT_RAPID_ACCEL = "rapid_acceleration"
EVENT_SHARP_TURN = "sharp_direction_change"
EVENT_CONVERGED = "players_converged"
EVENT_SEPARATED = "players_separated"

MOVEMENT_EVENTS = {
    EVENT_BECAME_STATIONARY,
    EVENT_STARTED_MOVING,
    EVENT_RAPID_ACCEL,
    EVENT_SHARP_TURN,
    EVENT_CONVERGED,
    EVENT_SEPARATED,
}

DEFAULT_MOVEMENT_CONFIG: dict[str, Any] = {
    "stationary_speed_max": 0.015,  # norm units / second
    "moving_speed_min": 0.040,
    "state_persist_points": 5,
    "accel_delta_min": 0.28,  # speed increase (norm/s)
    "accel_window_points": 4,
    "accel_cooldown_frames": 20,
    "turn_angle_deg_min": 70.0,
    "turn_min_speed": 0.05,
    "turn_cooldown_frames": 20,
    "converge_distance_max": 0.08,
    "converge_prior_min": 0.14,
    "separate_distance_min": 0.18,
    "separate_prior_max": 0.10,
    "pair_event_cooldown_frames": 20,
    "max_events_per_type_per_player": 12,
}


def merge_movement_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_MOVEMENT_CONFIG)
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


def _norm_dist(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float | None:
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos_v = (ax * bx + ay * by) / (na * nb)
    cos_v = max(-1.0, min(1.0, cos_v))
    return math.degrees(math.acos(cos_v))


def _raw_event(
    *,
    event_type: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    confidence: float,
    person_group_ids: Sequence[int],
    segment_id: int | None,
    details: dict[str, Any],
    sort_tie: int = 0,
) -> dict[str, Any]:
    return {
        "_sort": (int(start_frame), int(sort_tie)),
        "event_type": event_type,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "confidence": float(confidence),
        "person_group_ids": [int(x) for x in person_group_ids],
        "segment_id": segment_id,
        "details": {
            "coordinate_space": COORDINATE_SPACE,
            "units": "normalized_screen_space",
            "note": "Screen-space measurement only — not real-world speed/distance.",
            **details,
        },
    }


def _fragment_points(frag: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for p in frag.get("points") or []:
        if not isinstance(p, dict):
            continue
        if p.get("usable_for_analytics") is False:
            continue
        x = _as_float(p.get("x_normalized"))
        y = _as_float(p.get("y_normalized"))
        frame = _as_int(p.get("frame"))
        t = _as_float(p.get("time_seconds"))
        if None in (x, y, frame, t):
            continue
        points.append(
            {
                "frame": frame,
                "time_seconds": t,
                "x_normalized": x,
                "y_normalized": y,
                "segment_id": _as_int(p.get("segment_id")),
            }
        )
    points.sort(key=lambda row: (row["frame"], row["time_seconds"]))
    return points


def _speeds(points: list[dict[str, Any]]) -> list[float | None]:
    """Per-point instantaneous speed using previous sample (norm units / second)."""
    speeds: list[float | None] = [None]
    for i in range(1, len(points)):
        dt = points[i]["time_seconds"] - points[i - 1]["time_seconds"]
        if dt <= 1e-6:
            speeds.append(None)
            continue
        dist = _norm_dist(
            points[i - 1]["x_normalized"],
            points[i - 1]["y_normalized"],
            points[i]["x_normalized"],
            points[i]["y_normalized"],
        )
        speeds.append(dist / dt)
    return speeds


def detect_player_motion_events(
    *,
    person_group_id: int,
    fragment: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    points = _fragment_points(fragment)
    if len(points) < 3:
        return []
    cfg = config
    speeds = _speeds(points)
    stationary_max = float(cfg["stationary_speed_max"])
    moving_min = float(cfg["moving_speed_min"])
    persist = max(2, int(cfg["state_persist_points"]))
    accel_delta = float(cfg["accel_delta_min"])
    accel_window = max(2, int(cfg["accel_window_points"]))
    accel_cooldown = max(1, int(cfg.get("accel_cooldown_frames", 20)))
    turn_deg = float(cfg["turn_angle_deg_min"])
    turn_min_speed = float(cfg["turn_min_speed"])
    turn_cooldown = max(1, int(cfg.get("turn_cooldown_frames", 20)))
    max_per_type = max(1, int(cfg["max_events_per_type_per_player"]))

    events: list[dict[str, Any]] = []
    counts = {
        EVENT_BECAME_STATIONARY: 0,
        EVENT_STARTED_MOVING: 0,
        EVENT_RAPID_ACCEL: 0,
        EVENT_SHARP_TURN: 0,
    }
    last_accel_frame = -10_000
    last_turn_frame = -10_000

    # Classify each point: 0 stationary, 1 moving, -1 unknown
    labels: list[int] = []
    for speed in speeds:
        if speed is None:
            labels.append(-1)
        elif speed <= stationary_max:
            labels.append(0)
        elif speed >= moving_min:
            labels.append(1)
        else:
            labels.append(-1)

    def _run_state(end_idx: int, want: int) -> bool:
        start = max(0, end_idx - persist + 1)
        chunk = labels[start : end_idx + 1]
        return len(chunk) >= persist and all(v == want for v in chunk)

    prev_stable: int | None = None
    for i in range(persist - 1, len(points)):
        if _run_state(i, 0):
            if prev_stable == 1 and counts[EVENT_BECAME_STATIONARY] < max_per_type:
                p = points[i]
                events.append(
                    _raw_event(
                        event_type=EVENT_BECAME_STATIONARY,
                        start_frame=p["frame"],
                        end_frame=p["frame"],
                        start_time=p["time_seconds"],
                        end_time=p["time_seconds"],
                        confidence=0.78,
                        person_group_ids=[person_group_id],
                        segment_id=p.get("segment_id") or _as_int(fragment.get("segment_id")),
                        details={
                            "speed_norm_per_s": round(float(speeds[i] or 0.0), 6),
                            "stationary_speed_max": stationary_max,
                            "persist_points": persist,
                        },
                        sort_tie=2,
                    )
                )
                counts[EVENT_BECAME_STATIONARY] += 1
            prev_stable = 0
        elif _run_state(i, 1):
            if prev_stable == 0 and counts[EVENT_STARTED_MOVING] < max_per_type:
                p = points[i]
                events.append(
                    _raw_event(
                        event_type=EVENT_STARTED_MOVING,
                        start_frame=p["frame"],
                        end_frame=p["frame"],
                        start_time=p["time_seconds"],
                        end_time=p["time_seconds"],
                        confidence=0.78,
                        person_group_ids=[person_group_id],
                        segment_id=p.get("segment_id") or _as_int(fragment.get("segment_id")),
                        details={
                            "speed_norm_per_s": round(float(speeds[i] or 0.0), 6),
                            "moving_speed_min": moving_min,
                            "persist_points": persist,
                        },
                        sort_tie=2,
                    )
                )
                counts[EVENT_STARTED_MOVING] += 1
            prev_stable = 1

    # Rapid acceleration
    for i in range(accel_window, len(points)):
        s0 = speeds[i - accel_window]
        s1 = speeds[i]
        if s0 is None or s1 is None:
            continue
        delta = s1 - s0
        frame = points[i]["frame"]
        if (
            delta >= accel_delta
            and counts[EVENT_RAPID_ACCEL] < max_per_type
            and frame - last_accel_frame >= accel_cooldown
        ):
            p = points[i]
            events.append(
                _raw_event(
                    event_type=EVENT_RAPID_ACCEL,
                    start_frame=points[i - accel_window]["frame"],
                    end_frame=p["frame"],
                    start_time=points[i - accel_window]["time_seconds"],
                    end_time=p["time_seconds"],
                    confidence=min(0.9, 0.55 + min(delta, 0.4)),
                    person_group_ids=[person_group_id],
                    segment_id=p.get("segment_id") or _as_int(fragment.get("segment_id")),
                    details={
                        "speed_before_norm_per_s": round(s0, 6),
                        "speed_after_norm_per_s": round(s1, 6),
                        "delta_speed_norm_per_s": round(delta, 6),
                        "accel_delta_min": accel_delta,
                    },
                    sort_tie=3,
                )
            )
            counts[EVENT_RAPID_ACCEL] += 1
            last_accel_frame = frame

    # Sharp direction changes
    for i in range(2, len(points)):
        a0, a1, a2 = points[i - 2], points[i - 1], points[i]
        v1 = (a1["x_normalized"] - a0["x_normalized"], a1["y_normalized"] - a0["y_normalized"])
        v2 = (a2["x_normalized"] - a1["x_normalized"], a2["y_normalized"] - a1["y_normalized"])
        angle = _angle_deg(v1[0], v1[1], v2[0], v2[1])
        speed = speeds[i]
        if angle is None or speed is None or speed < turn_min_speed:
            continue
        frame = a2["frame"]
        if (
            angle >= turn_deg
            and counts[EVENT_SHARP_TURN] < max_per_type
            and frame - last_turn_frame >= turn_cooldown
        ):
            events.append(
                _raw_event(
                    event_type=EVENT_SHARP_TURN,
                    start_frame=a1["frame"],
                    end_frame=a2["frame"],
                    start_time=a1["time_seconds"],
                    end_time=a2["time_seconds"],
                    confidence=min(0.9, 0.5 + angle / 180.0),
                    person_group_ids=[person_group_id],
                    segment_id=a2.get("segment_id") or _as_int(fragment.get("segment_id")),
                    details={
                        "turn_angle_deg": round(angle, 3),
                        "turn_angle_deg_min": turn_deg,
                        "speed_norm_per_s": round(speed, 6),
                    },
                    sort_tie=3,
                )
            )
            counts[EVENT_SHARP_TURN] += 1
            last_turn_frame = frame

    return events


def _positions_by_frame(
    trajectories: Sequence[dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """frame -> person_group_id -> point."""
    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for row in trajectories:
        if not isinstance(row, dict):
            continue
        gid = _as_int(row.get("person_group_id"))
        if gid is None:
            continue
        fragments = row.get("path_fragments") or []
        if not isinstance(fragments, list):
            continue
        for frag in fragments:
            if not isinstance(frag, dict) or not frag.get("usable_for_analytics", True):
                continue
            for p in _fragment_points(frag):
                by_frame.setdefault(p["frame"], {})[gid] = {
                    **p,
                    "person_group_id": gid,
                }
    return by_frame


def detect_pair_distance_events(
    trajectories: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = config
    converge_max = float(cfg["converge_distance_max"])
    converge_prior = float(cfg["converge_prior_min"])
    separate_min = float(cfg["separate_distance_min"])
    separate_prior = float(cfg["separate_prior_max"])
    cooldown = max(1, int(cfg["pair_event_cooldown_frames"]))

    by_frame = _positions_by_frame(trajectories)
    frames = sorted(by_frame)
    if len(frames) < 2:
        return []

    prev_dist: dict[tuple[int, int], float] = {}
    last_event_frame: dict[tuple[str, int, int], int] = {}
    events: list[dict[str, Any]] = []

    for frame in frames:
        players = by_frame[frame]
        ids = sorted(players)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pa, pb = players[a], players[b]
                dist = _norm_dist(
                    pa["x_normalized"],
                    pa["y_normalized"],
                    pb["x_normalized"],
                    pb["y_normalized"],
                )
                key = (a, b)
                prior = prev_dist.get(key)
                prev_dist[key] = dist
                if prior is None:
                    continue

                # Converged: was far, now close
                if prior >= converge_prior and dist <= converge_max:
                    ek = (EVENT_CONVERGED, a, b)
                    if frame - last_event_frame.get(ek, -10_000) >= cooldown:
                        last_event_frame[ek] = frame
                        events.append(
                            _raw_event(
                                event_type=EVENT_CONVERGED,
                                start_frame=frame,
                                end_frame=frame,
                                start_time=float(pa["time_seconds"]),
                                end_time=float(pa["time_seconds"]),
                                confidence=0.72,
                                person_group_ids=[a, b],
                                segment_id=pa.get("segment_id") or pb.get("segment_id"),
                                details={
                                    "distance_norm": round(dist, 6),
                                    "prior_distance_norm": round(prior, 6),
                                    "converge_distance_max": converge_max,
                                    "converge_prior_min": converge_prior,
                                },
                                sort_tie=4,
                            )
                        )

                # Separated: was close, now far
                if prior <= separate_prior and dist >= separate_min:
                    ek = (EVENT_SEPARATED, a, b)
                    if frame - last_event_frame.get(ek, -10_000) >= cooldown:
                        last_event_frame[ek] = frame
                        events.append(
                            _raw_event(
                                event_type=EVENT_SEPARATED,
                                start_frame=frame,
                                end_frame=frame,
                                start_time=float(pa["time_seconds"]),
                                end_time=float(pa["time_seconds"]),
                                confidence=0.72,
                                person_group_ids=[a, b],
                                segment_id=pa.get("segment_id") or pb.get("segment_id"),
                                details={
                                    "distance_norm": round(dist, 6),
                                    "prior_distance_norm": round(prior, 6),
                                    "separate_distance_min": separate_min,
                                    "separate_prior_max": separate_prior,
                                },
                                sort_tie=4,
                            )
                        )
    return events


def detect_movement_events(
    trajectories: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return raw timeline events (with ``_sort``) from analytics-safe paths."""
    cfg = merge_movement_config(config)
    events: list[dict[str, Any]] = []
    for row in trajectories:
        if not isinstance(row, dict):
            continue
        gid = _as_int(row.get("person_group_id"))
        if gid is None:
            continue
        fragments = row.get("path_fragments") or []
        if not isinstance(fragments, list):
            continue
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            if not frag.get("usable_for_analytics", True):
                continue
            events.extend(
                detect_player_motion_events(
                    person_group_id=gid,
                    fragment=frag,
                    config=cfg,
                )
            )
    events.extend(detect_pair_distance_events(trajectories, config=cfg))
    logger.info("Detected %s movement events", len(events))
    return events
