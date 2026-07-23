"""Cautious MVP defensive events from team_id + nearest-opponent logic (Phase 11).

All distances are normalized screen-space separation — not official defensive stats.
Language in event details uses possible / likely wording only.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from src.analytics.trajectories import COORDINATE_SPACE

logger = logging.getLogger(__name__)

EVENT_PRESSURE_STARTED = "close_defensive_pressure_started"
EVENT_PRESSURE_ENDED = "close_defensive_pressure_ended"
EVENT_SEPARATION = "separation_threshold_exceeded"
EVENT_HELP_DEFENSE = "possible_help_defense"
EVENT_DOUBLE_TEAM = "possible_double_team"
EVENT_MATCHUP_CHANGED = "likely_matchup_changed"

DEFENSE_EVENTS = {
    EVENT_PRESSURE_STARTED,
    EVENT_PRESSURE_ENDED,
    EVENT_SEPARATION,
    EVENT_HELP_DEFENSE,
    EVENT_DOUBLE_TEAM,
    EVENT_MATCHUP_CHANGED,
}

DEFAULT_DEFENSE_CONFIG: dict[str, Any] = {
    "pressure_distance_max": 0.08,
    "pressure_clear_distance": 0.12,
    "pressure_persist_frames": 5,
    "clear_persist_frames": 5,
    "separation_distance_min": 0.18,
    "separation_prior_max": 0.10,
    "double_team_distance_max": 0.10,
    "double_team_persist_frames": 4,
    "help_defense_distance_max": 0.12,
    "help_defense_persist_frames": 3,
    "matchup_persist_frames": 5,
    "matchup_cooldown_frames": 30,
    "event_cooldown_frames": 20,
    "max_events_per_type_per_player": 12,
}


def merge_defense_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_DEFENSE_CONFIG)
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
    sort_tie: int = 5,
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
            "note": (
                "Possible/likely screen-space separation only — "
                "not official defensive stats or real-world distance."
            ),
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
        camera_state = p.get("camera_state")
        if camera_state is not None and camera_state != "wide_gameplay":
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


def _positions_by_frame(
    trajectories: Sequence[dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """frame -> person_group_id -> point (+ team_id)."""
    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for row in trajectories:
        if not isinstance(row, dict):
            continue
        gid = _as_int(row.get("person_group_id"))
        if gid is None:
            continue
        team_id = _as_int(row.get("team_id"))
        if team_id is None:
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
                    "team_id": team_id,
                }
    return by_frame


def _nearest_opponent(
    player: dict[str, Any],
    others: dict[int, dict[str, Any]],
) -> tuple[int | None, float | None]:
    best_id: int | None = None
    best_dist: float | None = None
    team = player["team_id"]
    for oid, other in others.items():
        if oid == player["person_group_id"]:
            continue
        if other.get("team_id") == team:
            continue
        dist = _norm_dist(
            player["x_normalized"],
            player["y_normalized"],
            other["x_normalized"],
            other["y_normalized"],
        )
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = oid
    return best_id, best_dist


def _opponents_within(
    player: dict[str, Any],
    others: dict[int, dict[str, Any]],
    radius: float,
) -> list[tuple[int, float]]:
    team = player["team_id"]
    hits: list[tuple[int, float]] = []
    for oid, other in others.items():
        if oid == player["person_group_id"]:
            continue
        if other.get("team_id") == team:
            continue
        dist = _norm_dist(
            player["x_normalized"],
            player["y_normalized"],
            other["x_normalized"],
            other["y_normalized"],
        )
        if dist <= radius:
            hits.append((oid, dist))
    hits.sort(key=lambda item: item[1])
    return hits


def detect_defense_events(
    trajectories: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return raw timeline events (with ``_sort``) from analytics-safe paths."""
    cfg = merge_defense_config(config)
    pressure_max = float(cfg["pressure_distance_max"])
    pressure_clear = float(cfg["pressure_clear_distance"])
    pressure_persist = max(1, int(cfg["pressure_persist_frames"]))
    clear_persist = max(1, int(cfg["clear_persist_frames"]))
    sep_min = float(cfg["separation_distance_min"])
    sep_prior = float(cfg["separation_prior_max"])
    double_max = float(cfg["double_team_distance_max"])
    double_persist = max(1, int(cfg["double_team_persist_frames"]))
    help_max = float(cfg["help_defense_distance_max"])
    help_persist = max(1, int(cfg["help_defense_persist_frames"]))
    matchup_persist = max(1, int(cfg["matchup_persist_frames"]))
    matchup_cooldown = max(1, int(cfg["matchup_cooldown_frames"]))
    event_cooldown = max(1, int(cfg["event_cooldown_frames"]))
    max_per_type = max(1, int(cfg["max_events_per_type_per_player"]))

    by_frame = _positions_by_frame(trajectories)
    frames = sorted(by_frame)
    if not frames:
        return []

    events: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = {}
    last_event_frame: dict[tuple[str, int], int] = {}

    # Per focal player state
    pressure_run: dict[int, int] = {}
    clear_run: dict[int, int] = {}
    pressure_active: dict[int, bool] = {}
    pressure_defender: dict[int, int | None] = {}
    pressure_start_meta: dict[int, dict[str, Any]] = {}
    was_close_dist: dict[int, float] = {}
    double_run: dict[int, int] = {}
    help_run: dict[int, int] = {}
    matchup_candidate: dict[int, int | None] = {}
    matchup_run: dict[int, int] = {}
    matchup_current: dict[int, int | None] = {}
    last_matchup_frame: dict[int, int] = {}

    def _can_emit(event_type: str, gid: int, frame: int) -> bool:
        key = (event_type, gid)
        if counts.get(key, 0) >= max_per_type:
            return False
        if frame - last_event_frame.get(key, -10_000) < event_cooldown:
            return False
        return True

    def _mark(event_type: str, gid: int, frame: int) -> None:
        key = (event_type, gid)
        counts[key] = counts.get(key, 0) + 1
        last_event_frame[key] = frame

    for frame in frames:
        players = by_frame[frame]
        for gid, player in players.items():
            nearest_id, nearest_dist = _nearest_opponent(player, players)
            t = float(player["time_seconds"])
            seg = player.get("segment_id")

            # --- Matchup change (nearest opp identity persists) ---
            if nearest_id is not None:
                prev_match = matchup_current.get(gid)
                cand = matchup_candidate.get(gid)
                if cand == nearest_id:
                    matchup_run[gid] = matchup_run.get(gid, 0) + 1
                else:
                    matchup_candidate[gid] = nearest_id
                    matchup_run[gid] = 1

                if (
                    matchup_run[gid] >= matchup_persist
                    and prev_match is not None
                    and nearest_id != prev_match
                    and frame - last_matchup_frame.get(gid, -10_000) >= matchup_cooldown
                    and _can_emit(EVENT_MATCHUP_CHANGED, gid, frame)
                ):
                    events.append(
                        _raw_event(
                            event_type=EVENT_MATCHUP_CHANGED,
                            start_frame=frame,
                            end_frame=frame,
                            start_time=t,
                            end_time=t,
                            confidence=0.62,
                            person_group_ids=[gid, nearest_id, prev_match],
                            segment_id=seg,
                            details={
                                "label": "likely_matchup_changed",
                                "previous_opponent_id": prev_match,
                                "new_opponent_id": nearest_id,
                                "screen_space_separation": round(float(nearest_dist or 0.0), 6),
                                "matchup_persist_frames": matchup_persist,
                            },
                            sort_tie=6,
                        )
                    )
                    _mark(EVENT_MATCHUP_CHANGED, gid, frame)
                    last_matchup_frame[gid] = frame
                    matchup_current[gid] = nearest_id
                elif matchup_run[gid] >= matchup_persist:
                    matchup_current[gid] = nearest_id
            else:
                matchup_candidate[gid] = None
                matchup_run[gid] = 0

            # --- Pressure start / end ---
            under_pressure = (
                nearest_id is not None
                and nearest_dist is not None
                and nearest_dist <= pressure_max
            )
            if under_pressure:
                pressure_run[gid] = pressure_run.get(gid, 0) + 1
                clear_run[gid] = 0
                pressure_defender[gid] = nearest_id
                was_close_dist[gid] = float(nearest_dist)
                if (
                    not pressure_active.get(gid)
                    and pressure_run[gid] >= pressure_persist
                    and _can_emit(EVENT_PRESSURE_STARTED, gid, frame)
                ):
                    events.append(
                        _raw_event(
                            event_type=EVENT_PRESSURE_STARTED,
                            start_frame=frame,
                            end_frame=frame,
                            start_time=t,
                            end_time=t,
                            confidence=0.68,
                            person_group_ids=[gid, int(nearest_id)],
                            segment_id=seg,
                            details={
                                "label": "possible_close_defensive_pressure",
                                "defender_person_group_id": int(nearest_id),
                                "screen_space_separation": round(float(nearest_dist), 6),
                                "pressure_distance_max": pressure_max,
                                "persist_frames": pressure_persist,
                            },
                        )
                    )
                    _mark(EVENT_PRESSURE_STARTED, gid, frame)
                    pressure_active[gid] = True
                    pressure_start_meta[gid] = {
                        "frame": frame,
                        "time": t,
                        "defender": int(nearest_id),
                    }
            else:
                pressure_run[gid] = 0
                cleared = (
                    nearest_dist is None
                    or nearest_dist >= pressure_clear
                )
                if pressure_active.get(gid) and cleared:
                    clear_run[gid] = clear_run.get(gid, 0) + 1
                    if (
                        clear_run[gid] >= clear_persist
                        and _can_emit(EVENT_PRESSURE_ENDED, gid, frame)
                    ):
                        def_id = pressure_defender.get(gid)
                        ids = [gid] + ([int(def_id)] if def_id is not None else [])
                        events.append(
                            _raw_event(
                                event_type=EVENT_PRESSURE_ENDED,
                                start_frame=frame,
                                end_frame=frame,
                                start_time=t,
                                end_time=t,
                                confidence=0.66,
                                person_group_ids=ids,
                                segment_id=seg,
                                details={
                                    "label": "possible_pressure_release",
                                    "defender_person_group_id": def_id,
                                    "screen_space_separation": (
                                        round(float(nearest_dist), 6)
                                        if nearest_dist is not None
                                        else None
                                    ),
                                    "pressure_clear_distance": pressure_clear,
                                    "clear_persist_frames": clear_persist,
                                },
                            )
                        )
                        _mark(EVENT_PRESSURE_ENDED, gid, frame)
                        pressure_active[gid] = False
                        pressure_start_meta.pop(gid, None)
                else:
                    clear_run[gid] = 0

            # --- Separation after being close ---
            prior_close = was_close_dist.get(gid)
            if (
                prior_close is not None
                and prior_close <= sep_prior
                and nearest_dist is not None
                and nearest_id is not None
                and nearest_dist >= sep_min
                and _can_emit(EVENT_SEPARATION, gid, frame)
            ):
                events.append(
                    _raw_event(
                        event_type=EVENT_SEPARATION,
                        start_frame=frame,
                        end_frame=frame,
                        start_time=t,
                        end_time=t,
                        confidence=0.64,
                        person_group_ids=[gid, int(nearest_id)],
                        segment_id=seg,
                        details={
                            "label": "screen_space_separation_threshold_exceeded",
                            "opponent_person_group_id": int(nearest_id),
                            "screen_space_separation": round(float(nearest_dist), 6),
                            "prior_screen_space_separation": round(float(prior_close), 6),
                            "separation_distance_min": sep_min,
                            "separation_prior_max": sep_prior,
                        },
                        sort_tie=5,
                    )
                )
                _mark(EVENT_SEPARATION, gid, frame)
                was_close_dist.pop(gid, None)

            # --- Double team ---
            close_opps = _opponents_within(player, players, double_max)
            if len(close_opps) >= 2:
                double_run[gid] = double_run.get(gid, 0) + 1
                if (
                    double_run[gid] >= double_persist
                    and _can_emit(EVENT_DOUBLE_TEAM, gid, frame)
                ):
                    opp_ids = [oid for oid, _ in close_opps[:3]]
                    events.append(
                        _raw_event(
                            event_type=EVENT_DOUBLE_TEAM,
                            start_frame=frame,
                            end_frame=frame,
                            start_time=t,
                            end_time=t,
                            confidence=0.58,
                            person_group_ids=[gid, *opp_ids],
                            segment_id=seg,
                            details={
                                "label": "possible_double_team",
                                "defender_person_group_ids": opp_ids,
                                "screen_space_separations": [
                                    round(d, 6) for _, d in close_opps[:3]
                                ],
                                "double_team_distance_max": double_max,
                                "persist_frames": double_persist,
                            },
                            sort_tie=5,
                        )
                    )
                    _mark(EVENT_DOUBLE_TEAM, gid, frame)
                    double_run[gid] = 0  # require a fresh run before re-fire
            else:
                double_run[gid] = 0

            # --- Help defense: already pressured + second defender nearby ---
            help_opps = _opponents_within(player, players, help_max)
            if pressure_active.get(gid) and len(help_opps) >= 2:
                help_run[gid] = help_run.get(gid, 0) + 1
                if (
                    help_run[gid] >= help_persist
                    and _can_emit(EVENT_HELP_DEFENSE, gid, frame)
                ):
                    primary = pressure_defender.get(gid)
                    helpers = [oid for oid, _ in help_opps if oid != primary][:2]
                    ids = [gid]
                    if primary is not None:
                        ids.append(int(primary))
                    ids.extend(helpers)
                    events.append(
                        _raw_event(
                            event_type=EVENT_HELP_DEFENSE,
                            start_frame=frame,
                            end_frame=frame,
                            start_time=t,
                            end_time=t,
                            confidence=0.55,
                            person_group_ids=ids,
                            segment_id=seg,
                            details={
                                "label": "possible_help_defense",
                                "primary_defender_person_group_id": primary,
                                "help_defender_person_group_ids": helpers,
                                "screen_space_separations": [
                                    round(d, 6) for _, d in help_opps[:3]
                                ],
                                "help_defense_distance_max": help_max,
                                "persist_frames": help_persist,
                            },
                            sort_tie=5,
                        )
                    )
                    _mark(EVENT_HELP_DEFENSE, gid, frame)
                    help_run[gid] = 0
            else:
                help_run[gid] = 0

    logger.info("Detected %s defense events", len(events))
    return events
