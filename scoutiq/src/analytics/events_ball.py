"""Cautious ball-related probable events from existing ball / hoop artifacts (Phase 12).

Soft-fails when ball tracks are missing. Never claims made/missed shots.
Distances are normalized screen-space units.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from src.analytics.trajectories import COORDINATE_SPACE

logger = logging.getLogger(__name__)

EVENT_POSSESSION_CHANGE = "probable_possession_change"
EVENT_BALL_NEAR_PLAYER = "ball_near_player"
EVENT_TOWARD_HOOP = "ball_moved_toward_hoop"
EVENT_SHOT_ATTEMPT = "possible_shot_attempt"

BALL_EVENTS = {
    EVENT_POSSESSION_CHANGE,
    EVENT_BALL_NEAR_PLAYER,
    EVENT_TOWARD_HOOP,
    EVENT_SHOT_ATTEMPT,
}

DEFAULT_BALL_CONFIG: dict[str, Any] = {
    "near_player_distance_max": 0.10,
    "near_player_persist_frames": 3,
    "near_player_cooldown_frames": 25,
    "possession_change_persist_frames": 5,
    "possession_change_cooldown_frames": 20,
    "toward_hoop_min_progress": 0.04,
    "toward_hoop_window_frames": 8,
    "toward_hoop_cooldown_frames": 25,
    "shot_hoop_distance_max": 0.18,
    "shot_toward_progress_min": 0.03,
    "shot_not_held_distance_min": 0.08,
    "shot_cooldown_frames": 40,
    "max_events_per_type": 40,
    "analytics_safe_only": True,
}


def merge_ball_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_BALL_CONFIG)
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


def _bbox_center_norm(
    bbox: Sequence[Any] | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    w = max(int(frame_width), 1)
    h = max(int(frame_height), 1)
    return ((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h


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
    sort_tie: int = 7,
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
                "Probable/possible screen-space ball event only — "
                "not official play-by-play; does not claim made/missed shots."
            ),
            **details,
        },
    }


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _usable_frame_lookup(
    segments: Sequence[dict[str, Any]] | None,
) -> dict[int, dict[str, Any]]:
    """frame -> segment meta for analytics-safe wide gameplay frames."""
    lookup: dict[int, dict[str, Any]] = {}
    if not segments:
        return lookup
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if not seg.get("usable_for_analytics", True):
            continue
        state = str(seg.get("camera_state") or "wide_gameplay")
        if state != "wide_gameplay":
            continue
        start = _as_int(seg.get("start_frame"))
        end = _as_int(seg.get("end_frame"))
        if start is None or end is None:
            continue
        meta = {
            "segment_id": _as_int(seg.get("segment_id")),
            "camera_state": state,
        }
        for frame in range(start, end + 1):
            lookup[frame] = meta
    return lookup


def _player_positions_by_frame(
    trajectories: Sequence[dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for row in trajectories:
        if not isinstance(row, dict):
            continue
        gid = _as_int(row.get("person_group_id"))
        if gid is None:
            continue
        team_id = _as_int(row.get("team_id"))
        fragments = row.get("path_fragments") or []
        if not isinstance(fragments, list):
            continue
        for frag in fragments:
            if not isinstance(frag, dict) or not frag.get("usable_for_analytics", True):
                continue
            for p in frag.get("points") or []:
                if not isinstance(p, dict) or p.get("usable_for_analytics") is False:
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
                by_frame.setdefault(frame, {})[gid] = {
                    "person_group_id": gid,
                    "team_id": team_id,
                    "x_normalized": x,
                    "y_normalized": y,
                    "time_seconds": t,
                    "segment_id": _as_int(p.get("segment_id")),
                }
    return by_frame


def _pick_ball(frame_row: dict[str, Any]) -> dict[str, Any] | None:
    balls = frame_row.get("balls") or []
    if not isinstance(balls, list) or not balls:
        return None
    return max(balls, key=lambda b: float(b.get("confidence") or 0.0))


def _hoop_center(
    hoops: Sequence[dict[str, Any]] | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float] | None:
    if not hoops:
        return None
    best = max(hoops, key=lambda h: float(h.get("confidence") or 0.0))
    return _bbox_center_norm(
        best.get("bbox_xyxy"), frame_width=frame_width, frame_height=frame_height
    )


def detect_ball_events(
    *,
    ball_frames: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]] | None = None,
    segments: Sequence[dict[str, Any]] | None = None,
    hoop_by_frame: dict[int, list[dict[str, Any]]] | None = None,
    possession_frames: Sequence[dict[str, Any]] | None = None,
    frame_width: int = 1280,
    frame_height: int = 720,
    fps: float | None = 30.0,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return raw timeline events (with ``_sort``). Empty if no ball frames."""
    cfg = merge_ball_config(config)
    if not ball_frames:
        return []

    near_max = float(cfg["near_player_distance_max"])
    near_persist = max(1, int(cfg["near_player_persist_frames"]))
    near_cooldown = max(1, int(cfg["near_player_cooldown_frames"]))
    poss_persist = max(1, int(cfg["possession_change_persist_frames"]))
    poss_cooldown = max(1, int(cfg["possession_change_cooldown_frames"]))
    toward_progress = float(cfg["toward_hoop_min_progress"])
    toward_window = max(2, int(cfg["toward_hoop_window_frames"]))
    toward_cooldown = max(1, int(cfg["toward_hoop_cooldown_frames"]))
    shot_hoop_max = float(cfg["shot_hoop_distance_max"])
    shot_progress = float(cfg["shot_toward_progress_min"])
    shot_not_held = float(cfg["shot_not_held_distance_min"])
    shot_cooldown = max(1, int(cfg["shot_cooldown_frames"]))
    max_per_type = max(1, int(cfg["max_events_per_type"]))
    analytics_safe_only = bool(cfg.get("analytics_safe_only", True))

    usable = _usable_frame_lookup(segments) if analytics_safe_only else {}
    players_by_frame = _player_positions_by_frame(trajectories or [])
    hoop_by_frame = hoop_by_frame or {}
    restrict_to_usable = bool(analytics_safe_only and segments is not None)

    poss_by_frame: dict[int, dict[str, Any]] = {}
    for row in possession_frames or []:
        if not isinstance(row, dict):
            continue
        frame = _as_int(row.get("frame_number") or row.get("frame"))
        if frame is not None:
            poss_by_frame[frame] = row

    events: list[dict[str, Any]] = []
    counts: dict[str, int] = {t: 0 for t in BALL_EVENTS}
    last_emit: dict[str, int] = {t: -10_000 for t in BALL_EVENTS}

    # Ball track history for toward-hoop / shot
    ball_hist: list[dict[str, Any]] = []
    near_run: dict[int, int] = {}
    # Possession change state
    held_team_candidate: int | None = None
    held_team_run = 0
    stable_held_team: int | None = None
    held_player_candidate: int | None = None

    def _frame_time(frame: int, fallback: float | None) -> float:
        if fallback is not None:
            return float(fallback)
        if fps and fps > 0:
            return float(frame) / float(fps)
        return 0.0

    def _can(event_type: str, frame: int) -> bool:
        return (
            counts[event_type] < max_per_type
            and frame - last_emit[event_type] >= (
                near_cooldown
                if event_type == EVENT_BALL_NEAR_PLAYER
                else poss_cooldown
                if event_type == EVENT_POSSESSION_CHANGE
                else toward_cooldown
                if event_type == EVENT_TOWARD_HOOP
                else shot_cooldown
            )
        )

    def _mark(event_type: str, frame: int) -> None:
        counts[event_type] += 1
        last_emit[event_type] = frame

    for row in ball_frames:
        if not isinstance(row, dict):
            continue
        frame = _as_int(row.get("frame_number") or row.get("frame"))
        if frame is None:
            continue
        if restrict_to_usable and frame not in usable:
            continue

        seg_meta = usable.get(frame) if usable else None
        segment_id = seg_meta.get("segment_id") if seg_meta else None
        t = _as_float(row.get("timestamp_seconds"))
        t = _frame_time(frame, t)

        ball = _pick_ball(row)
        if ball is None:
            near_run.clear()
            continue

        ball_xy = _bbox_center_norm(
            ball.get("bbox_xyxy"),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if ball_xy is None:
            continue
        ball_conf = float(ball.get("confidence") or 0.4)
        bx, by = ball_xy

        players = players_by_frame.get(frame) or {}
        # Nearest player to ball
        nearest_gid: int | None = None
        nearest_dist: float | None = None
        nearest_team: int | None = None
        for gid, p in players.items():
            dist = _norm_dist(bx, by, p["x_normalized"], p["y_normalized"])
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_gid = gid
                nearest_team = p.get("team_id")
                if segment_id is None:
                    segment_id = p.get("segment_id")

        # Prefer possession_frames when available for held team/player
        poss = poss_by_frame.get(frame)
        held_team: int | None = None
        held_player: int | None = None
        held_dist: float | None = nearest_dist
        if poss and poss.get("state") == "held":
            held_team = _as_int(poss.get("team_id"))
            held_player = _as_int(poss.get("person_group_id"))
            held_dist = _as_float(poss.get("distance_norm")) or nearest_dist
        elif nearest_dist is not None and nearest_dist <= near_max:
            held_team = nearest_team
            held_player = nearest_gid

        # --- ball_near_player ---
        if (
            nearest_gid is not None
            and nearest_dist is not None
            and nearest_dist <= near_max
        ):
            near_run[nearest_gid] = near_run.get(nearest_gid, 0) + 1
            for gid in list(near_run):
                if gid != nearest_gid:
                    near_run[gid] = 0
            if (
                near_run[nearest_gid] >= near_persist
                and _can(EVENT_BALL_NEAR_PLAYER, frame)
            ):
                events.append(
                    _raw_event(
                        event_type=EVENT_BALL_NEAR_PLAYER,
                        start_frame=frame,
                        end_frame=frame,
                        start_time=t,
                        end_time=t,
                        confidence=min(0.85, 0.4 + 0.45 * ball_conf),
                        person_group_ids=[nearest_gid],
                        segment_id=segment_id,
                        details={
                            "label": "ball_near_player",
                            "screen_space_separation": round(float(nearest_dist), 6),
                            "near_player_distance_max": near_max,
                            "ball_confidence": round(ball_conf, 4),
                        },
                    )
                )
                _mark(EVENT_BALL_NEAR_PLAYER, frame)
                near_run[nearest_gid] = 0
        else:
            near_run.clear()

        # --- probable_possession_change ---
        if held_team is not None:
            if held_team == held_team_candidate:
                held_team_run += 1
            else:
                held_team_candidate = held_team
                held_player_candidate = held_player
                held_team_run = 1
            if held_team_run >= poss_persist:
                if (
                    stable_held_team is not None
                    and held_team != stable_held_team
                    and _can(EVENT_POSSESSION_CHANGE, frame)
                ):
                    ids = []
                    if held_player_candidate is not None:
                        ids.append(int(held_player_candidate))
                    events.append(
                        _raw_event(
                            event_type=EVENT_POSSESSION_CHANGE,
                            start_frame=frame,
                            end_frame=frame,
                            start_time=t,
                            end_time=t,
                            confidence=min(0.8, 0.45 + 0.4 * ball_conf),
                            person_group_ids=ids,
                            segment_id=segment_id,
                            details={
                                "label": "probable_possession_change",
                                "from_team_id": stable_held_team,
                                "to_team_id": held_team,
                                "screen_space_separation": (
                                    round(float(held_dist), 6)
                                    if held_dist is not None
                                    else None
                                ),
                                "persist_frames": poss_persist,
                            },
                        )
                    )
                    _mark(EVENT_POSSESSION_CHANGE, frame)
                stable_held_team = held_team
        else:
            held_team_candidate = None
            held_team_run = 0

        # --- toward hoop / possible shot ---
        hoop_xy = _hoop_center(
            hoop_by_frame.get(frame),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        # Carry last hoop from history if current frame missing
        if hoop_xy is None and ball_hist:
            hoop_xy = ball_hist[-1].get("hoop_xy")

        hoop_dist = (
            _norm_dist(bx, by, hoop_xy[0], hoop_xy[1]) if hoop_xy is not None else None
        )
        ball_hist.append(
            {
                "frame": frame,
                "time": t,
                "xy": (bx, by),
                "hoop_xy": hoop_xy,
                "hoop_dist": hoop_dist,
                "nearest_dist": nearest_dist,
                "segment_id": segment_id,
                "ball_conf": ball_conf,
            }
        )
        if len(ball_hist) > toward_window + 2:
            ball_hist = ball_hist[-(toward_window + 2) :]

        if len(ball_hist) >= toward_window and hoop_dist is not None:
            older = ball_hist[-toward_window]
            if older.get("hoop_dist") is not None:
                progress = float(older["hoop_dist"]) - float(hoop_dist)
                if progress >= toward_progress and _can(EVENT_TOWARD_HOOP, frame):
                    events.append(
                        _raw_event(
                            event_type=EVENT_TOWARD_HOOP,
                            start_frame=int(older["frame"]),
                            end_frame=frame,
                            start_time=float(older["time"]),
                            end_time=t,
                            confidence=min(0.78, 0.4 + 0.35 * ball_conf + progress),
                            person_group_ids=(
                                [nearest_gid] if nearest_gid is not None else []
                            ),
                            segment_id=segment_id,
                            details={
                                "label": "ball_moved_toward_hoop",
                                "screen_space_separation_to_hoop": round(hoop_dist, 6),
                                "prior_separation_to_hoop": round(
                                    float(older["hoop_dist"]), 6
                                ),
                                "progress_norm": round(progress, 6),
                                "toward_hoop_min_progress": toward_progress,
                            },
                        )
                    )
                    _mark(EVENT_TOWARD_HOOP, frame)

                # possible shot: near hoop, progressing toward it, not glued to player
                not_held = (
                    nearest_dist is None or nearest_dist >= shot_not_held
                )
                if (
                    hoop_dist <= shot_hoop_max
                    and progress >= shot_progress
                    and not_held
                    and _can(EVENT_SHOT_ATTEMPT, frame)
                ):
                    events.append(
                        _raw_event(
                            event_type=EVENT_SHOT_ATTEMPT,
                            start_frame=int(older["frame"]),
                            end_frame=frame,
                            start_time=float(older["time"]),
                            end_time=t,
                            confidence=min(0.7, 0.35 + 0.35 * ball_conf + progress),
                            person_group_ids=(
                                [nearest_gid] if nearest_gid is not None else []
                            ),
                            segment_id=segment_id,
                            details={
                                "label": "possible_shot_attempt",
                                "screen_space_separation_to_hoop": round(hoop_dist, 6),
                                "progress_norm": round(progress, 6),
                                "player_ball_separation": (
                                    round(float(nearest_dist), 6)
                                    if nearest_dist is not None
                                    else None
                                ),
                                "outcome": "unknown",
                                "claim": "does_not_assert_make_or_miss",
                            },
                        )
                    )
                    _mark(EVENT_SHOT_ATTEMPT, frame)

    logger.info("Detected %s ball events", len(events))
    return events


def detect_ball_events_for_job(
    job_dir: str | Path,
    *,
    trajectories: Sequence[dict[str, Any]] | None = None,
    segments: Sequence[dict[str, Any]] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    fps: float | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load ball/hoop/possession artifacts from a job dir; soft-fail to []."""
    job_dir = Path(job_dir).resolve()
    ball_path = job_dir / "ball_tracks.jsonl.gz"
    if not ball_path.is_file():
        alt = job_dir / "ball_tracks.jsonl"
        ball_path = alt if alt.is_file() else ball_path
    ball_frames = _load_jsonl_gz(ball_path)
    if not ball_frames:
        logger.info("No ball tracks for %s — skipping ball events", job_dir.name)
        return []

    hoop_by_frame: dict[int, list[dict[str, Any]]] = {}
    for hoop_name in ("hoop_tracks.jsonl.gz", "hoop_tracks.jsonl"):
        hoop_path = job_dir / hoop_name
        if not hoop_path.is_file():
            continue
        for row in _load_jsonl_gz(hoop_path):
            frame = _as_int(row.get("frame_number"))
            if frame is not None:
                hoop_by_frame[frame] = list(row.get("hoops") or [])
        break

    possession_frames = _load_jsonl_gz(job_dir / "possession_frames.jsonl.gz")
    if not possession_frames:
        possession_frames = _load_jsonl_gz(job_dir / "possession_frames.jsonl")

    width = frame_width
    height = frame_height
    if not width or not height:
        summary_path = job_dir / "ball_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                width = width or _as_int(summary.get("frame_width"))
                height = height or _as_int(summary.get("frame_height"))
                if fps is None:
                    fps = _as_float(summary.get("fps"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    width = int(width or 1280)
    height = int(height or 720)

    return detect_ball_events(
        ball_frames=ball_frames,
        trajectories=trajectories,
        segments=segments,
        hoop_by_frame=hoop_by_frame,
        possession_frames=possession_frames,
        frame_width=width,
        frame_height=height,
        fps=fps,
        config=config,
    )
