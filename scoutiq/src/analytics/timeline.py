"""Job-local game timeline from segments / quality / screen-space events.

Phase 9–11: camera, tracking, movement, and cautious defensive proximity events.
No basketball shot/pass inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.analytics.events_ball import BALL_EVENTS
from src.analytics.events_defense import DEFENSE_EVENTS
from src.analytics.events_movement import MOVEMENT_EVENTS
from src.analytics.trajectories import COORDINATE_SPACE, load_segments_payload
from src.pipeline.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)

TIMELINE_FILENAME = "game_timeline.json"

EVENT_GAMEPLAY_STARTED = "gameplay_segment_started"
EVENT_GAMEPLAY_ENDED = "gameplay_segment_ended"
EVENT_ORIENTATION_CHANGED = "camera_orientation_changed"
EVENT_CLOSEUP_STARTED = "closeup_started"
EVENT_CLOSEUP_ENDED = "closeup_ended"
EVENT_GAMEPLAY_RESUMED = "gameplay_resumed"
EVENT_QUALITY_DROPPED = "tracking_quality_dropped"
EVENT_QUALITY_RECOVERED = "tracking_quality_recovered"

CAMERA_EVENTS = {
    EVENT_GAMEPLAY_STARTED,
    EVENT_GAMEPLAY_ENDED,
    EVENT_ORIENTATION_CHANGED,
    EVENT_CLOSEUP_STARTED,
    EVENT_CLOSEUP_ENDED,
    EVENT_GAMEPLAY_RESUMED,
}
TRACKING_EVENTS = {
    EVENT_QUALITY_DROPPED,
    EVENT_QUALITY_RECOVERED,
}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def make_event(
    *,
    event_id: str,
    event_type: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    confidence: float,
    person_group_ids: Sequence[int] | None = None,
    segment_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = float(max(0.0, min(1.0, confidence)))
    return {
        "event_id": event_id,
        "event_type": event_type,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": round(float(start_time), 6),
        "end_time": round(float(end_time), 6),
        "confidence": round(conf, 4),
        "confidence_label": confidence_label(conf),
        "person_group_ids": [int(x) for x in (person_group_ids or [])],
        "segment_id": segment_id,
        "details": details or {},
    }


def _frame_time(frame: int, frame_times: dict[int, float], fps: float | None) -> float:
    if frame in frame_times:
        return float(frame_times[frame])
    if fps and fps > 0:
        return float(frame) / float(fps)
    return 0.0


def events_from_segments(
    segments: Sequence[dict[str, Any]],
    *,
    base_confidence: float = 0.9,
) -> list[dict[str, Any]]:
    """Emit gameplay / closeup / resume events from camera segments."""
    events: list[dict[str, Any]] = []
    ordered = sorted(
        [s for s in segments if isinstance(s, dict)],
        key=lambda s: (_as_int(s.get("start_frame")) or 0),
    )

    prev_state: str | None = None
    for seg in ordered:
        sid = _as_int(seg.get("segment_id"))
        state = str(seg.get("camera_state") or "unknown")
        start_f = _as_int(seg.get("start_frame"))
        end_f = _as_int(seg.get("end_frame"))
        start_t = _as_float(seg.get("start_time")) or 0.0
        end_t = _as_float(seg.get("end_time")) or start_t
        if start_f is None or end_f is None:
            continue

        details = {
            "camera_state": state,
            "hoop_side": seg.get("hoop_side"),
            "usable_for_analytics": bool(seg.get("usable_for_analytics")),
        }

        if state == "wide_gameplay":
            events.append(
                {
                    "_sort": (start_f, 0),
                    "event_type": EVENT_GAMEPLAY_STARTED,
                    "start_frame": start_f,
                    "end_frame": start_f,
                    "start_time": start_t,
                    "end_time": start_t,
                    "confidence": base_confidence,
                    "segment_id": sid,
                    "details": details,
                }
            )
            events.append(
                {
                    "_sort": (end_f, 2),
                    "event_type": EVENT_GAMEPLAY_ENDED,
                    "start_frame": end_f,
                    "end_frame": end_f,
                    "start_time": end_t,
                    "end_time": end_t,
                    "confidence": base_confidence,
                    "segment_id": sid,
                    "details": details,
                }
            )
            if prev_state in {"closeup", "transition"}:
                events.append(
                    {
                        "_sort": (start_f, 1),
                        "event_type": EVENT_GAMEPLAY_RESUMED,
                        "start_frame": start_f,
                        "end_frame": start_f,
                        "start_time": start_t,
                        "end_time": start_t,
                        "confidence": base_confidence,
                        "segment_id": sid,
                        "details": {
                            **details,
                            "resumed_from": prev_state,
                        },
                    }
                )

        elif state == "closeup":
            events.append(
                {
                    "_sort": (start_f, 0),
                    "event_type": EVENT_CLOSEUP_STARTED,
                    "start_frame": start_f,
                    "end_frame": start_f,
                    "start_time": start_t,
                    "end_time": start_t,
                    "confidence": base_confidence,
                    "segment_id": sid,
                    "details": details,
                }
            )
            events.append(
                {
                    "_sort": (end_f, 2),
                    "event_type": EVENT_CLOSEUP_ENDED,
                    "start_frame": end_f,
                    "end_frame": end_f,
                    "start_time": end_t,
                    "end_time": end_t,
                    "confidence": base_confidence,
                    "segment_id": sid,
                    "details": details,
                }
            )

        prev_state = state

    return events


def events_from_orientation_flips(
    flips: Sequence[dict[str, Any]],
    *,
    frame_times: dict[int, float],
    fps: float | None,
    base_confidence: float = 0.92,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for flip in flips:
        if not isinstance(flip, dict):
            continue
        frame = _as_int(flip.get("confirmed_frame") or flip.get("transition_end_frame"))
        if frame is None:
            continue
        t = _frame_time(frame, frame_times, fps)
        events.append(
            {
                "_sort": (frame, 1),
                "event_type": EVENT_ORIENTATION_CHANGED,
                "start_frame": frame,
                "end_frame": frame,
                "start_time": t,
                "end_time": t,
                "confidence": base_confidence,
                "segment_id": None,
                "details": {
                    "from_side": flip.get("from_side"),
                    "to_side": flip.get("to_side"),
                    "transition_start_frame": flip.get("transition_start_frame"),
                    "transition_end_frame": flip.get("transition_end_frame"),
                    "coordinate_space": COORDINATE_SPACE,
                },
            }
        )
    return events


def events_from_tracking_quality(
    *,
    cut_frames: Sequence[int],
    frame_times: dict[int, float],
    fps: float | None,
    recovery_pad_frames: int = 3,
    base_confidence: float = 0.8,
) -> list[dict[str, Any]]:
    """Map hard cuts to tracking quality drop / recover pairs."""
    events: list[dict[str, Any]] = []
    pad = max(0, int(recovery_pad_frames))
    for cut in sorted({int(c) for c in cut_frames}):
        t0 = _frame_time(cut, frame_times, fps)
        recover_frame = cut + pad
        t1 = _frame_time(recover_frame, frame_times, fps)
        events.append(
            {
                "_sort": (cut, 0),
                "event_type": EVENT_QUALITY_DROPPED,
                "start_frame": cut,
                "end_frame": cut,
                "start_time": t0,
                "end_time": t0,
                "confidence": base_confidence,
                "segment_id": None,
                "details": {
                    "reason": "hard_cut_or_layout_discontinuity",
                    "cut_frame": cut,
                },
            }
        )
        events.append(
            {
                "_sort": (recover_frame, 1),
                "event_type": EVENT_QUALITY_RECOVERED,
                "start_frame": recover_frame,
                "end_frame": recover_frame,
                "start_time": t1,
                "end_time": t1,
                "confidence": base_confidence * 0.95,
                "segment_id": None,
                "details": {
                    "reason": "post_cut_stabilization_window",
                    "after_cut_frame": cut,
                    "recovery_pad_frames": pad,
                },
            }
        )
    return events


def finalize_events(raw_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        raw_events,
        key=lambda e: e.get("_sort", (int(e.get("start_frame") or 0), 0)),
    )
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(ordered, start=1):
        out.append(
            make_event(
                event_id=f"event_{idx:03d}",
                event_type=str(raw["event_type"]),
                start_frame=int(raw["start_frame"]),
                end_frame=int(raw["end_frame"]),
                start_time=float(raw["start_time"]),
                end_time=float(raw["end_time"]),
                confidence=float(raw.get("confidence") or 0.5),
                person_group_ids=raw.get("person_group_ids") or [],
                segment_id=_as_int(raw.get("segment_id")),
                details=raw.get("details") or {},
            )
        )
    return out


def build_game_timeline(
    *,
    job_id: str,
    frame_width: int,
    frame_height: int,
    fps: float | None,
    segments_payload: dict[str, Any],
    frame_times: dict[int, float] | None = None,
    trajectories: list[dict[str, Any]] | None = None,
    movement_config: dict[str, Any] | None = None,
    defense_config: dict[str, Any] | None = None,
    ball_config: dict[str, Any] | None = None,
    ball_frames: list[dict[str, Any]] | None = None,
    hoop_by_frame: dict[int, list[dict[str, Any]]] | None = None,
    possession_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame_times = frame_times or {}
    segments = list(segments_payload.get("segments") or [])
    flips = list(segments_payload.get("hoop_orientation_flips") or [])
    cut_frames = list(segments_payload.get("cut_frames") or [])

    conf_scale = 0.75 if segments_payload.get("confidence") == "reduced" else 1.0
    raw: list[dict[str, Any]] = []
    raw.extend(
        events_from_segments(segments, base_confidence=0.9 * conf_scale)
    )
    raw.extend(
        events_from_orientation_flips(
            flips,
            frame_times=frame_times,
            fps=fps,
            base_confidence=0.92 * conf_scale,
        )
    )
    raw.extend(
        events_from_tracking_quality(
            cut_frames=cut_frames,
            frame_times=frame_times,
            fps=fps,
            base_confidence=0.8 * conf_scale,
        )
    )
    if trajectories:
        from src.analytics.events_defense import detect_defense_events
        from src.analytics.events_movement import detect_movement_events

        movement_raw = detect_movement_events(
            trajectories, config=movement_config
        )
        for item in movement_raw:
            item["confidence"] = float(item.get("confidence") or 0.5) * conf_scale
        raw.extend(movement_raw)

        defense_raw = detect_defense_events(
            trajectories, config=defense_config
        )
        for item in defense_raw:
            item["confidence"] = float(item.get("confidence") or 0.5) * conf_scale
        raw.extend(defense_raw)

    # Ball events soft-fail: missing/empty ball_frames → no ball events.
    if ball_frames:
        try:
            from src.analytics.events_ball import detect_ball_events

            ball_raw = detect_ball_events(
                ball_frames=ball_frames,
                trajectories=trajectories or [],
                segments=segments,
                hoop_by_frame=hoop_by_frame,
                possession_frames=possession_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=fps,
                config=ball_config,
            )
            for item in ball_raw:
                item["confidence"] = float(item.get("confidence") or 0.5) * conf_scale
            raw.extend(ball_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ball event detection skipped: %s", exc)

    events = finalize_events(raw)

    by_type: dict[str, int] = {}
    for event in events:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1

    return {
        "job_id": job_id,
        "coordinate_space": COORDINATE_SPACE,
        "video": {
            "fps": round(float(fps), 6) if fps else None,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
        },
        "segments": segments,
        "events": events,
        "event_types_included": sorted(
            CAMERA_EVENTS
            | TRACKING_EVENTS
            | MOVEMENT_EVENTS
            | DEFENSE_EVENTS
            | BALL_EVENTS
        ),
        "summary": {
            "event_count": len(events),
            "segment_count": len(segments),
            "events_by_type": by_type,
            "camera_event_count": sum(
                1 for e in events if e["event_type"] in CAMERA_EVENTS
            ),
            "tracking_event_count": sum(
                1 for e in events if e["event_type"] in TRACKING_EVENTS
            ),
            "movement_event_count": sum(
                1 for e in events if e["event_type"] in MOVEMENT_EVENTS
            ),
            "defense_event_count": sum(
                1 for e in events if e["event_type"] in DEFENSE_EVENTS
            ),
            "ball_event_count": sum(
                1 for e in events if e["event_type"] in BALL_EVENTS
            ),
        },
        "note": (
            "Camera, tracking, screen-space movement, cautious defensive "
            "proximity, and probable ball events. Not official play-by-play. "
            "Distances are normalized screen-space (possible/likely); shot "
            "events do not claim make or miss."
        ),
    }


def _load_frame_times_from_tracks(job_dir: Path) -> dict[int, float]:
    from src.analytics.trajectories import iter_track_frames

    tracks = job_dir / "tracks.jsonl.gz"
    if not tracks.is_file():
        alt = job_dir / "tracks.jsonl"
        if not alt.is_file():
            return {}
        tracks = alt
    times: dict[int, float] = {}
    for row in iter_track_frames(tracks):
        frame = _as_int(row.get("frame_number"))
        ts = _as_float(row.get("timestamp_seconds"))
        if frame is not None and ts is not None:
            times[frame] = ts
    return times


def generate_timeline_for_job(
    job_dir: str | Path,
    *,
    segments_payload: dict[str, Any] | None = None,
    movement_config: dict[str, Any] | None = None,
    defense_config: dict[str, Any] | None = None,
    ball_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``analytics/game_timeline.json`` for a job."""
    job_dir = Path(job_dir).resolve()
    if segments_payload is None:
        segments_payload = load_segments_payload(
            job_dir / "analytics" / "game_segments.json"
        )
    if segments_payload is None:
        raise FileNotFoundError(
            f"game_segments.json missing for job {job_dir.name}; generate segments first."
        )

    width = _as_int(segments_payload.get("frame_width"))
    height = _as_int(segments_payload.get("frame_height"))
    if not width or not height:
        snapshot = job_dir / "snapshot.json"
        if snapshot.is_file():
            snap = json.loads(snapshot.read_text(encoding="utf-8"))
            width = width or _as_int(snap.get("frame_width"))
            height = height or _as_int(snap.get("frame_height"))
    if not width or not height:
        raise ValueError("frame_width/frame_height unavailable for timeline")

    if movement_config is None or defense_config is None or ball_config is None:
        try:
            from src.pipeline.player_track_pipeline import load_pipeline_config

            analytics_cfg = load_pipeline_config().get("analytics") or {}
            if movement_config is None:
                movement_config = analytics_cfg.get("movement_events") or {}
            if defense_config is None:
                defense_config = analytics_cfg.get("defense_events") or {}
            if ball_config is None:
                ball_config = analytics_cfg.get("ball_events") or {}
        except Exception:  # noqa: BLE001
            if movement_config is None:
                movement_config = {}
            if defense_config is None:
                defense_config = {}
            if ball_config is None:
                ball_config = {}

    trajectories: list[dict[str, Any]] = []
    try:
        from src.analytics.heatmap_density import load_trajectory_rows

        analytics_path = job_dir / "analytics" / "trajectories_analytics.jsonl.gz"
        raw_path = job_dir / "analytics" / "trajectories.jsonl.gz"
        if analytics_path.is_file():
            trajectories = load_trajectory_rows(analytics_path)
        elif raw_path.is_file():
            trajectories = load_trajectory_rows(raw_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load trajectories for timeline events: %s", exc)

    ball_frames: list[dict[str, Any]] = []
    hoop_by_frame: dict[int, list[dict[str, Any]]] = {}
    possession_frames: list[dict[str, Any]] = []
    try:
        from src.analytics.events_ball import _load_jsonl_gz

        for name in ("ball_tracks.jsonl.gz", "ball_tracks.jsonl"):
            path = job_dir / name
            if path.is_file():
                ball_frames = _load_jsonl_gz(path)
                break
        for name in ("hoop_tracks.jsonl.gz", "hoop_tracks.jsonl"):
            path = job_dir / name
            if path.is_file():
                for row in _load_jsonl_gz(path):
                    frame = _as_int(row.get("frame_number"))
                    if frame is not None:
                        hoop_by_frame[frame] = list(row.get("hoops") or [])
                break
        for name in ("possession_frames.jsonl.gz", "possession_frames.jsonl"):
            path = job_dir / name
            if path.is_file():
                possession_frames = _load_jsonl_gz(path)
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load ball artifacts for timeline: %s", exc)
        ball_frames = []

    fps = _as_float(segments_payload.get("fps_estimate"))
    frame_times = _load_frame_times_from_tracks(job_dir)
    timeline = build_game_timeline(
        job_id=job_dir.name,
        frame_width=int(width),
        frame_height=int(height),
        fps=fps,
        segments_payload=segments_payload,
        frame_times=frame_times,
        trajectories=trajectories,
        movement_config=movement_config,
        defense_config=defense_config,
        ball_config=ball_config,
        ball_frames=ball_frames or None,
        hoop_by_frame=hoop_by_frame or None,
        possession_frames=possession_frames or None,
    )
    out_path = job_dir / "analytics" / TIMELINE_FILENAME
    write_json_atomic(out_path, timeline)
    logger.info(
        "Wrote timeline with %s events → %s",
        timeline["summary"]["event_count"],
        out_path,
    )
    return timeline
