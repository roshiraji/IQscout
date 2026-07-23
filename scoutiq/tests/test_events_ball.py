"""Tests for Phase 12 probable ball events."""

from __future__ import annotations

from src.analytics.events_ball import (
    EVENT_BALL_NEAR_PLAYER,
    EVENT_POSSESSION_CHANGE,
    EVENT_SHOT_ATTEMPT,
    EVENT_TOWARD_HOOP,
    detect_ball_events,
    merge_ball_config,
)
from src.analytics.timeline import build_game_timeline


def _bbox(cx, cy, size=20):
    return [cx - size, cy - size, cx + size, cy + size]


def _ball_frame(frame, t, cx, cy, conf=0.8):
    return {
        "frame_number": frame,
        "timestamp_seconds": t,
        "balls": [{"bbox_xyxy": _bbox(cx, cy), "confidence": conf, "track_id": 1}],
    }


def _player(gid, team_id, points):
    return {
        "person_group_id": gid,
        "team_id": team_id,
        "path_fragments": [
            {
                "usable_for_analytics": True,
                "segment_id": 1,
                "points": points,
            }
        ],
    }


def _pt(frame, t, x, y):
    return {
        "frame": frame,
        "time_seconds": t,
        "x_normalized": x,
        "y_normalized": y,
        "segment_id": 1,
        "usable_for_analytics": True,
        "camera_state": "wide_gameplay",
    }


def _cfg(**overrides):
    base = {
        "near_player_distance_max": 0.12,
        "near_player_persist_frames": 2,
        "near_player_cooldown_frames": 1,
        "possession_change_persist_frames": 2,
        "possession_change_cooldown_frames": 1,
        "toward_hoop_min_progress": 0.03,
        "toward_hoop_window_frames": 3,
        "toward_hoop_cooldown_frames": 1,
        "shot_hoop_distance_max": 0.25,
        "shot_toward_progress_min": 0.02,
        "shot_not_held_distance_min": 0.08,
        "shot_cooldown_frames": 1,
        "max_events_per_type": 20,
        "analytics_safe_only": False,
    }
    base.update(overrides)
    return merge_ball_config(base)


def test_ball_near_player_and_possession_change():
    # Frame size 100x100 so norm coords are easy: player at (0.5,0.5)=50,50
    w, h = 100, 100
    ball_frames = [
        _ball_frame(0, 0.0, 20, 50),
        _ball_frame(1, 0.1, 48, 50),
        _ball_frame(2, 0.2, 50, 50),
        _ball_frame(3, 0.3, 50, 50),
        # switch to other side of court near team 1 player
        _ball_frame(4, 0.4, 80, 50),
        _ball_frame(5, 0.5, 82, 50),
        _ball_frame(6, 0.6, 85, 50),
    ]
    traj = [
        _player(1, 0, [_pt(i, i / 10.0, 0.50, 0.50) for i in range(0, 7)]),
        _player(2, 1, [_pt(i, i / 10.0, 0.85, 0.50) for i in range(0, 7)]),
    ]
    possession = [
        {"frame_number": 1, "state": "held", "team_id": 0, "person_group_id": 1, "distance_norm": 0.05},
        {"frame_number": 2, "state": "held", "team_id": 0, "person_group_id": 1, "distance_norm": 0.02},
        {"frame_number": 3, "state": "held", "team_id": 0, "person_group_id": 1, "distance_norm": 0.02},
        {"frame_number": 5, "state": "held", "team_id": 1, "person_group_id": 2, "distance_norm": 0.03},
        {"frame_number": 6, "state": "held", "team_id": 1, "person_group_id": 2, "distance_norm": 0.03},
    ]
    events = detect_ball_events(
        ball_frames=ball_frames,
        trajectories=traj,
        possession_frames=possession,
        frame_width=w,
        frame_height=h,
        config=_cfg(),
    )
    types = {e["event_type"] for e in events}
    assert EVENT_BALL_NEAR_PLAYER in types
    assert EVENT_POSSESSION_CHANGE in types
    change = next(e for e in events if e["event_type"] == EVENT_POSSESSION_CHANGE)
    assert change["details"]["from_team_id"] == 0
    assert change["details"]["to_team_id"] == 1
    assert "not claim" in change["details"]["note"].lower() or "not official" in change["details"]["note"].lower()


def test_toward_hoop_and_possible_shot():
    w, h = 100, 100
    # Hoop at top-center (50, 10); ball moves toward it from (50, 80)
    hoop_by_frame = {
        i: [{"bbox_xyxy": _bbox(50, 10, 8), "confidence": 0.9}] for i in range(0, 8)
    }
    ball_frames = [
        _ball_frame(0, 0.0, 50, 80),
        _ball_frame(1, 0.1, 50, 70),
        _ball_frame(2, 0.2, 50, 55),
        _ball_frame(3, 0.3, 50, 40),
        _ball_frame(4, 0.4, 50, 28),
        _ball_frame(5, 0.5, 50, 18),
    ]
    # Player stays far so shot_not_held passes
    traj = [_player(1, 0, [_pt(i, i / 10.0, 0.20, 0.80) for i in range(0, 6)])]
    events = detect_ball_events(
        ball_frames=ball_frames,
        trajectories=traj,
        hoop_by_frame=hoop_by_frame,
        frame_width=w,
        frame_height=h,
        config=_cfg(),
    )
    types = {e["event_type"] for e in events}
    assert EVENT_TOWARD_HOOP in types
    assert EVENT_SHOT_ATTEMPT in types
    shot = next(e for e in events if e["event_type"] == EVENT_SHOT_ATTEMPT)
    assert shot["details"]["outcome"] == "unknown"
    assert shot["details"]["claim"] == "does_not_assert_make_or_miss"


def test_missing_ball_frames_returns_empty():
    assert detect_ball_events(ball_frames=[], config=_cfg()) == []


def test_analytics_safe_filters_closeup_frames():
    w, h = 100, 100
    ball_frames = [_ball_frame(i, i / 10.0, 50, 50) for i in range(0, 5)]
    traj = [_player(1, 0, [_pt(i, i / 10.0, 0.50, 0.50) for i in range(0, 5)])]
    segments = [
        {
            "segment_id": 2,
            "start_frame": 0,
            "end_frame": 4,
            "camera_state": "closeup",
            "usable_for_analytics": False,
        }
    ]
    events = detect_ball_events(
        ball_frames=ball_frames,
        trajectories=traj,
        segments=segments,
        frame_width=w,
        frame_height=h,
        config=_cfg(analytics_safe_only=True),
    )
    assert events == []


def test_build_timeline_includes_ball_summary_and_soft_fail():
    # No ball_frames → timeline still builds, ball_event_count 0
    timeline = build_game_timeline(
        job_id="b1",
        frame_width=100,
        frame_height=100,
        fps=10.0,
        segments_payload={
            "confidence": "high",
            "segments": [
                {
                    "segment_id": 1,
                    "start_frame": 0,
                    "end_frame": 2,
                    "start_time": 0.0,
                    "end_time": 0.2,
                    "camera_state": "wide_gameplay",
                    "usable_for_analytics": True,
                }
            ],
            "hoop_orientation_flips": [],
            "cut_frames": [],
        },
        ball_frames=None,
    )
    assert "ball_event_count" in timeline["summary"]
    assert timeline["summary"]["ball_event_count"] == 0
    assert EVENT_SHOT_ATTEMPT in timeline["event_types_included"]
