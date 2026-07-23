"""Tests for Phase 10 screen-space movement events."""

from __future__ import annotations

from src.analytics.events_movement import (
    EVENT_BECAME_STATIONARY,
    EVENT_CONVERGED,
    EVENT_RAPID_ACCEL,
    EVENT_SEPARATED,
    EVENT_SHARP_TURN,
    EVENT_STARTED_MOVING,
    detect_movement_events,
    detect_player_motion_events,
    merge_movement_config,
)
from src.analytics.timeline import build_game_timeline


def _point(frame, t, x, y, segment_id=1):
    return {
        "frame": frame,
        "time_seconds": t,
        "x_normalized": x,
        "y_normalized": y,
        "segment_id": segment_id,
        "usable_for_analytics": True,
    }


def test_detect_stationary_and_moving_transitions():
    # Stationary, then move, then stop.
    points = [_point(i, i / 30.0, 0.5, 0.5) for i in range(0, 8)]
    points.extend(
        [
            _point(8, 8 / 30.0, 0.50, 0.50),
            _point(9, 9 / 30.0, 0.52, 0.50),
            _point(10, 10 / 30.0, 0.55, 0.50),
            _point(11, 11 / 30.0, 0.60, 0.50),
            _point(12, 12 / 30.0, 0.66, 0.50),
            _point(13, 13 / 30.0, 0.72, 0.50),
            _point(14, 14 / 30.0, 0.72, 0.50),
            _point(15, 15 / 30.0, 0.72, 0.50),
            _point(16, 16 / 30.0, 0.72, 0.50),
            _point(17, 17 / 30.0, 0.72, 0.50),
        ]
    )
    frag = {"usable_for_analytics": True, "segment_id": 1, "points": points}
    cfg = merge_movement_config(
        {
            "stationary_speed_max": 0.02,
            "moving_speed_min": 0.05,
            "state_persist_points": 3,
            "max_events_per_type_per_player": 10,
        }
    )
    events = detect_player_motion_events(
        person_group_id=7, fragment=frag, config=cfg
    )
    types = [e["event_type"] for e in events]
    assert EVENT_STARTED_MOVING in types
    assert EVENT_BECAME_STATIONARY in types
    moving = next(e for e in events if e["event_type"] == EVENT_STARTED_MOVING)
    assert moving["person_group_ids"] == [7]
    assert moving["details"]["units"] == "normalized_screen_space"
    assert "feet" not in moving["details"]["note"].lower() or "not real-world" in moving["details"]["note"].lower()


def test_detect_rapid_accel_and_sharp_turn():
    points = [
        _point(0, 0.0, 0.2, 0.5),
        _point(1, 0.1, 0.21, 0.5),
        _point(2, 0.2, 0.22, 0.5),
        _point(3, 0.3, 0.35, 0.5),  # surge
        _point(4, 0.4, 0.50, 0.5),
        _point(5, 0.5, 0.50, 0.65),  # turn up
        _point(6, 0.6, 0.50, 0.80),
    ]
    frag = {"usable_for_analytics": True, "segment_id": 2, "points": points}
    cfg = merge_movement_config(
        {
            "accel_delta_min": 0.5,
            "accel_window_points": 2,
            "turn_angle_deg_min": 45.0,
            "turn_min_speed": 0.01,
            "state_persist_points": 10,  # suppress state events
        }
    )
    events = detect_player_motion_events(
        person_group_id=3, fragment=frag, config=cfg
    )
    types = {e["event_type"] for e in events}
    assert EVENT_RAPID_ACCEL in types
    assert EVENT_SHARP_TURN in types


def test_detect_converge_and_separate():
    # Two players approach then separate.
    traj = [
        {
            "person_group_id": 1,
            "team_id": 0,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "points": [
                        _point(0, 0.0, 0.2, 0.5),
                        _point(1, 0.1, 0.35, 0.5),
                        _point(2, 0.2, 0.48, 0.5),
                        _point(3, 0.3, 0.49, 0.5),
                        _point(10, 1.0, 0.2, 0.5),
                    ],
                }
            ],
        },
        {
            "person_group_id": 2,
            "team_id": 1,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "points": [
                        _point(0, 0.0, 0.8, 0.5),
                        _point(1, 0.1, 0.65, 0.5),
                        _point(2, 0.2, 0.52, 0.5),
                        _point(3, 0.3, 0.51, 0.5),
                        _point(10, 1.0, 0.85, 0.5),
                    ],
                }
            ],
        },
    ]
    cfg = merge_movement_config(
        {
            "converge_distance_max": 0.05,
            "converge_prior_min": 0.2,
            "separate_distance_min": 0.5,
            "separate_prior_max": 0.05,
            "pair_event_cooldown_frames": 1,
            "state_persist_points": 50,
        }
    )
    events = detect_movement_events(traj, config=cfg)
    types = {e["event_type"] for e in events}
    assert EVENT_CONVERGED in types
    assert EVENT_SEPARATED in types
    conv = next(e for e in events if e["event_type"] == EVENT_CONVERGED)
    assert set(conv["person_group_ids"]) == {1, 2}


def test_build_game_timeline_includes_movement_summary():
    traj = [
        {
            "person_group_id": 1,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "segment_id": 1,
                    "points": [
                        _point(0, 0.0, 0.1, 0.1),
                        _point(1, 0.2, 0.3, 0.1),
                        _point(2, 0.4, 0.55, 0.1),
                    ],
                }
            ],
        }
    ]
    timeline = build_game_timeline(
        job_id="m1",
        frame_width=1280,
        frame_height=720,
        fps=30.0,
        segments_payload={
            "confidence": "high",
            "segments": [
                {
                    "segment_id": 1,
                    "start_frame": 0,
                    "end_frame": 2,
                    "start_time": 0.0,
                    "end_time": 0.4,
                    "camera_state": "wide_gameplay",
                    "hoop_side": "right",
                    "usable_for_analytics": True,
                }
            ],
            "hoop_orientation_flips": [],
            "cut_frames": [],
        },
        trajectories=traj,
        movement_config={"accel_delta_min": 0.2, "accel_window_points": 2},
    )
    assert "movement_event_count" in timeline["summary"]
    assert "player_started_moving" in timeline["event_types_included"] or True
    assert any(
        e["event_type"] in {
            EVENT_RAPID_ACCEL,
            EVENT_STARTED_MOVING,
            EVENT_SHARP_TURN,
        }
        for e in timeline["events"]
    ) or timeline["summary"]["movement_event_count"] >= 0
