"""Tests for Phase 11 cautious screen-space defensive events."""

from __future__ import annotations

from src.analytics.events_defense import (
    EVENT_DOUBLE_TEAM,
    EVENT_HELP_DEFENSE,
    EVENT_MATCHUP_CHANGED,
    EVENT_PRESSURE_ENDED,
    EVENT_PRESSURE_STARTED,
    EVENT_SEPARATION,
    detect_defense_events,
    merge_defense_config,
)
from src.analytics.timeline import build_game_timeline


def _point(frame, t, x, y, segment_id=1, usable=True, camera_state="wide_gameplay"):
    return {
        "frame": frame,
        "time_seconds": t,
        "x_normalized": x,
        "y_normalized": y,
        "segment_id": segment_id,
        "usable_for_analytics": usable,
        "camera_state": camera_state,
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


def _tight_cfg(**overrides):
    base = {
        "pressure_distance_max": 0.08,
        "pressure_clear_distance": 0.12,
        "pressure_persist_frames": 3,
        "clear_persist_frames": 3,
        "separation_distance_min": 0.20,
        "separation_prior_max": 0.08,
        "double_team_distance_max": 0.10,
        "double_team_persist_frames": 2,
        "help_defense_distance_max": 0.12,
        "help_defense_persist_frames": 2,
        "matchup_persist_frames": 3,
        "matchup_cooldown_frames": 1,
        "event_cooldown_frames": 1,
        "max_events_per_type_per_player": 10,
    }
    base.update(overrides)
    return merge_defense_config(base)


def test_pressure_start_and_end():
    # Offensive player at x=0.5; defender approaches then leaves.
    offense = []
    defense = []
    for i in range(0, 12):
        offense.append(_point(i, i / 30.0, 0.50, 0.50))
        if i < 3:
            defense.append(_point(i, i / 30.0, 0.70, 0.50))  # far
        elif i < 8:
            defense.append(_point(i, i / 30.0, 0.55, 0.50))  # close (~0.05)
        else:
            defense.append(_point(i, i / 30.0, 0.80, 0.50))  # far again

    traj = [_player(1, 0, offense), _player(2, 1, defense)]
    events = detect_defense_events(traj, config=_tight_cfg())
    types = [e["event_type"] for e in events]
    assert EVENT_PRESSURE_STARTED in types
    assert EVENT_PRESSURE_ENDED in types
    started = next(e for e in events if e["event_type"] == EVENT_PRESSURE_STARTED)
    assert set(started["person_group_ids"]) == {1, 2}
    assert "screen_space_separation" in started["details"]
    assert "possible" in started["details"]["label"]
    assert "not official" in started["details"]["note"].lower()


def test_double_team_and_help_defense():
    # Player 1 pressured by 2, then 3 also enters close range.
    frames = list(range(0, 10))
    p1 = [_point(i, i / 30.0, 0.50, 0.50) for i in frames]
    # Defender A close from start
    p2 = [_point(i, i / 30.0, 0.55, 0.50) for i in frames]
    # Defender B arrives mid-way
    p3 = []
    for i in frames:
        if i < 4:
            p3.append(_point(i, i / 30.0, 0.90, 0.50))
        else:
            p3.append(_point(i, i / 30.0, 0.52, 0.52))

    traj = [_player(1, 0, p1), _player(2, 1, p2), _player(3, 1, p3)]
    events = detect_defense_events(traj, config=_tight_cfg())
    types = {e["event_type"] for e in events}
    assert EVENT_DOUBLE_TEAM in types
    assert EVENT_HELP_DEFENSE in types or EVENT_PRESSURE_STARTED in types
    dt = next(e for e in events if e["event_type"] == EVENT_DOUBLE_TEAM)
    assert 1 in dt["person_group_ids"]
    assert len(dt["person_group_ids"]) >= 3


def test_matchup_changed_and_separation():
    # Nearest opp switches from 2 → 3 and persists; then separation from close.
    offense = []
    d_a = []
    d_b = []
    for i in range(0, 16):
        offense.append(_point(i, i / 30.0, 0.50, 0.50))
        if i < 6:
            d_a.append(_point(i, i / 30.0, 0.56, 0.50))  # nearest
            d_b.append(_point(i, i / 30.0, 0.85, 0.50))
        elif i < 12:
            d_a.append(_point(i, i / 30.0, 0.85, 0.50))
            d_b.append(_point(i, i / 30.0, 0.56, 0.50))  # new nearest
        else:
            d_a.append(_point(i, i / 30.0, 0.90, 0.50))
            d_b.append(_point(i, i / 30.0, 0.80, 0.50))  # both far → separation

    traj = [_player(1, 0, offense), _player(2, 1, d_a), _player(3, 1, d_b)]
    events = detect_defense_events(traj, config=_tight_cfg())
    types = {e["event_type"] for e in events}
    assert EVENT_MATCHUP_CHANGED in types
    assert EVENT_SEPARATION in types or EVENT_PRESSURE_ENDED in types
    match = next(e for e in events if e["event_type"] == EVENT_MATCHUP_CHANGED)
    assert match["details"]["previous_opponent_id"] == 2
    assert match["details"]["new_opponent_id"] == 3


def test_no_events_on_closeup_or_unusable_points():
    offense = [
        _point(i, i / 30.0, 0.50, 0.50, usable=False, camera_state="closeup")
        for i in range(0, 8)
    ]
    defense = [
        _point(i, i / 30.0, 0.55, 0.50, usable=False, camera_state="closeup")
        for i in range(0, 8)
    ]
    traj = [
        {
            "person_group_id": 1,
            "team_id": 0,
            "path_fragments": [
                {
                    "usable_for_analytics": False,
                    "segment_id": 2,
                    "points": offense,
                }
            ],
        },
        {
            "person_group_id": 2,
            "team_id": 1,
            "path_fragments": [
                {
                    "usable_for_analytics": False,
                    "segment_id": 2,
                    "points": defense,
                }
            ],
        },
    ]
    events = detect_defense_events(traj, config=_tight_cfg())
    assert events == []


def test_build_game_timeline_includes_defense_summary():
    traj = [
        _player(
            1,
            0,
            [_point(i, i / 30.0, 0.50, 0.50) for i in range(0, 6)],
        ),
        _player(
            2,
            1,
            [_point(i, i / 30.0, 0.55, 0.50) for i in range(0, 6)],
        ),
    ]
    timeline = build_game_timeline(
        job_id="d1",
        frame_width=1280,
        frame_height=720,
        fps=30.0,
        segments_payload={
            "confidence": "high",
            "segments": [
                {
                    "segment_id": 1,
                    "start_frame": 0,
                    "end_frame": 5,
                    "start_time": 0.0,
                    "end_time": 0.16,
                    "camera_state": "wide_gameplay",
                    "hoop_side": "right",
                    "usable_for_analytics": True,
                }
            ],
            "hoop_orientation_flips": [],
            "cut_frames": [],
        },
        trajectories=traj,
        defense_config=_tight_cfg(),
    )
    assert "defense_event_count" in timeline["summary"]
    assert EVENT_PRESSURE_STARTED in timeline["event_types_included"]
    assert timeline["summary"]["defense_event_count"] >= 1
    assert any(e["event_type"] == EVENT_PRESSURE_STARTED for e in timeline["events"])
