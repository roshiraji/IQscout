"""Tests for Phase 9 game timeline (camera + tracking events)."""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics.timeline import (
    EVENT_CLOSEUP_STARTED,
    EVENT_GAMEPLAY_RESUMED,
    EVENT_GAMEPLAY_STARTED,
    EVENT_ORIENTATION_CHANGED,
    EVENT_QUALITY_DROPPED,
    EVENT_QUALITY_RECOVERED,
    build_game_timeline,
    events_from_segments,
    finalize_events,
    generate_timeline_for_job,
    make_event,
)


def test_make_event_confidence_label():
    high = make_event(
        event_id="event_001",
        event_type=EVENT_GAMEPLAY_STARTED,
        start_frame=0,
        end_frame=0,
        start_time=0.0,
        end_time=0.0,
        confidence=0.95,
    )
    assert high["confidence_label"] == "high"
    assert high["person_group_ids"] == []
    assert "details" in high


def test_events_from_segments_emit_gameplay_closeup_and_resume():
    segments = [
        {
            "segment_id": 1,
            "start_frame": 0,
            "end_frame": 9,
            "start_time": 0.0,
            "end_time": 0.3,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
        },
        {
            "segment_id": 2,
            "start_frame": 10,
            "end_frame": 14,
            "start_time": 0.33,
            "end_time": 0.46,
            "camera_state": "closeup",
            "hoop_side": "right",
            "usable_for_analytics": False,
        },
        {
            "segment_id": 3,
            "start_frame": 15,
            "end_frame": 30,
            "start_time": 0.5,
            "end_time": 1.0,
            "camera_state": "wide_gameplay",
            "hoop_side": "left",
            "usable_for_analytics": True,
        },
    ]
    raw = events_from_segments(segments)
    types = [e["event_type"] for e in raw]
    assert EVENT_GAMEPLAY_STARTED in types
    assert EVENT_CLOSEUP_STARTED in types
    assert EVENT_GAMEPLAY_RESUMED in types
    events = finalize_events(raw)
    assert events[0]["event_id"] == "event_001"
    resume = next(e for e in events if e["event_type"] == EVENT_GAMEPLAY_RESUMED)
    assert resume["details"]["resumed_from"] == "closeup"


def test_build_game_timeline_includes_orientation_and_quality():
    payload = {
        "confidence": "high",
        "segments": [
            {
                "segment_id": 1,
                "start_frame": 0,
                "end_frame": 100,
                "start_time": 0.0,
                "end_time": 3.3,
                "camera_state": "wide_gameplay",
                "hoop_side": "right",
                "usable_for_analytics": True,
            },
            {
                "segment_id": 2,
                "start_frame": 101,
                "end_frame": 200,
                "start_time": 3.36,
                "end_time": 6.6,
                "camera_state": "wide_gameplay",
                "hoop_side": "left",
                "usable_for_analytics": True,
            },
        ],
        "hoop_orientation_flips": [
            {
                "from_side": "right",
                "to_side": "left",
                "transition_start_frame": 90,
                "transition_end_frame": 100,
                "confirmed_frame": 100,
            }
        ],
        "cut_frames": [50],
    }
    timeline = build_game_timeline(
        job_id="jobx",
        frame_width=1280,
        frame_height=720,
        fps=30.0,
        segments_payload=payload,
        frame_times={50: 1.666, 100: 3.333},
    )
    types = {e["event_type"] for e in timeline["events"]}
    assert EVENT_ORIENTATION_CHANGED in types
    assert EVENT_QUALITY_DROPPED in types
    assert EVENT_QUALITY_RECOVERED in types
    assert timeline["video"]["fps"] == 30.0
    assert timeline["summary"]["event_count"] == len(timeline["events"])
    assert "gameplay_segment_started" in timeline["event_types_included"]


def test_generate_timeline_for_job_writes_file(tmp_path: Path):
    job_dir = tmp_path / "tljob"
    analytics = job_dir / "analytics"
    analytics.mkdir(parents=True)
    segments_payload = {
        "frame_width": 640,
        "frame_height": 360,
        "fps_estimate": 30.0,
        "confidence": "reduced",
        "segments": [
            {
                "segment_id": 1,
                "start_frame": 0,
                "end_frame": 29,
                "start_time": 0.0,
                "end_time": 0.966,
                "camera_state": "wide_gameplay",
                "hoop_side": "unknown",
                "usable_for_analytics": True,
            }
        ],
        "hoop_orientation_flips": [],
        "cut_frames": [],
    }
    (analytics / "game_segments.json").write_text(
        json.dumps(segments_payload), encoding="utf-8"
    )
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 640, "frame_height": 360}),
        encoding="utf-8",
    )

    timeline = generate_timeline_for_job(job_dir)
    out = analytics / "game_timeline.json"
    assert out.is_file()
    assert timeline["summary"]["event_count"] >= 2
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["events"][0]["event_type"] == EVENT_GAMEPLAY_STARTED
