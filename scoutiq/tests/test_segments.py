"""Focused tests for gameplay segment detection (Phase 3)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.analytics.segments import (
    CAMERA_CLOSEUP,
    CAMERA_TRANSITION,
    CAMERA_WIDE,
    build_game_segments,
    build_transition_frames,
    classify_hoop_side,
    collect_cut_frames,
    detect_closeup_candidate_frames,
    detect_track_layout_cuts,
    generate_segments_for_job,
    is_closeup_candidate,
    persist_frame_flags,
    stabilize_hoop_sides,
)


def test_classify_hoop_side_thresholds():
    assert classify_hoop_side(0.20) == "left"
    assert classify_hoop_side(0.50) == "center"
    assert classify_hoop_side(0.80) == "right"
    assert classify_hoop_side(0.35) == "center"
    assert classify_hoop_side(0.65) == "center"


def test_stabilize_hoop_sides_requires_persistence_for_left_right_flip():
    # Stable right, one left blip (surrounded by rights), then a confirmed left run.
    observations = [(i, "right", 0.8) for i in range(0, 10)]
    observations.append((10, "left", 0.2))  # isolated — must not flip
    observations.extend((i, "right", 0.8) for i in range(11, 19))
    observations.extend((i, "left", 0.2) for i in range(20, 28))

    side_by_frame, flips = stabilize_hoop_sides(
        observations,
        frame_start=0,
        frame_end=40,
        window=8,
        min_votes=5,
    )

    assert side_by_frame[5] == "right"
    assert side_by_frame[10] == "right"  # blip ignored
    assert side_by_frame[30] == "left"
    assert len(flips) == 1
    assert flips[0]["from_side"] == "right"
    assert flips[0]["to_side"] == "left"
    assert flips[0]["transition_start_frame"] == 20


def test_detect_track_layout_cuts_on_sudden_loss_and_scale():
    frames = [
        {
            "frame_number": 0,
            "tracks": [
                {"bbox_xyxy": [0, 0, 10, 20]},
                {"bbox_xyxy": [20, 0, 30, 20]},
                {"bbox_xyxy": [40, 0, 50, 20]},
                {"bbox_xyxy": [60, 0, 70, 20]},
                {"bbox_xyxy": [80, 0, 90, 20]},
            ],
        },
        {
            "frame_number": 1,
            "tracks": [{"bbox_xyxy": [0, 0, 10, 20]}],  # sudden loss
        },
        {
            "frame_number": 2,
            "tracks": [{"bbox_xyxy": [0, 0, 200, 400]}],  # huge scale jump
        },
    ]
    cuts = detect_track_layout_cuts(
        frames,
        track_loss_ratio=0.45,
        track_loss_min_prev=4,
        bbox_scale_ratio=2.5,
    )
    assert 1 in cuts
    assert 2 in cuts


def test_build_transition_frames_pads_cuts_and_includes_hoop_flips():
    transitions = build_transition_frames(
        cut_frames=[10],
        hoop_flip_events=[
            {
                "transition_start_frame": 20,
                "transition_end_frame": 22,
            }
        ],
        frame_start=0,
        frame_end=30,
        pad_frames=2,
    )
    assert {8, 9, 10, 11, 12}.issubset(transitions)
    assert {20, 21, 22}.issubset(transitions)


def test_build_game_segments_marks_only_wide_usable():
    hoop_side = {f: "right" for f in range(0, 11)}
    for f in range(11, 21):
        hoop_side[f] = "left"
    transitions = {8, 9, 10}
    closeups = {12, 13, 14}
    frame_times = {f: f / 30.0 for f in range(0, 21)}

    segments = build_game_segments(
        frame_start=0,
        frame_end=20,
        frame_times=frame_times,
        hoop_side_by_frame=hoop_side,
        transition_frames=transitions,
        closeup_frames=closeups,
        fps=30.0,
    )

    assert segments[0]["camera_state"] == CAMERA_WIDE
    assert segments[0]["usable_for_analytics"] is True
    assert segments[0]["hoop_side"] == "right"

    transition_seg = next(s for s in segments if s["camera_state"] == CAMERA_TRANSITION)
    assert transition_seg["usable_for_analytics"] is False
    assert transition_seg["start_frame"] == 8
    assert transition_seg["end_frame"] == 10

    closeup_seg = next(s for s in segments if s["camera_state"] == CAMERA_CLOSEUP)
    assert closeup_seg["usable_for_analytics"] is False
    assert closeup_seg["start_frame"] == 12
    assert closeup_seg["end_frame"] == 14

    left_wide = [s for s in segments if s["hoop_side"] == "left" and s["camera_state"] == CAMERA_WIDE]
    assert left_wide
    assert left_wide[0]["usable_for_analytics"] is True


def test_is_closeup_candidate_starting_rule():
    assert is_closeup_candidate(
        player_count=1,
        largest_area_ratio=0.25,
        hoop_absent=True,
        tracks_dropped_sharply=False,
        feet_likely_missing=False,
    )
    assert not is_closeup_candidate(
        player_count=5,
        largest_area_ratio=0.25,
        hoop_absent=True,
        tracks_dropped_sharply=False,
        feet_likely_missing=False,
    )
    assert not is_closeup_candidate(
        player_count=1,
        largest_area_ratio=0.10,
        hoop_absent=True,
        tracks_dropped_sharply=False,
        feet_likely_missing=False,
    )
    assert not is_closeup_candidate(
        player_count=1,
        largest_area_ratio=0.25,
        hoop_absent=False,
        tracks_dropped_sharply=False,
        feet_likely_missing=False,
    )
    assert is_closeup_candidate(
        player_count=2,
        largest_area_ratio=0.22,
        hoop_absent=False,
        tracks_dropped_sharply=True,
        feet_likely_missing=False,
    )


def test_persist_frame_flags_ignores_isolated_blips():
    candidates = {10, 20, 21, 22, 23}
    confirmed = persist_frame_flags(
        candidates,
        frame_start=0,
        frame_end=30,
        window=5,
        min_votes=3,
    )
    assert 10 not in confirmed
    assert 22 in confirmed
    assert 23 in confirmed


def test_detect_closeup_candidates_and_job_output(tmp_path: Path):
    # 1000x500 frame → area 500000; box 400x300 = 120000 → ratio 0.24
    wide_tracks = [
        {"person_group_id": i, "bbox_xyxy": [i * 40, 10, i * 40 + 30, 80]}
        for i in range(6)
    ]
    closeup_box = [{"person_group_id": 1, "bbox_xyxy": [100, 50, 500, 350]}]

    frames = []
    for f in range(0, 12):
        frames.append(
            {
                "frame_number": f,
                "timestamp_seconds": f / 30.0,
                "tracks": list(wide_tracks),
            }
        )
    for f in range(12, 20):
        frames.append(
            {
                "frame_number": f,
                "timestamp_seconds": f / 30.0,
                "tracks": list(closeup_box),
            }
        )

    candidates, stats = detect_closeup_candidate_frames(
        frames,
        frame_width=1000,
        frame_height=500,
        hoop_present_frames=set(range(0, 12)),  # hoop present only on wide part
        hoop_available=True,
        max_players=2,
        min_area_ratio=0.20,
    )
    assert stats["candidate_frames"] >= 5
    assert 12 in candidates
    assert 0 not in candidates

    job_dir = tmp_path / "closeupjob"
    job_dir.mkdir()
    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row) + "\n")
    (job_dir / "quality.json").write_text(json.dumps({"cut_frames": []}), encoding="utf-8")
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 1000, "frame_height": 500}),
        encoding="utf-8",
    )
    # Hoop present on wide frames only.
    with gzip.open(job_dir / "hoop_tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for f in range(0, 12):
            handle.write(
                json.dumps(
                    {
                        "frame_number": f,
                        "hoops": [{"bbox_xyxy": [800, 40, 900, 100], "confidence": 0.9}],
                    }
                )
                + "\n"
            )
        for f in range(12, 20):
            handle.write(json.dumps({"frame_number": f, "hoops": []}) + "\n")

    payload = generate_segments_for_job(
        job_dir,
        config={
            "probe_video": False,
            "transition_pad_frames": 0,
            "closeup_max_players": 2,
            "closeup_min_area_ratio": 0.20,
            "closeup_persist_window": 5,
            "closeup_persist_min_votes": 3,
            "hoop_side_window": 8,
            "hoop_side_min_votes": 5,
        },
    )

    assert payload["closeup_detection"]["enabled"] is True
    assert payload["summary"]["closeup_segment_count"] >= 1
    closeup_segs = [s for s in payload["segments"] if s["camera_state"] == CAMERA_CLOSEUP]
    assert closeup_segs
    assert all(s["usable_for_analytics"] is False for s in closeup_segs)
    assert "closeup" in payload["signals_used"]


def test_collect_cut_frames_merges_sources():
    assert collect_cut_frames(
        quality_cuts=[3, 1],
        track_cuts=[1, 5],
        video_cuts=[5, 9],
    ) == [1, 3, 5, 9]


def test_generate_segments_for_job_without_hoop_is_reduced_confidence(tmp_path: Path):
    job_dir = tmp_path / "segjob"
    job_dir.mkdir()

    tracks = []
    for frame in range(0, 30):
        tracks.append(
            {
                "frame_number": frame,
                "timestamp_seconds": frame / 30.0,
                "tracks": [
                    {"person_group_id": 1, "bbox_xyxy": [10, 10, 40, 80]},
                    {"person_group_id": 2, "bbox_xyxy": [50, 10, 80, 80]},
                    {"person_group_id": 3, "bbox_xyxy": [90, 10, 120, 80]},
                    {"person_group_id": 4, "bbox_xyxy": [130, 10, 160, 80]},
                    {"person_group_id": 5, "bbox_xyxy": [170, 10, 200, 80]},
                ],
            }
        )
    # Inject a hard layout cut at frame 15.
    tracks[15]["tracks"] = [{"person_group_id": 1, "bbox_xyxy": [10, 10, 40, 80]}]

    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in tracks:
            handle.write(json.dumps(row) + "\n")

    (job_dir / "quality.json").write_text(
        json.dumps({"cut_frames": [10]}),
        encoding="utf-8",
    )
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 1280, "frame_height": 720}),
        encoding="utf-8",
    )

    payload = generate_segments_for_job(
        job_dir,
        config={
            "probe_video": False,
            "transition_pad_frames": 1,
            "track_loss_ratio": 0.45,
            "track_loss_min_prev": 4,
        },
    )

    out = job_dir / "analytics" / "game_segments.json"
    assert out.is_file()
    assert payload["confidence"] == "reduced"
    assert payload["hoop_available"] is False
    assert 10 in payload["cut_frames"]
    assert 15 in payload["cut_frames"]
    assert payload["summary"]["segment_count"] >= 1
    assert any(s["camera_state"] == CAMERA_TRANSITION for s in payload["segments"])
    assert all(
        s["usable_for_analytics"] is (s["camera_state"] == CAMERA_WIDE)
        for s in payload["segments"]
    )


def test_generate_segments_for_job_with_hoop_flip(tmp_path: Path):
    job_dir = tmp_path / "hoopjob"
    job_dir.mkdir()

    tracks = [
        {
            "frame_number": f,
            "timestamp_seconds": f / 30.0,
            "tracks": [{"person_group_id": 1, "bbox_xyxy": [10, 10, 30, 50]}],
        }
        for f in range(0, 40)
    ]
    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in tracks:
            handle.write(json.dumps(row) + "\n")

    (job_dir / "quality.json").write_text(json.dumps({"cut_frames": []}), encoding="utf-8")
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 1000, "frame_height": 500}),
        encoding="utf-8",
    )

    # Persistent right, then persistent left (5+ of 8).
    hoop_rows = []
    for f in range(0, 12):
        hoop_rows.append(
            {
                "frame_number": f,
                "hoops": [{"bbox_xyxy": [800, 50, 900, 120], "confidence": 0.9}],
            }
        )
    for f in range(20, 30):
        hoop_rows.append(
            {
                "frame_number": f,
                "hoops": [{"bbox_xyxy": [50, 50, 150, 120], "confidence": 0.9}],
            }
        )
    with gzip.open(job_dir / "hoop_tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in hoop_rows:
            handle.write(json.dumps(row) + "\n")

    payload = generate_segments_for_job(
        job_dir,
        config={
            "probe_video": False,
            "hoop_side_window": 8,
            "hoop_side_min_votes": 5,
            "transition_pad_frames": 0,
        },
    )

    assert payload["confidence"] == "high"
    assert payload["hoop_available"] is True
    assert payload["summary"]["hoop_flip_count"] == 1
    assert any(s["hoop_side"] == "right" and s["usable_for_analytics"] for s in payload["segments"])
    assert any(s["hoop_side"] == "left" and s["usable_for_analytics"] for s in payload["segments"])
    assert any(s["camera_state"] == CAMERA_TRANSITION for s in payload["segments"])
