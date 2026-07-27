"""Focused tests for analytics trajectory generation (Phase 2 + Phase 5)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.analytics.trajectories import (
    COORDINATE_SPACE,
    TrajectoryInputError,
    annotate_points_with_segments,
    attach_segments_and_fragments,
    bbox_bottom_center,
    build_frame_segment_index,
    build_trajectories,
    generate_trajectories_for_job,
    load_team_lookup,
    resolve_frame_size,
    resolve_tracks_path,
    split_path_fragments,
)


def test_bbox_bottom_center_uses_mid_x_and_bottom_y():
    assert bbox_bottom_center([100, 50, 200, 300]) == (150.0, 300.0)


@pytest.mark.parametrize(
    "bbox",
    [None, [], [1, 2, 3], [10, 10, 10, 20], [20, 10, 10, 20], "bad"],
)
def test_bbox_bottom_center_rejects_invalid(bbox):
    assert bbox_bottom_center(bbox) is None


def test_load_team_lookup_from_person_groups_payload(tmp_path: Path):
    path = tmp_path / "person_groups.json"
    path.write_text(
        json.dumps(
            {
                "person_groups": [
                    {"person_group_id": 7, "team_id": 1},
                    {"person_group_id": 8, "team_id": None},
                    {"person_group_id": 9},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_team_lookup(path) == {
        7: {
            "team_id": 1,
            "jersey_color_name": None,
            "team_label": None,
            "role": "player",
        },
        8: {
            "team_id": None,
            "jersey_color_name": None,
            "team_label": None,
            "role": "player",
        },
        9: {
            "team_id": None,
            "jersey_color_name": None,
            "team_label": None,
            "role": "player",
        },
    }


def test_load_team_lookup_infers_team_from_jersey(tmp_path: Path):
    path = tmp_path / "person_groups.json"
    path.write_text(
        json.dumps(
            {
                "person_groups": [
                    {"person_group_id": 1, "jersey_color_name": "red"},
                    {"person_group_id": 2, "jersey_color_name": "Blue", "team_id": None},
                ]
            }
        ),
        encoding="utf-8",
    )
    lookup = load_team_lookup(path)
    assert lookup[1]["team_id"] == 0
    assert lookup[1]["jersey_color_name"] == "red"
    assert lookup[2]["team_id"] == 1
    assert lookup[2]["jersey_color_name"] == "blue"


def test_build_trajectories_groups_by_person_group_id_and_skips_bad_rows():
    frames = [
        {
            "frame_number": 10,
            "timestamp_seconds": 0.333333,
            "tracks": [
                {
                    "person_group_id": 7,
                    "bbox_xyxy": [100, 50, 200, 300],
                },
                {
                    "person_group_id": 7,
                    "bbox_xyxy": [110, 60, 210, 310],
                },
                {
                    # missing identity
                    "bbox_xyxy": [10, 10, 20, 20],
                },
                {
                    "person_group_id": 3,
                    "bbox_xyxy": [0, 0, 0, 0],  # invalid
                },
                {
                    "person_group_id": 3,
                    "bbox_xyxy": [400, 100, 500, 400],
                },
            ],
        },
        {
            "frame_number": 11,
            "timestamp_seconds": 0.366667,
            "tracks": [
                {
                    "person_group_id": 3,
                    "bbox_xyxy": [410, 110, 510, 410],
                },
            ],
        },
    ]

    trajectories, summary = build_trajectories(
        frames,
        frame_width=1000,
        frame_height=500,
        team_lookup={7: 1, 3: 0},
    )

    assert [t["person_group_id"] for t in trajectories] == [3, 7]
    assert trajectories[0]["team_id"] == 0
    assert trajectories[1]["team_id"] == 1
    assert len(trajectories[0]["points"]) == 2
    assert len(trajectories[1]["points"]) == 2

    first = trajectories[1]["points"][0]
    assert first["frame"] == 10
    assert first["x"] == 150.0
    assert first["y"] == 300.0
    assert first["x_normalized"] == pytest.approx(0.15)
    assert first["y_normalized"] == pytest.approx(0.6)

    assert summary["tracked_players"] == 2
    assert summary["trajectory_points"] == 4
    assert summary["frame_range"] == {"start": 10, "end": 11}
    assert summary["coordinate_space"] == COORDINATE_SPACE
    assert summary["missing_data"]["missing_person_group_id"] == 1
    assert summary["missing_data"]["invalid_bbox"] == 1
    assert summary["missing_data"]["skipped_detections"] == 2


def test_resolve_tracks_path_prefers_locked_tracks_when_present(tmp_path: Path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "tracks.jsonl.gz").write_bytes(b"")
    assert resolve_tracks_path(job_dir) == job_dir / "tracks.jsonl.gz"

    (job_dir / "locked_tracks.jsonl.gz").write_bytes(b"")
    assert resolve_tracks_path(job_dir) == job_dir / "locked_tracks.jsonl.gz"

    # Explicit override always wins.
    assert resolve_tracks_path(job_dir, "tracks.jsonl.gz") == job_dir / "tracks.jsonl.gz"


def test_build_trajectories_prefers_mask_stance_point_over_bbox_bottom():
    # A mask with most points near the top of the box but a small cluster of
    # "feet" pixels lower than the box bottom-center would predict.
    mask_polygon = [
        [140, 60],
        [160, 62],
        [150, 61],
        [145, 58],
        [148, 100],
        [152, 100],
        [150, 99],
    ]
    frames = [
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "tracks": [
                {
                    "person_group_id": 1,
                    "bbox_xyxy": [100, 50, 200, 100],
                    "mask_polygon": mask_polygon,
                },
            ],
        }
    ]
    trajectories, summary = build_trajectories(frames, frame_width=1000, frame_height=500)
    point = trajectories[0]["points"][0]
    # bbox bottom-center would be (150, 100); mask stance should land close to
    # the lowest mask points, near (150, 99.7).
    assert point["point_source"] == "mask_bottom"
    assert point["x"] == pytest.approx(150.0, abs=1.0)
    assert point["y"] < 100.0
    assert summary["mask_stance_points"] == 1
    assert summary["bbox_fallback_points"] == 0


def test_build_trajectories_falls_back_to_bbox_without_mask():
    frames = [
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "tracks": [{"person_group_id": 1, "bbox_xyxy": [100, 50, 200, 300]}],
        }
    ]
    trajectories, summary = build_trajectories(frames, frame_width=1000, frame_height=500)
    point = trajectories[0]["points"][0]
    assert point["point_source"] == "bbox_bottom"
    assert point["interpolated"] is False
    assert summary["mask_stance_points"] == 0
    assert summary["bbox_fallback_points"] == 1


def test_split_path_fragments_interpolates_short_gaps():
    points = [
        {
            "frame": 0,
            "time_seconds": 0.0,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 0,
            "y": 0,
            "x_normalized": 0.0,
            "y_normalized": 0.0,
        },
        {
            # 4-frame gap (missed detections), same fragment.
            "frame": 4,
            "time_seconds": 0.12,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 40,
            "y": 40,
            "x_normalized": 0.4,
            "y_normalized": 0.4,
        },
    ]
    fragments = split_path_fragments(
        points, max_gap_frames=15, max_interpolate_gap_frames=6
    )
    assert len(fragments) == 1
    frag = fragments[0]
    assert frag["detected_point_count"] == 2
    assert frag["point_count"] == 5  # frames 0,1,2,3,4
    frames_in_order = [p["frame"] for p in frag["points"]]
    assert frames_in_order == [0, 1, 2, 3, 4]
    middle = frag["points"][2]
    assert middle["interpolated"] is True
    assert middle["x"] == pytest.approx(20.0)
    assert middle["y"] == pytest.approx(20.0)
    assert not frag["points"][0].get("interpolated")
    assert not frag["points"][-1].get("interpolated")


def test_split_path_fragments_does_not_interpolate_beyond_threshold():
    points = [
        {
            "frame": 0,
            "time_seconds": 0.0,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 0,
            "y": 0,
            "x_normalized": 0.0,
            "y_normalized": 0.0,
        },
        {
            # 10-frame gap exceeds max_interpolate_gap_frames (6) but stays
            # under max_gap_frames (15), so no interpolation, no new fragment.
            "frame": 10,
            "time_seconds": 0.33,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 100,
            "y": 100,
            "x_normalized": 1.0,
            "y_normalized": 1.0,
        },
    ]
    fragments = split_path_fragments(
        points, max_gap_frames=15, max_interpolate_gap_frames=6
    )
    assert len(fragments) == 1
    assert fragments[0]["point_count"] == 2
    assert fragments[0]["detected_point_count"] == 2


def test_split_path_fragments_smooths_per_frame_jitter():
    # A real player cannot reverse direction every single frame; a wobbling
    # detection box can. Simulate that sawtooth and confirm the smoothed
    # trail's amplitude collapses while it still moves in the right
    # direction overall.
    ys = [100, 140, 100, 140, 100, 140, 100]
    points = [
        {
            "frame": i,
            "time_seconds": i / 30.0,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": i * 10,
            "y": y,
            "x_normalized": i * 10 / 1000.0,
            "y_normalized": y / 500.0,
        }
        for i, y in enumerate(ys)
    ]

    fragments = split_path_fragments(
        points,
        max_gap_frames=15,
        max_interpolate_gap_frames=6,
        smoothing_window_frames=5,
    )
    assert len(fragments) == 1
    smoothed_ys = [p["y"] for p in fragments[0]["points"]]

    assert max(ys) - min(ys) == 40
    assert max(smoothed_ys) - min(smoothed_ys) < 20
    # Still ends up roughly centered between the two jitter extremes, not
    # dragged toward either one.
    assert 105 < sum(smoothed_ys) / len(smoothed_ys) < 125


def test_split_path_fragments_smoothing_disabled_when_window_is_one():
    ys = [100, 140, 100, 140, 100]
    points = [
        {
            "frame": i,
            "time_seconds": i / 30.0,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": i * 10,
            "y": y,
            "x_normalized": i * 10 / 1000.0,
            "y_normalized": y / 500.0,
        }
        for i, y in enumerate(ys)
    ]
    fragments = split_path_fragments(
        points,
        max_gap_frames=15,
        max_interpolate_gap_frames=6,
        smoothing_window_frames=1,
    )
    assert [p["y"] for p in fragments[0]["points"]] == ys


def test_split_path_fragments_breaks_on_segment_change_and_gap():
    points = [
        {
            "frame": 0,
            "time_seconds": 0.0,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 1,
            "y": 1,
            "x_normalized": 0.1,
            "y_normalized": 0.1,
        },
        {
            "frame": 1,
            "time_seconds": 0.03,
            "segment_id": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
            "x": 2,
            "y": 2,
            "x_normalized": 0.2,
            "y_normalized": 0.2,
        },
        {
            "frame": 2,
            "time_seconds": 0.06,
            "segment_id": 2,
            "camera_state": "transition",
            "hoop_side": "right",
            "usable_for_analytics": False,
            "x": 3,
            "y": 3,
            "x_normalized": 0.3,
            "y_normalized": 0.3,
        },
        {
            "frame": 20,
            "time_seconds": 0.66,
            "segment_id": 3,
            "camera_state": "wide_gameplay",
            "hoop_side": "left",
            "usable_for_analytics": True,
            "x": 4,
            "y": 4,
            "x_normalized": 0.4,
            "y_normalized": 0.4,
        },
        {
            "frame": 40,
            "time_seconds": 1.33,
            "segment_id": 3,
            "camera_state": "wide_gameplay",
            "hoop_side": "left",
            "usable_for_analytics": True,
            "x": 5,
            "y": 5,
            "x_normalized": 0.5,
            "y_normalized": 0.5,
        },
    ]

    all_frags = split_path_fragments(points, max_gap_frames=15, analytics_safe_only=False)
    assert len(all_frags) == 4  # seg1 block, transition, seg3 pt, gap-split seg3 pt

    safe = split_path_fragments(points, max_gap_frames=15, analytics_safe_only=True)
    assert len(safe) == 3
    assert all(f["usable_for_analytics"] for f in safe)
    assert safe[0]["point_count"] == 2
    assert safe[1]["point_count"] == 1
    assert safe[2]["point_count"] == 1


def test_attach_segments_and_fragments_summary_counts():
    trajectories = [
        {
            "person_group_id": 1,
            "team_id": 0,
            "points": [
                {"frame": 0, "time_seconds": 0.0, "x": 1, "y": 1, "x_normalized": 0.1, "y_normalized": 0.1},
                {"frame": 1, "time_seconds": 0.03, "x": 2, "y": 2, "x_normalized": 0.2, "y_normalized": 0.2},
                {"frame": 2, "time_seconds": 0.06, "x": 3, "y": 3, "x_normalized": 0.3, "y_normalized": 0.3},
                {"frame": 3, "time_seconds": 0.09, "x": 4, "y": 4, "x_normalized": 0.4, "y_normalized": 0.4},
            ],
        }
    ]
    segments = [
        {
            "segment_id": 1,
            "start_frame": 0,
            "end_frame": 1,
            "camera_state": "wide_gameplay",
            "hoop_side": "right",
            "usable_for_analytics": True,
        },
        {
            "segment_id": 2,
            "start_frame": 2,
            "end_frame": 2,
            "camera_state": "closeup",
            "hoop_side": "right",
            "usable_for_analytics": False,
        },
        {
            "segment_id": 3,
            "start_frame": 3,
            "end_frame": 3,
            "camera_state": "transition",
            "hoop_side": "left",
            "usable_for_analytics": False,
        },
    ]
    frame_index = build_frame_segment_index(segments)
    raw, analytics, stats = attach_segments_and_fragments(
        trajectories, frame_index=frame_index, max_gap_frames=15
    )

    assert stats["total_raw_points"] == 4
    assert stats["total_analytics_safe_points"] == 2
    assert stats["excluded_closeup_frames"] == 1
    assert stats["excluded_transition_frames"] == 1
    assert raw[0]["points"][0]["segment_id"] == 1
    assert raw[0]["points"][2]["camera_state"] == "closeup"
    assert analytics[0]["path_fragment_count"] == 1
    assert analytics[0]["point_count"] == 2


def test_generate_trajectories_for_job_writes_raw_and_analytics(tmp_path: Path):
    job_dir = tmp_path / "job123"
    job_dir.mkdir()

    tracks = []
    for frame in range(0, 10):
        tracks.append(
            {
                "frame_number": frame,
                "timestamp_seconds": frame / 30.0,
                "tracks": [
                    {"person_group_id": 1, "bbox_xyxy": [10, 20, 30, 40]},
                ],
            }
        )
    # Long gap then resume (should create a separate fragment).
    tracks.append(
        {
            "frame_number": 40,
            "timestamp_seconds": 40 / 30.0,
            "tracks": [
                {"person_group_id": 1, "bbox_xyxy": [12, 22, 32, 42]},
                {"person_group_id": 2, "bbox_xyxy": [100, 200, 140, 280]},
            ],
        }
    )
    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in tracks:
            handle.write(json.dumps(row) + "\n")

    (job_dir / "person_groups.json").write_text(
        json.dumps(
            {
                "person_groups": [
                    {"person_group_id": 1, "team_id": 0},
                    {"person_group_id": 2, "team_id": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 1920, "frame_height": 1080}),
        encoding="utf-8",
    )
    (job_dir / "quality.json").write_text(
        json.dumps({"cut_frames": [5]}),
        encoding="utf-8",
    )

    summary = generate_trajectories_for_job(
        job_dir,
        trajectory_config={"max_gap_frames": 15},
    )

    traj_path = job_dir / "analytics" / "trajectories.jsonl.gz"
    analytics_path = job_dir / "analytics" / "trajectories_analytics.jsonl.gz"
    summary_path = job_dir / "analytics" / "trajectory_summary.json"
    segments_path = job_dir / "analytics" / "game_segments.json"
    assert traj_path.is_file()
    assert analytics_path.is_file()
    assert summary_path.is_file()
    assert segments_path.is_file()
    assert summary["job_id"] == "job123"
    assert summary["tracked_players"] == 2
    assert summary["total_raw_points"] == 12
    assert "total_analytics_safe_points" in summary
    assert "path_fragment_count" in summary
    assert "excluded_transition_frames" in summary
    assert summary["path_fragment_count"] >= 2

    with gzip.open(traj_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 2
    assert rows[0]["person_group_id"] == 1
    assert "segment_id" in rows[0]["points"][0]
    assert "camera_state" in rows[0]["points"][0]
    assert "path_fragments" in rows[0]
    assert rows[0]["path_fragment_count"] >= 2

    with gzip.open(analytics_path, "rt", encoding="utf-8") as handle:
        safe_rows = [json.loads(line) for line in handle if line.strip()]
    assert safe_rows
    assert all("path_fragments" in row for row in safe_rows)
    assert all(
        frag["usable_for_analytics"]
        for row in safe_rows
        for frag in row["path_fragments"]
    )


def test_generate_trajectories_for_job_requires_tracks(tmp_path: Path):
    job_dir = tmp_path / "empty_job"
    job_dir.mkdir()
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 100, "frame_height": 100}),
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryInputError, match="Tracking output missing"):
        generate_trajectories_for_job(job_dir)


def test_resolve_frame_size_requires_snapshot_when_not_overridden(tmp_path: Path):
    with pytest.raises(TrajectoryInputError, match="Frame width/height"):
        resolve_frame_size(tmp_path)


def test_annotate_points_defaults_when_segment_missing():
    annotated = annotate_points_with_segments(
        [{"frame": 99, "time_seconds": 1.0, "x": 1, "y": 2, "x_normalized": 0.1, "y_normalized": 0.2}],
        {},
    )
    assert annotated[0]["segment_id"] is None
    assert annotated[0]["camera_state"] == "unknown"
    assert annotated[0]["usable_for_analytics"] is False
