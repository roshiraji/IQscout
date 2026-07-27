"""Court-band spatial filters for ball detections / preview."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.ball.spatial import (
    center_in_court_band,
    filter_balls_by_court_band,
    spatial_gates_from_cfg,
)
from src.analytics.service import get_ball_preview_for_job


def test_center_in_court_band_defaults():
    assert center_in_court_band(0.5, 0.5) is True
    assert center_in_court_band(0.5, 0.05) is False  # upper crowd
    assert center_in_court_band(0.5, 0.95) is False  # scorebug
    assert center_in_court_band(0.01, 0.5) is False  # far left


def test_filter_balls_by_court_band():
    dets = [
        {"bbox_xyxy": [40, 10, 60, 30], "confidence": 0.9},   # cy~20 → top
        {"bbox_xyxy": [40, 90, 60, 110], "confidence": 0.8},  # cy~100 mid of 200
        {"bbox_xyxy": [40, 180, 60, 200], "confidence": 0.7}, # bottom
    ]
    kept = filter_balls_by_court_band(
        dets, frame_width=200, frame_height=200, y_min=0.18, y_max=0.90
    )
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.8


def test_ball_preview_rejects_upper_crowd(tmp_path: Path):
    tracks = tmp_path / "ball_tracks.jsonl.gz"
    rows = [
        {
            "frame_number": 1,
            "timestamp_seconds": 0.05,
            "balls": [
                {
                    # Center near top of 100x100 frame → y=0.05
                    "bbox_xyxy": [40, 0, 60, 10],
                    "confidence": 0.95,
                    "source": "ball_yolo",
                }
            ],
        },
        {
            "frame_number": 2,
            "timestamp_seconds": 0.10,
            "balls": [
                {
                    "bbox_xyxy": [40, 40, 60, 60],
                    "confidence": 0.9,
                    "source": "ball_yolo",
                }
            ],
        },
    ]
    with gzip.open(tracks, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (tmp_path / "ball_summary.json").write_text(
        json.dumps({"frame_width": 100, "frame_height": 100, "fps": 20}),
        encoding="utf-8",
    )
    payload = get_ball_preview_for_job(tmp_path)
    assert payload["point_count"] == 1
    assert abs(payload["points"][0]["y_normalized"] - 0.5) < 1e-6
    assert payload["spatial_filter"]["enabled"] is True


def test_spatial_gates_from_cfg_defaults():
    gates = spatial_gates_from_cfg({})
    assert gates["enabled"] is True
    assert gates["y_min"] == 0.18
