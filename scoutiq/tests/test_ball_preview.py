"""Ball preview helper for trajectory UI (Segment 4)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.analytics.service import get_ball_preview_for_job


def test_ball_preview_soft_empty(tmp_path: Path):
    payload = get_ball_preview_for_job(tmp_path)
    assert payload["available"] is False
    assert payload["points"] == []
    assert payload["point_count"] == 0


def test_ball_preview_normalizes_centers(tmp_path: Path):
    tracks = tmp_path / "ball_tracks.jsonl.gz"
    with gzip.open(tracks, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "frame_number": 10,
                    "timestamp_seconds": 0.5,
                    "balls": [
                        {
                            "bbox_xyxy": [40, 40, 60, 60],
                            "confidence": 0.9,
                            "source": "ball_yolo",
                        }
                    ],
                }
            )
            + "\n"
        )
    (tmp_path / "ball_summary.json").write_text(
        json.dumps({"frame_width": 100, "frame_height": 100, "fps": 20}),
        encoding="utf-8",
    )
    payload = get_ball_preview_for_job(tmp_path)
    assert payload["available"] is True
    assert payload["point_count"] == 1
    pt = payload["points"][0]
    assert abs(pt["x_normalized"] - 0.5) < 1e-6
    assert abs(pt["y_normalized"] - 0.5) < 1e-6


def test_ball_preview_drops_soft_sources_and_breaks_jumps(tmp_path: Path):
    tracks = tmp_path / "ball_tracks.jsonl.gz"
    rows = [
        {
            "frame_number": 1,
            "timestamp_seconds": 0.05,
            "balls": [
                {
                    "bbox_xyxy": [40, 40, 60, 60],
                    "confidence": 0.9,
                    "source": "ball_yolo",
                }
            ],
        },
        {
            "frame_number": 2,
            "timestamp_seconds": 0.10,
            "balls": [
                {
                    "bbox_xyxy": [42, 42, 62, 62],
                    "confidence": 0.5,
                    "source": "coast",
                    "interpolated": True,
                }
            ],
        },
        {
            "frame_number": 3,
            "timestamp_seconds": 0.15,
            "balls": [
                {
                    # Far jump but still inside court band (y≈0.55).
                    "bbox_xyxy": [90, 45, 98, 65],
                    "confidence": 0.95,
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
    assert payload["point_count"] == 2
    assert payload["points"][0]["trail_break_before"] is False
    assert payload["points"][1]["trail_break_before"] is True
    assert all(p.get("source") == "ball_yolo" for p in payload["points"])