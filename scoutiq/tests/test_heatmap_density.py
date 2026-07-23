"""Tests for analytics-safe heatmap density (Phase 8)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.analytics.heatmap_density import (
    build_density_grid,
    generate_density_for_job,
    iter_analytics_points,
)


def test_build_density_grid_bins_normalized_points():
    points = [
        {"x_normalized": 0.1, "y_normalized": 0.1},
        {"x_normalized": 0.1, "y_normalized": 0.1},
        {"x_normalized": 0.9, "y_normalized": 0.9},
    ]
    density = build_density_grid(points, bins_x=10, bins_y=10)
    assert density["point_count"] == 3
    assert density["max_count"] == 2
    assert density["coordinate_space"] == "normalized screen-space coordinates"
    assert density["grid"][1][1] == 2
    assert density["grid"][9][9] == 1


def test_iter_analytics_points_excludes_unusable_and_honors_filters():
    trajectories = [
        {
            "person_group_id": 1,
            "team_id": 0,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "points": [
                        {
                            "x_normalized": 0.2,
                            "y_normalized": 0.3,
                            "time_seconds": 1.0,
                            "segment_id": 1,
                            "usable_for_analytics": True,
                        },
                        {
                            "x_normalized": 0.4,
                            "y_normalized": 0.5,
                            "time_seconds": 5.0,
                            "segment_id": 2,
                            "usable_for_analytics": True,
                        },
                    ],
                },
                {
                    "usable_for_analytics": False,
                    "points": [
                        {
                            "x_normalized": 0.8,
                            "y_normalized": 0.8,
                            "time_seconds": 2.0,
                            "segment_id": 9,
                            "usable_for_analytics": False,
                        }
                    ],
                },
            ],
        },
        {
            "person_group_id": 2,
            "team_id": 1,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "points": [
                        {
                            "x_normalized": 0.55,
                            "y_normalized": 0.55,
                            "time_seconds": 1.5,
                            "segment_id": 1,
                            "usable_for_analytics": True,
                        }
                    ],
                }
            ],
        },
    ]

    all_safe = iter_analytics_points(trajectories, analytics_safe_only=True)
    assert len(all_safe) == 3

    team_red = iter_analytics_points(
        trajectories, team_ids={0}, analytics_safe_only=True
    )
    assert len(team_red) == 2

    seg1 = iter_analytics_points(
        trajectories, segment_id=1, analytics_safe_only=True
    )
    assert len(seg1) == 2

    early = iter_analytics_points(
        trajectories, time_end=2.0, analytics_safe_only=True
    )
    assert len(early) == 2


def test_generate_density_for_job_writes_artifact(tmp_path: Path):
    job_dir = tmp_path / "densjob"
    analytics = job_dir / "analytics"
    analytics.mkdir(parents=True)
    rows = [
        {
            "person_group_id": 7,
            "team_id": 0,
            "path_fragments": [
                {
                    "usable_for_analytics": True,
                    "points": [
                        {
                            "x_normalized": 0.25,
                            "y_normalized": 0.5,
                            "usable_for_analytics": True,
                        },
                        {
                            "x_normalized": 0.26,
                            "y_normalized": 0.51,
                            "usable_for_analytics": True,
                        },
                    ],
                }
            ],
        }
    ]
    with gzip.open(analytics / "trajectories_analytics.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    payload = generate_density_for_job(
        job_dir, config={"bins_x": 20, "bins_y": 10}
    )
    out = analytics / "heatmap_density.json"
    assert out.is_file()
    assert payload["point_count"] == 2
    assert payload["bins_x"] == 20
    assert payload["analytics_safe_only"] is True
    assert "normalized screen-space" in payload["coordinate_space"]
