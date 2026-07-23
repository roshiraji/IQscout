"""Tests for analytics API service + FastAPI routes (Phase 6)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.analytics.service import (
    AnalyticsServiceError,
    generate_analytics_for_job,
    get_segments_for_job,
    get_trajectories_for_job,
)


def _write_minimal_job(job_dir: Path, *, with_cut: bool = True) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for frame in range(0, 20):
        tracks.append(
            {
                "frame_number": frame,
                "timestamp_seconds": frame / 30.0,
                "tracks": [
                    {"person_group_id": 1, "bbox_xyxy": [10, 20, 40, 80]},
                    {"person_group_id": 2, "bbox_xyxy": [50, 20, 80, 80]},
                    {"person_group_id": 3, "bbox_xyxy": [90, 20, 120, 80]},
                    {"person_group_id": 4, "bbox_xyxy": [130, 20, 160, 80]},
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
                    {"person_group_id": 2, "team_id": 0},
                    {"person_group_id": 3, "team_id": 1},
                    {"person_group_id": 4, "team_id": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 1280, "frame_height": 720}),
        encoding="utf-8",
    )
    (job_dir / "quality.json").write_text(
        json.dumps({"cut_frames": [8] if with_cut else []}),
        encoding="utf-8",
    )


def test_generate_analytics_for_job_soft_fails_without_hoop(tmp_path: Path):
    job_dir = tmp_path / "job_a"
    _write_minimal_job(job_dir)
    result = generate_analytics_for_job(job_dir)
    assert result["ok"] is True
    assert result["hoop_available"] is False
    assert result["segments_confidence"] == "reduced"
    assert result["warnings"]
    assert result["trajectory_summary"]["total_analytics_safe_points"] > 0
    assert (job_dir / "analytics" / "trajectories_analytics.jsonl.gz").is_file()
    assert (job_dir / "analytics" / "game_segments.json").is_file()


def test_generate_analytics_missing_tracks(tmp_path: Path):
    job_dir = tmp_path / "empty"
    job_dir.mkdir()
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": 100, "frame_height": 100}),
        encoding="utf-8",
    )
    with pytest.raises(AnalyticsServiceError) as exc:
        generate_analytics_for_job(job_dir)
    assert exc.value.status_code == 404
    assert "Tracking output is missing" in exc.value.detail


def test_generate_analytics_missing_metadata(tmp_path: Path):
    job_dir = tmp_path / "nometa"
    job_dir.mkdir()
    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "frame_number": 0,
                    "timestamp_seconds": 0.0,
                    "tracks": [{"person_group_id": 1, "bbox_xyxy": [1, 2, 3, 4]}],
                }
            )
            + "\n"
        )
    with pytest.raises(AnalyticsServiceError) as exc:
        generate_analytics_for_job(job_dir)
    assert exc.value.status_code == 422
    assert "Video metadata" in exc.value.detail


def test_generate_analytics_no_usable_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_dir = tmp_path / "nousable"
    _write_minimal_job(job_dir, with_cut=False)

    def _fake_segments(_job_dir, **_kwargs):
        return {
            "confidence": "reduced",
            "segments": [
                {
                    "segment_id": 1,
                    "start_frame": 0,
                    "end_frame": 19,
                    "camera_state": "transition",
                    "hoop_side": "unknown",
                    "usable_for_analytics": False,
                }
            ],
            "summary": {"usable_segment_count": 0},
        }

    monkeypatch.setattr(
        "src.analytics.service.generate_segments_for_job",
        _fake_segments,
    )
    with pytest.raises(AnalyticsServiceError) as exc:
        generate_analytics_for_job(job_dir)
    assert exc.value.status_code == 422
    assert "No usable gameplay segment" in exc.value.detail


def test_get_trajectories_and_segments_after_generate(tmp_path: Path):
    job_dir = tmp_path / "job_b"
    _write_minimal_job(job_dir)
    generate_analytics_for_job(job_dir)

    segs = get_segments_for_job(job_dir)
    assert "segments" in segs
    analytics = get_trajectories_for_job(job_dir, variant="analytics")
    assert analytics["variant"] == "analytics"
    assert analytics["trajectory_count"] >= 1
    raw = get_trajectories_for_job(job_dir, variant="raw")
    assert raw["variant"] == "raw"
    assert raw["trajectories"][0]["points"][0]["segment_id"] is not None


def test_fastapi_analytics_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "apijob123"
    job_dir = tmp_path / job_id
    _write_minimal_job(job_dir)

    class _FakeManager:
        def artifact_dir(self, jid: str) -> Path:
            return tmp_path / Path(jid).name

        def get(self, jid: str) -> dict:
            return {"job_id": jid, "status": "ready"}

    monkeypatch.setattr("app.api_server.get_job_manager", lambda: _FakeManager())

    # Import after path setup used by api_server.
    from app import api_server

    client = TestClient(api_server.app)

    missing = client.get("/api/jobs/missingjob/analytics/segments")
    assert missing.status_code == 404

    before = client.get(f"/api/jobs/{job_id}/analytics/trajectories")
    assert before.status_code == 404

    generated = client.post(f"/api/jobs/{job_id}/analytics/trajectories/generate")
    assert generated.status_code == 200, generated.text
    body = generated.json()
    assert body["ok"] is True
    assert body["trajectory_summary"]["total_raw_points"] > 0

    segs = client.get(f"/api/jobs/{job_id}/analytics/segments")
    assert segs.status_code == 200
    assert isinstance(segs.json().get("segments"), list)

    traj = client.get(f"/api/jobs/{job_id}/analytics/trajectories?variant=analytics")
    assert traj.status_code == 200
    assert traj.json()["trajectory_count"] >= 1

    timeline = client.get(f"/api/jobs/{job_id}/analytics/timeline")
    assert timeline.status_code == 200
    body = timeline.json()
    assert "events" in body
    assert body["summary"]["event_count"] >= 1

    bad_variant = client.get(f"/api/jobs/{job_id}/analytics/trajectories?variant=court")
    assert bad_variant.status_code == 400
