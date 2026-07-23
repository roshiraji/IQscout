"""API-facing helpers for job analytics (trajectories + segments)."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Literal

from src.analytics.segments import SEGMENTS_FILENAME, generate_segments_for_job
from src.analytics.trajectories import (
    SUMMARY_FILENAME,
    TRAJECTORIES_ANALYTICS_FILENAME,
    TRAJECTORIES_FILENAME,
    TrajectoryInputError,
    generate_trajectories_for_job,
    load_segments_payload,
)

logger = logging.getLogger(__name__)

Variant = Literal["raw", "analytics"]


class AnalyticsServiceError(Exception):
    """Mapped to HTTP errors by the FastAPI layer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalyticsServiceError(404, f"Missing artifact: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyticsServiceError(500, f"Could not read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalyticsServiceError(500, f"Invalid JSON object in {path.name}")
    return payload


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalyticsServiceError(404, f"Missing artifact: {path.name}")
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyticsServiceError(500, f"Could not read {path.name}: {exc}") from exc
    return rows


def require_tracking_artifacts(job_dir: Path) -> Path:
    tracks = job_dir / "tracks.jsonl.gz"
    if not tracks.is_file():
        alt = job_dir / "tracks.jsonl"
        if alt.is_file():
            return alt
        raise AnalyticsServiceError(
            404,
            "Tracking output is missing (tracks.jsonl.gz). Run Analyze first.",
        )
    return tracks


def require_frame_metadata(job_dir: Path) -> None:
    snapshot = job_dir / "snapshot.json"
    if not snapshot.is_file():
        raise AnalyticsServiceError(
            422,
            "Video metadata is unavailable (snapshot.json with frame_width/frame_height).",
        )
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyticsServiceError(
            422,
            "Video metadata is unavailable (could not read snapshot.json).",
        ) from exc
    width = payload.get("frame_width")
    height = payload.get("frame_height")
    try:
        ok = int(width) > 0 and int(height) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise AnalyticsServiceError(
            422,
            "Video metadata is unavailable (snapshot.json missing frame_width/frame_height).",
        )


def hoop_status(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "hoop_tracks.jsonl.gz"
    available = path.is_file()
    return {
        "hoop_available": available,
        "warning": (
            None
            if available
            else "Hoop detections unavailable; segmentation used scene/player-layout signals with reduced confidence."
        ),
    }


def generate_analytics_for_job(job_dir: str | Path) -> dict[str, Any]:
    """Generate segments (if needed) + trajectories. Does not re-run YOLO/ByteTrack."""
    job_dir = Path(job_dir).resolve()
    require_tracking_artifacts(job_dir)
    require_frame_metadata(job_dir)
    hoop = hoop_status(job_dir)

    try:
        segments_payload = generate_segments_for_job(job_dir)
    except FileNotFoundError as exc:
        raise AnalyticsServiceError(404, str(exc)) from exc
    except TrajectoryInputError as exc:
        raise AnalyticsServiceError(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Segment generation failed for %s", job_dir.name)
        raise AnalyticsServiceError(500, f"Segment generation failed: {exc}") from exc

    usable_segments = [
        s
        for s in (segments_payload.get("segments") or [])
        if isinstance(s, dict) and s.get("usable_for_analytics")
    ]
    if not usable_segments:
        raise AnalyticsServiceError(
            422,
            "No usable gameplay segment found (no wide_gameplay segments).",
        )

    try:
        summary = generate_trajectories_for_job(
            job_dir,
            ensure_segments=False,
            segments_payload=segments_payload,
        )
    except TrajectoryInputError as exc:
        raise AnalyticsServiceError(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise AnalyticsServiceError(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Trajectory generation failed for %s", job_dir.name)
        raise AnalyticsServiceError(500, f"Trajectory generation failed: {exc}") from exc

    if int(summary.get("total_analytics_safe_points") or 0) <= 0:
        raise AnalyticsServiceError(
            422,
            "No usable gameplay segment points found after filtering closeups/transitions.",
        )

    density_payload = None
    try:
        from src.analytics.heatmap_density import generate_density_for_job

        density_cfg = {}
        try:
            from src.pipeline.player_track_pipeline import load_pipeline_config

            density_cfg = (load_pipeline_config().get("analytics") or {}).get(
                "heatmap_density"
            ) or {}
        except Exception:  # noqa: BLE001
            density_cfg = {}
        density_payload = generate_density_for_job(job_dir, config=density_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Density generation skipped for %s: %s", job_dir.name, exc)

    timeline_payload = None
    try:
        from src.analytics.timeline import generate_timeline_for_job

        timeline_payload = generate_timeline_for_job(
            job_dir, segments_payload=segments_payload
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Timeline generation skipped for %s: %s", job_dir.name, exc)

    warnings: list[str] = []
    if hoop["warning"]:
        warnings.append(str(hoop["warning"]))
    if segments_payload.get("confidence") == "reduced":
        warnings.append("Segment confidence is reduced.")

    artifacts = dict(summary.get("artifacts") or {})
    if density_payload is not None:
        artifacts["heatmap_density"] = "analytics/heatmap_density.json"
    if timeline_payload is not None:
        artifacts["game_timeline"] = "analytics/game_timeline.json"

    return {
        "job_id": job_dir.name,
        "ok": True,
        "warnings": warnings,
        "hoop_available": hoop["hoop_available"],
        "segments_confidence": segments_payload.get("confidence"),
        "segment_summary": segments_payload.get("summary"),
        "trajectory_summary": summary,
        "heatmap_density": (
            {
                "point_count": density_payload.get("point_count"),
                "bins_x": density_payload.get("bins_x"),
                "bins_y": density_payload.get("bins_y"),
                "max_count": density_payload.get("max_count"),
                "coordinate_space": density_payload.get("coordinate_space"),
            }
            if density_payload
            else None
        ),
        "timeline_summary": (
            timeline_payload.get("summary") if timeline_payload else None
        ),
        "artifacts": artifacts,
    }


def get_segments_for_job(job_dir: str | Path) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    path = job_dir / "analytics" / SEGMENTS_FILENAME
    payload = load_segments_payload(path)
    if payload is None:
        raise AnalyticsServiceError(
            404,
            "Segments not found. POST /api/jobs/{job_id}/analytics/trajectories/generate first.",
        )
    return payload


def get_trajectories_for_job(
    job_dir: str | Path,
    *,
    variant: str = "analytics",
) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    variant_norm = (variant or "analytics").strip().lower()
    if variant_norm not in {"raw", "analytics"}:
        raise AnalyticsServiceError(400, "variant must be 'raw' or 'analytics'")

    summary_path = job_dir / "analytics" / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise AnalyticsServiceError(
            404,
            "Trajectories not found. POST /api/jobs/{job_id}/analytics/trajectories/generate first.",
        )
    summary = _read_json(summary_path)

    filename = (
        TRAJECTORIES_FILENAME
        if variant_norm == "raw"
        else TRAJECTORIES_ANALYTICS_FILENAME
    )
    rows = load_jsonl_gz(job_dir / "analytics" / filename)
    return {
        "job_id": job_dir.name,
        "variant": variant_norm,
        "coordinate_space": summary.get("coordinate_space"),
        "summary": summary,
        "trajectories": rows,
        "trajectory_count": len(rows),
    }


def get_timeline_for_job(job_dir: str | Path) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    path = job_dir / "analytics" / "game_timeline.json"
    if not path.is_file():
        raise AnalyticsServiceError(
            404,
            "Timeline not found. POST /api/jobs/{job_id}/analytics/trajectories/generate first.",
        )
    return _read_json(path)
