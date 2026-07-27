"""
ScoutIQ FastAPI backend for the React dashboard.

Upload MP4 → intake validation → optional YOLO player labeling.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import threading  # noqa: E402 — used by /api/shutdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
# scoutiq/ on path enables `from src.pipeline...` (seg tracker + showcase reuse);
# src/ on path keeps the flat MVP imports (track_players, etc.) working.
for _p in (PROJECT_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Pin bundled ffmpeg before any video encode path runs (imageio / OpenCV helpers).
import os  # noqa: E402

_ffmpeg_bin = (
    PROJECT_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "imageio_ffmpeg"
    / "binaries"
    / "ffmpeg-win-x86_64-v7.1.exe"
)
if _ffmpeg_bin.is_file():
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(_ffmpeg_bin.resolve())

from heatmap import generate_player_heatmap  # noqa: E402
from player_labeling import run_player_labeling  # noqa: E402
from player_snapshot_selector import run_snapshot_selection  # noqa: E402
from track_players import render_selected_tracks, run_player_tracking  # noqa: E402
from video_intake_filter import save_intake_report, validate_video  # noqa: E402

from src.pipeline.atomic_json import write_json_atomic  # noqa: E402
from src.pipeline.player_track_pipeline import run_player_seg_tracking  # noqa: E402
from src.pipeline.job_manager import get_job_manager  # noqa: E402
from src.pipeline.resolve_job_identity import resolve_job_identity, resolve_manual_jersey  # noqa: E402
from src.analytics.service import (  # noqa: E402
    AnalyticsServiceError,
    generate_analytics_for_job,
    get_ball_preview_for_job,
    get_segments_for_job,
    get_timeline_for_job,
    get_trajectories_for_job,
)

INPUT_DIR = PROJECT_ROOT / "data" / "input_videos"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "intake_reports"
ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "annotated_videos"
HEATMAPS_DIR = PROJECT_ROOT / "outputs" / "heatmaps"
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "player_snapshots"
TRACKING_SELECTIONS_DIR = PROJECT_ROOT / "outputs" / "tracking_selections"
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"
TRACKING_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "tracking_videos"


class TrackingSelectionRequest(BaseModel):
    """User's chosen Person # / track boxes from the snapshot GUI."""

    job_id: str | None = None
    video_path: str | None = None
    snapshot_frame_number: int | None = None
    timestamp_seconds: float | None = None
    selected_person_group_ids: list[int] = Field(default_factory=list)
    selected_track_ids: list[int] = Field(default_factory=list)
    selected_players: list[dict] = Field(default_factory=list)
    run_tracking: bool = True


class AutoTrackRequest(BaseModel):
    """Auto-track every player with the trained seg model (best_player_seg.pt)."""

    video_path: str
    focus_labels: list[str] = Field(default_factory=list)
    max_frames: int | None = None


class ManualJerseyRequest(BaseModel):
    """User-entered jersey number for one tracked person."""

    track_id: int
    jersey_number: str


class OffenseTeamOverrideRequest(BaseModel):
    """Manual offense-team override (0/1) or auto (null) — no re-detect."""

    offense_team_id: int | None = None
    mode: str = "manual"  # "manual" | "auto"

app = FastAPI(title="ScoutIQ API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict[str, str | list[str]]:
    return {
        "status": "ok",
        "features": [
            "background_tracking_jobs",
            "canonical_snapshot",
            "tracking_selection",
            "person_group_stitching",
            "selected_player_retention",
            "coach_insights",
            "ball_tracking",
            "possession_offense_defense",
            "offense_team_override",
            "clean_isolate_ai",
            "full_clip_player_tracking",
            "identity_attachments",
            "identity_resolve",
            "intake",
            "app_shutdown",
            "wnba_team_roster",
        ],
    }


@app.get("/api/wnba/teams")
def list_wnba_roster_teams() -> dict:
    """Current WNBA teams from outputs/wnba_players.db for Home/Away pickers."""
    from wnba_player_database import list_wnba_teams

    teams = list_wnba_teams()
    return {
        "available": bool(teams),
        "team_count": len(teams),
        "teams": teams,
        "note": (
            "Select abbreviations (e.g. IND) so roster identity matching does "
            "not fail on typos like 'Indiana Fevers'."
            if teams
            else "Roster DB missing — rebuild with python -m src.wnba_player_database."
        ),
    }


@app.post("/api/shutdown")
def shutdown_app() -> JSONResponse:
    """Signal the desktop launcher to quit (writes quit.requested flag)."""
    runtime_dir = PROJECT_ROOT / ".scoutiq_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    quit_flag = runtime_dir / "quit.requested"
    quit_flag.write_text("1", encoding="utf-8")

    def _delayed_stop() -> None:
        # Give the HTTP response time to reach the browser, then free ports.
        time.sleep(0.8)
        stop_script = PROJECT_ROOT / "scripts" / "stop_services.ps1"
        if stop_script.is_file():
            try:
                import subprocess

                flags = 0
                if hasattr(subprocess, "CREATE_NO_WINDOW"):
                    flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-WindowStyle",
                        "Hidden",
                        "-File",
                        str(stop_script),
                    ],
                    cwd=str(PROJECT_ROOT),
                    creationflags=flags,
                )
            except Exception:
                pass

    threading.Thread(target=_delayed_stop, daemon=True).start()
    return JSONResponse({"status": "shutting_down"})


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    run_labeling: bool = False,
    run_snapshot: bool = True,
    league: str | None = Form(None),
    game_date: str | None = Form(None),
    home_team: str | None = Form(None),
    away_team: str | None = Form(None),
) -> dict:
    """Save/validate upload and enqueue one canonical full-clip tracking job."""
    _ensure_directories()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    # Unique storage prevents a later upload with the same filename from
    # overwriting a queued/running job's source video.
    destination = INPUT_DIR / f"{uuid.uuid4().hex[:10]}_{Path(file.filename).name}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    destination.write_bytes(content)

    validation = validate_video(destination)
    report_path = save_intake_report(
        validation,
        original_filename=file.filename,
        reports_dir=REPORTS_DIR,
        selected_file_path=destination,
    )
    if validation.get("metadata"):
        validation["metadata"]["filename"] = file.filename

    response: dict = {
        "validation": validation,
        "report_path": str(report_path.resolve()),
        "saved_path": str(destination.resolve()),
        "job_id": None,
        "job": None,
        "snapshot": None,
        "labeling": None,
    }

    if validation["usable"]:
        game_context = {
            key: value
            for key, value in {
                "league": league,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
            }.items()
            if value
        }
        job = get_job_manager().submit(
            destination,
            game_context=game_context,
        )
        response["job_id"] = job["job_id"]
        response["job"] = job

    if validation["usable"] and run_labeling:
        try:
            labeling = run_player_labeling(destination)
            response["labeling"] = labeling

            metadata = validation.get("metadata", {})
            width = int(metadata.get("width") or 1280)
            height = int(metadata.get("height") or 720)
            heatmap_path = HEATMAPS_DIR / f"{destination.stem}_heatmap.png"
            generate_player_heatmap(
                Path(labeling["detections_csv"]),
                heatmap_path,
                frame_width=width,
                frame_height=height,
            )
            response["labeling"]["heatmap_png"] = str(heatmap_path.resolve())
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Labeling failed after validation passed: {exc}",
            ) from exc

    return response


def _enrich_selected_players_with_identity(
    job_id: str, selected_players: list[dict]
) -> list[dict]:
    """Attach resolved roster names from identity_attachments onto Confirm payload.

    Cheap JSON read only — no OCR/roster recompute. Ensures focused preview
    labels use player_name when identity was resolved before Confirm.
    """
    if not selected_players:
        return selected_players
    identity_path = get_job_manager().artifact_dir(Path(job_id).name) / "identity_attachments.json"
    if not identity_path.is_file():
        return selected_players
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return selected_players

    by_track: dict[int, dict] = {}
    for item in payload.get("attachments", []) or []:
        try:
            by_track[int(item["track_id"])] = item
        except (KeyError, TypeError, ValueError):
            continue

    enriched: list[dict] = []
    for player in selected_players:
        row = dict(player)
        tid = row.get("track_id", row.get("person_group_id"))
        if tid is None:
            enriched.append(row)
            continue
        try:
            attachment = by_track.get(int(tid))
        except (TypeError, ValueError):
            attachment = None
        if attachment and attachment.get("player_name"):
            row["player_name"] = attachment.get("player_name")
            if attachment.get("jersey_number") is not None:
                row["jersey_number"] = attachment.get("jersey_number")
            if attachment.get("team_id") is not None:
                row["team_id"] = attachment.get("team_id")
            if attachment.get("status") is not None:
                row["identity_status"] = attachment.get("status")
            if attachment.get("confidence") is not None:
                row["identity_confidence"] = attachment.get("confidence")
        enriched.append(row)
    return enriched


@app.get("/api/jobs/{job_id}")
def get_tracking_job(job_id: str) -> dict:
    """Return durable background progress and canonical artifacts."""
    state = get_job_manager().get(Path(job_id).name)
    if state is None:
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    result = state.get("result")
    if result and result.get("snapshot"):
        identity_path = get_job_manager().artifact_dir(job_id) / "identity_attachments.json"
        if identity_path.is_file():
            try:
                payload = json.loads(identity_path.read_text(encoding="utf-8"))
                by_track = {
                    int(item["track_id"]): item
                    for item in payload.get("attachments", [])
                }
                for player in result["snapshot"].get("players", []):
                    attachment = by_track.get(int(player["track_id"]))
                    if attachment:
                        player.update(
                            {
                                "identity_status": attachment.get("status", "pending"),
                                "player_name": attachment.get("player_name"),
                                "team_id": attachment.get("team_id"),
                                "jersey_number": attachment.get("jersey_number"),
                                "identity_confidence": attachment.get("confidence"),
                                "identity_reason": attachment.get("reason"),
                            }
                        )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return state


def _identity_response(job_id: str) -> dict:
    artifact_dir = get_job_manager().artifact_dir(Path(job_id).name)
    identity_path = artifact_dir / "identity_attachments.json"
    if not identity_path.is_file():
        raise HTTPException(status_code=404, detail="Identity results not ready.")
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read identity results.") from exc

    identities = []
    for item in payload.get("attachments", []):
        evidence = item.get("evidence") or {}
        identities.append(
            {
                "track_id": item.get("track_id"),
                "status": item.get("status", "pending"),
                "player_name": item.get("player_name"),
                "player_id": item.get("player_id"),
                "team_id": item.get("team_id"),
                "jersey_number": item.get("jersey_number"),
                "confidence": item.get("confidence"),
                "score_margin": item.get("score_margin"),
                "reason": item.get("reason"),
                "ocr_jersey": evidence.get("ocr_jersey"),
                "ocr_consistency": evidence.get("ocr_consistency"),
                "identity_evidence_status": evidence.get("identity_evidence_status"),
                "identity_evidence_reason": evidence.get("identity_evidence_reason"),
                "candidate_scores": evidence.get("candidate_scores") or [],
            }
        )
    return {
        "job_id": job_id,
        "game_context": payload.get("game_context") or {},
        "identities": identities,
    }


@app.get("/api/identity/{job_id}")
def get_job_identity(job_id: str) -> dict:
    """Return roster/OCR identity attachments for a tracking job."""
    if get_job_manager().get(Path(job_id).name) is None:
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    return _identity_response(job_id)


@app.post("/api/identity/{job_id}/resolve")
def resolve_job_identity_endpoint(job_id: str) -> dict:
    """Run jersey OCR + roster matching for a completed tracking job."""
    state = get_job_manager().get(Path(job_id).name)
    if state is None:
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    if state.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Tracking job is {state.get('status')}; wait until ready.",
        )
    try:
        summary = resolve_job_identity(get_job_manager().artifact_dir(Path(job_id).name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Identity resolution failed: {exc}",
        ) from exc
    return {
        "message": "Identity resolution complete (best-frame jersey OCR).",
        "summary": summary,
        "identity": _identity_response(job_id),
    }


@app.post("/api/identity/{job_id}/manual-jersey")
def resolve_manual_jersey_endpoint(job_id: str, body: ManualJerseyRequest) -> dict:
    """Resolve one person from a manually entered jersey number."""
    state = get_job_manager().get(Path(job_id).name)
    if state is None:
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    if state.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Tracking job is {state.get('status')}; wait until ready.",
        )
    jersey = body.jersey_number.strip()
    if not jersey:
        raise HTTPException(status_code=400, detail="Jersey number is required.")
    try:
        result = resolve_manual_jersey(
            get_job_manager().artifact_dir(Path(job_id).name),
            track_id=body.track_id,
            jersey_number=jersey,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Manual jersey resolution failed: {exc}",
        ) from exc
    return {
        "message": f"Resolved person #{body.track_id} as {result['player_name']}.",
        "result": result,
        "identity": _identity_response(job_id),
    }


@app.get("/api/jobs/{job_id}/isolate-status")
def get_isolate_status(job_id: str, generation: str | None = None) -> dict:
    """Return readiness/progress for the latest temporal isolate render."""
    state = get_job_manager().get_isolate(Path(job_id).name)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="No isolate render has been queued. Re-Confirm the selection.",
        )
    if generation and state.get("generation") != generation:
        raise HTTPException(
            status_code=409,
            detail="This isolate render was replaced by a newer selection.",
        )
    return state


def _write_confirm_status(job_id: str, payload: dict) -> None:
    artifact_dir = get_job_manager().artifact_dir(Path(job_id).name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "confirm_status.json"
    data = {
        **payload,
        "job_id": Path(job_id).name,
        "updated_at": time.time(),
    }
    write_json_atomic(path, data)


@app.get("/api/jobs/{job_id}/confirm-status")
def get_confirm_status(job_id: str) -> dict:
    """Live Confirm progress while focused preview is rendering."""
    path = get_job_manager().artifact_dir(Path(job_id).name) / "confirm_status.json"
    if not path.is_file():
        return {
            "job_id": Path(job_id).name,
            "status": "idle",
            "stage": "idle",
            "progress": 0.0,
            "detail": "Waiting for Confirm…",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/possession")
def get_job_possession(job_id: str) -> dict:
    """Return possession / offense-defense summary for a job."""
    artifact_dir = get_job_manager().artifact_dir(Path(job_id).name)
    path = artifact_dir / "possession.json"
    override_path = artifact_dir / "offense_team_override.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Possession not available yet.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    override = None
    if override_path.is_file():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            override = None
    return {"job_id": job_id, "possession": data, "override": override}


def _analytics_job_dir(job_id: str) -> Path:
    safe_id = Path(job_id).name
    return get_job_manager().artifact_dir(safe_id)


def _raise_analytics_http(exc: AnalyticsServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/api/jobs/{job_id}/analytics/trajectories/generate")
def generate_job_analytics_trajectories(job_id: str) -> dict:
    """Build segments + trajectories from existing tracking output (no YOLO re-run)."""
    job_dir = _analytics_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    try:
        return generate_analytics_for_job(job_dir)
    except AnalyticsServiceError as exc:
        _raise_analytics_http(exc)
        raise  # pragma: no cover


@app.get("/api/jobs/{job_id}/analytics/trajectories")
def get_job_analytics_trajectories(job_id: str, variant: str = "analytics") -> dict:
    """Return generated trajectories (``variant=analytics`` or ``raw``)."""
    job_dir = _analytics_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    try:
        return get_trajectories_for_job(job_dir, variant=variant)
    except AnalyticsServiceError as exc:
        _raise_analytics_http(exc)
        raise  # pragma: no cover


@app.get("/api/jobs/{job_id}/analytics/segments")
def get_job_analytics_segments(job_id: str) -> dict:
    """Return gameplay camera segments for a job."""
    job_dir = _analytics_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    try:
        return get_segments_for_job(job_dir)
    except AnalyticsServiceError as exc:
        _raise_analytics_http(exc)
        raise  # pragma: no cover


@app.get("/api/jobs/{job_id}/analytics/timeline")
def get_job_analytics_timeline(job_id: str) -> dict:
    """Return camera/tracking game timeline events for a job."""
    job_dir = _analytics_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    try:
        return get_timeline_for_job(job_dir)
    except AnalyticsServiceError as exc:
        _raise_analytics_http(exc)
        raise  # pragma: no cover


@app.get("/api/jobs/{job_id}/analytics/ball-preview")
def get_job_analytics_ball_preview(job_id: str) -> dict:
    """Return normalized ball centers for trajectory preview (empty if unavailable)."""
    job_dir = _analytics_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    return get_ball_preview_for_job(job_dir)


@app.post("/api/jobs/{job_id}/offense-team")
def set_offense_team(job_id: str, body: OffenseTeamOverrideRequest) -> dict:
    """Set offense team (0/1) or Auto — recomputes Matchup roles only."""
    manager = get_job_manager()
    state = manager.get(Path(job_id).name)
    if state is None:
        raise HTTPException(status_code=404, detail="Tracking job not found.")
    if state.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Tracking job is {state.get('status')}; wait until ready.",
        )
    mode = (body.mode or "manual").lower()
    if mode not in {"manual", "auto"}:
        raise HTTPException(status_code=400, detail="mode must be manual or auto")
    offense_id = None if mode == "auto" else body.offense_team_id
    if mode == "manual" and offense_id not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail="offense_team_id must be 0 or 1 for manual override (or mode=auto).",
        )

    artifact_dir = manager.artifact_dir(Path(job_id).name)
    try:
        from src.ball.live import apply_offense_team_override
        from src.pipeline.coach_insights import compute_coach_insights
        from src.pipeline.player_track_pipeline import load_pipeline_config

        cfg = load_pipeline_config()
        possession_cfg = (cfg.get("ball") or {}).get("possession") or {}
        possession = apply_offense_team_override(
            artifact_dir,
            offense_team_id=offense_id,
            mode=mode,
            possession_cfg=possession_cfg,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Offense override failed: {exc}"
        ) from exc

    # Recompute Matchup Outcomes from existing selection without re-detect.
    coach_insights = None
    coach_narrative = None
    selection_summary_path = artifact_dir / "selection_summary.json"
    selected_csv = artifact_dir / "selected_tracks.csv"
    if selected_csv.is_file() and selection_summary_path.is_file():
        try:
            import csv

            rows = list(csv.DictReader(selected_csv.open(encoding="utf-8")))
            # Normalize numeric fields used by coach insights.
            for row in rows:
                for key in (
                    "frame_number",
                    "person_group_id",
                    "track_id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "center_x",
                    "center_y",
                    "stance_x",
                    "stance_y",
                    "timestamp_seconds",
                ):
                    if row.get(key) not in (None, ""):
                        try:
                            row[key] = float(row[key]) if "." in str(row[key]) else int(float(row[key]))
                        except (TypeError, ValueError):
                            pass
            summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))
            gids = [int(g) for g in (summary.get("selected_person_group_ids") or [])]
            label_by = {
                int(v): k for k, v in (summary.get("track_mapping") or {}).items()
            }
            # Team meta from person_groups / snapshot.
            team_meta: dict[int, dict] = {}
            groups_path = artifact_dir / "person_groups.json"
            if groups_path.is_file():
                groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
                for g in groups_payload.get("person_groups") or []:
                    gid = int(g["person_group_id"])
                    team_meta[gid] = {
                        "team_id": g.get("team_id"),
                        "jersey_color_name": g.get("jersey_color_name"),
                        "role": g.get("role"),
                        "player_label": label_by.get(gid) or g.get("label"),
                    }
            width = int((summary.get("quality") or {}).get("frame_width") or 1920)
            height = int((summary.get("quality") or {}).get("frame_height") or 1080)
            # Prefer snapshot dimensions if present.
            snap = artifact_dir / "snapshot.json"
            if snap.is_file():
                snap_data = json.loads(snap.read_text(encoding="utf-8"))
                width = int(snap_data.get("frame_width") or width)
                height = int(snap_data.get("frame_height") or height)
            coach_insights = compute_coach_insights(
                rows,
                source_frame_count=int(summary.get("source_frame_count") or 1),
                frame_width=width,
                frame_height=height,
                label_by_group=label_by,
                team_meta_by_group=team_meta,
                primary_pair=tuple(sorted(gids)[:2]) if len(gids) >= 2 else None,
                possession=possession,
            )
            coach_narrative = coach_insights.get("coach_narrative")
            (artifact_dir / "coach_insights.json").write_text(
                json.dumps(coach_insights, indent=2), encoding="utf-8"
            )
            summary["coach_insights"] = coach_insights
            summary["coach_narrative"] = coach_narrative
            summary["possession"] = possession
            summary["offense_team_id"] = possession.get("offense_team_id")
            summary["defense_team_id"] = possession.get("defense_team_id")
            selection_summary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            # Override still saved; insights refresh is best-effort.
            coach_insights = {"error": str(exc)}

    return {
        "job_id": job_id,
        "message": (
            "Offense team set to Auto."
            if mode == "auto"
            else (
                "Offense team set to Team Red."
                if offense_id == 0
                else "Offense team set to Team Blue."
                if offense_id == 1
                else f"Offense team set to team_id={offense_id}."
            )
        ),
        "possession": possession,
        "coach_insights": coach_insights,
        "coach_narrative": coach_narrative,
    }


@app.get("/api/jobs/{job_id}/files/{kind}")
def get_job_file(job_id: str, kind: str) -> FileResponse:
    """Serve canonical/focused artifacts without exposing arbitrary paths."""
    filenames = {
        "snapshot": ("snapshot.jpg", "image/jpeg"),
        "snapshot_masks": ("snapshot_masks.jpg", "image/jpeg"),
        "all": ("all_players.mp4", "video/mp4"),
        "focused": ("focused_players.mp4", "video/mp4"),
        "focused_masks": ("focused_masks.mp4", "video/mp4"),
        "focused_isolated": ("focused_isolated.mp4", "video/mp4"),
        "tracks": ("tracks.csv", "text/csv"),
        "selected_tracks": ("selected_tracks.csv", "text/csv"),
        "pair_distances": ("pair_distances.csv", "text/csv"),
        "quality": ("quality.json", "application/json"),
        "identity": ("identity_attachments.json", "application/json"),
        "person_groups": ("person_groups.json", "application/json"),
        "selection_summary": ("selection_summary.json", "application/json"),
        "coach_insights": ("coach_insights.json", "application/json"),
        "possession": ("possession.json", "application/json"),
        "ball_summary": ("ball_summary.json", "application/json"),
    }
    if kind not in filenames:
        raise HTTPException(status_code=404, detail="Unknown job artifact.")
    filename, media_type = filenames[kind]
    path = get_job_manager().artifact_dir(Path(job_id).name) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Job artifact not ready.")
    # Focused/selection artifacts are rewritten on each Confirm — avoid browsers
    # replaying a stale mp4 when the URL path is unchanged.
    headers = None
    if kind in {
        "focused",
        "focused_masks",
        "focused_isolated",
        "selected_tracks",
        "pair_distances",
        "coach_insights",
        "selection_summary",
        "possession",
    }:
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        headers=headers,
    )


@app.post("/api/tracking/selection")
def save_tracking_selection(body: TrackingSelectionRequest) -> dict:
    """Filter stitched person groups and render without rerunning inference."""
    _ensure_directories()

    selected_group_ids = list(body.selected_person_group_ids)
    if not selected_group_ids:
        selected_group_ids = [
            int(player["person_group_id"])
            for player in body.selected_players
            if player.get("person_group_id") is not None
        ]
    selected_ids = list(body.selected_track_ids)
    if not selected_ids:
        selected_ids = [
            int(player["track_id"])
            for player in body.selected_players
            if player.get("track_id") is not None
        ]
    if not selected_group_ids and not selected_ids and not body.selected_players:
        raise HTTPException(
            status_code=400,
            detail="Select at least two players for coach insights.",
        )

    effective_count = len(selected_group_ids) or len(selected_ids) or len(
        body.selected_players
    )
    if body.job_id and effective_count < 2:
        raise HTTPException(
            status_code=400,
            detail="Select at least two person groups for coach insights.",
        )

    if body.job_id:
        manager = get_job_manager()
        state = manager.get(body.job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Tracking job not found.")
        if state.get("status") != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Tracking job is {state.get('status')}; wait until ready.",
            )
        selected_players = _enrich_selected_players_with_identity(
            body.job_id, list(body.selected_players)
        )
        selection_path = manager.artifact_dir(body.job_id) / "selection.json"
        payload = {
            "job_id": body.job_id,
            "selected_person_group_ids": selected_group_ids,
            "selected_track_ids": selected_ids,
            "selected_players": selected_players,
            "selected_labels": [
                player.get("player_name") or player.get("label")
                for player in selected_players
            ],
        }
        selection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_confirm_status(
            body.job_id,
            {
                "status": "processing",
                "stage": "saving_selection",
                "progress": 0.02,
                "detail": f"Confirming {effective_count} selected player(s)…",
            },
        )

        def _confirm_progress(stage: str, progress: float, detail: str | None = None) -> None:
            _write_confirm_status(
                body.job_id,
                {
                    "status": "processing",
                    "stage": stage,
                    "progress": progress,
                    "detail": detail or stage.replace("_", " "),
                },
            )

        try:
            # When person groups are provided, do not also pass track_ids —
            # snapshot track_id mirrors person_group_id and must not be remapped
            # through raw ByteTrack track_to_group.
            tracking = render_selected_tracks(
                manager.artifact_dir(body.job_id),
                None if selected_group_ids else selected_ids,
                selected_person_group_ids=selected_group_ids or None,
                selected_players=selected_players,
                quiet=True,
                progress_cb=_confirm_progress,
            )
            _write_confirm_status(
                body.job_id,
                {
                    "status": "processing",
                    "stage": "queuing_isolate",
                    "progress": 0.97,
                    "detail": "Queuing isolate preview in the background…",
                },
            )
            isolate_generation = uuid.uuid4().hex
            from src.pipeline.player_track_pipeline import load_pipeline_config

            pipeline_cfg = load_pipeline_config()
            isolate_cfg = dict(pipeline_cfg.get("isolate") or {})
            isolate_state = manager.submit_isolate(
                body.job_id,
                set(tracking["selected_person_group_ids"]),
                isolate_generation,
                isolate_cfg=isolate_cfg,
            )
            tracking.update(
                {
                    "isolated_tracking_video": None,
                    "isolation_available": False,
                    "isolation_status": isolate_state["status"],
                    "isolation_generation": isolate_generation,
                    "isolation_mode": isolate_state.get("effective_mode"),
                    "isolation_requested_mode": isolate_state.get("requested_mode"),
                    "isolation_method": isolate_state.get("method"),
                    "isolation_ai_fallback": isolate_state.get("ai_fallback"),
                }
            )
            summary_path = manager.artifact_dir(body.job_id) / "selection_summary.json"
            summary_path.write_text(json.dumps(tracking, indent=2), encoding="utf-8")
            _write_confirm_status(
                body.job_id,
                {
                    "status": "ready",
                    "stage": "ready",
                    "progress": 1.0,
                    "detail": "Focused preview ready. Isolate continues in the background.",
                },
            )
        except ValueError as exc:
            _write_confirm_status(
                body.job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "progress": 0.0,
                    "detail": str(exc),
                    "error": str(exc),
                },
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _write_confirm_status(
                body.job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "progress": 0.0,
                    "detail": f"Focused preview rendering failed: {exc}",
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=500,
                detail=f"Focused preview rendering failed: {exc}",
            ) from exc
        player_count = len(
            tracking.get("selected_person_group_ids")
            or selected_group_ids
            or selected_ids
        )
        retention = tracking.get("selected_player_retention")
        retention_note = (
            f" Retention {round(float(retention) * 100)}%."
            if retention is not None
            else ""
        )
        return {
            "message": (
                f"Focused full-clip preview ready for {player_count} person group(s); "
                f"canonical inference was reused.{retention_note}"
            ),
            "selection_path": str(selection_path.resolve()),
            "selected_labels": payload["selected_labels"],
            "player_count": player_count,
            "tracking": tracking,
            "coach_insights": tracking.get("coach_insights"),
            "coach_narrative": tracking.get("coach_narrative"),
        }

    if not body.video_path:
        raise HTTPException(status_code=400, detail="video_path is required for legacy selection.")
    video_path = Path(body.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")

    stem = video_path.stem
    selection_path = TRACKING_SELECTIONS_DIR / f"{stem}_tracking_selection.json"

    payload = {
        "video_path": body.video_path,
        "snapshot_frame_number": body.snapshot_frame_number,
        "timestamp_seconds": body.timestamp_seconds,
        "selected_players": body.selected_players,
        "selected_labels": [player["label"] for player in body.selected_players],
    }

    selection_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    response = {
        "message": "Tracking selection saved.",
        "selection_path": str(selection_path.resolve()),
        "selected_labels": payload["selected_labels"],
        "player_count": len(body.selected_players),
        "tracking": None,
    }

    if body.run_tracking:
        try:
            response["tracking"] = run_player_tracking(
                selection_data=payload,
                quiet=True,
            )
            response["message"] = (
                f"Tracking complete for {len(response['tracking']['track_mapping'])} player(s)."
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Tracking failed after saving selection: {exc}",
            ) from exc

    return response


@app.post("/api/track/auto")
def auto_track_players(body: AutoTrackRequest) -> dict:
    """Detect + segment + track ALL players with best_player_seg.pt (no GDINO)."""
    _ensure_directories()

    video_path = Path(body.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")

    try:
        summary = run_player_seg_tracking(
            video_path,
            focus_labels=body.focus_labels or None,
            max_frames=body.max_frames,
            quiet=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Auto tracking failed: {exc}",
        ) from exc

    video_name = Path(summary["tracking_video"]).name
    return {
        "message": (
            f"Auto-tracked {len(summary['track_mapping'])} player(s) "
            f"across {summary['frames_tracked']} frame(s)."
        ),
        "tracking": summary,
        "tracking_video_filename": video_name,
    }


@app.get("/api/files/tracking/{filename}")
def get_tracking_video(filename: str) -> FileResponse:
    """Serve a tracked-player preview MP4."""
    safe_name = Path(filename).name
    video_path = TRACKING_VIDEOS_DIR / safe_name
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Tracking video not found.")
    return FileResponse(video_path, media_type="video/mp4", filename=safe_name)


@app.get("/api/files/snapshot/{filename}")
def get_snapshot_image(filename: str) -> FileResponse:
    """Serve a labeled player snapshot JPG."""
    safe_name = Path(filename).name
    image_path = SNAPSHOT_DIR / safe_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot image not found.")
    return FileResponse(image_path, media_type="image/jpeg", filename=safe_name)


@app.get("/api/files/heatmap/{filename}")
def get_heatmap_image(filename: str) -> FileResponse:
    """Serve a generated player-position heatmap PNG."""
    safe_name = Path(filename).name
    image_path = HEATMAPS_DIR / safe_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not found.")
    return FileResponse(image_path, media_type="image/png", filename=safe_name)


@app.get("/api/files/annotated/{filename}")
def get_annotated_video(filename: str) -> FileResponse:
    """Stream a processed annotated MP4 to the React video player."""
    safe_name = Path(filename).name
    video_path = ANNOTATED_DIR / safe_name
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Annotated video not found.")
    return FileResponse(video_path, media_type="video/mp4", filename=safe_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(PROJECT_ROOT),
    )
