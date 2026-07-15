"""
ScoutIQ FastAPI backend for the React dashboard.

Upload MP4 → intake validation → optional YOLO player labeling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
# scoutiq/ on path enables `from src.pipeline...` (seg tracker + showcase reuse);
# src/ on path keeps the flat MVP imports (track_players, etc.) working.
for _p in (PROJECT_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heatmap import generate_player_heatmap  # noqa: E402
from player_labeling import run_player_labeling  # noqa: E402
from player_snapshot_selector import run_snapshot_selection  # noqa: E402
from track_players import run_player_tracking  # noqa: E402
from video_intake_filter import save_intake_report, validate_video  # noqa: E402

from src.pipeline.player_track_pipeline import run_player_seg_tracking  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "data" / "input_videos"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "intake_reports"
ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "annotated_videos"
HEATMAPS_DIR = PROJECT_ROOT / "outputs" / "heatmaps"
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "player_snapshots"
TRACKING_SELECTIONS_DIR = PROJECT_ROOT / "outputs" / "tracking_selections"
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"
TRACKING_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "tracking_videos"


class TrackingSelectionRequest(BaseModel):
    """User's chosen Player_# boxes from the snapshot GUI."""

    video_path: str
    snapshot_frame_number: int
    timestamp_seconds: float
    selected_players: list[dict] = Field(default_factory=list)
    run_tracking: bool = True


class AutoTrackRequest(BaseModel):
    """Auto-track every player with the trained seg model (best_player_seg.pt)."""

    video_path: str
    focus_labels: list[str] = Field(default_factory=list)
    max_frames: int | None = None

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
        "features": ["snapshot", "tracking_selection", "player_tracking", "intake", "labeling"],
    }


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    run_labeling: bool = False,
    run_snapshot: bool = True,
) -> dict:
    """Save upload, validate intake, snapshot players, optional full labeling."""
    _ensure_directories()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    destination = INPUT_DIR / Path(file.filename).name
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

    response: dict = {
        "validation": validation,
        "report_path": str(report_path.resolve()),
        "saved_path": str(destination.resolve()),
        "snapshot": None,
        "labeling": None,
    }

    if validation["usable"] and run_snapshot:
        try:
            response["snapshot"] = run_snapshot_selection(destination, quiet=True)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Player snapshot selection failed: {exc}",
            ) from exc

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


@app.post("/api/tracking/selection")
def save_tracking_selection(body: TrackingSelectionRequest) -> dict:
    """Save selected players and run ByteTrack from the snapshot frame onward."""
    _ensure_directories()

    if not body.selected_players:
        raise HTTPException(status_code=400, detail="Select at least one player to track.")

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
