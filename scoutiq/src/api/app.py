"""FastAPI endpoints for resolved clips search."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPS_MANIFEST = APP_ROOT / "outputs" / "identity" / "clips" / "clips_manifest.json"
DEFAULT_ROSTER_JSONL = APP_ROOT / "outputs" / "identity" / "rosters" / "wnba_rosters_2026.jsonl"

app = FastAPI(title="ScoutIQ Identity API", version="0.1.0")


def _load_clips(path: Path = DEFAULT_CLIPS_MANIFEST) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("clips", [])


def _load_rosters(path: Path = DEFAULT_ROSTER_JSONL) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@app.get("/search/clips")
def search_clips(
    player_name: str | None = Query(default=None),
    team_id: str | None = Query(default=None),
    jersey_number: str | None = Query(default=None),
    video_id: str | None = Query(default=None),
) -> dict:
    clips = _load_clips()
    rosters = _load_rosters()
    name_by_player_id = {r["player_id"]: r.get("full_name") for r in rosters}
    team_by_player_id = {r["player_id"]: r.get("team_id") for r in rosters}
    jersey_by_player_id = {r["player_id"]: r.get("jersey_number") for r in rosters}

    out = []
    for clip in clips:
        pid = clip.get("resolved_player_id")
        clip_player = name_by_player_id.get(pid)
        clip_team = team_by_player_id.get(pid)
        clip_jersey = jersey_by_player_id.get(pid)

        if player_name and (not clip_player or player_name.lower() not in clip_player.lower()):
            continue
        if team_id and str(clip_team) != str(team_id):
            continue
        if jersey_number and str(clip_jersey) != str(jersey_number):
            continue
        if video_id and clip.get("video_id") != video_id:
            continue
        out.append({**clip, "player_name": clip_player, "team_id": clip_team, "jersey_number": clip_jersey})
    return {"count": len(out), "clips": out}


@app.get("/search/health")
def health() -> dict:
    return {"status": "ok", "clips_manifest": str(DEFAULT_CLIPS_MANIFEST)}


@app.post("/resolve/tracklet")
def resolve_tracklet_debug(payload: dict) -> dict:
    if "tracklet_id" not in payload:
        raise HTTPException(status_code=400, detail="tracklet_id is required.")
    # Debug placeholder for direct resolver introspection.
    return {"status": "review", "message": "Use resolver CLI for full scoring.", "input": payload}
