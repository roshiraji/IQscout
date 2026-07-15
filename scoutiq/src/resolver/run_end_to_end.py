"""Orchestrate end-to-end identity pipeline on one video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.clips.cut_resolved_clips import cut_resolved_clips
from src.crops.build_tracklet_crops import build_tracklet_crops
from src.db.ingest_rosters import ingest_rosters
from src.face.embed_tracklets import embed_tracklets as embed_faces
from src.ocr.run_tracklet_ocr import run_tracklet_ocr
from src.reid.embed_tracklets import embed_tracklets as embed_reid
from src.resolver.resolve_game import resolve_game


def run_pipeline(video: Path, season: int, weights: str, output_root: Path, team_hint: str | None = None) -> dict:
    video_id = video.stem
    run_dir = output_root / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tracking_dir = run_dir / "tracking"
    crops_dir = run_dir / "crops"
    evidence_dir = run_dir / "evidence"
    resolve_dir = run_dir / "resolve"
    clips_dir = run_dir / "clips"
    roster_dir = output_root / "rosters"

    # Run detector/tracker in a subprocess so native crashes do not kill the orchestrator.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.track.run",
            "--video",
            str(video),
            "--weights",
            weights,
            "--output",
            str(tracking_dir),
        ],
        check=False,
    )
    tracking_summary_path = tracking_dir / "tracking_summary.json"
    if not tracking_summary_path.is_file():
        raise RuntimeError("Tracking stage failed: tracking_summary.json was not created.")
    t_summary = json.loads(tracking_summary_path.read_text(encoding="utf-8"))
    c_summary = build_tracklet_crops(
        video=video,
        tracks_json=tracking_dir / "tracks.json",
        output_dir=crops_dir,
        top_k=10,
        min_spacing_frames=8,
    )
    roster_summary = ingest_rosters(season=season, output_path=roster_dir / f"wnba_rosters_{season}.jsonl")
    o_summary = run_tracklet_ocr(crops_dir / "tracklet_keyframes.json", evidence_dir / "ocr.json")
    f_summary = embed_faces(crops_dir / "tracklet_keyframes.json", evidence_dir / "face.json")
    r_summary = embed_reid(crops_dir / "tracklet_keyframes.json", evidence_dir / "reid.json")
    z_summary = resolve_game(
        video_id=video_id,
        tracklets_manifest=crops_dir / "tracklet_keyframes.json",
        ocr_json=evidence_dir / "ocr.json",
        face_json=evidence_dir / "face.json",
        reid_json=evidence_dir / "reid.json",
        roster_jsonl=Path(roster_summary["output"]),
        output_path=resolve_dir / "resolved_tracklets.json",
        team_hint=team_hint,
    )
    clip_summary = cut_resolved_clips(
        video=video,
        tracks_json=tracking_dir / "tracks.json",
        resolved_json=resolve_dir / "resolved_tracklets.json",
        output_dir=clips_dir,
        min_status="review",
    )
    summary = {
        "video": str(video.resolve()),
        "season": season,
        "tracking": t_summary,
        "crops": c_summary,
        "rosters": roster_summary,
        "ocr": o_summary,
        "face": f_summary,
        "reid": r_summary,
        "resolver": z_summary,
        "clips": clip_summary,
        "run_dir": str(run_dir.resolve()),
    }
    (run_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end WNBA identity resolver on one video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--output-root", default="outputs/identity")
    parser.add_argument("--team-hint", default=None)
    args = parser.parse_args()

    summary = run_pipeline(
        video=Path(args.video),
        season=args.season,
        weights=args.weights,
        output_root=Path(args.output_root),
        team_hint=args.team_hint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
