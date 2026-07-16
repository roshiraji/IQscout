"""Resolve tracklets into player IDs using OCR-first, roster-gated scoring."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common_models import ResolvedTracklet
from src.resolver.team_normalizer import normalize_team_id


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_roster(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _jersey_numbers_match(ocr_number: str | None, roster_number: str | None) -> bool:
    if not ocr_number or roster_number is None:
        return False
    return str(roster_number).lstrip("0") == str(ocr_number).lstrip("0")


def _mean_cos(a: list[list[float]], b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    b_arr = np.array(b, dtype=float)
    b_norm = np.linalg.norm(b_arr) or 1.0
    scores = []
    for row in a:
        r = np.array(row, dtype=float)
        r_norm = np.linalg.norm(r) or 1.0
        scores.append(float(np.dot(r, b_arr) / (r_norm * b_norm)))
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _candidate_pool(
    *,
    by_team: dict[str, list[dict]],
    roster: list[dict],
    team_hint: str | None,
) -> list[dict]:
    if team_hint:
        return by_team.get(team_hint, [])
    return roster


def resolve_game(
    *,
    video_id: str,
    tracklets_manifest: Path,
    ocr_json: Path,
    face_json: Path,
    reid_json: Path,
    roster_jsonl: Path,
    output_path: Path,
    team_hint: str | None = None,
) -> dict:
    tracklets = _read_json(tracklets_manifest)
    ocr = {int(t["track_id"]): t for t in _read_json(ocr_json).get("tracklets", [])}
    face = {int(t["track_id"]): t for t in _read_json(face_json).get("tracklets", [])}
    reid = {int(t["track_id"]): t for t in _read_json(reid_json).get("tracklets", [])}
    roster = _load_roster(roster_jsonl)

    by_team = defaultdict(list)
    for player in roster:
        by_team[player.get("team_id")].append(player)

    normalized_team_hint = normalize_team_id(team_hint)

    results: list[dict] = []
    for track in tracklets:
        tid = int(track["track_id"])
        track_team_hint = normalize_team_id(track.get("team_hint")) or normalized_team_hint
        ocr_row = ocr.get(tid, {})
        ocr_number = ocr_row.get("jersey_prediction")
        ocr_consistency = float(ocr_row.get("ocr_consistency") or 0.0)
        face_vecs = [row["embedding"] for row in face.get(tid, {}).get("face_embeddings", [])]
        reid_vecs = [row["embedding"] for row in reid.get(tid, {}).get("reid_embeddings", [])]

        candidates = _candidate_pool(
            by_team=by_team,
            roster=roster,
            team_hint=track_team_hint,
        )

        exact_matches = [
            player
            for player in candidates
            if _jersey_numbers_match(ocr_number, player.get("jersey_number"))
        ]
        if len(exact_matches) == 1 and ocr_consistency >= 0.60:
            exact_player = exact_matches[0]
            resolved = ResolvedTracklet(
                video_id=video_id,
                tracklet_id=f"{video_id}_t_{tid:06d}",
                resolved_player_id=exact_player["player_id"],
                resolver_confidence=round(ocr_consistency, 4),
                resolver_score=1.0,
                score_margin=1.0,
                candidate_scores=[
                    {
                        "player_id": exact_player["player_id"],
                        "full_name": exact_player["full_name"],
                        "team_id": exact_player["team_id"],
                        "jersey_number": exact_player["jersey_number"],
                        "score": 1.0,
                    }
                ],
                status="resolved",
                reason="unique_team_jersey_match",
            )
            results.append(resolved.model_dump())
            continue

        scored = []
        for player in candidates:
            team_id = player.get("team_id")
            jersey = player.get("jersey_number")
            s_number = 1.0 if _jersey_numbers_match(ocr_number, jersey) else 0.0
            s_face = _mean_cos(face_vecs, player.get("vectors", {}).get("face"))
            s_reid = _mean_cos(reid_vecs, player.get("vectors", {}).get("reid_proto"))
            s_team = 1.0 if (track_team_hint and team_id == track_team_hint) else 0.0
            score = (
                0.40 * s_number
                + 0.25 * s_face
                + 0.15 * s_reid
                + 0.10 * s_team
                + 0.05 * 1.0
                + 0.05 * ocr_consistency
            )
            scored.append(
                {
                    "player_id": player.get("player_id"),
                    "full_name": player.get("full_name"),
                    "team_id": team_id,
                    "jersey_number": jersey,
                    "score": round(float(score), 6),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        top1 = scored[0] if scored else None
        top2 = scored[1] if len(scored) > 1 else {"score": -1.0}
        margin = (top1["score"] - top2["score"]) if top1 else 0.0
        conf = max(0.0, min(1.0, (top1["score"] if top1 else 0.0)))

        if top1 and conf >= 0.85 and margin >= 0.15 and len(track.get("keyframes", [])) >= 3:
            status = "resolved"
            reason = None
        elif top1 and conf >= 0.60:
            status = "review"
            reason = "medium_confidence"
        else:
            status = "unresolved"
            reason = "low_confidence"

        resolved = ResolvedTracklet(
            video_id=video_id,
            tracklet_id=f"{video_id}_t_{tid:06d}",
            resolved_player_id=top1["player_id"] if top1 else None,
            resolver_confidence=round(conf, 4),
            resolver_score=top1["score"] if top1 else None,
            score_margin=round(margin, 4),
            candidate_scores=scored[:3],
            status=status,
            reason=reason,
        )
        results.append(resolved.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"video_id": video_id, "resolved_tracklets": results}, indent=2), encoding="utf-8")
    return {
        "video_id": video_id,
        "tracklets": len(results),
        "resolved": sum(1 for r in results if r["status"] == "resolved"),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve tracklets into player identities.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--reid", required=True)
    parser.add_argument("--rosters", required=True, help="Path to roster jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--team-hint", default=None)
    args = parser.parse_args()
    summary = resolve_game(
        video_id=args.video_id,
        tracklets_manifest=Path(args.tracklets),
        ocr_json=Path(args.ocr),
        face_json=Path(args.face),
        reid_json=Path(args.reid),
        roster_jsonl=Path(args.rosters),
        output_path=Path(args.output),
        team_hint=args.team_hint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
