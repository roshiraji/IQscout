"""Run roster-gated identity resolution on a completed tracking job."""

from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path

from src.crops.build_tracklet_crops import build_tracklet_crops
from src.face.embed_tracklets import embed_tracklets as embed_faces
from src.ocr.run_tracklet_ocr import run_tracklet_ocr
from src.pipeline.identity_contracts import IdentityEvidence, TrackIdentityAttachment
from src.pipeline.player_track_pipeline import load_pipeline_config
from src.reid.embed_tracklets import embed_tracklets as embed_reid
from src.resolver.resolve_game import _jersey_numbers_match, resolve_game
from src.resolver.team_normalizer import roster_team_from_jersey_color

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_config(config: dict | None = None) -> dict:
    """Load identity settings from player_tracking_mvp.yaml."""
    cfg = config if config is not None else load_pipeline_config()
    identity = cfg.get("identity") or {}
    return {
        "auto_resolve_enabled": bool(identity.get("auto_resolve_enabled", False)),
        "jersey_ocr": dict(identity.get("jersey_ocr") or {}),
        "face_embeddings": dict(identity.get("face_embeddings") or {}),
        "reid_embeddings": dict(identity.get("reid_embeddings") or {}),
    }


def _jersey_ocr_config(config: dict | None = None) -> dict:
    return _identity_config(config)["jersey_ocr"]


def is_auto_identity_resolve_enabled(config: dict | None = None) -> bool:
    """Whether the pipeline should launch OCR automatically after tracking."""
    return _identity_config(config)["auto_resolve_enabled"]


def _write_skipped_embeddings(
    manifest_path: Path,
    output_path: Path,
    *,
    field_name: str,
    backend: str,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        {"track_id": int(track["track_id"]), field_name: []}
        for track in manifest
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"backend": backend, "tracklets": rows}, indent=2),
        encoding="utf-8",
    )
    return {
        "backend": backend,
        "tracklets": len(rows),
        "output": str(output_path.resolve()),
        "skipped": True,
    }


def _log_identity_timing_summary(
    *,
    job_id: str,
    per_track: list[dict],
    stage_seconds: dict[str, float],
) -> None:
    total = stage_seconds.get("total", 0.0)
    logger.info("Identity pipeline timing summary (job=%s)", job_id)
    logger.info(
        "Candidate selection: %.2fs\n"
        "PaddleOCR initialization: %.2fs\n"
        "OCR (all players): %.2fs\n"
        "Face embeddings: %.2fs\n"
        "ReID embeddings: %.2fs\n"
        "Final identity resolution: %.2fs\n"
        "Total identity pipeline: %.2fs",
        stage_seconds.get("candidate_selection", 0.0),
        stage_seconds.get("ocr_init", 0.0),
        stage_seconds.get("ocr_total", 0.0),
        stage_seconds.get("face_embeddings", 0.0),
        stage_seconds.get("reid_embeddings", 0.0),
        stage_seconds.get("final_resolution", 0.0),
        total,
    )
    for row in per_track:
        logger.info(
            "Player %s\n"
            "---------\n"
            "Candidate selection: %.2fs\n"
            "OCR: %.2fs (%d calls)\n"
            "Face embeddings: %.2fs\n"
            "ReID embeddings: %.2fs\n"
            "Total: %.2fs",
            row["track_id"],
            row.get("candidate_selection_seconds", 0.0),
            row.get("ocr_seconds", 0.0),
            row.get("ocr_calls", 0),
            row.get("face_seconds", 0.0),
            row.get("reid_seconds", 0.0),
            row.get("total_seconds", 0.0),
        )


def _official_group_ids(artifact_dir: Path) -> set[int]:
    """Person groups marked official (or white jersey) — skip jersey OCR."""
    excluded: set[int] = set()
    person_groups_path = artifact_dir / "person_groups.json"
    if person_groups_path.is_file():
        payload = _read_json(person_groups_path)
        for group in payload.get("person_groups") or []:
            role = str(group.get("role") or "").lower()
            color = str(group.get("jersey_color_name") or "").lower()
            if role == "official" or color == "white":
                excluded.add(int(group["person_group_id"]))
        return excluded

    snapshot_path = artifact_dir / "snapshot.json"
    if snapshot_path.is_file():
        snapshot = _read_json(snapshot_path)
        for player in snapshot.get("players") or []:
            role = str(player.get("role") or "").lower()
            color = str(player.get("jersey_color_name") or "").lower()
            if role == "official" or color == "white":
                gid = player.get("person_group_id") or player.get("track_id")
                if gid is not None:
                    excluded.add(int(gid))
    return excluded


def _export_person_group_tracks(tracks_jsonl: Path, output_path: Path) -> Path:
    records: list[dict] = []
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            frame_idx = int(frame["frame_number"])
            timestamp = float(frame.get("timestamp_seconds") or 0.0)
            for det in frame.get("tracks") or []:
                gid = det.get("person_group_id")
                if gid is None:
                    continue
                bbox = det.get("bbox_xyxy")
                if not bbox:
                    continue
                records.append(
                    {
                        "track_id": int(gid),
                        "frame_idx": frame_idx,
                        "timestamp_seconds": timestamp,
                        "bbox_xyxy": bbox,
                        "det_conf": float(det.get("confidence") or 0.0),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def _team_hint_by_group(artifact_dir: Path, game_context: dict) -> dict[int, str]:
    hints: dict[int, str] = {}
    person_groups_path = artifact_dir / "person_groups.json"
    if person_groups_path.is_file():
        payload = _read_json(person_groups_path)
        for group in payload.get("person_groups") or []:
            gid = int(group["person_group_id"])
            roster_id = group.get("roster_team_id")
            if roster_id:
                hints[gid] = str(roster_id)
                continue
            color = group.get("jersey_color_name")
            mapped = roster_team_from_jersey_color(color, game_context)
            if mapped:
                hints[gid] = mapped
        return hints

    snapshot_path = artifact_dir / "snapshot.json"
    if snapshot_path.is_file():
        snapshot = _read_json(snapshot_path)
        for player in snapshot.get("players") or []:
            gid = int(player.get("person_group_id") or player.get("track_id"))
            roster_id = player.get("roster_team_id")
            if roster_id:
                hints[gid] = str(roster_id)
                continue
            mapped = roster_team_from_jersey_color(
                player.get("jersey_color_name"),
                game_context,
            )
            if mapped:
                hints[gid] = mapped
    return hints


def _apply_team_hints(manifest_path: Path, team_hints: dict[int, str]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for track in manifest:
        tid = int(track["track_id"])
        hint = team_hints.get(tid)
        if hint:
            track["team_hint"] = hint
            track["jersey_color"] = track.get("jersey_color")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _roster_lookup(roster_jsonl: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with roster_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[row["player_id"]] = row
    return lookup


def _update_identity_attachments(
    *,
    artifact_dir: Path,
    job_id: str,
    game_context: dict,
    resolved_payload: dict,
    ocr_payload: dict,
    roster_lookup: dict[str, dict],
) -> Path:
    identity_path = artifact_dir / "identity_attachments.json"
    if identity_path.is_file():
        identity_doc = _read_json(identity_path)
    else:
        identity_doc = {"job_id": job_id, "game_context": game_context, "attachments": []}

    ocr_by_track = {
        int(row["track_id"]): row for row in ocr_payload.get("tracklets", [])
    }
    resolved_by_track: dict[int, dict] = {}
    for row in resolved_payload.get("resolved_tracklets", []):
        tracklet_id = row.get("tracklet_id") or ""
        try:
            tid = int(str(tracklet_id).rsplit("_t_", 1)[-1])
        except (IndexError, ValueError):
            continue
        resolved_by_track[tid] = row

    attachments: list[dict] = []
    existing = {
        int(item["track_id"]): item
        for item in identity_doc.get("attachments", [])
    }
    track_ids = sorted(set(existing) | set(resolved_by_track))
    for tid in track_ids:
        resolved = resolved_by_track.get(tid, {})
        ocr_row = ocr_by_track.get(tid, {})
        top = (resolved.get("candidate_scores") or [{}])[0]
        player_id = resolved.get("resolved_player_id")
        roster_row = roster_lookup.get(player_id or "", {})
        evidence = IdentityEvidence(
            ocr_jersey=ocr_row.get("jersey_prediction"),
            ocr_consistency=ocr_row.get("ocr_consistency"),
            identity_evidence_status=ocr_row.get("identity_evidence_status"),
            identity_evidence_reason=ocr_row.get("identity_evidence_reason"),
            team_hint=top.get("team_id") or ocr_row.get("team_hint"),
            candidate_scores=resolved.get("candidate_scores") or [],
            keyframe_paths=[
                kf.get("crop_path", "")
                for kf in (resolved.get("keyframes") or [])
                if kf.get("crop_path")
            ],
        )
        attachment = TrackIdentityAttachment(
            job_id=job_id,
            track_id=tid,
            status=resolved.get("status") or existing.get(tid, {}).get("status") or "pending",
            player_id=player_id,
            player_name=top.get("full_name") or roster_row.get("full_name"),
            team_id=top.get("team_id") or roster_row.get("team_id"),
            jersey_number=top.get("jersey_number") or roster_row.get("jersey_number"),
            confidence=resolved.get("resolver_confidence"),
            score_margin=resolved.get("score_margin"),
            reason=resolved.get("reason"),
            evidence=evidence,
        )
        attachments.append(attachment.model_dump())

    identity_doc["game_context"] = game_context
    identity_doc["attachments"] = attachments
    identity_path.write_text(json.dumps(identity_doc, indent=2), encoding="utf-8")
    return identity_path


def resolve_job_identity(
    artifact_dir: Path,
    *,
    season: int = 2026,
    roster_dir: Path | None = None,
) -> dict:
    artifact_dir = artifact_dir.resolve()
    job_id = artifact_dir.name
    summary_path = artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing tracking summary: {summary_path}")

    summary = _read_json(summary_path)
    video_path = Path(summary["video_path"])
    tracks_jsonl = artifact_dir / "tracks.jsonl.gz"
    if not tracks_jsonl.is_file():
        raise FileNotFoundError(f"Missing tracks artifact: {tracks_jsonl}")

    identity_path = artifact_dir / "identity_attachments.json"
    game_context = {}
    if identity_path.is_file():
        game_context = _read_json(identity_path).get("game_context") or {}
    game_context = game_context or summary.get("game_context") or {}

    roster_root = roster_dir or (artifact_dir.parents[1] / "identity" / "rosters")
    roster_jsonl = roster_root / f"wnba_rosters_{season}.jsonl"
    if not roster_jsonl.is_file():
        from src.db.ingest_rosters import ingest_rosters

        ingest_rosters(season, roster_jsonl)

    jersey_ocr = _jersey_ocr_config()
    identity_cfg = _identity_config()
    face_cfg = identity_cfg["face_embeddings"]
    reid_cfg = identity_cfg["reid_embeddings"]
    face_enabled = bool(face_cfg.get("enabled", False))
    reid_enabled = bool(reid_cfg.get("enabled", False))

    identity_work = artifact_dir / "identity"
    crops_dir = identity_work / "crops"
    evidence_dir = identity_work / "evidence"
    tracks_json = identity_work / "tracks.json"
    pipeline_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    _export_person_group_tracks(tracks_jsonl, tracks_json)

    exclude_ids = _official_group_ids(artifact_dir)
    crop_start = time.perf_counter()
    build_tracklet_crops(
        video=video_path,
        tracks_json=tracks_json,
        output_dir=crops_dir,
        top_k=int(jersey_ocr.get("max_candidates_per_person") or 5),
        min_spacing_frames=int(jersey_ocr.get("min_spacing_frames") or 15),
        exclude_track_ids=exclude_ids,
        jersey_ocr=jersey_ocr,
    )
    stage_seconds["candidate_selection"] = time.perf_counter() - crop_start
    manifest_path = crops_dir / "tracklet_keyframes.json"
    team_hints = _team_hint_by_group(artifact_dir, game_context)
    _apply_team_hints(manifest_path, team_hints)

    ocr_start = time.perf_counter()
    ocr_summary = run_tracklet_ocr(
        manifest_path,
        evidence_dir / "ocr.json",
        jersey_ocr=jersey_ocr,
        profile=True,
    )
    stage_seconds["ocr_total"] = time.perf_counter() - ocr_start
    stage_seconds["ocr_init"] = float(ocr_summary.get("ocr_init_seconds") or 0.0)

    manifest = _read_json(manifest_path)
    track_count = max(len(manifest), 1)
    candidate_share = stage_seconds["candidate_selection"] / track_count

    if face_enabled:
        face_start = time.perf_counter()
        face_summary = embed_faces(
            manifest_path,
            evidence_dir / "face.json",
            use_insightface=bool(face_cfg.get("use_insightface", False)),
        )
        stage_seconds["face_embeddings"] = time.perf_counter() - face_start
    else:
        face_summary = _write_skipped_embeddings(
            manifest_path,
            evidence_dir / "face.json",
            field_name="face_embeddings",
            backend="disabled",
        )
        stage_seconds["face_embeddings"] = 0.0

    if reid_enabled:
        reid_start = time.perf_counter()
        reid_summary = embed_reid(manifest_path, evidence_dir / "reid.json")
        stage_seconds["reid_embeddings"] = time.perf_counter() - reid_start
    else:
        reid_summary = _write_skipped_embeddings(
            manifest_path,
            evidence_dir / "reid.json",
            field_name="reid_embeddings",
            backend="disabled",
        )
        stage_seconds["reid_embeddings"] = 0.0

    resolve_start = time.perf_counter()
    resolved_path = identity_work / "resolved_tracklets.json"
    resolver_summary = resolve_game(
        video_id=job_id,
        tracklets_manifest=manifest_path,
        ocr_json=evidence_dir / "ocr.json",
        face_json=evidence_dir / "face.json",
        reid_json=evidence_dir / "reid.json",
        roster_jsonl=roster_jsonl,
        output_path=resolved_path,
        team_hint=None,
    )
    stage_seconds["final_resolution"] = time.perf_counter() - resolve_start
    stage_seconds["total"] = time.perf_counter() - pipeline_start

    per_track_profiles: list[dict] = []
    ocr_by_track = {
        int(row["track_id"]): row
        for row in ocr_summary.get("per_track_profiles") or []
    }
    for track in manifest:
        tid = int(track["track_id"])
        ocr_row = ocr_by_track.get(tid, {})
        per_track_profiles.append(
            {
                "track_id": tid,
                "candidate_selection_seconds": round(candidate_share, 2),
                "ocr_seconds": float(ocr_row.get("ocr_seconds") or 0.0),
                "ocr_calls": int(ocr_row.get("ocr_calls") or 0),
                "face_seconds": round(
                    stage_seconds["face_embeddings"] / track_count, 2
                )
                if face_enabled
                else 0.0,
                "reid_seconds": round(
                    stage_seconds["reid_embeddings"] / track_count, 2
                )
                if reid_enabled
                else 0.0,
                "total_seconds": round(
                    candidate_share
                    + float(ocr_row.get("ocr_seconds") or 0.0)
                    + (
                        stage_seconds["face_embeddings"] / track_count
                        if face_enabled
                        else 0.0
                    )
                    + (
                        stage_seconds["reid_embeddings"] / track_count
                        if reid_enabled
                        else 0.0
                    ),
                    2,
                ),
            }
        )
    _log_identity_timing_summary(
        job_id=job_id,
        per_track=per_track_profiles,
        stage_seconds=stage_seconds,
    )

    resolved_payload = _read_json(resolved_path)
    ocr_payload = _read_json(evidence_dir / "ocr.json")
    roster_lookup = _roster_lookup(roster_jsonl)
    updated_identity = _update_identity_attachments(
        artifact_dir=artifact_dir,
        job_id=job_id,
        game_context=game_context,
        resolved_payload=resolved_payload,
        ocr_payload=ocr_payload,
        roster_lookup=roster_lookup,
    )

    return {
        "job_id": job_id,
        "identity_attachments": str(updated_identity.resolve()),
        "resolved_tracklets": str(resolved_path.resolve()),
        "ocr": ocr_summary,
        "face": face_summary,
        "reid": reid_summary,
        "resolver": resolver_summary,
        "timing": stage_seconds,
        "per_track_profiles": per_track_profiles,
        "face_embeddings_enabled": face_enabled,
        "reid_embeddings_enabled": reid_enabled,
    }


def resolve_manual_jersey(
    artifact_dir: Path,
    *,
    track_id: int,
    jersey_number: str,
    season: int = 2026,
    roster_dir: Path | None = None,
) -> dict:
    """Resolve one tracked person from a user-entered jersey number."""
    artifact_dir = artifact_dir.resolve()
    job_id = artifact_dir.name
    identity_path = artifact_dir / "identity_attachments.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"Missing identity attachments: {identity_path}")

    identity_doc = _read_json(identity_path)
    game_context = identity_doc.get("game_context") or {}
    team_hints = _team_hint_by_group(artifact_dir, game_context)
    team_hint = team_hints.get(int(track_id))
    if not team_hint:
        raise ValueError(
            f"No roster team mapping for person #{track_id}. "
            "Enter home/away teams when uploading the clip."
        )

    roster_root = roster_dir or (artifact_dir.parents[1] / "identity" / "rosters")
    roster_jsonl = roster_root / f"wnba_rosters_{season}.jsonl"
    if not roster_jsonl.is_file():
        from src.db.ingest_rosters import ingest_rosters

        ingest_rosters(season, roster_jsonl)

    candidates = []
    with roster_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            player = json.loads(line)
            if player.get("team_id") != team_hint:
                continue
            if _jersey_numbers_match(jersey_number, player.get("jersey_number")):
                candidates.append(player)

    if len(candidates) != 1:
        raise ValueError(
            f"Jersey #{jersey_number} on {team_hint} matched "
            f"{len(candidates)} roster players; expected exactly one."
        )

    player = candidates[0]
    candidate_score = {
        "player_id": player["player_id"],
        "full_name": player["full_name"],
        "team_id": player["team_id"],
        "jersey_number": player["jersey_number"],
        "score": 1.0,
    }
    attachments = []
    updated = False
    for item in identity_doc.get("attachments", []):
        if int(item["track_id"]) != int(track_id):
            attachments.append(item)
            continue
        evidence = item.get("evidence") or {}
        attachments.append(
            TrackIdentityAttachment(
                job_id=job_id,
                track_id=int(track_id),
                status="resolved",
                player_id=player["player_id"],
                player_name=player["full_name"],
                team_id=player["team_id"],
                jersey_number=player["jersey_number"],
                confidence=1.0,
                score_margin=1.0,
                reason="manual_jersey_entry",
                evidence=IdentityEvidence(
                    ocr_jersey=str(jersey_number),
                    ocr_consistency=1.0,
                    identity_evidence_status="readable",
                    identity_evidence_reason="manual_jersey_entry",
                    team_hint=team_hint,
                    candidate_scores=[candidate_score],
                ),
            ).model_dump()
        )
        updated = True

    if not updated:
        attachments.append(
            TrackIdentityAttachment(
                job_id=job_id,
                track_id=int(track_id),
                status="resolved",
                player_id=player["player_id"],
                player_name=player["full_name"],
                team_id=player["team_id"],
                jersey_number=player["jersey_number"],
                confidence=1.0,
                score_margin=1.0,
                reason="manual_jersey_entry",
                evidence=IdentityEvidence(
                    ocr_jersey=str(jersey_number),
                    ocr_consistency=1.0,
                    identity_evidence_status="readable",
                    identity_evidence_reason="manual_jersey_entry",
                    team_hint=team_hint,
                    candidate_scores=[candidate_score],
                ),
            ).model_dump()
        )

    identity_doc["attachments"] = attachments
    identity_path.write_text(json.dumps(identity_doc, indent=2), encoding="utf-8")
    return {
        "job_id": job_id,
        "track_id": int(track_id),
        "player_name": player["full_name"],
        "team_id": player["team_id"],
        "jersey_number": player["jersey_number"],
        "status": "resolved",
        "reason": "manual_jersey_entry",
    }
