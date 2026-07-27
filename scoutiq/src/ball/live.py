"""Live ball track pass + possession inference over job artifacts."""

from __future__ import annotations

import gzip
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ball.detect import detect_balls
from src.ball.fusion import compute_possession_windows
from src.ball.hoop import detect_hoops
from src.ball.possession import (
    assign_frame_possession,
    load_possession_override,
    summarize_possession,
    write_possession_json,
)
from src.ball.track import BallTracker, interpolate_ball_track_gaps

logger = logging.getLogger(__name__)


def run_ball_tracking_on_video(
    video: Path,
    *,
    artifact_dir: Path,
    ball_cfg: dict[str, Any],
    cut_frames: list[int] | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Detect + track ball; write ball_tracks.jsonl.gz + ball_summary.json."""
    if not bool(ball_cfg.get("enabled", False)):
        return {"enabled": False, "skipped": True}

    weights = ball_cfg.get("weights")
    weights_path = Path(weights) if weights else None
    if weights_path is not None and not weights_path.is_absolute():
        # Resolve relative to scoutiq/ (parent of src/).
        scoutiq = Path(__file__).resolve().parents[2]
        candidate = scoutiq / weights_path
        weights_path = candidate if candidate.is_file() else weights_path

    conf = float(ball_cfg.get("conf", 0.25))
    device = str(ball_cfg.get("device", "cpu"))
    every_n = max(int(ball_cfg.get("every_n_frames", 1)), 1)
    use_hsv = bool(ball_cfg.get("hsv_fallback", True))
    track_cfg = ball_cfg.get("track") or {}
    spatial_filter = dict(ball_cfg.get("spatial_filter") or {})
    cut_set = set(int(c) for c in (cut_frames or []))

    # Optional hoop detector (WS3) for attacking-direction.
    hoop_cfg = ball_cfg.get("hoop") or {}
    hoop_enabled = bool(hoop_cfg.get("enabled", False))
    hoop_weights = hoop_cfg.get("weights")
    hoop_weights_path = Path(hoop_weights) if hoop_weights else None
    if hoop_weights_path is not None and not hoop_weights_path.is_absolute():
        scoutiq = Path(__file__).resolve().parents[2]
        cand = scoutiq / hoop_weights_path
        hoop_weights_path = cand if cand.is_file() else hoop_weights_path
    hoop_every_n = max(int(hoop_cfg.get("every_n_frames", 15)), 1)
    hoop_conf = float(hoop_cfg.get("conf", 0.30))
    hoop_ready = hoop_enabled and hoop_weights_path is not None and hoop_weights_path.is_file()
    hoops_path = artifact_dir / "hoop_tracks.jsonl.gz"
    hoop_handle = gzip.open(hoops_path, "wt", encoding="utf-8") if hoop_ready else None
    last_hoops: list[dict] = []
    frames_with_hoop = 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for ball pass: {video}")
    fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(width or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(height or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    tracker = BallTracker(
        iou_threshold=float(track_cfg.get("iou_threshold", 0.15)),
        max_missed=int(track_cfg.get("max_missed", 12)),
    )

    tracks_path = artifact_dir / "ball_tracks.jsonl.gz"
    summary_path = artifact_dir / "ball_summary.json"
    frames_with_ball = 0
    frames_processed = 0
    conf_sum = 0.0
    conf_n = 0
    resets = 0
    sources: dict[str, int] = {}
    start = time.perf_counter()
    last_tracked: list[dict] = []

    with gzip.open(tracks_path, "wt", encoding="utf-8") as handle:
        frame_id = -1
        while True:
            ret, image = cap.read()
            if not ret:
                break
            frame_id += 1
            if max_frames is not None and frames_processed >= int(max_frames):
                break
            frames_processed += 1
            timestamp = frame_id / fps if fps else 0.0

            if frame_id in cut_set:
                tracker.reset()
                resets += 1
                last_tracked = []
                last_hoops = []

            if hoop_ready and (frame_id % hoop_every_n == 0 or not last_hoops):
                try:
                    detected = detect_hoops(
                        image,
                        weights=hoop_weights_path,
                        conf=hoop_conf,
                        device=device,
                    )
                    # Carry last-known rim across missed frames / sparse cadence.
                    if detected:
                        last_hoops = detected
                except Exception:  # noqa: BLE001
                    pass
            if hoop_handle is not None:
                if last_hoops:
                    frames_with_hoop += 1
                hoop_handle.write(
                    json.dumps(
                        {
                            "frame_number": frame_id,
                            "hoops": [
                                {"bbox_xyxy": h.get("bbox_xyxy"), "confidence": h.get("confidence")}
                                for h in last_hoops
                            ],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            run_detect = frame_id % every_n == 0 or not last_tracked
            if run_detect:
                dets = detect_balls(
                    image,
                    weights=weights_path if weights_path and weights_path.is_file() else None,
                    conf=conf,
                    device=device,
                    use_hsv_fallback=use_hsv,
                    spatial_filter=spatial_filter,
                )
                tracked = tracker.update(dets)
                last_tracked = tracked
            else:
                tracked = list(last_tracked)

            if tracked:
                frames_with_ball += 1
                for det in tracked:
                    conf_sum += float(det.get("confidence") or 0.0)
                    conf_n += 1
                    src = str(det.get("source") or "unknown")
                    sources[src] = sources.get(src, 0) + 1

            handle.write(
                json.dumps(
                    {
                        "frame_number": frame_id,
                        "timestamp_seconds": round(timestamp, 4),
                        "balls": [
                            {
                                "track_id": d.get("track_id"),
                                "confidence": d.get("confidence"),
                                "bbox_xyxy": d.get("bbox_xyxy"),
                                "source": d.get("source"),
                            }
                            for d in tracked
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    cap.release()
    if hoop_handle is not None:
        hoop_handle.close()

    # WS1 post-pass: linear-interpolate short ball gaps between real detections.
    interp_cfg = track_cfg.get("interpolate") or {}
    filled = 0
    if bool(interp_cfg.get("enabled", True)):
        try:
            filled = interpolate_ball_track_gaps(
                tracks_path,
                max_gap=int(interp_cfg.get("max_gap", 5)),
                min_conf=float(interp_cfg.get("min_conf", 0.2)),
            )
            frames_with_ball += filled
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ball interpolation skipped: %s", exc)

    coverage = frames_with_ball / max(frames_processed, 1)
    summary = {
        "enabled": True,
        "weights": str(weights_path) if weights_path else None,
        "weights_loaded": bool(weights_path and weights_path.is_file()),
        "device": device,
        "conf": conf,
        "every_n_frames": every_n,
        "hsv_fallback": use_hsv,
        "hoop_enabled": hoop_ready,
        "hoop_weights": str(hoop_weights_path) if hoop_weights_path else None,
        "frames_with_hoop": frames_with_hoop,
        "interpolated_frames": filled,
        "frames_processed": frames_processed,
        "frames_with_ball": frames_with_ball,
        "coverage": round(coverage, 4),
        "mean_confidence": round(conf_sum / max(conf_n, 1), 4) if conf_n else None,
        "lost_ball_rate": round(1.0 - coverage, 4),
        "scene_cut_resets": resets,
        "sources": sources,
        "ball_tracks": str(tracks_path.resolve()),
        "runtime_seconds": round(time.perf_counter() - start, 2),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not quiet:
        print(
            f"[ball_track] coverage={summary['coverage']} "
            f"mean_conf={summary['mean_confidence']} resets={resets} "
            f"weights_loaded={summary['weights_loaded']}"
        )
    logger.info("Ball tracking complete → %s", summary_path)
    return summary


def _load_person_frame_index(
    tracks_jsonl: Path,
    person_groups: list[dict],
) -> dict[int, list[dict]]:
    """frame_number → person rows with team meta."""
    meta_by_gid = {
        int(g["person_group_id"]): g for g in person_groups if g.get("person_group_id") is not None
    }
    by_frame: dict[int, list[dict]] = {}
    if not tracks_jsonl.is_file():
        return by_frame
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            fid = int(frame["frame_number"])
            rows: list[dict] = []
            for det in frame.get("tracks") or []:
                tid = int(det.get("track_id") or -1)
                gid = det.get("person_group_id")
                if gid is None:
                    continue
                gid = int(gid)
                meta = meta_by_gid.get(gid) or {}
                box = det.get("bbox_xyxy") or []
                row = {
                    "person_group_id": gid,
                    "track_id": tid,
                    "bbox_xyxy": box,
                    "center_x": (box[0] + box[2]) / 2.0 if len(box) >= 4 else None,
                    "center_y": (box[1] + box[3]) / 2.0 if len(box) >= 4 else None,
                    "stance_x": det.get("stance_x"),
                    "stance_y": det.get("stance_y"),
                    "team_id": meta.get("team_id"),
                    "jersey_color_name": meta.get("jersey_color_name"),
                    "role": meta.get("role") or "player",
                }
                rows.append(row)
            by_frame[fid] = rows
    return by_frame


def infer_possession_for_job(
    artifact_dir: Path,
    *,
    possession_cfg: dict[str, Any] | None = None,
    override_offense_team_id: int | None = None,
) -> dict[str, Any]:
    """Join ball tracks + person groups → possession.json."""
    artifact_dir = Path(artifact_dir)
    possession_cfg = possession_cfg or {}
    ball_tracks = artifact_dir / "ball_tracks.jsonl.gz"
    tracks_jsonl = artifact_dir / "tracks.jsonl.gz"
    groups_path = artifact_dir / "person_groups.json"
    override_path = artifact_dir / "offense_team_override.json"

    if override_offense_team_id is None:
        if override_path.is_file():
            try:
                ov = json.loads(override_path.read_text(encoding="utf-8"))
                if ov.get("offense_team_id") is not None:
                    override_offense_team_id = int(ov["offense_team_id"])
                elif ov.get("mode") == "auto":
                    override_offense_team_id = None
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        else:
            override_offense_team_id = load_possession_override(artifact_dir / "possession.json")

    person_groups: list[dict] = []
    if groups_path.is_file():
        try:
            payload = json.loads(groups_path.read_text(encoding="utf-8"))
            person_groups = payload.get("person_groups") or payload if isinstance(payload, list) else (
                payload.get("person_groups") or []
            )
        except (OSError, json.JSONDecodeError):
            person_groups = []

    by_frame = _load_person_frame_index(tracks_jsonl, person_groups)
    ball_summary = {}
    ball_sum_path = artifact_dir / "ball_summary.json"
    if ball_sum_path.is_file():
        try:
            ball_summary = json.loads(ball_sum_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ball_summary = {}

    width = int(ball_summary.get("frame_width") or 1920)
    height = int(ball_summary.get("frame_height") or 1080)
    diag = math.hypot(width, height)
    fps = float(ball_summary.get("fps") or 30.0)
    max_dist = float(possession_cfg.get("max_dist_norm", 0.12))
    min_held = float(possession_cfg.get("min_held_fraction", 0.12))
    min_conf = float(possession_cfg.get("min_confidence", 0.45))

    # Hoop tracks (optional).
    hoops_by_frame: dict[int, list[dict]] = {}
    hoops_path = artifact_dir / "hoop_tracks.jsonl.gz"
    if hoops_path.is_file():
        with gzip.open(hoops_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                hf = json.loads(line)
                hoops_by_frame[int(hf["frame_number"])] = hf.get("hoops") or []

    # Scene cuts from player quality.json (for window boundaries).
    cut_frames: list[int] = []
    quality_path = artifact_dir / "quality.json"
    if quality_path.is_file():
        try:
            q = json.loads(quality_path.read_text(encoding="utf-8"))
            cut_frames = [int(c) for c in (q.get("cut_frames") or [])]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            cut_frames = []

    frame_possessions: list[dict] = []
    frame_records: list[dict] = []
    if ball_tracks.is_file():
        with gzip.open(ball_tracks, "rt", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                fid = int(frame["frame_number"])
                balls = frame.get("balls") or []
                ball = max(balls, key=lambda b: float(b.get("confidence") or 0.0)) if balls else None
                persons = by_frame.get(fid) or []
                poss = assign_frame_possession(
                    ball,
                    persons,
                    max_dist_norm=max_dist,
                    frame_diagonal=diag,
                )
                poss["frame_number"] = fid
                frame_possessions.append(poss)
                frame_records.append(
                    {
                        "frame_number": fid,
                        "ball": ball,
                        "persons": persons,
                        "hoops": hoops_by_frame.get(fid) or [],
                    }
                )

    # Legacy clip-wide vote (kept for back-compat / debug).
    legacy = summarize_possession(
        frame_possessions,
        min_held_fraction=min_held,
        min_confidence=min_conf,
        override_offense_team_id=override_offense_team_id,
    )

    # WS2+WS4: fused multi-signal per-window offense/defense.
    fused = compute_possession_windows(
        frame_records,
        cut_frames=cut_frames,
        fps=fps,
        diag=diag,
        weights=possession_cfg.get("fusion_weights"),
        max_dist_norm=max_dist,
        min_window_frames=int(possession_cfg.get("min_window_frames", 8)),
        min_confidence=min_conf,
        margin=float(possession_cfg.get("margin", 0.12)),
    )

    # Decide the clip-level O/D: manual override > fused > legacy.
    if override_offense_team_id is not None:
        offense_team_id = int(override_offense_team_id)
        defense_team_id = 1 - offense_team_id if offense_team_id in (0, 1) else None
        source = "manual_override"
        confidence = 1.0
    elif fused.get("offense_team_id") is not None:
        offense_team_id = fused["offense_team_id"]
        defense_team_id = fused["defense_team_id"]
        source = "fused_windows"
        confidence = fused["confidence"]
    else:
        offense_team_id = legacy.get("offense_team_id")
        defense_team_id = legacy.get("defense_team_id")
        source = legacy.get("source")
        confidence = legacy.get("confidence")

    summary = {
        "offense_team_id": offense_team_id,
        "defense_team_id": defense_team_id,
        "source": source,
        "confidence": confidence,
        "windows": fused.get("windows"),
        "n_windows": fused.get("n_windows"),
        "fused": {k: fused[k] for k in ("offense_team_id", "defense_team_id", "confidence", "source")},
        "legacy": {k: legacy.get(k) for k in ("offense_team_id", "defense_team_id", "confidence", "source", "team_held_counts", "held_fraction")},
        "states": legacy.get("states"),
        "held_fraction": legacy.get("held_fraction"),
        "frame_possessions_sample": [p for p in frame_possessions if p.get("state") == "held"][:40],
        "ball_coverage": ball_summary.get("coverage"),
        "artifact_dir": str(artifact_dir.resolve()),
        "note": (
            "Offense = team with ball possession when confident (fused across "
            "possession + attacking direction + spread + continuity, per window). "
            "Jersey color still defines sides. Manual override always wins."
        ),
    }
    write_possession_json(artifact_dir / "possession.json", summary)

    # Also write compact frame log for debugging.
    with gzip.open(artifact_dir / "possession_frames.jsonl.gz", "wt", encoding="utf-8") as handle:
        for p in frame_possessions:
            handle.write(json.dumps(p, separators=(",", ":")) + "\n")

    return summary


def apply_offense_team_override(
    artifact_dir: Path,
    *,
    offense_team_id: int | None,
    mode: str = "manual",
    possession_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist override and recompute possession without re-running detectors."""
    artifact_dir = Path(artifact_dir)
    override_path = artifact_dir / "offense_team_override.json"
    payload = {
        "mode": mode if offense_team_id is None else "manual",
        "offense_team_id": offense_team_id,
        "manual": offense_team_id is not None,
    }
    override_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return infer_possession_for_job(
        artifact_dir,
        possession_cfg=possession_cfg,
        override_offense_team_id=offense_team_id if mode != "auto" else None,
    )
