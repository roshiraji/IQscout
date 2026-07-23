"""Online player detect + segment + track for the React dashboard.

User uploads a clip -> best_player_seg.pt (player-only YOLO-seg) -> exclusions
-> ByteTrack (stable IDs) -> semi-transparent masks + bbox + ID overlay ->
browser-safe H.264 preview + tracks JSON/CSV.

NO Grounding DINO / SAM2 in this path. Reuses the lean showcase detection and
overlay logic (src.showcase.run_showcase) plus the shared ByteTrack tracker.

Run standalone (from scoutiq/):
  python -m src.pipeline.player_track_pipeline --video path/to/clip.mp4 --max-frames 60
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from src.object_tracker import ObjectTracker
from src.pipeline.identity_contracts import pending_attachments
from src.pipeline.jersey_team_colors import (
    assign_jersey_teams,
    pick_dense_frame_candidates,
)
from src.pipeline.tracklet_stitcher import (
    annotate_tracks_jsonl,
    stitch_tracklets,
)
from src.pipeline.tracking_quality import evaluate_quality
from src.showcase.exclusions import (
    filter_detections,
    load_exclusion_set,
    resolve_exclusion_path,
)
from src.showcase.run_showcase import (
    _smooth_polygon,
    detect_players_seg,
    draw_showcase_frame,
)
from src.video_encoding import encode_for_browser

logger = logging.getLogger(__name__)

# scoutiq/ (…/src/pipeline/player_track_pipeline.py -> parents[2]).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "player_tracking_mvp.yaml"

TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"
TRACKING_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "tracking_videos"
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "player_snapshots"
JOBS_DIR = PROJECT_ROOT / "outputs" / "jobs"

ProgressCallback = Callable[[int, int, str], None]


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_pipeline_config(config_path: str | Path | None = None) -> dict:
    """Load runtime config with safe defaults if the YAML is missing."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    defaults = {
        "weights": "outputs/training_loop/best_player_seg.pt",
        "conf": 0.25,
        "iou": 0.70,
        "device": "0",
        "exclusions_dir": "data/exclusions",
        "mask_overlay": True,
        "mask_smooth_alpha": 0.35,
        "min_track_frames": 3,
        "max_frames": None,
        "quality": {},
        "tracker": {
            "tracker_type": "bytetrack",
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 60,
            "minimum_matching_threshold": 0.75,
        },
        "stitch": {
            "enabled": True,
            "max_gap_seconds": 1.5,
            "max_center_distance_ratio": 0.18,
            "color_weight": 0.65,
            "appearance_weight": 0.35,
            "spatial_weight": 0.25,
            "min_merge_score": 0.55,
            "min_group_frames": 8,
            "keyframe_samples": 3,
            "use_osnet": True,
        },
        "jersey_teams": {
            "enabled": True,
            "top_frame_fraction": 0.92,
            "max_candidate_frames": 12,
        },
        "identity": {
            "auto_resolve_enabled": False,
            "jersey_ocr": {
                "max_candidates_per_person": 5,
                "min_spacing_frames": 15,
                "min_agreeing_reads": 2,
                "min_consistency": 0.60,
                "min_read_confidence": 0.55,
                "max_consecutive_frame_failures": 3,
                "max_processed_frames": 5,
                "max_ocr_calls_per_person": 8,
                "use_angle_cls": False,
                "save_debug_crops": False,
                "min_bbox_height": 100,
                "min_bbox_width": 45,
                "min_crop_height": 70,
                "min_crop_width": 35,
                "min_sharpness": 25,
                "min_contrast": 18,
            },
            "face_embeddings": {"enabled": False},
            "reid_embeddings": {"enabled": False},
        },
        "ball": {
            "enabled": False,
            "weights": "outputs/training_loop_ball/best_ball_det.pt",
            "conf": 0.25,
            "device": "cpu",
            "every_n_frames": 1,
            "hsv_fallback": True,
            "track": {
                "iou_threshold": 0.15,
                "max_missed": 12,
                "interpolate": {"enabled": True, "max_gap": 5, "min_conf": 0.2},
            },
            "hoop": {
                "enabled": True,
                "weights": "outputs/training_loop_hoop_ref/best_hoop_ref_seg.pt",
                "conf": 0.40,
                "every_n_frames": 15,
            },
            "possession": {
                "max_dist_norm": 0.12,
                "min_held_fraction": 0.12,
                "min_confidence": 0.45,
                "min_window_frames": 8,
                "margin": 0.12,
                "fusion_weights": {
                    "possession": 0.55,
                    "direction": 0.25,
                    "spread": 0.08,
                    "continuity": 0.12,
                },
            },
        },
    }
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        defaults.update({k: v for k, v in loaded.items() if v is not None or k == "max_frames"})
    else:
        logger.warning("Config %s not found — using built-in defaults.", path)
    return defaults


def _apply_device(model, device: str | None) -> None:
    """Best-effort move the YOLO model onto the requested device."""
    if not device or str(device).lower() in ("auto", ""):
        return
    target = f"cuda:{device}" if str(device).isdigit() else str(device)
    try:
        model.to(target)
    except Exception as exc:  # noqa: BLE001 — device hints are non-fatal
        logger.warning("Could not move model to %s (%s) — using Ultralytics default.", target, exc)


def _slugify(name: str) -> str:
    try:
        from src.showcase.ingest import slugify

        return slugify(name)
    except Exception:  # noqa: BLE001
        return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def run_player_seg_tracking(
    video_path: str | Path,
    *,
    config_path: str | Path | None = None,
    clip_id: str | None = None,
    focus_labels: list[str] | None = None,
    max_frames: int | None = None,
    quiet: bool = False,
    output_dir: str | Path | None = None,
    job_id: str | None = None,
    game_context: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Detect + segment + track every player in a clip and render a preview.

    Args:
        video_path: MP4 to process.
        config_path: player_tracking_mvp.yaml (defaults to configs/).
        clip_id: exclusion lookup key (defaults to a slug of the filename).
        focus_labels: optional ["Player_3", ...] to highlight only those IDs.
        max_frames: cap frames (smoke tests); overrides config when given.
        quiet: suppress stdout progress.

    Returns:
        Summary dict compatible with the FrontEnd TrackingResults component.
    """
    cfg = load_pipeline_config(config_path)
    if max_frames is not None:
        cfg["max_frames"] = max_frames

    video = _resolve(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    weights = _resolve(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Train the showcase model first (best_player_seg.pt)."
        )

    conf = float(cfg["conf"])
    iou = float(cfg["iou"])
    device = str(cfg.get("device", "0"))
    mask_overlay = bool(cfg.get("mask_overlay", True))
    mask_smooth_alpha = float(cfg.get("mask_smooth_alpha", 0.35)) if mask_overlay else 0.0
    min_track_frames = int(cfg.get("min_track_frames", 3))
    cap_frames = cfg.get("max_frames")

    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cid = clip_id or _slugify(video.name)
    excl_path = resolve_exclusion_path(_resolve(cfg["exclusions_dir"]), cid)
    exclusions = load_exclusion_set(excl_path, width=width, height=height, clip_id=cid)

    model = YOLO(str(weights))
    _apply_device(model, device)

    logger.info(
        "Player-seg tracking | video=%s | weights=%s | device=%s | conf=%.2f iou=%.2f | exclusions=%s",
        video.name, weights.name, device, conf, iou, excl_path or "defaults",
    )
    if not quiet:
        print(f"[player_track] weights={weights} device={device} conf={conf} iou={iou}")

    stem = video.stem
    if output_dir is not None:
        artifact_dir = _resolve(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tracking_video_path = artifact_dir / "all_players.mp4"
        tracks_csv_path = artifact_dir / "tracks.csv"
        tracks_jsonl_path = artifact_dir / "tracks.jsonl.gz"
        summary_json_path = artifact_dir / "summary.json"
        quality_json_path = artifact_dir / "quality.json"
        snapshot_path = artifact_dir / "snapshot.jpg"
        snapshot_json_path = artifact_dir / "snapshot.json"
        identity_path = artifact_dir / "identity_attachments.json"
        person_groups_path = artifact_dir / "person_groups.json"
    else:
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        TRACKING_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        artifact_dir = TRACKING_DIR
        tracking_video_path = TRACKING_VIDEOS_DIR / f"{stem}_seg_tracked.mp4"
        tracks_csv_path = TRACKING_DIR / f"{stem}_seg_tracks.csv"
        tracks_jsonl_path = TRACKING_DIR / f"{stem}_seg_tracks.jsonl.gz"
        summary_json_path = TRACKING_DIR / f"{stem}_seg_tracking_summary.json"
        quality_json_path = TRACKING_DIR / f"{stem}_seg_quality.json"
        snapshot_path = SNAPSHOT_DIR / f"{stem}_seg_snapshot.jpg"
        snapshot_json_path = SNAPSHOT_DIR / f"{stem}_seg_snapshot.json"
        identity_path = TRACKING_DIR / f"{stem}_identity_attachments.json"
        person_groups_path = TRACKING_DIR / f"{stem}_person_groups.json"

    tmp_video = tracking_video_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {tmp_video}")

    focus_ids: set[int] = set()
    if focus_labels:
        for lbl in focus_labels:
            try:
                focus_ids.add(int(str(lbl).split("_")[-1]))
            except (ValueError, IndexError):
                continue

    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = source_frame_count
    if cap_frames is not None:
        total_frames = min(total_frames, int(cap_frames))

    tracker_cfg = cfg.get("tracker") or {}
    tracker = ObjectTracker(
        frame_rate=fps,
        track_activation_threshold=float(
            tracker_cfg.get("track_activation_threshold", 0.25)
        ),
        lost_track_buffer=int(tracker_cfg.get("lost_track_buffer", 60)),
        minimum_matching_threshold=float(
            tracker_cfg.get("minimum_matching_threshold", 0.75)
        ),
    )
    mask_memory: dict[int, list[list[float]]] = {}
    track_lengths: dict[int, int] = {}
    track_stats: dict[int, dict] = {}
    frames_processed = 0
    frames_with_players = 0
    total_positions = 0
    confidence_sum = 0.0
    confidence_min = 1.0
    confidence_max = 0.0
    cut_frames: list[int] = []
    previous_probe: np.ndarray | None = None
    start = time.perf_counter()

    fieldnames = [
        "frame_number", "timestamp_seconds", "track_id", "person_group_id",
        "player_label", "confidence", "x1", "y1", "x2", "y2", "center_x",
        "center_y", "box_width", "box_height", "has_mask",
    ]
    with tracks_csv_path.open("w", newline="", encoding="utf-8") as csv_handle, gzip.open(
        tracks_jsonl_path, "wt", encoding="utf-8"
    ) as jsonl_handle:
        csv_writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        frame_id = -1
        while True:
            ret, image = cap.read()
            if not ret:
                break
            frame_id += 1
            if cap_frames is not None and frames_processed >= int(cap_frames):
                break
            frames_processed += 1
            timestamp = frame_id / fps if fps else 0.0

            probe = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36)
            )
            if previous_probe is not None:
                cut_score = float(
                    np.mean(
                        np.abs(probe.astype(np.float32) - previous_probe.astype(np.float32))
                    )
                )
                if cut_score >= 45.0:
                    cut_frames.append(frame_id)
            previous_probe = probe

            dets = detect_players_seg(model, image, conf=conf, iou=iou)
            dets = filter_detections(dets, exclusions, class_names=("player",))
            tracked = tracker.update(dets)

            visible: list[dict] = []
            for det in tracked:
                tid = int(det.get("track_id", -1))
                if tid < 0 or (focus_ids and tid not in focus_ids):
                    continue
                track_lengths[tid] = track_lengths.get(tid, 0) + 1
                if mask_smooth_alpha > 0:
                    smoothed = _smooth_polygon(
                        mask_memory.get(tid),
                        det.get("mask_polygon"),
                        alpha=mask_smooth_alpha,
                    )
                    if smoothed is not None:
                        det["mask_polygon"] = smoothed
                        mask_memory[tid] = smoothed
                if not mask_overlay:
                    det = {**det, "mask_polygon": None}
                visible.append(det)

                score = float(det.get("confidence") or 0.0)
                confidence_sum += score
                confidence_min = min(confidence_min, score)
                confidence_max = max(confidence_max, score)
                stats = track_stats.setdefault(
                    tid,
                    {
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                        "detections": 0,
                        "gap_frames": 0,
                        "max_gap": 0,
                        "confidence_sum": 0.0,
                    },
                )
                gap = max(frame_id - int(stats["last_frame"]) - 1, 0)
                stats["gap_frames"] += gap
                stats["max_gap"] = max(int(stats["max_gap"]), gap)
                stats["last_frame"] = frame_id
                stats["detections"] += 1
                stats["confidence_sum"] += score

            if visible:
                frames_with_players += 1

            compact_tracks: list[dict] = []
            for det in visible:
                tid = int(det["track_id"])
                box = det.get("bbox_xyxy") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                record = {
                    "frame_number": frame_id,
                    "timestamp_seconds": round(timestamp, 4),
                    "track_id": tid,
                    "person_group_id": "",
                    "player_label": f"Track #{tid}",
                    "confidence": round(float(det.get("confidence") or 0.0), 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": round((x1 + x2) / 2, 2),
                    "center_y": round((y1 + y2) / 2, 2),
                    "box_width": x2 - x1,
                    "box_height": y2 - y1,
                    "has_mask": bool(det.get("mask_polygon")),
                }
                csv_writer.writerow(record)
                total_positions += 1
                compact_tracks.append(
                    {
                        "track_id": tid,
                        "confidence": record["confidence"],
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "mask_polygon": det.get("mask_polygon"),
                    }
                )

            jsonl_handle.write(
                json.dumps(
                    {
                        "frame_number": frame_id,
                        "timestamp_seconds": round(timestamp, 4),
                        "tracks": compact_tracks,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

            annotated = draw_showcase_frame(
                image, visible, frame_id=frame_id, timestamp=timestamp
            )
            writer.write(annotated)

            if progress_callback and (
                frames_processed == 1
                or frames_processed % 5 == 0
                or frames_processed == total_frames
            ):
                progress_callback(frames_processed, total_frames, "tracking")

    cap.release()
    writer.release()

    # mp4v -> browser-safe H.264.
    if tmp_video.is_file():
        tmp_video.replace(tracking_video_path)
    try:
        encode_for_browser(tracking_video_path, quiet=quiet)
    except Exception as exc:  # noqa: BLE001 — encoding must never fail the job
        logger.warning("Browser encode skipped after tracking: %s", exc)

    if progress_callback:
        progress_callback(frames_processed, total_frames, "stitching")

    stitch_payload = stitch_tracklets(
        tracks_jsonl_path,
        video,
        fps=fps,
        width=width,
        height=height,
        stitch_cfg=cfg.get("stitch"),
        device=device,
    )
    track_to_group = {
        int(k): int(v) for k, v in (stitch_payload.get("track_to_group") or {}).items()
    }
    if not track_to_group and track_lengths:
        # Stitch disabled / empty: 1:1 person groups so selection still works.
        track_to_group = {int(tid): int(tid) for tid in track_lengths}
        stitch_payload = {
            **stitch_payload,
            "track_to_group": {str(k): v for k, v in track_to_group.items()},
            "person_groups": [
                {
                    "person_group_id": tid,
                    "label": f"Person #{tid}",
                    "member_track_ids": [tid],
                    "first_frame": 0,
                    "last_frame": max(frames_processed - 1, 0),
                    "detections": int(track_lengths[tid]),
                    "span_frames": int(track_lengths[tid]),
                    "mean_confidence": 0.0,
                    "raw_tracklet_count": 1,
                }
                for tid in sorted(track_lengths)
            ],
            "unique_raw_tracks": len(track_lengths),
            "unique_person_groups": len(track_lengths),
            "average_id_switches_per_person_pre": 0.0,
            "average_id_switches_per_person_post": 0.0,
            "merges_accepted": 0,
        }
    person_groups_path.write_text(
        json.dumps(stitch_payload, indent=2), encoding="utf-8"
    )
    if track_to_group:
        annotate_tracks_jsonl(tracks_jsonl_path, track_to_group)
        _rewrite_tracks_csv(tracks_csv_path, tracks_jsonl_path, track_to_group)

    snapshot_payload = _build_group_snapshot(
        video=video,
        tracks_jsonl=tracks_jsonl_path,
        person_groups=stitch_payload.get("person_groups") or [],
        track_to_group=track_to_group,
        snapshot_path=snapshot_path,
        job_id=job_id,
        width=width,
        height=height,
        fps=fps,
        min_track_frames=min_track_frames,
        game_context=game_context,
        jersey_cfg=cfg.get("jersey_teams"),
    )
    if snapshot_payload:
        # Persist team_ids onto person_groups for metrics / focused overlays.
        team_by_gid = {
            int(p["person_group_id"]): p
            for p in (snapshot_payload.get("players") or [])
        }
        for group in stitch_payload.get("person_groups") or []:
            meta = team_by_gid.get(int(group["person_group_id"]))
            if not meta:
                continue
            group["team_id"] = meta.get("team_id")
            group["roster_team_id"] = meta.get("roster_team_id")
            group["team_label"] = meta.get("team_label")
            group["jersey_color_name"] = meta.get("jersey_color_name")
            group["role"] = meta.get("role") or "player"
            group["overlay_color_hex"] = meta.get("overlay_color_hex")
            group["overlay_color_bgr"] = meta.get("overlay_color_bgr")
        person_groups_path.write_text(
            json.dumps(stitch_payload, indent=2), encoding="utf-8"
        )
        snapshot_json_path.write_text(
            json.dumps(snapshot_payload, indent=2), encoding="utf-8"
        )

    tracked_ids = sorted(track_lengths)
    if focus_ids:
        tracked_ids = [t for t in tracked_ids if t in focus_ids]
    group_ids = sorted({track_to_group.get(tid, tid) for tid in tracked_ids})
    track_mapping = {
        f"Person #{gid}": gid for gid in group_ids
    }

    short_track_threshold = max(int(round(fps)), 1)
    per_track_quality = {}
    for tid, stats in track_stats.items():
        count = int(stats["detections"])
        per_track_quality[str(tid)] = {
            **stats,
            "person_group_id": track_to_group.get(tid),
            "mean_confidence": round(
                float(stats["confidence_sum"]) / max(count, 1), 4
            ),
            "short_track": count < short_track_threshold,
        }
    short_tracks = [
        int(tid)
        for tid, stats in per_track_quality.items()
        if stats["short_track"]
    ]
    quality = {
        "frame_index_base": 0,
        "source_frame_count": source_frame_count,
        "frames_processed": frames_processed,
        "complete_clip": frames_processed == source_frame_count,
        "frame_coverage": round(
            frames_with_players / max(frames_processed, 1), 4
        ),
        "total_positions": total_positions,
        "unique_tracks": len(track_stats),
        "unique_raw_tracks": stitch_payload.get("unique_raw_tracks", len(track_stats)),
        "unique_person_groups": stitch_payload.get(
            "unique_person_groups", len(group_ids)
        ),
        "average_id_switches_per_person_pre": stitch_payload.get(
            "average_id_switches_per_person_pre", 0.0
        ),
        "average_id_switches_per_person_post": stitch_payload.get(
            "average_id_switches_per_person_post", 0.0
        ),
        "detection_confidence": {
            "mean": round(confidence_sum / max(total_positions, 1), 4),
            "min": round(confidence_min, 4) if total_positions else None,
            "max": round(confidence_max, 4) if total_positions else None,
            "inference_threshold": conf,
            "note": "Detection confidence is not tracking continuity or training mAP.",
        },
        "cut_frames": cut_frames,
        "short_track_ids": short_tracks,
        "fragmentation_warning": bool(short_tracks or cut_frames),
        "track_stats": per_track_quality,
        "stitch": {
            "enabled": stitch_payload.get("enabled", False),
            "appearance_backend": stitch_payload.get("appearance_backend"),
            "merges_accepted": stitch_payload.get("merges_accepted", 0),
        },
    }
    quality["gate"] = evaluate_quality(quality, cfg.get("quality"))
    quality_json_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    identity_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "game_context": game_context or {},
                "attachments": pending_attachments(job_id or stem, group_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ball_summary = None
    possession_summary = None
    ball_cfg = cfg.get("ball") or {}
    if bool(ball_cfg.get("enabled", False)):
        if progress_callback:
            progress_callback(frames_processed, total_frames, "ball_tracking")
        try:
            from src.ball.live import infer_possession_for_job, run_ball_tracking_on_video

            ball_summary = run_ball_tracking_on_video(
                video,
                artifact_dir=artifact_dir,
                ball_cfg=ball_cfg,
                cut_frames=cut_frames,
                width=width,
                height=height,
                fps=fps,
                max_frames=int(cap_frames) if cap_frames is not None else None,
                quiet=quiet,
            )
            possession_summary = infer_possession_for_job(
                artifact_dir,
                possession_cfg=ball_cfg.get("possession") or {},
            )
        except Exception as exc:  # noqa: BLE001 — ball must never fail Analyze
            logger.warning("Ball / possession pass soft-failed: %s", exc)
            ball_summary = {"enabled": True, "error": str(exc)}
            possession_summary = {"offense_team_id": None, "source": "error"}

    runtime = round(time.perf_counter() - start, 2)
    summary = {
        "job_id": job_id,
        "video_path": str(video.resolve()),
        "weights": str(weights.resolve()),
        "model": weights.name,
        "device": device,
        "snapshot_frame_number": (
            snapshot_payload["snapshot_frame_number"] if snapshot_payload else None
        ),
        "timestamp_seconds": (
            snapshot_payload["timestamp_seconds"] if snapshot_payload else None
        ),
        "frame_index_base": 0,
        "source_frame_count": source_frame_count,
        "complete_clip": frames_processed == source_frame_count,
        "selected_labels": list(track_mapping.keys()),
        "track_mapping": track_mapping,
        "unique_tracks": len(track_lengths),
        "unique_raw_tracks": quality["unique_raw_tracks"],
        "unique_person_groups": quality["unique_person_groups"],
        "frames_tracked": frames_processed,
        "total_positions": total_positions,
        "frame_coverage": round(frames_with_players / max(frames_processed, 1), 4),
        "unmatched_labels": [],
        "tracks_csv": str(tracks_csv_path.resolve()),
        "tracking_video": str(tracking_video_path.resolve()),
        "snapshot_image": str(snapshot_path.resolve()) if snapshot_payload else None,
        "snapshot": snapshot_payload,
        "tracks_jsonl": str(tracks_jsonl_path.resolve()),
        "person_groups_json": str(person_groups_path.resolve()),
        "person_groups": stitch_payload.get("person_groups") or [],
        "quality_json": str(quality_json_path.resolve()),
        "quality": quality,
        "identity_attachments": str(identity_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "total_runtime_seconds": runtime,
        "auto_tracked": True,
        "focus_labels": focus_labels or [],
        "ball": ball_summary,
        "possession": possession_summary,
        "offense_team_id": (
            possession_summary.get("offense_team_id") if possession_summary else None
        ),
        "defense_team_id": (
            possession_summary.get("defense_team_id") if possession_summary else None
        ),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json_path.resolve())

    if not quiet:
        print(
            f"[player_track] raw={quality['unique_raw_tracks']} "
            f"groups={quality['unique_person_groups']} "
            f"frames={frames_processed} coverage={summary['frame_coverage']}"
        )
        print(f"[player_track] video -> {tracking_video_path.resolve()}")

    logger.info(
        "Player-seg tracking complete → %s | raw=%d groups=%d coverage=%.3f",
        tracking_video_path,
        quality["unique_raw_tracks"],
        quality["unique_person_groups"],
        summary["frame_coverage"],
    )
    return summary


def _rewrite_tracks_csv(
    csv_path: Path,
    tracks_jsonl: Path,
    track_to_group: dict[int, int],
) -> None:
    fieldnames = [
        "frame_number", "timestamp_seconds", "track_id", "person_group_id",
        "player_label", "confidence", "x1", "y1", "x2", "y2", "center_x",
        "center_y", "box_width", "box_height", "has_mask",
    ]
    rows: list[dict] = []
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            fid = int(frame["frame_number"])
            ts = float(frame.get("timestamp_seconds") or 0.0)
            for det in frame.get("tracks") or []:
                tid = int(det["track_id"])
                gid = int(det.get("person_group_id") or track_to_group.get(tid) or 0)
                box = [int(v) for v in (det.get("bbox_xyxy") or [0, 0, 0, 0])[:4]]
                x1, y1, x2, y2 = box
                rows.append(
                    {
                        "frame_number": fid,
                        "timestamp_seconds": ts,
                        "track_id": tid,
                        "person_group_id": gid,
                        "player_label": f"Person #{gid}" if gid else f"Track #{tid}",
                        "confidence": round(float(det.get("confidence") or 0.0), 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": round((x1 + x2) / 2, 2),
                        "center_y": round((y1 + y2) / 2, 2),
                        "box_width": x2 - x1,
                        "box_height": y2 - y1,
                        "has_mask": bool(det.get("mask_polygon")),
                    }
                )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_group_snapshot(
    *,
    video: Path,
    tracks_jsonl: Path,
    person_groups: list[dict],
    track_to_group: dict[int, int],
    snapshot_path: Path,
    job_id: str | None,
    width: int,
    height: int,
    fps: float,
    min_track_frames: int,
    game_context: dict | None = None,
    jersey_cfg: dict | None = None,
) -> dict | None:
    """Pick a dense multi-player frame; color masks by jersey team."""
    mature_groups = {
        int(g["person_group_id"])
        for g in person_groups
        if int(g.get("detections") or 0) >= min_track_frames
    }
    if not mature_groups and person_groups:
        mature_groups = {int(g["person_group_id"]) for g in person_groups}

    frame_scores: list[tuple[int, float, list[dict]]] = []
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            dets = []
            seen_groups: set[int] = set()
            for det in frame.get("tracks") or []:
                tid = int(det.get("track_id", -1))
                gid = int(det.get("person_group_id") or track_to_group.get(tid) or -1)
                if gid < 0 or gid not in mature_groups or gid in seen_groups:
                    continue
                seen_groups.add(gid)
                labeled = dict(det)
                labeled["person_group_id"] = gid
                labeled["track_id"] = gid
                dets.append(labeled)
            if not dets:
                continue
            score = len(dets) * 100.0 + sum(
                float(d.get("confidence") or 0.0) for d in dets
            )
            frame_scores.append((int(frame["frame_number"]), score, dets))

    if not frame_scores:
        return None

    jcfg = {
        "enabled": True,
        "top_frame_fraction": 0.92,
        "max_candidate_frames": 12,
        **(jersey_cfg or {}),
    }
    seed = abs(hash(str(job_id) or str(video))) % (2**31)
    candidates = pick_dense_frame_candidates(
        frame_scores,
        top_fraction=float(jcfg["top_frame_fraction"]),
        max_candidates=int(jcfg["max_candidate_frames"]),
        seed=seed,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    best_frame = None
    best_tracks: list[dict] = []
    best_image = None
    best_team_assign: dict = {"teams": [], "group_to_team": {}, "separation_score": 0.0}
    best_rank = -1.0

    try:
        for frame_id, _score, dets in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            if not jcfg.get("enabled", True):
                best_frame, best_tracks, best_image = frame_id, dets, image
                break
            assignment = assign_jersey_teams(
                image, dets, game_context=game_context, seed=seed + frame_id
            )
            sep = float(assignment.get("separation_score") or 0.0)
            teams = assignment.get("teams") or []
            red_n = blue_n = 0
            for team in teams:
                name = (team.get("jersey_color_name") or "").lower()
                n = len(team.get("member_group_ids") or [])
                if name == "red":
                    red_n = n
                elif name == "blue":
                    blue_n = n
            # Prefer true 5v5-style splits over "everyone red".
            balance = 1.0 - abs(red_n - blue_n) / max(red_n + blue_n, 1)
            rank = sep * 20.0 + balance * 30.0 + len(dets)
            if best_frame is None or rank > best_rank:
                best_rank = rank
                best_frame = frame_id
                best_tracks = dets
                best_image = image
                best_team_assign = assignment
    finally:
        cap.release()

    if best_frame is None or best_image is None:
        return None

    timestamp = best_frame / fps if fps else 0.0
    group_to_team = best_team_assign.get("group_to_team") or {}
    draw_tracks = []
    for det in best_tracks:
        gid = int(det["person_group_id"])
        team = group_to_team.get(gid) or {}
        color_name = team.get("jersey_color_name")
        role = team.get("role")
        if role == "official" or color_name == "white":
            name_label = f"Person #{gid} · Official"
        elif color_name:
            name_label = f"Person #{gid} · {str(color_name).title()}"
        else:
            name_label = f"Person #{gid}"
        draw_tracks.append(
            {
                "track_id": gid,
                "confidence": det.get("confidence"),
                "bbox_xyxy": det.get("bbox_xyxy"),
                "mask_polygon": det.get("mask_polygon"),
                "overlay_color_bgr": team.get("overlay_color_bgr"),
                "label": name_label,
            }
        )
    annotated = draw_showcase_frame(
        best_image,
        draw_tracks,
        frame_id=best_frame,
        timestamp=timestamp,
        draw_hud=False,
        draw_boxes=True,
        draw_labels=True,
        draw_masks=True,
    )
    masks_only = draw_showcase_frame(
        best_image,
        draw_tracks,
        frame_id=best_frame,
        timestamp=timestamp,
        draw_hud=False,
        draw_boxes=False,
        draw_labels=False,
        draw_masks=True,
    )
    cv2.imwrite(str(snapshot_path), annotated)
    masks_path = snapshot_path.with_name(
        snapshot_path.stem.replace("_seg_snapshot", "_seg_snapshot_masks")
        if "_seg_snapshot" in snapshot_path.stem
        else f"{snapshot_path.stem}_masks"
    ).with_suffix(".jpg")
    if snapshot_path.name == "snapshot.jpg":
        masks_path = snapshot_path.with_name("snapshot_masks.jpg")
    cv2.imwrite(str(masks_path), masks_only)

    players = []
    for det in sorted(best_tracks, key=lambda item: (item.get("bbox_xyxy") or [0])[0]):
        gid = int(det["person_group_id"])
        team = group_to_team.get(gid) or {}
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"][:4]]
        raw_members = next(
            (
                g.get("member_track_ids")
                for g in person_groups
                if int(g["person_group_id"]) == gid
            ),
            [],
        )
        color_name = team.get("jersey_color_name")
        role = team.get("role")
        if role == "official" or color_name == "white":
            label = f"Person #{gid} · Official"
        elif color_name:
            label = f"Person #{gid} · {str(color_name).title()}"
        else:
            label = f"Person #{gid}"
        players.append(
            {
                "person_group_id": gid,
                "track_id": gid,
                "member_track_ids": raw_members,
                "label": label,
                "team_id": team.get("team_id"),
                "roster_team_id": team.get("roster_team_id"),
                "team_label": team.get("team_label"),
                "jersey_color_name": color_name,
                "role": role or "player",
                "overlay_color_hex": team.get("overlay_color_hex"),
                "overlay_color_bgr": team.get("overlay_color_bgr"),
                "identity_status": "pending",
                "player_name": None,
                "confidence": round(float(det.get("confidence") or 0.0), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round((x1 + x2) / 2, 2),
                "center_y": round((y1 + y2) / 2, 2),
            }
        )

    return {
        "job_id": job_id,
        "video_path": str(video.resolve()),
        "snapshot_frame_number": best_frame,
        "frame_index_base": 0,
        "timestamp_seconds": round(timestamp, 4),
        "frame_width": width,
        "frame_height": height,
        "players": players,
        "player_count": len(players),
        "teams": best_team_assign.get("teams") or [],
        "jersey_separation_score": best_team_assign.get("separation_score"),
        "image_path": str(snapshot_path.resolve()),
        "image_filename": snapshot_path.name,
        "masks_image_path": str(masks_path.resolve()),
        "masks_image_filename": masks_path.name,
        "selection_unit": "person_group_id",
    }


def _write_tracking_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "frame_number", "timestamp_seconds", "track_id", "player_label",
        "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
        "box_width", "box_height", "has_mask",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Player detect+segment+track (best_player_seg.pt).")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--config", default=None, help="Path to player_tracking_mvp.yaml")
    parser.add_argument("--clip-id", default=None, help="Exclusion lookup key")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames for smoke tests")
    parser.add_argument("--focus", nargs="*", default=None, help="Highlight only these Player_# IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run_player_seg_tracking(
            args.video,
            config_path=args.config,
            clip_id=args.clip_id,
            focus_labels=args.focus,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
