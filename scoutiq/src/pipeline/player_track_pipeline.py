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
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.object_tracker import update_tracker
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

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    stem = video.stem
    tracking_video_path = TRACKING_VIDEOS_DIR / f"{stem}_seg_tracked.mp4"
    tracks_csv_path = TRACKING_DIR / f"{stem}_seg_tracks.csv"
    summary_json_path = TRACKING_DIR / f"{stem}_seg_tracking_summary.json"
    snapshot_path = SNAPSHOT_DIR / f"{stem}_seg_snapshot.jpg"

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

    mask_memory: dict[int, list[list[float]]] = {}
    track_lengths: dict[int, int] = {}
    track_rows: list[dict] = []
    frames_processed = 0
    frames_with_players = 0
    snapshot_written = False
    start = time.perf_counter()

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

        dets = detect_players_seg(model, image, conf=conf, iou=iou)
        dets = filter_detections(dets, exclusions, class_names=("player",))
        tracked = update_tracker(dets)

        for det in tracked:
            tid = int(det.get("track_id", -1))
            if tid >= 0:
                track_lengths[tid] = track_lengths.get(tid, 0) + 1
            if mask_smooth_alpha > 0 and tid >= 0:
                smoothed = _smooth_polygon(
                    mask_memory.get(tid), det.get("mask_polygon"), alpha=mask_smooth_alpha
                )
                if smoothed is not None:
                    det["mask_polygon"] = smoothed
                    mask_memory[tid] = smoothed

        # Anti-flicker: drop brand-new low-confidence tracks.
        visible: list[dict] = []
        for det in tracked:
            tid = int(det.get("track_id", -1))
            age = track_lengths.get(tid, 0)
            if age < min_track_frames and float(det.get("confidence") or 0) < conf + 0.1:
                continue
            if focus_ids and tid not in focus_ids:
                continue
            if not mask_overlay:
                det = {**det, "mask_polygon": None}
            visible.append(det)

        if visible:
            frames_with_players += 1

        for det in visible:
            tid = int(det.get("track_id", -1))
            box = det.get("bbox_xyxy") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            track_rows.append({
                "frame_number": frame_id,
                "timestamp_seconds": round(timestamp, 4),
                "track_id": tid,
                "player_label": f"Player_{tid}",
                "confidence": round(float(det.get("confidence") or 0.0), 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": round((x1 + x2) / 2, 2),
                "center_y": round((y1 + y2) / 2, 2),
                "box_width": x2 - x1,
                "box_height": y2 - y1,
                "has_mask": bool(det.get("mask_polygon")),
            })

        annotated = draw_showcase_frame(
            image, visible, frame_id=frame_id, timestamp=timestamp
        )
        writer.write(annotated)

        if not snapshot_written and visible:
            cv2.imwrite(str(snapshot_path), annotated)
            snapshot_written = True

    cap.release()
    writer.release()

    # mp4v -> browser-safe H.264.
    if tmp_video.is_file():
        tmp_video.replace(tracking_video_path)
    encode_for_browser(tracking_video_path, quiet=quiet)

    _write_tracking_csv(tracks_csv_path, track_rows)

    tracked_ids = sorted(track_lengths)
    if focus_ids:
        tracked_ids = [t for t in tracked_ids if t in focus_ids]
    track_mapping = {f"Player_{tid}": tid for tid in tracked_ids}

    runtime = round(time.perf_counter() - start, 2)
    summary = {
        "video_path": str(video.resolve()),
        "weights": str(weights.resolve()),
        "model": weights.name,
        "device": device,
        "snapshot_frame_number": 1,
        "timestamp_seconds": 0.0,
        "selected_labels": list(track_mapping.keys()),
        "track_mapping": track_mapping,
        "unique_tracks": len(track_lengths),
        "frames_tracked": frames_processed,
        "total_positions": len(track_rows),
        "frame_coverage": round(frames_with_players / max(frames_processed, 1), 4),
        "unmatched_labels": [],
        "tracks_csv": str(tracks_csv_path.resolve()),
        "tracking_video": str(tracking_video_path.resolve()),
        "snapshot_image": str(snapshot_path.resolve()) if snapshot_written else None,
        "total_runtime_seconds": runtime,
        "auto_tracked": True,
        "focus_labels": focus_labels or [],
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json_path.resolve())

    if not quiet:
        print(f"[player_track] tracks={len(track_mapping)} frames={frames_processed} "
              f"positions={len(track_rows)} coverage={summary['frame_coverage']}")
        print(f"[player_track] video -> {tracking_video_path.resolve()}")

    logger.info(
        "Player-seg tracking complete → %s | tracks=%d coverage=%.3f",
        tracking_video_path, len(track_mapping), summary["frame_coverage"],
    )
    return summary


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
