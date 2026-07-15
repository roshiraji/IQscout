"""Orchestrates the full video-to-overlay pipeline for one clip."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from court_roi import apply_nms, filter_detections_by_roi, load_or_default_roi
from frame_extractor import extract_all_frames, get_video_info
from locateanything_runner import locate_anything_available, run_locateanything_on_frame
from object_tracker import update_tracker
from render_tracking_overlay import render_video
from sam2_segmenter import refine_detections_with_sam2, sam2_available
from yolo11_seg_runner import detect_frame

logger = logging.getLogger(__name__)

_SCOUTIQ_ROOT = Path(__file__).resolve().parents[1]

# ---- paths -----------------------------------------------------------------
VIDEO_IN = _SCOUTIQ_ROOT / "data" / "input_videos" / "new clip for tunning.mp4"
FRAMES_DIR = _SCOUTIQ_ROOT / "data" / "working_frames" / "new_clip"
LA_JSON_DIR = _SCOUTIQ_ROOT / "data" / "auto_labels" / "locateanything" / "new_clip"
SAM2_DIR = _SCOUTIQ_ROOT / "data" / "auto_labels" / "sam2_masks" / "new_clip"
VIDEO_OUT = _SCOUTIQ_ROOT / "data" / "output_videos" / "new_clip_tracking_overlay.mp4"
TRACKING_DIR = _SCOUTIQ_ROOT / "data" / "output_tracking"
ROI_PATH = _SCOUTIQ_ROOT / "config" / "court_roi.json"

# ---- tunables --------------------------------------------------------------
YOLO_WEIGHTS = "yolo11n-seg.pt"
SAM2_WEIGHTS = "sam2.1_b.pt"
NMS_IOU = 0.45
MIN_BALL_CONF = 0.18
MIN_PERSON_CONF = 0.28
SAM2_BATCH = 8  # process SAM2 in mini-batches to avoid OOM on 8 GB VRAM


def _merge_detections(la_dets: list[dict], yolo_dets: list[dict]) -> list[dict]:
    """Merge LocateAnything + YOLO; prefer LA for grounding, YOLO for shapes."""
    if not la_dets:
        return yolo_dets
    if not yolo_dets:
        return la_dets
    # Simple union with NMS; LocateAnything boxes get slightly boosted conf.
    combined = []
    for d in la_dets:
        d = {**d, "confidence": min(float(d.get("confidence", 0.5)) * 1.05, 1.0)}
        combined.append(d)
    combined.extend(yolo_dets)
    return combined


def _save_tracking_csvs(all_tracked: dict[int, list[dict]]) -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    tracks_json_rows: list[dict] = []
    tracks_csv_rows: list[dict] = []
    ball_rows: list[dict] = []
    player_rows: list[dict] = []

    for frame_id in sorted(all_tracked):
        fps = 30.0
        ts = round(frame_id / fps, 4)
        for det in all_tracked[frame_id]:
            box = det.get("bbox_xyxy", [0, 0, 1, 1])
            x1, y1, x2, y2 = box[:4]
            bw, bh = x2 - x1, y2 - y1
            cx = round((x1 + x2) / 2, 2)
            cy = round((y1 + y2) / 2, 2)
            tid = det.get("track_id", -1)
            conf = round(float(det.get("confidence", 0.5)), 4)
            cname = det.get("class_name", "unknown")

            row = {"frame_id": frame_id, "timestamp": ts, "track_id": tid,
                   "class_name": cname, "x_center": cx, "y_center": cy,
                   "bbox_width": bw, "bbox_height": bh, "confidence": conf}
            tracks_csv_rows.append(row)
            tracks_json_rows.append(row)

            if cname == "ball":
                ball_rows.append({"frame_id": frame_id, "timestamp": ts,
                                   "x_center": cx, "y_center": cy,
                                   "confidence": conf, "visible": 1})
            else:
                player_rows.append(row)

    # tracks.json
    (TRACKING_DIR / "new_clip_tracks.json").write_text(
        json.dumps(tracks_json_rows, indent=2), encoding="utf-8"
    )

    # tracks.csv
    _write_csv(TRACKING_DIR / "new_clip_tracks.csv", tracks_csv_rows,
               ["frame_id", "timestamp", "track_id", "class_name",
                "x_center", "y_center", "bbox_width", "bbox_height", "confidence"])

    # ball_movement.csv (add speed estimate)
    prev_ball: dict | None = None
    for row in ball_rows:
        speed = 0.0
        if prev_ball is not None:
            dx = row["x_center"] - prev_ball["x_center"]
            dy = row["y_center"] - prev_ball["y_center"]
            dt = max(row["timestamp"] - prev_ball["timestamp"], 1e-3)
            speed = round((dx**2 + dy**2) ** 0.5 / dt, 2)
        row["speed_estimate"] = speed
        prev_ball = row
    _write_csv(TRACKING_DIR / "new_clip_ball_movement.csv", ball_rows,
               ["frame_id", "timestamp", "x_center", "y_center",
                "confidence", "speed_estimate", "visible"])

    _write_csv(TRACKING_DIR / "new_clip_player_movement.csv", player_rows,
               ["frame_id", "timestamp", "track_id", "class_name",
                "x_center", "y_center", "bbox_width", "bbox_height", "confidence"])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_pipeline(
    video_path: Path = VIDEO_IN,
    output_path: Path = VIDEO_OUT,
    use_sam2: bool = True,
    use_locateanything: bool = True,
) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {video_path}\n"
            "Expected: data/input_videos/new clip for tunning.mp4"
        )

    info = get_video_info(video_path)
    logger.info("Video: %s — %d frames @ %.1f fps (%dx%d)",
                video_path.name, info.frame_count, info.fps, info.width, info.height)

    # ---- Step 1: Extract frames ----
    frame_paths = extract_all_frames(video_path, FRAMES_DIR)

    # ---- Load ROI ----
    roi = load_or_default_roi(ROI_PATH, width=info.width, height=info.height)
    logger.info("Court ROI polygon: %s", roi.tolist())

    la_available = use_locateanything and locate_anything_available()
    if not la_available:
        logger.warning(
            "LocateAnything not available — using YOLO11-seg as primary detector.\n"
            "To enable: set LOCATEANYTHING_EXE=<path to executable>"
        )

    sam2_ok = use_sam2 and sam2_available()
    if not sam2_ok:
        logger.warning("SAM2 unavailable — masks will use YOLO11-seg polygons only.")

    all_tracked: dict[int, list[dict]] = {}

    logger.info("Processing %d frames...", len(frame_paths))
    for frame_path in tqdm(frame_paths, desc="Detecting + tracking", unit="frame"):
        try:
            fid = int(frame_path.stem.split("_")[-1])
        except ValueError:
            fid = 0
        ts = fid / info.fps

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            logger.warning("Cannot read frame %s — skipping.", frame_path.name)
            all_tracked[fid] = []
            continue

        # ---- Step 3: LocateAnything (if available) ----
        la_dets = []
        if la_available:
            la_dets = run_locateanything_on_frame(
                image_path=frame_path,
                frame_id=fid,
                timestamp=ts,
                width=info.width,
                height=info.height,
            )

        # ---- Step 7: YOLO11-seg ----
        yolo_dets = detect_frame(
            image_bgr=image_bgr,
            frame_id=fid,
            timestamp=ts,
            image_name=frame_path.name,
            weights=YOLO_WEIGHTS,
        )

        # ---- Merge ----
        merged = _merge_detections(la_dets, yolo_dets)

        # ---- Step 5: ROI filter + NMS ----
        merged = filter_detections_by_roi(merged, roi)
        merged = apply_nms(merged, iou_threshold=NMS_IOU)

        # ---- Save LocateAnything JSON ----
        if la_available:
            la_out = LA_JSON_DIR / f"{frame_path.stem}.json"
            la_out.write_text(json.dumps({"frame_id": fid, "detections": la_dets}, indent=2), encoding="utf-8")

        # ---- Step 6: SAM2 masks ----
        if sam2_ok and merged:
            merged = refine_detections_with_sam2(image_bgr, merged, SAM2_WEIGHTS)

        # ---- Step 9: Track ----
        tracked = update_tracker(merged)
        all_tracked[fid] = tracked

    # ---- Step 11: Save tracking CSVs ----
    logger.info("Saving tracking data...")
    _save_tracking_csvs(all_tracked)

    # ---- Step 10: Render final video ----
    logger.info("Rendering annotated video → %s", output_path)
    render_video(
        frame_paths=frame_paths,
        detections_by_frame=all_tracked,
        output_path=output_path,
        fps=info.fps,
        width=info.width,
        height=info.height,
    )

    logger.info("Pipeline complete. Output: %s", output_path)
    return output_path
