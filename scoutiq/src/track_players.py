"""
Player tracking for user-selected targets.

After the GUI snapshot step, the user picks Player_# boxes. This module:
1. Runs YOLO person detection on every frame
2. Associates detections with ByteTrack IDs
3. Locks onto the selected players at the snapshot frame
4. Exports per-frame positions for the rest of the clip

Output feeds future ScoutIQ metrics (speed, spacing, screens, etc.).
No spacing or tactical scoring in this step yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from player_snapshot_selector import DEFAULT_MODEL, resolve_device, validate_video
from video_encoding import encode_for_browser
from yolo_config import PERSON_CLASS_ID

TRACKING_DIR = Path("outputs/tracking")
TRACKING_VIDEOS_DIR = Path("outputs/tracking_videos")
SELECTIONS_DIR = Path("outputs/tracking_selections")

CONFIDENCE_THRESHOLD = 0.35
MIN_MATCH_IOU = 0.30

# Distinct BGR colors per tracked player for the preview video
TRACK_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 255, 0),
    (180, 180, 255),
]


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_tracking_selection(selection_path: str | Path) -> dict:
    """Load a saved GUI tracking-selection JSON file."""
    path = Path(selection_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking selection not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _match_selected_to_tracks(
    selected_players: list[dict],
    detections: sv.Detections,
) -> dict[int, str]:
    """
    At the snapshot frame, map ByteTrack IDs to user labels (Player_2, etc.)
    using IoU matching between selected boxes and tracked detections.
    """
    if detections.tracker_id is None or len(detections) == 0:
        return {}

    n_selected = len(selected_players)
    n_detections = len(detections)
    iou_matrix = np.zeros((n_selected, n_detections), dtype=np.float32)

    for row_idx, player in enumerate(selected_players):
        selected_box = [player["x1"], player["y1"], player["x2"], player["y2"]]
        for col_idx in range(n_detections):
            det_box = detections.xyxy[col_idx].tolist()
            iou_matrix[row_idx, col_idx] = _box_iou(selected_box, det_box)

    cost_matrix = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    label_by_track_id: dict[int, str] = {}
    for row_idx, col_idx in zip(row_indices, col_indices):
        if iou_matrix[row_idx, col_idx] < MIN_MATCH_IOU:
            continue
        track_id = int(detections.tracker_id[col_idx])
        label_by_track_id[track_id] = selected_players[row_idx]["label"]

    return label_by_track_id


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Stable color per Player_# label."""
    try:
        number = int(label.split("_")[1])
    except (IndexError, ValueError):
        number = 1
    return TRACK_COLORS[(number - 1) % len(TRACK_COLORS)]


def _write_tracking_csv(path: Path, rows: list[dict]) -> None:
    """Write per-frame tracked positions."""
    fieldnames = [
        "frame_number",
        "timestamp_seconds",
        "player_label",
        "track_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "box_width",
        "box_height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_tracked_frame(
    frame,
    detections: sv.Detections,
    label_by_track_id: dict[int, str],
) -> None:
    """Draw only the user-selected tracked players on a frame."""
    if detections.tracker_id is None:
        return

    for index in range(len(detections)):
        track_id = int(detections.tracker_id[index])
        label = label_by_track_id.get(track_id)
        if label is None:
            continue

        x1, y1, x2, y2 = detections.xyxy[index].astype(int).tolist()
        confidence = float(detections.confidence[index]) if detections.confidence is not None else 0.0
        color = _color_for_label(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def run_player_tracking(
    selection_path: str | Path | None = None,
    selection_data: dict | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    *,
    quiet: bool = False,
) -> dict:
    """
    Track user-selected players from the snapshot frame through the end of the clip.

    Returns paths and summary stats for API / CLI use.
    """
    if selection_data is None:
        if selection_path is None:
            raise ValueError("Provide selection_path or selection_data.")
        selection_data = load_tracking_selection(selection_path)

    video_path = Path(selection_data["video_path"])
    metadata = validate_video(video_path)

    selected_players = selection_data.get("selected_players", [])
    if not selected_players:
        raise ValueError("No selected players found in tracking selection.")

    anchor_frame = int(selection_data["snapshot_frame_number"])
    if anchor_frame < 1 or anchor_frame > metadata.frame_count:
        raise ValueError(f"Snapshot frame {anchor_frame} is outside the video range.")

    inference_device = resolve_device() if device is None else device
    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    tracks_csv_path = TRACKING_DIR / f"{stem}_tracks.csv"
    summary_json_path = TRACKING_DIR / f"{stem}_tracking_summary.json"
    tracking_video_path = TRACKING_VIDEOS_DIR / f"{stem}_tracked_players.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    writer = cv2.VideoWriter(
        str(tracking_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create tracking video: {tracking_video_path}")

    label_by_track_id: dict[int, str] = {}
    track_rows: list[dict] = []
    frames_exported = 0
    pipeline_start = time.perf_counter()

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp_seconds = round((frame_number - 1) / metadata.fps, 4)

        results = model(
            frame,
            classes=[PERSON_CLASS_ID],
            device=inference_device,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)

        if frame_number == anchor_frame:
            label_by_track_id = _match_selected_to_tracks(selected_players, detections)
            if not label_by_track_id:
                cap.release()
                writer.release()
                raise RuntimeError(
                    "Could not match any selected players to tracker IDs at the snapshot frame."
                )

        if frame_number >= anchor_frame and label_by_track_id and detections.tracker_id is not None:
            for index in range(len(detections)):
                track_id = int(detections.tracker_id[index])
                label = label_by_track_id.get(track_id)
                if label is None:
                    continue

                x1, y1, x2, y2 = detections.xyxy[index].astype(int).tolist()
                confidence = (
                    round(float(detections.confidence[index]), 4)
                    if detections.confidence is not None
                    else 0.0
                )
                center_x = round((x1 + x2) / 2, 2)
                center_y = round((y1 + y2) / 2, 2)

                track_rows.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp_seconds,
                        "player_label": label,
                        "track_id": track_id,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": center_x,
                        "center_y": center_y,
                        "box_width": x2 - x1,
                        "box_height": y2 - y1,
                    }
                )

            frames_exported += 1
            annotated = frame.copy()
            _draw_tracked_frame(annotated, detections, label_by_track_id)
            writer.write(annotated)
        else:
            writer.write(frame)

    cap.release()
    writer.release()
    encode_for_browser(tracking_video_path, quiet=quiet)
    total_runtime = round(time.perf_counter() - pipeline_start, 2)

    if not track_rows:
        raise RuntimeError("Tracking finished without exporting any player positions.")

    _write_tracking_csv(tracks_csv_path, track_rows)

    unmatched_labels = [
        player["label"]
        for player in selected_players
        if player["label"] not in label_by_track_id.values()
    ]

    summary = {
        "video_path": str(video_path.resolve()),
        "snapshot_frame_number": anchor_frame,
        "timestamp_seconds": selection_data.get("timestamp_seconds"),
        "selected_labels": [player["label"] for player in selected_players],
        "track_mapping": {label: track_id for track_id, label in label_by_track_id.items()},
        "frames_tracked": frames_exported,
        "total_positions": len(track_rows),
        "unmatched_labels": unmatched_labels,
        "tracks_csv": str(tracks_csv_path.resolve()),
        "tracking_video": str(tracking_video_path.resolve()),
        "total_runtime_seconds": total_runtime,
        "device": inference_device,
        "model": model_name,
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json_path.resolve())

    if not quiet:
        print(f"Tracked {len(label_by_track_id)} player(s) from frame {anchor_frame}.")
        print(f"  Positions exported: {len(track_rows)}")
        print(f"  Tracks CSV:         {tracks_csv_path.resolve()}")
        print(f"  Tracking video:     {tracking_video_path.resolve()}")
        print(f"  Summary JSON:       {summary_json_path.resolve()}")
        if unmatched_labels:
            print(f"  Warning: could not match {unmatched_labels} at anchor frame.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track user-selected players from a saved GUI selection JSON."
    )
    parser.add_argument(
        "selection_path",
        help="Path to outputs/tracking_selections/<clip>_tracking_selection.json",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO weights filename (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    try:
        run_player_tracking(selection_path=args.selection_path, model_name=args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("Error: CUDA out of memory during tracking. Retry with a smaller model.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
