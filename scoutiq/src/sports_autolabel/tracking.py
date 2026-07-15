"""Tracking outputs with graceful fallback to simple IoU tracking."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .contracts import parse_frame_json
from .settings import PipelinePaths
from .utils import write_json


@dataclass(slots=True)
class TrackRow:
    frame_id: int
    timestamp: float
    track_id: int
    class_name: str
    x_center: float
    y_center: float
    bbox_width: float
    bbox_height: float
    confidence: float


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)
    area_b = max(bx2 - bx1, 1) * max(by2 - by1, 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_and_size(box: list[int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return x1 + w / 2, y1 + h / 2, w, h


def run_simple_iou_tracking(paths: PipelinePaths) -> dict:
    frame_paths = sorted(paths.raw_predictions_dir.glob("frame_*.json"))
    if not frame_paths:
        raise FileNotFoundError("No frame detections found. Run autolabel first.")

    next_track_id = 1
    previous_tracks: dict[int, list[int]] = {}
    track_rows: list[TrackRow] = []
    tracks_json_rows: list[dict] = []

    for frame_path in frame_paths:
        frame = parse_frame_json(frame_path)
        current_tracks: dict[int, list[int]] = {}
        for det in frame.detections:
            if det.class_name not in {"player", "ball"}:
                continue
            best_track = None
            best_iou = 0.0
            for track_id, prev_box in previous_tracks.items():
                iou = _iou(det.bbox_xyxy, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            if best_track is None or best_iou < 0.25:
                track_id = next_track_id
                next_track_id += 1
            else:
                track_id = best_track
            current_tracks[track_id] = det.bbox_xyxy
            xc, yc, bw, bh = _center_and_size(det.bbox_xyxy)
            row = TrackRow(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                track_id=track_id,
                class_name=det.class_name,
                x_center=xc,
                y_center=yc,
                bbox_width=bw,
                bbox_height=bh,
                confidence=det.confidence,
            )
            track_rows.append(row)
            tracks_json_rows.append(
                {
                    "frame_id": row.frame_id,
                    "timestamp": row.timestamp,
                    "track_id": row.track_id,
                    "class_name": row.class_name,
                    "bbox_xyxy": det.bbox_xyxy,
                    "confidence": row.confidence,
                }
            )
        previous_tracks = current_tracks

    paths.tracking_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = paths.tracking_dir / "tracks.csv"
    with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "timestamp",
                "track_id",
                "class_name",
                "x_center",
                "y_center",
                "bbox_width",
                "bbox_height",
                "confidence",
            ]
        )
        for row in track_rows:
            writer.writerow(
                [
                    row.frame_id,
                    f"{row.timestamp:.3f}",
                    row.track_id,
                    row.class_name,
                    f"{row.x_center:.2f}",
                    f"{row.y_center:.2f}",
                    f"{row.bbox_width:.2f}",
                    f"{row.bbox_height:.2f}",
                    f"{row.confidence:.4f}",
                ]
            )

    tracks_json = paths.tracking_dir / "tracks.json"
    write_json(tracks_json, tracks_json_rows)

    # Movement exports requested for downstream analytics.
    ball_rows = [row for row in track_rows if row.class_name == "ball"]
    player_rows = [row for row in track_rows if row.class_name == "player"]

    ball_csv = paths.tracking_dir / "ball_movement.csv"
    with ball_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "timestamp", "x_center", "y_center", "confidence", "speed_estimate", "visible"])
        prev = None
        for row in ball_rows:
            speed = 0.0
            if prev is not None:
                dx = row.x_center - prev.x_center
                dy = row.y_center - prev.y_center
                dt = max(row.timestamp - prev.timestamp, 1e-3)
                speed = ((dx * dx + dy * dy) ** 0.5) / dt
            writer.writerow(
                [
                    row.frame_id,
                    f"{row.timestamp:.3f}",
                    f"{row.x_center:.2f}",
                    f"{row.y_center:.2f}",
                    f"{row.confidence:.4f}",
                    f"{speed:.2f}",
                    "1",
                ]
            )
            prev = row

    player_csv = paths.tracking_dir / "player_movement.csv"
    with player_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "timestamp",
                "track_id",
                "class_name",
                "x_center",
                "y_center",
                "bbox_width",
                "bbox_height",
                "confidence",
            ]
        )
        for row in player_rows:
            writer.writerow(
                [
                    row.frame_id,
                    f"{row.timestamp:.3f}",
                    row.track_id,
                    row.class_name,
                    f"{row.x_center:.2f}",
                    f"{row.y_center:.2f}",
                    f"{row.bbox_width:.2f}",
                    f"{row.bbox_height:.2f}",
                    f"{row.confidence:.4f}",
                ]
            )

    summary = {
        "tracks_csv": str(tracks_csv),
        "tracks_json": str(tracks_json),
        "ball_movement_csv": str(ball_csv),
        "player_movement_csv": str(player_csv),
        "rows_total": len(track_rows),
        "tracker": "simple_iou_fallback",
    }
    write_json(paths.tracking_dir / "tracking_summary.json", summary)
    return summary
