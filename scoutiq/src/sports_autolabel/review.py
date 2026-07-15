"""Review queue creation and placeholder manual review store."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .contracts import CorrectedLabelRecord, ReviewQueueItem, parse_frame_json
from .settings import PipelinePaths
from .utils import read_json, write_json


def score_for_review(paths: PipelinePaths) -> dict[int, tuple[float, list[str]]]:
    frame_files = sorted(paths.raw_predictions_dir.glob("frame_*.json"))
    scores: dict[int, tuple[float, list[str]]] = {}
    previous_player_count: int | None = None

    for json_path in frame_files:
        frame = parse_frame_json(json_path)
        reasons: list[str] = []
        score = 0.0
        players = [d for d in frame.detections if d.class_name == "player"]
        refs = [d for d in frame.detections if d.class_name == "referee"]
        balls = [d for d in frame.detections if d.class_name == "ball"]
        low_conf = [d for d in frame.detections if d.confidence < 0.45]

        if not 8 <= len(players) <= 12:
            score += 0.35
            reasons.append("player_count_outside_expected_range")
        if len(refs) >= max(len(players), 1):
            score += 0.25
            reasons.append("referee_count_too_high")
        if len(balls) == 0:
            score += 0.22
            reasons.append("ball_missing")
        if low_conf:
            score += 0.2
            reasons.append("low_confidence_detections")
        if previous_player_count is not None and abs(len(players) - previous_player_count) >= 4:
            score += 0.28
            reasons.append("sudden_player_count_shift")

        previous_player_count = len(players)
        scores[frame.frame_id] = (score, reasons)

    return scores


def build_review_queue(paths: PipelinePaths, percent: int) -> dict:
    percent = min(max(percent, 1), 100)
    scored = score_for_review(paths)
    frame_rows = []
    for json_path in sorted(paths.raw_predictions_dir.glob("frame_*.json")):
        frame = parse_frame_json(json_path)
        score, reasons = scored.get(frame.frame_id, (0.0, []))
        frame_rows.append(
            ReviewQueueItem(
                frame_path=str((paths.frames_dir / frame.image).resolve()),
                frame_id=frame.frame_id,
                score=score,
                reasons=reasons if reasons else ["sampled_for_quality_check"],
            )
        )

    frame_rows.sort(key=lambda row: row.score, reverse=True)
    pick = max(int(round(len(frame_rows) * (percent / 100.0))), 1)
    selected = frame_rows[:pick]
    payload = {
        "requested_percent": percent,
        "selected_count": len(selected),
        "total_frames": len(frame_rows),
        "items": [item.to_dict() for item in selected],
    }
    write_json(paths.review_queue_path, payload)
    return payload


def write_placeholder_review_file(paths: PipelinePaths) -> dict:
    """
    Manual review should save corrected labels separately from raw predictions.
    """
    corrected_path = paths.corrected_predictions_dir / "corrected_labels.json"
    if not corrected_path.is_file():
        write_json(corrected_path, [])
    return {"corrected_labels_path": str(corrected_path)}


def append_corrected_label(paths: PipelinePaths, record: CorrectedLabelRecord) -> None:
    corrected_path = paths.corrected_predictions_dir / "corrected_labels.json"
    rows = read_json(corrected_path, default=[])
    rows.append(record.to_dict())
    write_json(corrected_path, rows)
