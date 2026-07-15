"""Strict data contracts for auto-label pipeline artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DetectionRecord:
    class_id: int
    class_name: str
    prompt: str
    confidence: float
    bbox_xyxy: list[int]
    source: str
    review_status: str = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy"] = [int(v) for v in self.bbox_xyxy]
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


@dataclass(slots=True)
class FrameDetections:
    image: str
    frame_id: int
    timestamp: float
    width: int
    height: int
    detections: list[DetectionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "frame_id": int(self.frame_id),
            "timestamp": round(float(self.timestamp), 3),
            "width": int(self.width),
            "height": int(self.height),
            "detections": [item.to_dict() for item in self.detections],
        }


@dataclass(slots=True)
class CorrectedLabelRecord:
    frame_path: str
    source_model: str
    original_prediction: dict[str, Any]
    corrected_prediction: dict[str, Any]
    review_decision: str
    reviewer_timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewQueueItem:
    frame_path: str
    frame_id: int
    score: float
    reasons: list[str]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


def parse_frame_json(path: Path) -> FrameDetections:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        DetectionRecord(
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            prompt=str(row["prompt"]),
            confidence=float(row["confidence"]),
            bbox_xyxy=[int(v) for v in row["bbox_xyxy"]],
            source=str(row["source"]),
            review_status=str(row.get("review_status", "unreviewed")),
        )
        for row in payload.get("detections", [])
    ]
    return FrameDetections(
        image=str(payload["image"]),
        frame_id=int(payload["frame_id"]),
        timestamp=float(payload["timestamp"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        detections=items,
    )
