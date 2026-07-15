"""YOLO person detections used as SAM box prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from src.yolo_config import PERSON_CLASS_ID, load_yolo_model


@dataclass
class BoxPrompt:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def detect_people(
    model: YOLO,
    frame_bgr,
    *,
    device: str,
    conf: float,
    imgsz: int,
) -> list[BoxPrompt]:
    results = model(
        frame_bgr,
        classes=[PERSON_CLASS_ID],
        device=device,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    boxes: list[BoxPrompt] = []
    for box in results[0].boxes:
        score = float(box.conf[0])
        if score < conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        boxes.append(
            BoxPrompt(
                label="",
                confidence=round(score, 4),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )

    boxes.sort(key=lambda b: (b.x1 + b.x2) / 2)
    for index, item in enumerate(boxes, start=1):
        item.label = f"Player_{index}"
    return boxes


def load_snapshot_boxes(snapshot_json: Path) -> tuple[int, float, list[BoxPrompt]]:
    import json

    payload = json.loads(snapshot_json.read_text(encoding="utf-8"))
    frame_number = int(payload["snapshot_frame_number"])
    timestamp = float(payload.get("timestamp_seconds", 0.0))
    players = []
    for player in payload.get("players", []):
        players.append(
            BoxPrompt(
                label=str(player["label"]),
                confidence=float(player.get("confidence", 0.0)),
                x1=int(player["x1"]),
                y1=int(player["y1"]),
                x2=int(player["x2"]),
                y2=int(player["y2"]),
            )
        )
    return frame_number, timestamp, players


def read_frame(video_path: Path, frame_number: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def iter_sample_frames(video_path: Path, stride: int, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_idx = 0
    yielded = 0
    while yielded < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            timestamp = frame_idx / fps if fps > 0 else 0.0
            yield frame_idx, timestamp, frame
            yielded += 1
        frame_idx += 1
    cap.release()


def load_yolo(weights: str) -> YOLO:
    return load_yolo_model(weights)
