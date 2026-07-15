"""Frame extraction and synthetic demo frame generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import write_json


@dataclass(slots=True)
class ExtractionSummary:
    video: str
    output_dir: str
    source_fps: float
    sample_fps: float
    frame_count: int
    frames_saved: int

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "output_dir": self.output_dir,
            "source_fps": round(float(self.source_fps), 3),
            "sample_fps": round(float(self.sample_fps), 3),
            "frame_count": int(self.frame_count),
            "frames_saved": int(self.frames_saved),
        }


def extract_frames(video_path: Path, output_dir: Path, fps_sample: float) -> ExtractionSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(int(round(source_fps / max(fps_sample, 0.1))), 1)

    saved = 0
    frame_idx = 0
    progress = tqdm(total=frame_count if frame_count > 0 else None, desc="Extracting frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            out = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
        frame_idx += 1
        progress.update(1)
    progress.close()
    cap.release()

    summary = ExtractionSummary(
        video=str(video_path.resolve()),
        output_dir=str(output_dir.resolve()),
        source_fps=source_fps,
        sample_fps=fps_sample,
        frame_count=frame_count,
        frames_saved=saved,
    )
    write_json(output_dir / "extract_summary.json", summary.to_dict())
    return summary


def generate_demo_frames(output_dir: Path, frame_count: int = 16, width: int = 1280, height: int = 720) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in range(frame_count):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (15, 90, 30)

        # Court rectangle and center line.
        cv2.rectangle(canvas, (80, 80), (width - 80, height - 80), (160, 190, 170), 2)
        cv2.line(canvas, (width // 2, 80), (width // 2, height - 80), (180, 200, 180), 1)
        cv2.circle(canvas, (width // 2, height // 2), 84, (180, 200, 180), 1)

        # Hoop placeholder.
        hoop_x = int(width * 0.82)
        hoop_y = int(height * 0.18)
        cv2.rectangle(canvas, (hoop_x - 35, hoop_y - 10), (hoop_x + 35, hoop_y + 30), (220, 220, 220), 2)

        # Moving player dots.
        for player_id in range(10):
            x = 180 + (player_id * 90) + ((frame_id * (player_id % 3 + 1) * 3) % 40)
            y = 200 + ((player_id * 47 + frame_id * 9) % 320)
            cv2.rectangle(canvas, (x - 18, y - 36), (x + 18, y + 36), (220, 180, 120), 2)

        # Ball.
        ball_x = int(width * 0.3 + frame_id * 20)
        ball_y = int(height * 0.3 + ((frame_id % 6) - 3) * 8)
        cv2.circle(canvas, (ball_x, ball_y), 7, (30, 120, 240), -1)

        label = f"demo frame {frame_id:03d}"
        cv2.putText(canvas, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"frame_{frame_id:06d}.jpg"), canvas)
