"""Extract every frame from a video, preserving original FPS and resolution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=frame_count / fps if fps > 0 else 0.0,
    )


def extract_all_frames(video_path: Path, output_dir: Path) -> list[Path]:
    """Extract every frame as frame_XXXXXX.jpg and return sorted list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info = get_video_info(video_path)

    # Skip if already extracted (idempotent).
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) == info.frame_count and info.frame_count > 0:
        logger.info("Frames already extracted (%d found). Skipping.", len(existing))
        return existing

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_paths: list[Path] = []
    frame_idx = 0
    bar = tqdm(total=info.frame_count, desc="Extracting frames", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(out)
        frame_idx += 1
        bar.update(1)
    bar.close()
    cap.release()

    meta = {
        "video": str(video_path.resolve()),
        "frames_extracted": len(frame_paths),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Extracted %d frames → %s", len(frame_paths), output_dir)
    return frame_paths
