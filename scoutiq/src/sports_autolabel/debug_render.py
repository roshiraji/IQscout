"""Debug frame drawing and annotated video rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .contracts import parse_frame_json
from .settings import PipelinePaths
from .utils import write_json

COLORS = {
    "player": (80, 200, 80),
    "referee": (70, 160, 240),
    "ball": (40, 120, 240),
    "hoop": (240, 220, 80),
}


def draw_debug_frames(paths: PipelinePaths) -> dict:
    paths.debug_frames_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for json_path in tqdm(sorted(paths.raw_predictions_dir.glob("frame_*.json")), desc="Drawing debug"):
        frame = parse_frame_json(json_path)
        image_path = paths.frames_dir / frame.image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for det in frame.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = COLORS.get(det.class_name, (220, 220, 220))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(image, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        out = paths.debug_frames_dir / frame.image
        cv2.imwrite(str(out), image)
        saved += 1
    summary = {"frames_drawn": saved, "debug_dir": str(paths.debug_frames_dir)}
    write_json(paths.debug_frames_dir / "debug_summary.json", summary)
    return summary


def _write_video_from_frames(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise ValueError("No frames to render.")
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()
        return

    # Fallback for environments where OpenCV mp4 writer is unavailable.
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as io_writer:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                io_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to render video {output_path}: {exc}") from exc


def render_outputs(paths: PipelinePaths, fps: float = 10.0) -> dict:
    paths.final_videos_dir.mkdir(parents=True, exist_ok=True)
    debug_frames = sorted(paths.debug_frames_dir.glob("frame_*.jpg"))
    if not debug_frames:
        debug_frames = sorted(paths.frames_dir.glob("frame_*.jpg"))
        if not debug_frames:
            raise FileNotFoundError("No frames found for rendering. Run extract/debug first.")

    boxes_video = paths.final_videos_dir / "annotated_play_boxes.mp4"
    _write_video_from_frames(debug_frames, boxes_video, fps=fps)

    # If masks/tracking are not available yet, fallback to boxes stream.
    masks_video = paths.final_videos_dir / "annotated_play_masks.mp4"
    tracking_video = paths.final_videos_dir / "annotated_play_tracking.mp4"
    _write_video_from_frames(debug_frames, masks_video, fps=fps)
    _write_video_from_frames(debug_frames, tracking_video, fps=fps)

    summary = {
        "boxes_video": str(boxes_video),
        "masks_video": str(masks_video),
        "tracking_video": str(tracking_video),
    }
    write_json(paths.final_videos_dir / "render_summary.json", summary)
    return summary


def draw_roi_preview(image_path: Path, polygon: np.ndarray, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    cv2.polylines(image, [polygon.astype(np.int32)], isClosed=True, color=(0, 220, 255), thickness=2)
    cv2.imwrite(str(output_path), image)
