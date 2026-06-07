"""
YOLO-based player labeling for WNBA basketball clips.

Detects people in each frame with YOLO26 (Ultralytics), draws bounding boxes
labeled Player_1, Player_2, etc., and exports an annotated video plus CSV.

Note: labels reset every frame. Player_1 in frame 10 may not be the same
person as Player_1 in frame 11. Tracking (ByteTrack/DeepSORT) comes later.

# Phase 2:
# ByteTrack tracking

# Phase 3:
# Jersey number recognition

# Phase 4:
# Player identification

# Phase 5:
# Spacing and play analytics
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from yolo_config import (
    AVAILABLE_IMGSZ,
    AVAILABLE_YOLO_MODELS,
    DEFAULT_YOLO_MODEL,
    InferenceProfile,
    LOW_CONF_REVIEW_MIN,
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    QUALITY_MODE,
    get_inference_profile,
    load_yolo_model,
    model_display_name,
)

# Output folders (relative to scoutiq/ project root)
ANNOTATED_DIR = Path("outputs/annotated_videos")
DETECTIONS_DIR = Path("outputs/detections")


@dataclass
class VideoMetadata:
    """Basic video properties read through OpenCV."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def resolve_device(requested: str | None = None) -> str:
    """Choose the best YOLO inference device."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def describe_device(device: str) -> str:
    """Return a human-readable device label for logs and UI."""
    if device == "cpu":
        return "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device) if device.isdigit() else 0)
        return f"GPU ({gpu_name})"
    return f"GPU (device {device})"


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Confirm the input file exists, is MP4, and can be read by OpenCV."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Unsupported file type '{video_path.suffix}'. Only .mp4 is supported.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, _ = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Video is unreadable or has no frames: {video_path}")

    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("Video metadata is invalid (FPS, width, height, or frame count is zero).")

    return VideoMetadata(
        path=video_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def validate_video_path(video_path: Path) -> VideoMetadata:
    """CLI-friendly wrapper around get_video_metadata that exits on error."""
    try:
        return get_video_metadata(video_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def print_metadata(metadata: VideoMetadata) -> None:
    """Print source video details before YOLO processing begins."""
    print("\n--- Source video metadata ---")
    print(f"  filename:     {metadata.path.name}")
    print(f"  FPS:          {metadata.fps:.2f}")
    print(f"  frame count:  {metadata.frame_count}")
    print(f"  duration:     {metadata.duration_seconds:.2f} seconds")
    print(f"  width:        {metadata.width} px")
    print(f"  height:       {metadata.height} px")
    print("-----------------------------\n")


def _sort_detections_left_to_right(boxes: list[dict]) -> list[dict]:
    """Order detections top-to-bottom, left-to-right for stable per-frame labels."""
    return sorted(boxes, key=lambda b: (b["y1"], b["x1"]))


def _parse_person_boxes(result) -> list[dict]:
    """Parse all person detections from one YOLO result (already filtered at review floor)."""
    detections: list[dict] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detections.append(
            {
                "class_id": class_id,
                "class_name": PERSON_CLASS_NAME,
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "box_width": x2 - x1,
                "box_height": y2 - y1,
            }
        )

    return detections


def _split_detections(
    boxes: list[dict],
    primary_conf: float,
) -> tuple[list[dict], list[dict]]:
    """
    Split detections into primary (used for labels/video) and review candidates.

    Review candidates may represent distant or partially occluded players
    recoverable later via ByteTrack.
    """
    primary: list[dict] = []
    review: list[dict] = []

    for box in boxes:
        if box["confidence"] >= primary_conf:
            primary.append(box)
        elif LOW_CONF_REVIEW_MIN <= box["confidence"] < primary_conf:
            review.append(box)

    return _sort_detections_left_to_right(primary), review


def _frame_statistics_row(frame_number: int, primary: list[dict]) -> dict:
    """Build per-frame detection statistics from primary detections."""
    if not primary:
        return {
            "frame_number": frame_number,
            "player_count": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
        }

    confidences = [det["confidence"] for det in primary]
    return {
        "frame_number": frame_number,
        "player_count": len(primary),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "min_confidence": round(min(confidences), 4),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _encode_for_browser(video_path: Path) -> Path:
    """Re-encode the annotated MP4 to H.264 for browser and Streamlit playback."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("Note: install imageio-ffmpeg for in-browser video playback.")
        return video_path

    temp_path = video_path.with_suffix(".browser.mp4")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and temp_path.is_file() and temp_path.stat().st_size > 0:
        video_path.unlink()
        temp_path.rename(video_path)
        print("Encoded annotated video for browser playback (H.264).")
    else:
        temp_path.unlink(missing_ok=True)
        print("Warning: Browser video encoding failed; Streamlit playback may not work.")

    return video_path


def _draw_labels(frame, detections: list[dict]) -> None:
    """Draw bounding boxes and Player_N labels on a single frame."""
    for index, det in enumerate(detections, start=1):
        label = f"Player_{index} {det['confidence']:.2f}"
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _run_inference(model: YOLO, frame, device: str, profile: InferenceProfile):
    """Run YOLO on one frame with the active quality/speed profile."""
    return model(
        frame,
        classes=[PERSON_CLASS_ID],
        device=device,
        imgsz=profile.imgsz,
        conf=LOW_CONF_REVIEW_MIN,
        augment=profile.augment,
        iou=profile.iou,
        verbose=False,
    )


def process_video(
    metadata: VideoMetadata,
    model: YOLO,
    annotated_path: Path,
    player_labels_path: Path,
    bytetrack_path: Path,
    frame_stats_path: Path,
    low_conf_path: Path,
    device: str = "cpu",
    model_name: str | None = None,
    profile: InferenceProfile | None = None,
) -> dict[str, float | int]:
    """
    Run YOLO on every frame individually and write all detection outputs.
    """
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    inference_profile = profile or get_inference_profile()

    cap = cv2.VideoCapture(str(metadata.path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {annotated_path}")

    player_label_rows: list[dict] = []
    bytetrack_rows: list[dict] = []
    frame_stat_rows: list[dict] = []
    low_conf_rows: list[dict] = []
    inference_times: list[float] = []

    frame_number = 0
    pipeline_start = time.perf_counter()

    print("Starting YOLO26 player detection (every frame)...")
    print(f"Model: {model_display_name(model_name)}")
    print(f"Mode:  {inference_profile.mode_label}")
    print(
        f"Settings: imgsz={inference_profile.imgsz}, primary_conf={inference_profile.conf}, "
        f"augment={inference_profile.augment}, iou={inference_profile.iou}, "
        f"review_floor={LOW_CONF_REVIEW_MIN}"
    )
    print(f"Device: {describe_device(device)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp_seconds = round((frame_number - 1) / metadata.fps, 4)

        infer_start = time.perf_counter()
        try:
            results = _run_inference(model, frame, device, inference_profile)
        except torch.cuda.OutOfMemoryError:
            if inference_profile.imgsz > 1280:
                print(
                    "CUDA out of memory at imgsz="
                    f"{inference_profile.imgsz}. Retry with --imgsz 1280."
                )
            raise
        infer_elapsed = time.perf_counter() - infer_start
        inference_times.append(infer_elapsed)

        all_boxes = _parse_person_boxes(results[0])
        primary, review = _split_detections(all_boxes, inference_profile.conf)

        frame_stat_rows.append(_frame_statistics_row(frame_number, primary))

        for index, det in enumerate(primary, start=1):
            player_label_rows.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "label": f"Player_{index}",
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                    "center_x": det["center_x"],
                    "center_y": det["center_y"],
                    "box_width": det["box_width"],
                    "box_height": det["box_height"],
                }
            )
            bytetrack_rows.append(
                {
                    "frame": frame_number,
                    "class_id": det["class_id"],
                    "confidence": det["confidence"],
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                }
            )

        for det in review:
            low_conf_rows.append(
                {
                    "frame_number": frame_number,
                    "confidence": det["confidence"],
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                }
            )

        _draw_labels(frame, primary)
        writer.write(frame)

        if frame_number % 100 == 0:
            print(f"  Processed {frame_number} frames...")

    cap.release()
    writer.release()
    total_runtime = time.perf_counter() - pipeline_start

    _encode_for_browser(annotated_path)

    _write_csv(
        player_labels_path,
        [
            "frame_number",
            "timestamp_seconds",
            "label",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "box_width",
            "box_height",
        ],
        player_label_rows,
    )
    _write_csv(
        bytetrack_path,
        ["frame", "class_id", "confidence", "x1", "y1", "x2", "y2"],
        bytetrack_rows,
    )
    _write_csv(
        frame_stats_path,
        ["frame_number", "player_count", "avg_confidence", "max_confidence", "min_confidence"],
        frame_stat_rows,
    )
    _write_csv(
        low_conf_path,
        ["frame_number", "confidence", "x1", "y1", "x2", "y2"],
        low_conf_rows,
    )

    avg_inference_ms = (
        (sum(inference_times) / len(inference_times)) * 1000 if inference_times else 0.0
    )
    pipeline_fps = frame_number / total_runtime if total_runtime > 0 else 0.0

    print(f"\nDone. Processed {frame_number} frames.")
    print(f"Annotated video:          {annotated_path.resolve()}")
    print(f"Player labels CSV:        {player_labels_path.resolve()}")
    print(f"ByteTrack detections CSV: {bytetrack_path.resolve()}")
    print(f"Frame statistics CSV:     {frame_stats_path.resolve()}")
    print(f"Low-confidence CSV:       {low_conf_path.resolve()}")
    print(f"Primary detections:       {len(player_label_rows)}")
    print(f"Review candidates:        {len(low_conf_rows)}")
    print("--- Timing ---")
    print(f"  Total runtime:            {total_runtime:.2f} s")
    print(f"  Avg inference per frame:  {avg_inference_ms:.1f} ms")
    print(f"  Pipeline FPS:             {pipeline_fps:.2f}")

    return {
        "frames_processed": frame_number,
        "total_detections": len(player_label_rows),
        "review_candidates": len(low_conf_rows),
        "total_runtime_seconds": round(total_runtime, 2),
        "avg_inference_ms_per_frame": round(avg_inference_ms, 2),
        "pipeline_fps": round(pipeline_fps, 2),
    }


def run_player_labeling(
    video_path: str | Path,
    model: YOLO | None = None,
    device: str | None = None,
    model_name: str | None = None,
    quality_mode: bool | None = None,
    imgsz: int | None = None,
) -> dict[str, str | int | float | bool]:
    """Run the full YOLO player-labeling pipeline for UI and script integrations."""
    path = Path(video_path)
    metadata = get_video_metadata(path)
    inference_device = resolve_device(device)
    profile = get_inference_profile(quality_mode=quality_mode, imgsz=imgsz)

    stem = path.stem
    annotated_path = ANNOTATED_DIR / f"{stem}_labeled_players.mp4"
    player_labels_path = DETECTIONS_DIR / f"{stem}_player_labels.csv"
    bytetrack_path = DETECTIONS_DIR / f"{stem}_bytetrack_detections.csv"
    frame_stats_path = DETECTIONS_DIR / f"{stem}_frame_statistics.csv"
    low_conf_path = DETECTIONS_DIR / f"{stem}_low_confidence_candidates.csv"

    weights = model_name or DEFAULT_YOLO_MODEL
    yolo_model = model or load_yolo_model(weights)
    stats = process_video(
        metadata,
        yolo_model,
        annotated_path,
        player_labels_path,
        bytetrack_path,
        frame_stats_path,
        low_conf_path,
        device=inference_device,
        model_name=weights,
        profile=profile,
    )

    return {
        "annotated_video": str(annotated_path.resolve()),
        "detections_csv": str(player_labels_path.resolve()),
        "bytetrack_csv": str(bytetrack_path.resolve()),
        "frame_statistics_csv": str(frame_stats_path.resolve()),
        "low_confidence_csv": str(low_conf_path.resolve()),
        "frames_processed": stats["frames_processed"],
        "total_detections": stats["total_detections"],
        "review_candidates": stats["review_candidates"],
        "total_runtime_seconds": stats["total_runtime_seconds"],
        "avg_inference_ms_per_frame": stats["avg_inference_ms_per_frame"],
        "pipeline_fps": stats["pipeline_fps"],
        "device": inference_device,
        "device_label": describe_device(inference_device),
        "model": weights,
        "model_label": model_display_name(weights),
        "quality_mode": profile.quality_mode,
        "imgsz": profile.imgsz,
        "primary_conf": profile.conf,
        "augment": profile.augment,
        "iou": profile.iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label basketball players in an MP4 clip with YOLO26."
    )
    parser.add_argument("video_path", help="Path to the input .mp4 file")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "0"],
        default=None,
        help="Inference device. Defaults to GPU when CUDA is available, otherwise CPU.",
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_YOLO_MODELS,
        default=DEFAULT_YOLO_MODEL,
        metavar="WEIGHTS",
        help=(
            f"YOLO26 weights (default: {DEFAULT_YOLO_MODEL}). "
            f"Choices: {', '.join(AVAILABLE_YOLO_MODELS)}"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quality",
        action="store_true",
        help="Use quality-first inference settings (default).",
    )
    mode_group.add_argument(
        "--fast",
        action="store_true",
        help="Use speed-first inference settings.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=AVAILABLE_IMGSZ,
        default=None,
        help="Inference resolution for quality mode (1280 default, 1920 for max quality).",
    )
    args = parser.parse_args()

    if args.fast:
        quality_mode = False
    elif args.quality:
        quality_mode = True
    else:
        quality_mode = QUALITY_MODE

    video_path = Path(args.video_path)
    device = resolve_device(args.device)
    profile = get_inference_profile(quality_mode=quality_mode, imgsz=args.imgsz)

    metadata = validate_video_path(video_path)
    print_metadata(metadata)

    stem = video_path.stem
    annotated_path = ANNOTATED_DIR / f"{stem}_labeled_players.mp4"
    player_labels_path = DETECTIONS_DIR / f"{stem}_player_labels.csv"
    bytetrack_path = DETECTIONS_DIR / f"{stem}_bytetrack_detections.csv"
    frame_stats_path = DETECTIONS_DIR / f"{stem}_frame_statistics.csv"
    low_conf_path = DETECTIONS_DIR / f"{stem}_low_confidence_candidates.csv"

    try:
        print(f"Loading model: {model_display_name(args.model)}")
        print(f"Using device: {describe_device(device)}")
        model = load_yolo_model(args.model)
    except Exception as exc:
        print(f"Error: Failed to load YOLO model '{args.model}': {exc}")
        sys.exit(1)

    try:
        process_video(
            metadata,
            model,
            annotated_path,
            player_labels_path,
            bytetrack_path,
            frame_stats_path,
            low_conf_path,
            device=device,
            model_name=args.model,
            profile=profile,
        )
    except Exception as exc:
        print(f"Error during player labeling: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
