"""
YOLO-based player labeling for WNBA basketball clips.

Detects people in each frame with YOLOv8 (Ultralytics), draws bounding boxes
labeled Player_1, Player_2, etc., and exports an annotated video plus CSV.

Note: labels reset every frame. Player_1 in frame 10 may not be the same
person as Player_1 in frame 11. Tracking (ByteTrack/DeepSORT) comes later.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# --- Detection settings ---
DEFAULT_MODEL = "yolov8n.pt"
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"
CONFIDENCE_THRESHOLD = 0.35

# Output folders (relative to scoutiq/ project root)
ANNOTATED_DIR = Path("outputs/annotated_videos")
DETECTIONS_DIR = Path("outputs/detections")


def resolve_device(requested: str | None = None) -> str:
    """
    Choose the best YOLO inference device.

    Uses GPU (CUDA) when available unless the caller forces --device cpu.
    """
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


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """
    Confirm the input file exists, is MP4, and can be read by OpenCV.
    Raises FileNotFoundError or ValueError with a clear message on failure.
    """
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

    # Verify at least one frame is readable
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
    """Order detections left-to-right so Player_1, Player_2 labels are stable within a frame."""
    return sorted(boxes, key=lambda b: (b["y1"], b["x1"]))


def _extract_person_boxes(result, confidence_threshold: float) -> list[dict]:
    """Pull person detections above the confidence threshold from one YOLO result."""
    detections: list[dict] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id != PERSON_CLASS_ID or confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_width = x2 - x1
        box_height = y2 - y1

        detections.append(
            {
                "class_name": PERSON_CLASS_NAME,
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "box_width": box_width,
                "box_height": box_height,
            }
        )

    return _sort_detections_left_to_right(detections)


def _encode_for_browser(video_path: Path) -> Path:
    """
    Re-encode the annotated MP4 to H.264 for browser and Streamlit playback.

    OpenCV writes mp4v by default, which most browsers cannot play.
    """
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


def process_video(
    metadata: VideoMetadata,
    model: YOLO,
    annotated_path: Path,
    csv_path: Path,
    device: str = "cpu",
) -> dict[str, int]:
    """
    Run YOLO on every frame, write annotated video and detection CSV.
    Returns summary stats for UI and pipeline integrations.
    """
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

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

    csv_rows: list[dict] = []
    frame_number = 0

    print("Starting YOLO player detection...")
    print(f"Model: {DEFAULT_MODEL} | class: {PERSON_CLASS_NAME} | min confidence: {CONFIDENCE_THRESHOLD}")
    print(f"Device: {describe_device(device)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp_seconds = round((frame_number - 1) / metadata.fps, 4)

        # Run YOLO on the current frame (person class only) on CPU or GPU
        results = model(frame, classes=[PERSON_CLASS_ID], device=device, verbose=False)
        detections = _extract_person_boxes(results[0], CONFIDENCE_THRESHOLD)

        # Assign per-frame Player_1, Player_2, ... labels
        for index, det in enumerate(detections, start=1):
            csv_rows.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "label": f"Player_{index}",
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

        _draw_labels(frame, detections)
        writer.write(frame)

        if frame_number % 100 == 0:
            print(f"  Processed {frame_number} frames...")

    cap.release()
    writer.release()

    # Re-encode so the annotated clip plays in browsers (Streamlit st.video)
    _encode_for_browser(annotated_path)

    # Write CSV with one row per detected player per frame
    fieldnames = [
        "frame_number",
        "timestamp_seconds",
        "label",
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
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f"\nDone. Processed {frame_number} frames.")
    print(f"Annotated video: {annotated_path.resolve()}")
    print(f"Detection CSV:   {csv_path.resolve()}")
    print(f"Total detections written: {len(csv_rows)}")

    return {
        "frames_processed": frame_number,
        "total_detections": len(csv_rows),
    }


def run_player_labeling(
    video_path: str | Path,
    model: YOLO | None = None,
    device: str | None = None,
) -> dict[str, str | int]:
    """
    Run the full YOLO player-labeling pipeline for UI and script integrations.

    Returns paths and summary stats. Raises on invalid input or processing errors.
    """
    path = Path(video_path)
    metadata = get_video_metadata(path)
    inference_device = resolve_device(device)

    stem = path.stem
    annotated_path = ANNOTATED_DIR / f"{stem}_labeled_players.mp4"
    csv_path = DETECTIONS_DIR / f"{stem}_player_labels.csv"

    yolo_model = model or YOLO(DEFAULT_MODEL)
    stats = process_video(
        metadata,
        yolo_model,
        annotated_path,
        csv_path,
        device=inference_device,
    )

    return {
        "annotated_video": str(annotated_path.resolve()),
        "detections_csv": str(csv_path.resolve()),
        "frames_processed": stats["frames_processed"],
        "total_detections": stats["total_detections"],
        "device": inference_device,
        "device_label": describe_device(inference_device),
    }


def main() -> None:
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Label basketball players in an MP4 clip with YOLO.")
    parser.add_argument("video_path", help="Path to the input .mp4 file")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "0"],
        default=None,
        help="Inference device. Defaults to GPU when CUDA is available, otherwise CPU.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    device = resolve_device(args.device)

    # 2. Validate input and print metadata
    metadata = validate_video_path(video_path)
    print_metadata(metadata)

    # 3. Build output paths from the input filename
    stem = video_path.stem
    annotated_path = ANNOTATED_DIR / f"{stem}_labeled_players.mp4"
    csv_path = DETECTIONS_DIR / f"{stem}_player_labels.csv"

    # 4. Load YOLO model (downloads yolov8n.pt on first run)
    try:
        print(f"Loading YOLO model: {DEFAULT_MODEL}")
        print(f"Using device: {describe_device(device)}")
        model = YOLO(DEFAULT_MODEL)
    except Exception as exc:
        print(f"Error: Failed to load YOLO model '{DEFAULT_MODEL}': {exc}")
        sys.exit(1)

    # 5. Process every frame and save outputs
    try:
        process_video(metadata, model, annotated_path, csv_path, device=device)
    except Exception as exc:
        print(f"Error during player labeling: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
