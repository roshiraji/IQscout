"""
Video Intake Filter — quality-control gate before the vision pipeline.

Validates uploaded basketball footage (MP4) for format, size, duration,
resolution, and readability. Does NOT run YOLO or tracking.

Future flow: upload → validate_video() → if approved → player_labeling.py (YOLO26) → tracking
"""

from __future__ import annotations

import json
import math
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

# --- MVP intake standards (adjust here as requirements evolve) ---
ALLOWED_EXTENSIONS = {".mp4"}
ALLOWED_MIME_TYPES = {"video/mp4", "application/mp4"}
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 15
MIN_WIDTH = 640
MIN_HEIGHT = 360
PREFERRED_WIDTH = 1280
PREFERRED_HEIGHT = 720
MAX_FILE_SIZE_MB = 500
MIN_FPS = 15
MAX_FPS = 120


def _aspect_ratio_label(width: int, height: int) -> str:
    """Return a readable aspect ratio string such as 16:9."""
    if width <= 0 or height <= 0:
        return "unknown"
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _resolution_label(width: int, height: int) -> str:
    """Map pixel dimensions to a common resolution label."""
    if height >= 1080:
        return "1080p"
    if height >= 720:
        return "720p"
    if height >= 480:
        return "480p"
    if height >= 360:
        return "360p"
    return "below_360p"


def _guess_mime_type(path: Path) -> str | None:
    """Return guessed MIME type from the file extension, if available."""
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type


def _empty_metadata(filename: str = "") -> dict[str, Any]:
    """Return a metadata dict with safe default values."""
    return {
        "filename": filename,
        "file_size_mb": 0.0,
        "duration_seconds": 0.0,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "aspect_ratio": "unknown",
        "resolution_label": "unknown",
        "mime_type": None,
    }


def _build_result(
    usable: bool,
    status: str,
    message: str,
    metadata: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """Assemble the standard validation response dictionary."""
    return {
        "usable": usable,
        "status": status,
        "message": message,
        "metadata": metadata,
        "warnings": warnings,
        "errors": errors,
    }


def _read_video_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    """
    Open the video with OpenCV and extract technical metadata.

    Returns metadata dict and any errors encountered while reading.
    """
    errors: list[str] = []
    metadata = _empty_metadata(path.name)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        errors.append("Video is unreadable or corrupted (OpenCV could not open the file).")
        return metadata, errors

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try reading the first frame to catch some corrupt files
        ret, _ = cap.read()
        if not ret:
            errors.append("Video is unreadable or corrupted (no frames could be read).")
            return metadata, errors

        duration_seconds = frame_count / fps if fps > 0 else 0.0
        file_size_mb = path.stat().st_size / (1024 * 1024)

        metadata.update(
            {
                "filename": path.name,
                "file_size_mb": round(file_size_mb, 2),
                "duration_seconds": round(duration_seconds, 2),
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "aspect_ratio": _aspect_ratio_label(width, height),
                "resolution_label": _resolution_label(width, height),
                "mime_type": _guess_mime_type(path),
            }
        )
        return metadata, errors
    finally:
        cap.release()


def validate_video(video_path: str | Path) -> dict[str, Any]:
    """
    Run all intake checks on a video file and return a structured result.

    This is the main entry point for the intake filter. Call it before
    process_video.py or any YOLO/tracking steps.
    """
    path = Path(video_path)
    warnings: list[str] = []
    errors: list[str] = []

    # 1. File must exist on disk
    if not path.is_file():
        return _build_result(
            usable=False,
            status="rejected",
            message="Video is not usable.",
            metadata=_empty_metadata(path.name),
            warnings=warnings,
            errors=[f"File not found: {path}"],
        )

    metadata = _empty_metadata(path.name)
    metadata["mime_type"] = _guess_mime_type(path)

    # 2. Extension must be .mp4
    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        errors.append(f"Invalid file extension '{extension}'. Only .mp4 is allowed for the MVP.")
        return _build_result(
            usable=False,
            status="rejected",
            message="Video is not usable.",
            metadata=metadata,
            warnings=warnings,
            errors=errors,
        )

    # 3. MIME type check (when the OS can detect it)
    if metadata["mime_type"] and metadata["mime_type"] not in ALLOWED_MIME_TYPES:
        errors.append(
            f"Invalid MIME type '{metadata['mime_type']}'. Expected video/mp4."
        )

    # 4. File size must be within the MVP limit
    file_size_mb = path.stat().st_size / (1024 * 1024)
    metadata["file_size_mb"] = round(file_size_mb, 2)
    if file_size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB MVP limit."
        )

    # 5. OpenCV readability and stream metadata
    stream_metadata, read_errors = _read_video_metadata(path)
    metadata.update(stream_metadata)
    errors.extend(read_errors)

    if not read_errors:
        fps = metadata["fps"]
        width = metadata["width"]
        height = metadata["height"]
        duration = metadata["duration_seconds"]

        # 6. FPS range
        if fps <= 0:
            errors.append("Video FPS could not be determined or is zero.")
        elif fps < MIN_FPS or fps > MAX_FPS:
            errors.append(
                f"FPS ({fps}) is outside the allowed range ({MIN_FPS}–{MAX_FPS})."
            )

        # 7. Minimum dimensions (hard reject)
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            errors.append(
                f"Resolution ({width}x{height}) is below the minimum "
                f"({MIN_WIDTH}x{MIN_HEIGHT})."
            )

        # 8. Duration limits
        if duration < MIN_DURATION_SECONDS:
            errors.append(
                f"Video duration ({duration:.1f}s) is shorter than the "
                f"{MIN_DURATION_SECONDS}s minimum."
            )
        if duration > MAX_DURATION_SECONDS:
            errors.append(
                f"Video duration ({duration:.1f}s) exceeds the "
                f"{MAX_DURATION_SECONDS}s MVP limit."
            )

        # 9. Preferred resolution warning (soft — still usable)
        if width < PREFERRED_WIDTH or height < PREFERRED_HEIGHT:
            warnings.append(
                "Video resolution is below 720p; tracking quality may be reduced."
            )

        # 10. Zero or missing frame count
        if metadata["frame_count"] <= 0:
            errors.append("Video reports zero frames and may be corrupted.")

    # Final decision
    if errors:
        return _build_result(
            usable=False,
            status="rejected",
            message="Video is not usable.",
            metadata=metadata,
            warnings=warnings,
            errors=errors,
        )

    message = "Video is usable for processing."
    if warnings:
        message = "Video is usable for processing, but review the warnings below."

    return _build_result(
        usable=True,
        status="approved",
        message=message,
        metadata=metadata,
        warnings=warnings,
        errors=errors,
    )


def save_intake_report(
    validation_result: dict[str, Any],
    original_filename: str,
    reports_dir: str | Path,
    selected_file_path: str | Path | None = None,
) -> Path:
    """
    Save a JSON intake report for an uploaded or manually selected video.

    Returns the path to the written report file.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    stem = Path(original_filename).stem
    report_path = reports_path / f"{stem}_intake_report.json"

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_filename": original_filename,
        "usable": validation_result["usable"],
        "status": validation_result["status"],
        "message": validation_result["message"],
        "metadata": validation_result["metadata"],
        "warnings": validation_result["warnings"],
        "errors": validation_result["errors"],
    }
    if selected_file_path is not None:
        report["selected_file_path"] = str(Path(selected_file_path).resolve())

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/video_intake_filter.py <path_to_video.mp4>")
        raise SystemExit(1)

    result = validate_video(sys.argv[1])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["usable"] else 1)
