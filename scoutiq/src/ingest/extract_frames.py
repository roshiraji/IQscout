"""Extract video frames for annotation or debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def extract_frames(video: Path, output_dir: Path, fps_sample: float | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = source_fps if source_fps > 0 else 30.0
    every_frame = fps_sample is None
    sample_every = 1 if every_frame else max(int(round(source_fps / max(fps_sample, 0.1))), 1)

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            out = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
        idx += 1
    cap.release()

    summary = {
        "video": str(video.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_fps": round(source_fps, 3),
        "sample_fps": "all" if every_frame else fps_sample,
        "frame_count": frame_count,
        "frames_saved": saved,
    }
    (output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video as JPG images.")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--output", required=True, help="Output directory for JPG frames")
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Sampling fps (default: 2.0). Ignored when --all-frames is set.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Export every video frame (best for manual annotation).",
    )
    args = parser.parse_args()

    fps_sample = None if args.all_frames else args.fps
    summary = extract_frames(Path(args.video), Path(args.output), fps_sample)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
