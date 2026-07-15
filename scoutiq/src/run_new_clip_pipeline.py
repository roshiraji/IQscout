"""
Run the tracking overlay pipeline on the fine-tune clip.

Usage (from scoutiq/):
    python src/run_new_clip_pipeline.py
    python src/run_new_clip_pipeline.py --no-sam2
    python src/run_new_clip_pipeline.py --video "data/input_videos/other.mp4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src directory importable.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_overlay_pipeline import VIDEO_IN, VIDEO_OUT, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Basketball tracking overlay pipeline")
    parser.add_argument("--video", type=Path, default=None,
                        help="Override input video (default: data/input_videos/new clip for tunning.mp4)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output MP4 path")
    parser.add_argument("--no-sam2", action="store_true", help="Skip SAM2 masks")
    parser.add_argument("--no-locateanything", action="store_true",
                        help="Skip LocateAnything (use YOLO11-seg only)")
    args = parser.parse_args()

    video_path = args.video or VIDEO_IN
    output_path = args.output or VIDEO_OUT

    try:
        out = run_pipeline(
            video_path=video_path,
            output_path=output_path,
            use_sam2=not args.no_sam2,
            use_locateanything=not args.no_locateanything,
        )
        print(f"\nDone. Final video saved to:\n  {out}")
        return 0
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
