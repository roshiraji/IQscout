"""Smoke-test SAM install and checkpoint for ScoutIQ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCOUTIQ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOUTIQ_ROOT))

from src.segmentation import (  # noqa: E402
    get_sam_checkpoint_path,
    load_sam_predictor,
    predict_mask_from_box,
)


def _default_image() -> Path | None:
    candidates = [
        SCOUTIQ_ROOT / "outputs" / "player_snapshots",
        SCOUTIQ_ROOT / "data" / "input_videos",
        SCOUTIQ_ROOT.parent / "assets",
    ]
    for folder in candidates:
        if not folder.is_dir():
            continue
        for pattern in ("*_snapshot_labeled.jpg", "*.jpg", "*.png"):
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Segment Anything is ready.")
    parser.add_argument("--image", type=Path, help="BGR image path for a box-prompt test.")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "--output",
        type=Path,
        default=SCOUTIQ_ROOT / "outputs" / "sam_test_mask.png",
        help="Where to save the overlay preview.",
    )
    args = parser.parse_args()

    checkpoint = get_sam_checkpoint_path(args.model_type)
    if not checkpoint.is_file():
        print(f"Missing checkpoint: {checkpoint}")
        print(f"Run: scoutiq\\scripts\\download_sam_checkpoint.bat {args.model_type}")
        return 1

    print(f"Checkpoint: {checkpoint} ({checkpoint.stat().st_size // (1024 * 1024)} MB)")
    print("Loading SAM predictor...")
    predictor = load_sam_predictor(args.model_type)
    print(f"Predictor ready (device: {predictor.model.device})")

    image_path = args.image or _default_image()
    if image_path is None or not image_path.is_file():
        print("No test image found. Pass --image path/to/frame.jpg after download succeeds.")
        return 0

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    h, w = image.shape[:2]
    box = [w * 0.35, h * 0.2, w * 0.65, h * 0.85]
    mask = predict_mask_from_box(image, box, model_type=args.model_type)
    pixels = int(mask.sum())
    print(f"Image: {image_path.name} ({w}x{h})")
    print(f"Mask pixels: {pixels}")

    overlay = image.copy()
    overlay[mask] = (0, 255, 0)
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), blended)
    print(f"Saved preview: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
