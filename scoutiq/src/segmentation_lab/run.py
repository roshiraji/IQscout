"""
Segmentation lab CLI — evaluate SAM on the fine-tune clip in isolation.

Example (from scoutiq/):
  python -m src.segmentation_lab.run
  python -m src.segmentation_lab.run --video "data/input_videos/new tune.mp4"
  python -m src.segmentation_lab.run --use-snapshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.segmentation_lab.config import SegmentationLabConfig, load_config
from src.segmentation_lab.report import build_summary, write_html_gallery, write_json_report
from src.segmentation_lab.sam_eval import segment_box_prompts
from src.segmentation_lab.visualize import build_comparison_panel, save_mask_png
from src.segmentation_lab.yolo_prompts import (
    detect_people,
    iter_sample_frames,
    load_snapshot_boxes,
    load_yolo,
    read_frame,
    resolve_device,
)


def _default_snapshot_json(video_path: Path, scoutiq_root: Path) -> Path:
    return scoutiq_root / "outputs" / "player_snapshots" / f"{video_path.stem}_snapshot_players.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Segment Anything on the fine-tune clip (YOLO boxes as prompts)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to segmentation_lab.yaml (default: configs/segmentation_lab.yaml)",
    )
    parser.add_argument("--video", type=Path, help="Override fine_tune_clip from config.")
    parser.add_argument("--output", type=Path, help="Override output_dir from config.")
    parser.add_argument("--sam-model", choices=["vit_b", "vit_l", "vit_h"], help="SAM checkpoint variant.")
    parser.add_argument("--yolo-model", help="YOLO weights for box prompts.")
    parser.add_argument("--frame-stride", type=int, help="Sample every Nth frame.")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to evaluate.")
    parser.add_argument(
        "--use-snapshot",
        action="store_true",
        help="Only run on the snapshot frame + boxes from outputs/player_snapshots/*.json",
    )
    parser.add_argument("--snapshot-json", type=Path, help="Explicit snapshot players JSON path.")
    parser.add_argument("--open-report", action="store_true", help="Open index.html when finished (Windows).")
    return parser.parse_args(argv)


def _apply_overrides(config: SegmentationLabConfig, args: argparse.Namespace) -> SegmentationLabConfig:
    video = args.video or config.fine_tune_clip
    output = args.output or config.output_dir
    return SegmentationLabConfig(
        fine_tune_clip=video.resolve() if not video.is_absolute() else video,
        yolo_model=args.yolo_model or config.yolo_model,
        yolo_imgsz=config.yolo_imgsz,
        yolo_conf=config.yolo_conf,
        sam_model=args.sam_model or config.sam_model,
        frame_stride=args.frame_stride or config.frame_stride,
        max_frames=args.max_frames or config.max_frames,
        output_dir=output.resolve() if not output.is_absolute() else output,
        use_snapshot=args.use_snapshot or config.use_snapshot,
        snapshot_json=args.snapshot_json or config.snapshot_json,
    )


def run_lab(config: SegmentationLabConfig) -> Path:
    if not config.fine_tune_clip.is_file():
        raise FileNotFoundError(
            f"Fine-tune clip not found: {config.fine_tune_clip}\n"
            "Place your MP4 in data/input_videos/ and update configs/segmentation_lab.yaml"
        )

    run_dir = config.output_dir / config.fine_tune_clip.stem
    frames_dir = run_dir / "frames"
    masks_dir = run_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    yolo = load_yolo(config.yolo_model)
    frame_reports: list[dict] = []

    def process_frame(frame_number: int, timestamp: float, frame_bgr, boxes) -> None:
        if not boxes:
            frame_reports.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 3),
                    "player_count": 0,
                    "masks": [],
                    "comparison_image": "",
                }
            )
            return

        segmented = segment_box_prompts(
            frame_bgr,
            boxes,
            model_type=config.sam_model,  # type: ignore[arg-type]
        )
        mask_arrays = [mask for _, mask in segmented]
        results = [result for result, _ in segmented]

        frame_tag = f"frame_{frame_number:06d}"
        for result, mask in segmented:
            mask_path = masks_dir / f"{frame_tag}_{result.label}.png"
            save_mask_png(mask_path, mask)
            result.mask_path = str(mask_path.relative_to(run_dir))

        comparison = build_comparison_panel(frame_bgr, boxes, mask_arrays, results)
        comparison_path = frames_dir / f"{frame_tag}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)

        frame_reports.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": round(timestamp, 3),
                "player_count": len(boxes),
                "comparison_image": str(comparison_path.relative_to(run_dir)),
                "masks": [result.to_dict() for result in results],
            }
        )
        print(
            f"  frame {frame_number}: {len(boxes)} players, "
            f"avg SAM {sum(r.sam_score for r in results) / len(results):.2f}"
        )

    print(f"Segmentation lab — {config.fine_tune_clip.name}")
    print(f"YOLO: {config.yolo_model} ({device}) · SAM: {config.sam_model}")
    print(f"Output: {run_dir}")

    if config.use_snapshot:
        snapshot_json = config.snapshot_json or _default_snapshot_json(
            config.fine_tune_clip, config.scoutiq_root
        )
        if not snapshot_json.is_file():
            raise FileNotFoundError(
                f"Snapshot JSON not found: {snapshot_json}\n"
                "Run Analyze in the dashboard first, or pass --snapshot-json"
            )
        frame_number, timestamp, boxes = load_snapshot_boxes(snapshot_json)
        frame_bgr = read_frame(config.fine_tune_clip, frame_number)
        print(f"Snapshot mode: frame {frame_number} from {snapshot_json.name}")
        process_frame(frame_number, timestamp, frame_bgr, boxes)
    else:
        print(f"Sampling every {config.frame_stride} frames (max {config.max_frames})...")
        for frame_number, timestamp, frame_bgr in iter_sample_frames(
            config.fine_tune_clip,
            config.frame_stride,
            config.max_frames,
        ):
            boxes = detect_people(
                yolo,
                frame_bgr,
                device=device,
                conf=config.yolo_conf,
                imgsz=config.yolo_imgsz,
            )
            process_frame(frame_number, timestamp, frame_bgr, boxes)

    summary = build_summary(
        video_path=config.fine_tune_clip,
        sam_model=config.sam_model,
        yolo_model=config.yolo_model,
        frame_reports=frame_reports,
    )
    write_json_report(run_dir / "report.json", summary)
    write_html_gallery(run_dir / "index.html", summary, run_dir)

    agg = summary["aggregate"]
    print("Done.")
    print(f"  Avg SAM score: {agg['avg_sam_score']}")
    print(f"  Avg fill ratio: {agg['avg_fill_ratio']}")
    print(f"  Report: {run_dir / 'index.html'}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        config = _apply_overrides(config, args)
        if not config.fine_tune_clip.is_absolute():
            config = SegmentationLabConfig(
                fine_tune_clip=(config.scoutiq_root / config.fine_tune_clip).resolve(),
                yolo_model=config.yolo_model,
                yolo_imgsz=config.yolo_imgsz,
                yolo_conf=config.yolo_conf,
                sam_model=config.sam_model,
                frame_stride=config.frame_stride,
                max_frames=config.max_frames,
                output_dir=(
                    config.output_dir
                    if config.output_dir.is_absolute()
                    else (config.scoutiq_root / config.output_dir).resolve()
                ),
                use_snapshot=config.use_snapshot,
                snapshot_json=(
                    (config.scoutiq_root / config.snapshot_json).resolve()
                    if config.snapshot_json and not config.snapshot_json.is_absolute()
                    else config.snapshot_json
                ),
            )
        run_dir = run_lab(config)
        if args.open_report:
            import os
            import subprocess

            index = run_dir / "index.html"
            if sys.platform == "win32":
                os.startfile(index)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(index)], check=False)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
