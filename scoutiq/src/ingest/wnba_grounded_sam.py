"""Bridge ScoutIQ WNBA video frames to the Grounded SAM auto-annotation pipeline.

Extract frames from an MP4 (or reuse an existing frame folder), then run
Grounding DINO + SAM to annotate players, referees, and the ball.

Examples (from scoutiq/):

  python -m src.ingest.wnba_grounded_sam ^
    --video "data/input_videos/new tune.mp4" ^
  --clip-name new_tune --fps 2 --limit 5 --overwrite

  python -m src.ingest.wnba_grounded_sam ^
    --frames-dir "data/annotation_frames/new tune" ^
    --clip-name new_tune --limit 5 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCOUTIQ_ROOT = REPO_ROOT / "scoutiq"
GROUNDED_SAM_ROOT = REPO_ROOT / "grounded_sam_pipeline"
WNBA_CONFIG = GROUNDED_SAM_ROOT / "config" / "auto_annotation_wnba.yaml"

if str(GROUNDED_SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDED_SAM_ROOT))

from auto_annotation.config import AppConfig, load_config, validate_config  # noqa: E402
from auto_annotation.device import select_device, validate_environment  # noqa: E402
from auto_annotation.model_manager import ensure_checkpoints, load_models  # noqa: E402
from auto_annotation.pipeline import run_pipeline  # noqa: E402
from auto_annotation.utils import setup_logging  # noqa: E402

from src.ingest.extract_frames import extract_frames  # noqa: E402


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "clip"


def set_run_paths(cfg: AppConfig, images_dir: Path, output_root: Path) -> AppConfig:
    cfg.images_directory = images_dir.resolve()
    cfg.output_root = output_root.resolve()
    cfg.previews_directory = output_root / "previews"
    cfg.masks_directory = output_root / "masks"
    cfg.annotations_directory = output_root / "annotations"
    cfg.reports_directory = output_root / "reports"
    return cfg


def resolve_frames_dir(
    *,
    video: Path | None,
    frames_dir: Path | None,
    clip_name: str,
    fps_sample: float | None,
    skip_extract: bool,
) -> tuple[Path, dict | None]:
    if frames_dir is not None:
        frames_dir = frames_dir.resolve()
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        summary_path = frames_dir / "extract_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        return frames_dir, summary

    if video is None:
        raise ValueError("Provide --video or --frames-dir")

    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    frames_dir = (SCOUTIQ_ROOT / "data" / "annotation_frames" / clip_name).resolve()
    summary = None
    if skip_extract and frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
        summary_path = frames_dir / "extract_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = extract_frames(video, frames_dir, fps_sample)
    return frames_dir, summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract WNBA video frames and auto-annotate players with Grounded SAM",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="WNBA MP4 under scoutiq/data/input_videos/")
    src.add_argument("--frames-dir", type=Path, help="Existing extracted frames folder")

    p.add_argument("--clip-name", help="Slug for output folders (default: video/frames folder name)")
    p.add_argument(
        "--output-root",
        type=Path,
        help="Override annotation output root (default: data/annotations/wnba_grounded_sam/<clip>)",
    )
    p.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate when extracting (default: 2)")
    p.add_argument("--all-frames", action="store_true", help="Extract every frame from video")
    p.add_argument("--skip-extract", action="store_true", help="Reuse frames if folder already exists")
    p.add_argument("--limit", type=int, help="Annotate at most N frames (good for testing)")
    p.add_argument("--single-frame", type=Path, help="Annotate one frame file name or path")
    p.add_argument("--overwrite", action="store_true", help="Re-annotate all frames")
    p.add_argument("--resume", action="store_true", help="Skip frames already in progress file")
    p.add_argument("--mock", action="store_true", help="Mock inference (pipeline structure only)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--box-threshold", type=float)
    p.add_argument("--download-weights", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    clip_name = args.clip_name
    if not clip_name:
        if args.video is not None:
            clip_name = slugify(args.video.stem)
        else:
            clip_name = slugify(args.frames_dir.name)

    fps_sample = None if args.all_frames else args.fps
    frames_dir, extract_summary = resolve_frames_dir(
        video=args.video,
        frames_dir=args.frames_dir,
        clip_name=clip_name,
        fps_sample=fps_sample,
        skip_extract=args.skip_extract,
    )

    output_root = args.output_root or (SCOUTIQ_ROOT / "data" / "annotations" / "wnba_grounded_sam" / clip_name)
    output_root = output_root.resolve()

    cfg = load_config(WNBA_CONFIG, GROUNDED_SAM_ROOT)
    cfg = set_run_paths(cfg, frames_dir, output_root)
    if args.device:
        cfg.device = args.device
    if args.box_threshold is not None:
        cfg.box_threshold = args.box_threshold

    log_file = cfg.reports_directory / "pipeline.log"
    setup_logging(cfg.log_level, log_file)

    single_image = None
    if args.single_frame:
        candidate = Path(args.single_frame)
        single_image = candidate if candidate.is_file() else frames_dir / candidate.name
        if not single_image.is_file():
            raise FileNotFoundError(f"Frame not found: {args.single_frame}")

    require_models = not args.mock
    if args.download_weights:
        ensure_checkpoints(cfg, allow_download=True)

    errors = validate_config(cfg, require_models=require_models, require_input=single_image is None)
    if errors and not args.mock:
        print("Configuration errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    env_report = validate_environment(cfg, require_models=require_models) if not args.mock else {"mock": True}

    print("WNBA Grounded SAM annotation")
    print(f"  Clip:        {clip_name}")
    print(f"  Frames:      {frames_dir}")
    print(f"  Output:      {output_root}")
    if extract_summary:
        print(f"  Frames saved: {extract_summary.get('frames_saved', '?')}")

    models = None
    if not args.mock:
        dev = select_device(cfg.device)
        models = load_models(cfg, dev.device)

    summary_path = run_pipeline(
        cfg,
        models,
        limit=args.limit,
        single_image=single_image,
        resume=args.resume,
        overwrite=args.overwrite,
        mock=args.mock,
        env_report=env_report,
    )

    bridge_meta = {
        "clip_name": clip_name,
        "frames_directory": str(frames_dir),
        "output_root": str(output_root),
        "extract_summary": extract_summary,
        "wnba_training_config": "configs/wnba_players.yaml",
        "classes": cfg.classes,
        "run_summary": str(summary_path),
        "yolo_labels": str(cfg.yolo_seg_directory / "labels"),
        "previews": str(cfg.previews_directory),
    }
    meta_path = output_root / "wnba_bridge_summary.json"
    meta_path.write_text(json.dumps(bridge_meta, indent=2), encoding="utf-8")

    print(f"\nDone. Summary: {summary_path}")
    print(f"Previews:     {cfg.previews_directory}")
    print(f"YOLO labels:  {cfg.yolo_seg_directory / 'labels'}")
    print(f"Bridge meta:  {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
