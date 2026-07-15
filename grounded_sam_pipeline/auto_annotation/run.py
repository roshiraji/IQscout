"""CLI entry point for the auto-annotation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_annotation.config import PROJECT_ROOT, apply_cli_overrides, load_config, validate_config
from auto_annotation.device import select_device, validate_environment
from auto_annotation.model_manager import ensure_checkpoints, load_models
from auto_annotation.pipeline import run_pipeline
from auto_annotation.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grounding DINO + SAM automated instance-segmentation annotation pipeline",
    )
    p.add_argument("--config", default="config/auto_annotation.yaml", help="Path to YAML config")
    p.add_argument("--input", dest="input_dir", help="Override input images directory")
    p.add_argument("--classes", help="Comma-separated class names")
    p.add_argument("--box-threshold", type=float)
    p.add_argument("--text-threshold", type=float)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Compute device")
    p.add_argument("--save-previews", action="store_true", default=None)
    p.add_argument("--no-previews", action="store_false", dest="save_previews")
    p.add_argument("--validate-only", action="store_true", help="Check environment and exit")
    p.add_argument("--single-image", type=Path, help="Process one image only")
    p.add_argument("--limit", type=int, help="Process at most N images")
    p.add_argument("--resume", action="store_true", help="Skip already-completed images")
    p.add_argument("--overwrite", action="store_true", help="Reprocess all images")
    p.add_argument("--download-weights", action="store_true", help="Download missing checkpoints")
    p.add_argument("--mock", action="store_true", help="Mock inference (no models, for testing)")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    cfg = load_config(cfg_path, PROJECT_ROOT)
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    cfg = apply_cli_overrides(
        cfg,
        input_dir=args.input_dir,
        classes=classes,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        save_previews=args.save_previews,
    )
    if args.log_level:
        cfg.log_level = args.log_level

    log_file = cfg.reports_directory / "pipeline.log"
    setup_logging(cfg.log_level, log_file)

    require_models = not args.mock and not args.validate_only
    require_input = args.single_image is None

    if args.download_weights:
        ensure_checkpoints(cfg, allow_download=True)

    errors = validate_config(cfg, require_models=require_models, require_input=require_input)
    if errors and not args.mock:
        print("Configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    env_report = validate_environment(
        cfg, require_models=require_models, require_input=require_input,
    ) if not args.mock else {"mock": True}

    if args.validate_only:
        print("Validation passed.")
        print(f"  Device: {env_report.get('device', 'mock')}")
        print(f"  Input:  {cfg.images_directory}")
        print(f"  Output: {cfg.output_root}")
        return 0

    models = None
    if not args.mock:
        ensure_checkpoints(cfg, allow_download=args.download_weights)
        dev = select_device(cfg.device)
        models = load_models(cfg, dev.device)

    summary_path = run_pipeline(
        cfg,
        models,
        limit=args.limit,
        single_image=args.single_image,
        resume=args.resume,
        overwrite=args.overwrite,
        mock=args.mock,
        env_report=env_report,
    )
    print(f"\nDone. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
