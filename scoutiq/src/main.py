"""Staged sports auto-label CLI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sports_autolabel.autolabel import autolabel_frames
from sports_autolabel.dataset import build_dataset_version
from sports_autolabel.debug_render import draw_debug_frames, render_outputs
from sports_autolabel.exporters import (
    export_coco,
    export_yolo_detection,
    export_yolo_seg,
    merge_corrected_labels,
    validate_dataset,
)
from sports_autolabel.extract import extract_frames, generate_demo_frames
from sports_autolabel.review import build_review_queue, write_placeholder_review_file
from sports_autolabel.roi import set_roi_from_image
from sports_autolabel.settings import RuntimeSettings, build_paths, ensure_base_dirs
from sports_autolabel.tracking import run_simple_iou_tracking
from sports_autolabel.training import run_inference, train_yolo
from sports_autolabel.utils import configure_logging, write_default_classes_yaml, write_json


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    paths = build_paths(_project_root())
    ensure_base_dirs(paths)
    return paths


def _locateanything_executable() -> Path | None:
    env_path = os.environ.get("LOCATEANYTHING_EXE", "").strip()
    if not env_path:
        return None
    return Path(env_path).expanduser().resolve()


def cmd_init(args: argparse.Namespace) -> int:
    paths = _default_paths()
    write_default_classes_yaml(paths)
    write_placeholder_review_file(paths)
    if not paths.court_roi_path.is_file():
        # Created lazily by set-roi, but record placeholder metadata.
        write_json(
            paths.court_roi_path,
            {
                "image": "",
                "width": 0,
                "height": 0,
                "polygon": [],
                "note": "Run: python src/main.py set-roi --image data/frames/frame_000001.jpg",
            },
        )
    summary = {
        "project_root": str(paths.root),
        "cuda_available": torch.cuda.is_available(),
        "paths": paths.as_serializable(),
    }
    write_json(paths.outputs_dir / "init_summary.json", summary)
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    paths = _default_paths()
    video = Path(args.video)
    if not video.is_absolute():
        video = (paths.root / video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    summary = extract_frames(video, paths.frames_dir, args.fps)
    print(yaml.safe_dump(summary.to_dict(), sort_keys=False))
    return 0


def cmd_set_roi(args: argparse.Namespace) -> int:
    paths = _default_paths()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (paths.root / image_path).resolve()
    payload = set_roi_from_image(paths, image_path)
    print(yaml.safe_dump(payload, sort_keys=False))
    return 0


def cmd_autolabel(args: argparse.Namespace) -> int:
    paths = _default_paths()
    source = args.source.lower().strip()
    if source not in {"locateanything", "mock"}:
        raise ValueError("source must be 'locateanything' or 'mock'")
    summary = autolabel_frames(
        paths=paths,
        source=source,
        locateanything_executable=_locateanything_executable(),
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    paths = _default_paths()
    summary = draw_debug_frames(paths)
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_export_yolo_det(args: argparse.Namespace) -> int:
    paths = _default_paths()
    summary = export_yolo_detection(paths)
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_export_yolo_seg(args: argparse.Namespace) -> int:
    paths = _default_paths()
    sam2_available = False
    if args.sam2 is not None:
        sam2_available = bool(args.sam2)
    summary = export_yolo_seg(paths, sam2_available=sam2_available)
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_export_coco(args: argparse.Namespace) -> int:
    paths = _default_paths()
    output = Path(args.output) if args.output else None
    if output and not output.is_absolute():
        output = (paths.root / output).resolve()
    summary = export_coco(paths, output_path=output)
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


def cmd_review_select(args: argparse.Namespace) -> int:
    paths = _default_paths()
    payload = build_review_queue(paths, percent=int(args.percent))
    print(yaml.safe_dump(payload, sort_keys=False))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    paths = _default_paths()
    payload = write_placeholder_review_file(paths)
    payload["note"] = (
        "Manual review UI not yet connected. Save corrections to "
        f"{paths.corrected_predictions_dir / 'corrected_labels.json'}"
    )
    print(yaml.safe_dump(payload, sort_keys=False))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    paths = _default_paths()
    merge_corrected_labels(paths)
    result = build_dataset_version(paths, version=str(args.version))
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    paths = _default_paths()
    settings = RuntimeSettings()
    result = train_yolo(
        paths=paths,
        dataset_version=str(args.version),
        epochs=int(args.epochs),
        segmentation=bool(args.segmentation),
        settings=settings,
    )
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    paths = _default_paths()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (paths.root / weights).resolve()
    video = Path(args.video)
    if not video.is_absolute():
        video = (paths.root / video).resolve()
    result = run_inference(paths, weights=weights, video=video)
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    paths = _default_paths()
    result = run_simple_iou_tracking(paths)
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    paths = _default_paths()
    result = render_outputs(paths, fps=float(args.fps))
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    paths = _default_paths()
    logger = configure_logging()
    logger.warning("Running staged full pipeline with graceful fallbacks.")
    cmd_init(args)
    cmd_extract(args)
    cmd_autolabel(argparse.Namespace(source="locateanything"))
    cmd_export_yolo_det(args)
    cmd_debug(args)
    cmd_review_select(argparse.Namespace(percent=args.review_percent))
    cmd_render(argparse.Namespace(fps=8.0))
    return 0


def cmd_demo_mock(args: argparse.Namespace) -> int:
    paths = _default_paths()
    logger = configure_logging()
    cmd_init(args)

    # 1) create fake frames or use existing.
    existing = list(paths.frames_dir.glob("frame_*.jpg"))
    if not existing:
        generate_demo_frames(paths.frames_dir, frame_count=12)
        logger.info("Generated synthetic demo frames.")

    # 2) mock detections.
    cmd_autolabel(argparse.Namespace(source="mock"))

    # 3) YOLO detection labels.
    cmd_export_yolo_det(args)

    # 4) Debug frames.
    cmd_debug(args)

    # 5) Review queue.
    cmd_review_select(argparse.Namespace(percent=20))

    # 6) Build tiny dataset.
    cmd_export_yolo_seg(argparse.Namespace(sam2=False))
    cmd_build_dataset(argparse.Namespace(version="v001"))

    # 7) Render short demo video.
    cmd_render(argparse.Namespace(fps=8.0))
    print("demo-mock completed successfully.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = _default_paths()
    dataset_dir = paths.datasets_root / (
        args.version if str(args.version).startswith("basketball_yolo_seg_") else f"basketball_yolo_seg_{args.version}"
    )
    result = validate_dataset(
        dataset_dir=dataset_dir,
        classes_yaml_path=paths.classes_yaml_path,
        segmentation=bool(args.segmentation),
    )
    print(yaml.safe_dump(result, sort_keys=False))
    return 0 if result["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sports auto-labeling staged CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--video", required=True)
    p_extract.add_argument("--fps", type=float, default=2.0)

    p_set_roi = sub.add_parser("set-roi")
    p_set_roi.add_argument("--image", required=True)

    p_autolabel = sub.add_parser("autolabel")
    p_autolabel.add_argument("--source", required=True, choices=["locateanything", "mock"])

    sub.add_parser("debug")
    sub.add_parser("export-yolo-det")
    p_seg = sub.add_parser("export-yolo-seg")
    p_seg.add_argument("--sam2", action="store_true")
    p_coco = sub.add_parser("export-coco")
    p_coco.add_argument("--output")

    p_review_select = sub.add_parser("review-select")
    p_review_select.add_argument("--percent", type=int, default=20)
    sub.add_parser("review")

    p_build = sub.add_parser("build-dataset")
    p_build.add_argument("--version", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--version", required=True)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--segmentation", action="store_true")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--weights", required=True)
    p_infer.add_argument("--video", required=True)

    sub.add_parser("track")
    p_render = sub.add_parser("render")
    p_render.add_argument("--fps", type=float, default=8.0)

    p_full = sub.add_parser("full")
    p_full.add_argument("--video", required=True)
    p_full.add_argument("--fps", type=float, default=2.0)
    p_full.add_argument("--review_percent", type=int, default=20)

    sub.add_parser("demo-mock")

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--version", required=True)
    p_validate.add_argument("--segmentation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "init": cmd_init,
        "extract": cmd_extract,
        "set-roi": cmd_set_roi,
        "autolabel": cmd_autolabel,
        "debug": cmd_debug,
        "export-yolo-det": cmd_export_yolo_det,
        "export-yolo-seg": cmd_export_yolo_seg,
        "export-coco": cmd_export_coco,
        "review-select": cmd_review_select,
        "review": cmd_review,
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "infer": cmd_infer,
        "track": cmd_track,
        "render": cmd_render,
        "full": cmd_full,
        "demo-mock": cmd_demo_mock,
        "validate": cmd_validate,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
