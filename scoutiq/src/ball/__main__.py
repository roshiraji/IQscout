"""CLI for dedicated ball detection dataset + training.

Usage (from scoutiq/):
  python -m src.ball autolabel --manifest data/frames/manifest.csv [--max-frames N]
  python -m src.ball build-dataset
  python -m src.ball train --config configs/showcase_ball_train.yaml
  python -m src.ball smoke   # tiny autolabel + 1-epoch train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.ml_warnings import configure_ml_runtime

configure_ml_runtime()

SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ball")


def _resolve(path_str: str, base: Path = SCOUTIQ_ROOT) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def cmd_autolabel(args: argparse.Namespace) -> int:
    from src.ball.autolabel import autolabel_ball_manifest

    manifest = _resolve(args.manifest)
    output = _resolve(args.output)
    frames_root = _resolve(args.frames_root) if args.frames_root else SCOUTIQ_ROOT / "data" / "frames"
    stats = autolabel_ball_manifest(
        manifest_path=manifest,
        output_root=output,
        frames_root=frames_root,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        use_coco_ball=not args.no_coco,
        skip_existing=not args.force,
        use_sam_refine=getattr(args, "sam_refine", False),
        mine_hard_negatives=not getattr(args, "no_hard_negatives", False),
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_hoop_autolabel(args: argparse.Namespace) -> int:
    from src.ball.hoop import autolabel_hoop_manifest

    stats = autolabel_hoop_manifest(
        manifest_path=_resolve(args.manifest),
        output_root=_resolve(args.output),
        frames_root=_resolve(args.frames_root),
        max_frames=args.max_frames,
        skip_existing=not args.force,
        stride=args.stride,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_hoop_build_dataset(args: argparse.Namespace) -> int:
    from src.ball.hoop import build_hoop_yolo_dataset

    data_yaml = build_hoop_yolo_dataset(
        autolabel_root=_resolve(args.autolabel),
        dataset_root=_resolve(args.dataset),
        frames_root=_resolve(args.frames_root),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Dataset yaml: {data_yaml}")
    return 0


def cmd_hoop_train(args: argparse.Namespace) -> int:
    from src.ball.train import run_ball_training_from_config

    cfg = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8")) or {}
    if args.smoke:
        cfg.setdefault("training", {})["epochs"] = 1
        cfg.setdefault("training", {})["max_rounds"] = 1
    weights = run_ball_training_from_config(cfg, SCOUTIQ_ROOT)
    print(f"Hoop weights: {weights}")
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from src.ball.autolabel import build_ball_yolo_dataset

    data_yaml = build_ball_yolo_dataset(
        autolabel_root=_resolve(args.autolabel),
        dataset_root=_resolve(args.dataset),
        frames_root=_resolve(args.frames_root),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Dataset yaml: {data_yaml}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from src.ball.train import run_ball_training_from_config

    cfg = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8")) or {}
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.smoke:
        cfg.setdefault("training", {})["epochs"] = 1
        cfg.setdefault("training", {})["max_rounds"] = 1
    weights = run_ball_training_from_config(cfg, SCOUTIQ_ROOT)
    print(f"Ball weights: {weights}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end tiny path: label a few frames → dataset → 1-epoch train."""
    args.manifest = args.manifest or "data/frames/manifest.csv"
    args.output = "data/auto_labels_ball"
    args.frames_root = "data/frames"
    args.max_frames = args.max_frames or 24
    args.min_confidence = 0.28
    args.no_coco = False
    args.force = False
    cmd_autolabel(args)

    args.autolabel = "data/auto_labels_ball"
    args.dataset = "data/datasets/wnba_ball_det"
    args.val_ratio = 0.25
    args.seed = 42
    cmd_build_dataset(args)

    args.config = "configs/showcase_ball_train.yaml"
    args.epochs = 1
    args.smoke = True
    return cmd_train(args)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ball")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_al = sub.add_parser("autolabel", help="HSV(+COCO) ball proposals → labels")
    p_al.add_argument("--manifest", default="data/frames/manifest.csv")
    p_al.add_argument("--output", default="data/auto_labels_ball")
    p_al.add_argument("--frames-root", default="data/frames")
    p_al.add_argument("--max-frames", type=int, default=None)
    p_al.add_argument("--min-confidence", type=float, default=0.32)
    p_al.add_argument("--no-coco", action="store_true")
    p_al.add_argument("--sam-refine", action="store_true", help="Tighten ball boxes with SAM")
    p_al.add_argument("--no-hard-negatives", action="store_true")
    p_al.add_argument("--force", action="store_true")
    p_al.set_defaults(func=cmd_autolabel)

    p_ha = sub.add_parser("hoop-autolabel", help="GDINO hoop boxes → labels")
    p_ha.add_argument("--manifest", default="data/frames/manifest.csv")
    p_ha.add_argument("--output", default="data/auto_labels_hoop")
    p_ha.add_argument("--frames-root", default="data/frames")
    p_ha.add_argument("--max-frames", type=int, default=None)
    p_ha.add_argument("--stride", type=int, default=3)
    p_ha.add_argument("--force", action="store_true")
    p_ha.set_defaults(func=cmd_hoop_autolabel)

    p_hd = sub.add_parser("hoop-build-dataset", help="Scene-segment hoop split")
    p_hd.add_argument("--autolabel", default="data/auto_labels_hoop")
    p_hd.add_argument("--dataset", default="data/datasets/wnba_hoop_det")
    p_hd.add_argument("--frames-root", default="data/frames")
    p_hd.add_argument("--val-ratio", type=float, default=0.15)
    p_hd.add_argument("--seed", type=int, default=42)
    p_hd.set_defaults(func=cmd_hoop_build_dataset)

    p_ht = sub.add_parser("hoop-train", help="Train best_hoop_det.pt")
    p_ht.add_argument("--config", default="configs/showcase_hoop_train.yaml")
    p_ht.add_argument("--smoke", action="store_true")
    p_ht.set_defaults(func=cmd_hoop_train)

    p_ds = sub.add_parser("build-dataset", help="Scene-segment YOLO detect split")
    p_ds.add_argument("--autolabel", default="data/auto_labels_ball")
    p_ds.add_argument("--dataset", default="data/datasets/wnba_ball_det")
    p_ds.add_argument("--frames-root", default="data/frames")
    p_ds.add_argument("--val-ratio", type=float, default=0.15)
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.set_defaults(func=cmd_build_dataset)

    p_tr = sub.add_parser("train", help="Train best_ball_det.pt")
    p_tr.add_argument("--config", default="configs/showcase_ball_train.yaml")
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--smoke", action="store_true")
    p_tr.set_defaults(func=cmd_train)

    p_sm = sub.add_parser("smoke", help="Tiny label+train path")
    p_sm.add_argument("--manifest", default="data/frames/manifest.csv")
    p_sm.add_argument("--max-frames", type=int, default=24)
    p_sm.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
