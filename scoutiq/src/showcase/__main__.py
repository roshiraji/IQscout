"""CLI for showcase pipeline stages.

Usage (from scoutiq/):
  python -m src.showcase init
  python -m src.showcase ingest --config configs/showcase_ingest.yaml
  python -m src.showcase autolabel --config configs/showcase_ingest.yaml
  python -m src.showcase build-dataset --config configs/showcase_ingest.yaml
  python -m src.showcase train --config configs/showcase_train.yaml
  python -m src.showcase run --config configs/showcase_player_seg.yaml
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


def _resolve(path_str: str, base: Path = SCOUTIQ_ROOT) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def cmd_init(_args: argparse.Namespace) -> int:
    from src.showcase.exclusions import write_default_exclusion_template

    data = SCOUTIQ_ROOT / "data"
    for sub in (
        "footage/train",
        "footage/val",
        "footage/holdout",
        "frames",
        "exclusions",
        "datasets/nba_player_seg",
    ):
        (data / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("auto_labels", "training_loop", "showcase"):
        (SCOUTIQ_ROOT / "outputs" / sub).mkdir(parents=True, exist_ok=True)

    write_default_exclusion_template(data / "exclusions" / "default.json")
    readme = data / "README_SHOWCASE.md"
    if not readme.is_file():
        readme.write_text(_SHOWCASE_DATA_README, encoding="utf-8")
    print(f"Scaffolded data/ and outputs/ under {SCOUTIQ_ROOT}")
    print(f"Edit exclusions: {data / 'exclusions' / 'default.json'}")
    print(f"See: {readme}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from src.showcase.ingest import ingest_from_config

    cfg = _load_cfg(_resolve(args.config))
    manifest = ingest_from_config(cfg, SCOUTIQ_ROOT)
    print(f"Manifest: {manifest}")
    return 0


def cmd_autolabel(args: argparse.Namespace) -> int:
    from src.fusion.grounding_dino_runner import GroundingDinoSettings
    from src.showcase.autolabel import AutolabelSettings, autolabel_manifest

    cfg = _load_cfg(_resolve(args.config))
    data_root = _resolve(cfg.get("data_root", "data"))
    manifest = data_root / "frames" / "manifest.csv"
    if args.manifest:
        manifest = _resolve(args.manifest)
    output_root = _resolve(cfg.get("autolabel", {}).get("output_root", "outputs/auto_labels"))
    exclusions_dir = data_root / "exclusions"

    al = cfg.get("autolabel") or {}
    settings = AutolabelSettings(
        box_threshold=float(al.get("box_threshold", 0.30)),
        text_threshold=float(al.get("text_threshold", 0.25)),
        sam2_weights=str(al.get("sam2_weights", "sam2.1_b.pt")),
        min_confidence=float(al.get("min_confidence", 0.25)),
        write_qa_previews=bool(al.get("write_qa_previews", True)),
        max_frames=args.max_frames or al.get("max_frames"),
        gdino_device=al.get("gdino_device"),
        sam_device=al.get("sam_device"),
        skip_existing=bool(al.get("skip_existing", True)),
    )
    repo = SCOUTIQ_ROOT.parent
    gd = GroundingDinoSettings(
        config_path=repo / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path=repo / "weights/groundingdino_swint_ogc.pth",
        box_threshold=settings.box_threshold,
        text_threshold=settings.text_threshold,
    )
    stats = autolabel_manifest(
        manifest_path=manifest,
        output_root=output_root,
        exclusions_dir=exclusions_dir,
        gdino_settings=gd,
        settings=settings,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from src.showcase.autolabel import build_yolo_seg_dataset

    cfg = _load_cfg(_resolve(args.config))
    data_root = _resolve(cfg.get("data_root", "data"))
    autolabel_root = _resolve(cfg.get("autolabel", {}).get("output_root", "outputs/auto_labels"))
    dataset_root = data_root / "datasets" / "nba_player_seg"

    holdout: set[str] = set()
    gate = data_root / "frames" / "holdout_gate.json"
    if gate.is_file():
        holdout = set(json.loads(gate.read_text(encoding="utf-8")).get("holdout_clip_ids") or [])

    data_yaml = build_yolo_seg_dataset(
        autolabel_root=autolabel_root,
        dataset_root=dataset_root,
        holdout_clip_ids=holdout,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    print(f"Dataset: {data_yaml}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from src.showcase.train_loop import run_training_from_config

    cfg = _load_cfg(_resolve(args.config))
    weights = run_training_from_config(cfg, SCOUTIQ_ROOT)
    print(f"Best weights: {weights}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from src.showcase.run_showcase import run_showcase

    cfg = _load_cfg(_resolve(args.config))
    video = _resolve(args.video or cfg["video"])
    weights = _resolve(cfg.get("weights", "outputs/training_loop/best_player_seg.pt"))
    output_video = _resolve(cfg.get("output_video", "outputs/showcase/unseen_showcase.mp4"))
    frames_dir = _resolve(cfg.get("frames_dir", f"data/frames/showcase/{video.stem[:40]}"))
    exclusions_dir = _resolve(cfg.get("exclusions_dir", "data/exclusions"))

    track = cfg.get("tracker") or {}
    run_showcase(
        video=video,
        weights=weights,
        output_video=output_video,
        frames_dir=frames_dir,
        exclusions_dir=exclusions_dir,
        clip_id=cfg.get("clip_id"),
        conf=float(track.get("conf", cfg.get("conf", 0.25))),
        iou=float(track.get("iou", cfg.get("iou", 0.70))),
        min_track_frames=int(track.get("min_track_frames", 3)),
        mask_smooth_alpha=float(cfg.get("mask_smooth_alpha", 0.35)),
        max_frames=args.max_frames or cfg.get("max_frames"),
        write_tracks_json=bool(cfg.get("write_tracks_json", True)),
    )
    print(f"Showcase video: {output_video}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ScoutIQ player seg showcase pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="Create data/outputs folders + default exclusions")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("ingest", help="Extract frames + write manifest")
    s.add_argument("--config", default="configs/showcase_ingest.yaml")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("autolabel", help="GDINO->SAM2->YOLO-seg labels (factory only)")
    s.add_argument("--config", default="configs/showcase_ingest.yaml")
    s.add_argument("--manifest")
    s.add_argument("--max-frames", type=int)
    s.set_defaults(func=cmd_autolabel)

    s = sub.add_parser("build-dataset", help="Assemble YOLO-seg train/val (excludes holdout)")
    s.add_argument("--config", default="configs/showcase_ingest.yaml")
    s.set_defaults(func=cmd_build_dataset)

    s = sub.add_parser("train", help="Train yolo11n-seg -> best_player_seg.pt")
    s.add_argument("--config", default="configs/showcase_train.yaml")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("run", help="Showcase inference on unseen clip (no GDINO)")
    s.add_argument("--config", default="configs/showcase_player_seg.yaml")
    s.add_argument("--video", help="Override video path (holdout)")
    s.add_argument("--max-frames", type=int)
    s.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


_SHOWCASE_DATA_README = """# Showcase data layout

```
data/
  footage/{train,val,holdout}/   # optional copies of source mp4s
  frames/<game_id>/<clip_id>/    # frame_XXXXXX.jpg (scales toward 1M frames)
  frames/manifest.csv
  frames/holdout_gate.json       # holdout clip IDs — never train on these
  exclusions/<clip_id>.json      # court ROI + shot clock / scorebug / crowd
  datasets/nba_player_seg/       # YOLO-seg train/val

outputs/
  auto_labels/                   # factory labels (GDINO+SAM2)
  training_loop/best_player_seg.pt
  showcase/                      # final unseen overlay mp4
```

## Commands (from scoutiq/)

```powershell
..\\.venv\\Scripts\\python.exe -m src.showcase init
..\\.venv\\Scripts\\python.exe -m src.showcase ingest --config configs/showcase_ingest.yaml
..\\.venv\\Scripts\\python.exe -m src.showcase autolabel --config configs/showcase_ingest.yaml
..\\.venv\\Scripts\\python.exe -m src.showcase build-dataset --config configs/showcase_ingest.yaml
..\\.venv\\Scripts\\python.exe -m src.showcase train --config configs/showcase_train.yaml
..\\.venv\\Scripts\\python.exe -m src.showcase run --config configs/showcase_player_seg.yaml
```

## Unseen clip gate

1. Put the holdout video in `clips:` with `split: holdout` in `showcase_ingest.yaml` (or only pass it to `run`).
2. `holdout_gate.json` records those clip IDs; `build-dataset` refuses them.
3. Never point training `data.yaml` at holdout frames.
4. Showcase `run` uses only `best_player_seg.pt` — no Grounding DINO.

Edit `data/exclusions/default.json` (or per-clip JSON) for shot clock / scorebug rectangles.
"""


if __name__ == "__main__":
    raise SystemExit(main())
