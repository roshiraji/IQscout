"""
Iterative NBA player detector training loop.

Merges multi-format annotation data, trains YOLO, evaluates on val/test,
and runs tracking on source video until player-tracking confidence targets are met.

Usage (from scoutiq/):
  python -m src.training.nba_player_loop --config configs/nba_player_loop.yaml
  python -m src.training.nba_player_loop --ingest-only
  python -m src.training.nba_player_loop --max-rounds 1 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from ultralytics import YOLO

from src.training.nba_data_ingest import DEFAULT_CLASSES, build_merged_dataset

SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LoopTargets:
    min_player_map50: float = 0.45
    min_player_precision: float = 0.40
    min_player_recall: float = 0.40
    min_avg_player_confidence: float = 0.30
    min_players_per_frame: int = 4
    min_track_length_frames: int = 10
    min_frame_coverage: float = 0.60


@dataclass
class RoundResult:
    round_id: int
    weights: Path
    val_metrics: dict[str, float]
    tracking_metrics: dict[str, float]
    passed: bool
    notes: list[str]


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def ingest_from_config(cfg: dict[str, Any]) -> Path:
    sources = cfg["data_sources"]
    frames_dirs = [_resolve(p, SCOUTIQ_ROOT) for p in sources.get("frames_dirs", [])]
    out = _resolve(cfg["output"]["merged_dataset"], SCOUTIQ_ROOT)

    classes_raw = cfg.get("classes", DEFAULT_CLASSES)
    classes = {int(k): v for k, v in classes_raw.items()}

    legacy_zip = sources.get("legacy_yolo_zip")
    data_yaml, stats = build_merged_dataset(
        output_dir=out,
        classes=classes,
        yolo_dataset=_resolve(sources["yolo_dataset"], SCOUTIQ_ROOT) if sources.get("yolo_dataset") else None,
        raw_labels=_resolve(sources["raw_labels"], SCOUTIQ_ROOT) if sources.get("raw_labels") else None,
        frames_dirs=frames_dirs,
        annotations_csv=_resolve(sources["annotations_csv"], SCOUTIQ_ROOT) if sources.get("annotations_csv") else None,
        legacy_yolo_zip=_resolve(legacy_zip, SCOUTIQ_ROOT) if legacy_zip else None,
        train_ratio=float(cfg.get("split", {}).get("train", 0.80)),
        val_ratio=float(cfg.get("split", {}).get("val", 0.15)),
        seed=int(cfg.get("split", {}).get("seed", 42)),
    )
    print(f"Ingested {stats.images} image/label pairs from {stats.sources}")
    return data_yaml


def _extract_class_metrics(metrics_obj: Any, player_class_id: int = 0) -> dict[str, float]:
    """Pull player-class metrics from Ultralytics validation results."""
    result = {
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    if metrics_obj is None:
        return result

    box = getattr(metrics_obj, "box", None)
    if box is None:
        return result

    result["map50"] = float(getattr(box, "map50", 0.0) or 0.0)
    result["map50_95"] = float(getattr(box, "map", 0.0) or 0.0)
    result["precision"] = float(getattr(box, "mp", 0.0) or 0.0)
    result["recall"] = float(getattr(box, "mr", 0.0) or 0.0)

    # Per-class arrays when available (index = class id)
    maps = getattr(box, "maps", None)
    if maps is not None and len(maps) > player_class_id:
        result["player_map50_95"] = float(maps[player_class_id])
    ap50 = getattr(box, "ap50", None)
    if ap50 is not None and len(ap50) > player_class_id:
        result["player_map50"] = float(ap50[player_class_id])
    return result


def train_round(
    *,
    data_yaml: Path,
    round_id: int,
    runs_root: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    resume_weights: Path | None = None,
) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_name = f"round_{round_id:03d}"
    device = 0 if torch.cuda.is_available() else "cpu"

    weights = resume_weights if resume_weights and resume_weights.is_file() else model_name
    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(runs_root),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    best = runs_root / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce weights: {best}")
    return best


def validate_round(weights: Path, data_yaml: Path, imgsz: int) -> dict[str, float]:
    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    return _extract_class_metrics(metrics)


def evaluate_tracking(
    *,
    weights: Path,
    video: Path,
    player_class_id: int = 0,
    conf: float = 0.25,
    iou: float = 0.70,
    tracker: str = "botsort.yaml",
    max_frames: int | None = None,
    min_players_per_frame: int = 4,
) -> dict[str, float]:
    """Run BoT-SORT and score player detection + track persistence."""
    if not video.is_file():
        return {"error": 1.0, "frames_processed": 0.0}

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"error": 1.0, "frames_processed": 0.0}

    frame_idx = 0
    player_confs: list[float] = []
    players_per_frame: list[int] = []
    track_lengths: dict[int, int] = defaultdict(int)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou, verbose=False)[0]
        boxes = result.boxes
        frame_players = 0
        if boxes is not None and boxes.id is not None:
            for det_conf, cls, tid in zip(
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.id.cpu().numpy(),
            ):
                if int(cls) == player_class_id:
                    frame_players += 1
                    player_confs.append(float(det_conf))
                    track_lengths[int(tid)] += 1
        players_per_frame.append(frame_players)
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return {"error": 1.0, "frames_processed": 0.0}

    good_frames = sum(1 for n in players_per_frame if n >= min_players_per_frame)
    avg_track_len = (
        sum(track_lengths.values()) / len(track_lengths) if track_lengths else 0.0
    )
    return {
        "frames_processed": float(frame_idx),
        "avg_player_confidence": float(sum(player_confs) / len(player_confs)) if player_confs else 0.0,
        "avg_players_per_frame": float(sum(players_per_frame) / len(players_per_frame)),
        "frame_coverage": float(good_frames / frame_idx),
        "avg_track_length": float(avg_track_len),
        "unique_tracks": float(len(track_lengths)),
    }


def _check_targets(
    val_metrics: dict[str, float],
    tracking: dict[str, float],
    targets: LoopTargets,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    passed = True

    player_map50 = val_metrics.get("player_map50", val_metrics.get("map50", 0.0))
    checks = [
        ("player_map50", player_map50, targets.min_player_map50),
        ("precision", val_metrics.get("precision", 0.0), targets.min_player_precision),
        ("recall", val_metrics.get("recall", 0.0), targets.min_player_recall),
        ("avg_player_confidence", tracking.get("avg_player_confidence", 0.0), targets.min_avg_player_confidence),
        ("frame_coverage", tracking.get("frame_coverage", 0.0), targets.min_frame_coverage),
        ("avg_track_length", tracking.get("avg_track_length", 0.0), targets.min_track_length_frames),
    ]
    for name, value, threshold in checks:
        ok = value >= threshold
        notes.append(f"{name}: {value:.3f} (need >= {threshold:.3f}) {'PASS' if ok else 'FAIL'}")
        passed = passed and ok
    return passed, notes


def run_loop(cfg: dict[str, Any], *, max_rounds: int | None = None, epochs: int | None = None) -> Path:
    targets = LoopTargets(**cfg.get("confidence_targets", {}))
    train_cfg = cfg["training"]
    runs_root = _resolve(cfg["output"]["loop_runs"], SCOUTIQ_ROOT)
    runs_root.mkdir(parents=True, exist_ok=True)

    data_yaml = ingest_from_config(cfg)
    video = _resolve(cfg["data_sources"]["video"], SCOUTIQ_ROOT)
    rounds = max_rounds or int(train_cfg.get("max_rounds", 5))
    epochs_per_round = epochs or int(train_cfg.get("epochs_per_round", 50))
    imgsz = int(train_cfg.get("imgsz", 960))
    base_model = str(train_cfg.get("model", "yolo11n.pt"))
    track_cfg = cfg.get("tracking", {})

    history: list[dict[str, Any]] = []
    best_weights: Path | None = None
    best_score = -1.0
    resume: Path | None = None

    for round_id in range(1, rounds + 1):
        print(f"\n{'=' * 60}\nRound {round_id}/{rounds}\n{'=' * 60}")
        t0 = time.perf_counter()

        weights = train_round(
            data_yaml=data_yaml,
            round_id=round_id,
            runs_root=runs_root,
            model_name=base_model,
            epochs=epochs_per_round,
            imgsz=imgsz,
            resume_weights=resume,
        )
        val_metrics = validate_round(weights, data_yaml, imgsz)
        tracking = evaluate_tracking(
            weights=weights,
            video=video,
            conf=float(track_cfg.get("conf", 0.25)),
            iou=float(track_cfg.get("iou", 0.70)),
            tracker=str(track_cfg.get("tracker", "botsort.yaml")),
            max_frames=track_cfg.get("max_frames"),
            min_players_per_frame=targets.min_players_per_frame,
        )
        passed, notes = _check_targets(val_metrics, tracking, targets)

        score = (
            val_metrics.get("player_map50", val_metrics.get("map50", 0.0)) * 0.4
            + tracking.get("frame_coverage", 0.0) * 0.3
            + tracking.get("avg_player_confidence", 0.0) * 0.3
        )
        if score > best_score:
            best_score = score
            best_weights = weights
            export_path = runs_root / "best_player_detector.pt"
            export_path.write_bytes(weights.read_bytes())

        round_record = {
            "round": round_id,
            "weights": str(weights),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "val_metrics": val_metrics,
            "tracking_metrics": tracking,
            "passed": passed,
            "notes": notes,
            "score": round(score, 4),
        }
        history.append(round_record)
        (runs_root / f"round_{round_id:03d}_report.json").write_text(
            json.dumps(round_record, indent=2), encoding="utf-8",
        )
        for line in notes:
            print(line)

        if passed:
            print(f"\n✅ Confidence targets met after round {round_id}.")
            break

        resume = weights
        print(f"\n↻ Targets not met — continuing with weights from round {round_id}.")

    summary = {
        "rounds_completed": len(history),
        "best_weights": str(best_weights) if best_weights else None,
        "best_score": best_score,
        "targets_met": history[-1]["passed"] if history else False,
        "history": history,
    }
    summary_path = runs_root / "loop_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nLoop summary: {summary_path}")
    if best_weights:
        print(f"Best weights: {runs_root / 'best_player_detector.pt'}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NBA multi-source training loop for player tracking")
    p.add_argument("--config", default="configs/nba_player_loop.yaml")
    p.add_argument("--ingest-only", action="store_true", help="Only merge data sources, skip training")
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--epochs", type=int, help="Override epochs per round")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = _resolve(args.config, SCOUTIQ_ROOT)
    cfg = _load_config(cfg_path)

    if args.ingest_only:
        data_yaml = ingest_from_config(cfg)
        print(f"Dataset ready: {data_yaml}")
        return 0

    run_loop(cfg, max_rounds=args.max_rounds, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
