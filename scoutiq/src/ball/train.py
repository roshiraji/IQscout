"""Train dedicated YOLO detect weights for basketball (separate from players)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    os.environ.setdefault("USE_LIBUV", "0")


def resolve_device(cfg_device: Any = None) -> int | str | list[int]:
    if cfg_device is not None:
        return cfg_device
    if not torch.cuda.is_available():
        return "cpu"
    return 0


def train_ball_det(
    *,
    data_yaml: Path,
    runs_root: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 60,
    imgsz: int = 960,
    batch: int = 8,
    device: Any = None,
    resume_weights: Path | None = None,
    project_name: str = "ball_det",
    workers: int = 2,
    export_name: str = "best_ball_det.pt",
) -> Path:
    from ultralytics import YOLO

    runs_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    weights = resume_weights if resume_weights and resume_weights.is_file() else model_name
    logger.info(
        "Ball train %s | device=%s | epochs=%d | batch=%d | imgsz=%d",
        weights,
        device,
        epochs,
        batch,
        imgsz,
    )

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(runs_root),
        name=project_name,
        exist_ok=True,
        verbose=True,
    )
    best = runs_root / project_name / "weights" / "best.pt"
    if not best.is_file():
        candidates = list(runs_root.glob("**/weights/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Ball training produced no weights under {runs_root}")
        best = max(candidates, key=lambda p: p.stat().st_mtime)

    export = runs_root / export_name
    shutil.copy2(best, export)
    last_name = export_name.replace("best_", "last_")
    lasts = sorted(runs_root.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    if lasts:
        shutil.copy2(lasts[-1], runs_root / last_name)
    logger.info("Promoted weights → %s", export)
    return export


def validate_ball_det(
    weights: Path,
    data_yaml: Path,
    *,
    imgsz: int = 960,
    device: Any = None,
) -> dict[str, float]:
    from ultralytics import YOLO

    device = resolve_device(device)
    if isinstance(device, list):
        device = device[0]
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    out: dict[str, float] = {}
    try:
        box = metrics.box
        out["box_map50"] = float(getattr(box, "map50", 0.0) or 0.0)
        out["box_map"] = float(getattr(box, "map", 0.0) or 0.0)
        out["box_precision"] = float(getattr(box, "mp", 0.0) or 0.0)
        out["box_recall"] = float(getattr(box, "mr", 0.0) or 0.0)
    except Exception:
        pass
    return out


def run_ball_training_from_config(cfg: dict, scoutiq_root: Path) -> Path:
    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (scoutiq_root / p).resolve()

    train_cfg = cfg.get("training") or {}
    out_cfg = cfg.get("output", {})
    data_yaml = _resolve(cfg["dataset"]["data_yaml"])
    runs_root = _resolve(out_cfg.get("runs_root", "outputs/training_loop_ball"))
    export_name = str(out_cfg.get("weights_name", "best_ball_det.pt"))
    epochs = int(train_cfg.get("epochs", 60))
    imgsz = int(train_cfg.get("imgsz", 960))
    batch = int(train_cfg.get("batch", 8))
    workers = int(train_cfg.get("workers", 2))
    model_name = str(train_cfg.get("model", "yolo11n.pt"))
    device = train_cfg.get("device")
    rounds = int(train_cfg.get("max_rounds", 1))
    targets = cfg.get("targets") or {}
    min_box = float(targets.get("min_box_map50", 0.35))

    history: list[dict] = []
    best_export: Path | None = None
    resume: Path | None = None

    for round_id in range(1, rounds + 1):
        logger.info("=== Ball det train round %d/%d ===", round_id, rounds)
        weights = train_ball_det(
            data_yaml=data_yaml,
            runs_root=runs_root,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            resume_weights=resume,
            project_name=f"round_{round_id:03d}",
            workers=workers,
            export_name=export_name,
        )
        metrics = validate_ball_det(weights, data_yaml, imgsz=imgsz, device=device)
        passed = metrics.get("box_map50", 0.0) >= min_box
        record = {
            "round": round_id,
            "weights": str(weights),
            "metrics": metrics,
            "passed": passed,
        }
        history.append(record)
        (runs_root / f"round_{round_id:03d}_report.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        best_export = weights
        if passed:
            break
        resume = weights

    summary = {
        "history": history,
        "best_weights": str(best_export) if best_export else None,
        "note": "Separate from player seg weights (best_player_seg_v2.pt).",
    }
    (runs_root / "loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_export or runs_root / "best_ball_det.pt"
