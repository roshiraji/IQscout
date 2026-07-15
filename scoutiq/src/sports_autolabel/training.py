"""Training/inference wrappers with GPU-safe fallbacks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from .exporters import validate_dataset
from .settings import PipelinePaths, RuntimeSettings
from .utils import read_json, write_json


def _next_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    runs = sorted(path for path in root.glob("run_*") if path.is_dir())
    if not runs:
        return root / "run_001"
    latest = runs[-1].name.split("_")[-1]
    try:
        next_id = int(latest) + 1
    except ValueError:
        next_id = len(runs) + 1
    return root / f"run_{next_id:03d}"


def _load_yolo():
    from ultralytics import YOLO

    return YOLO


def train_yolo(
    *,
    paths: PipelinePaths,
    dataset_version: str,
    epochs: int,
    segmentation: bool,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    dataset_dir = paths.datasets_root / (
        dataset_version if dataset_version.startswith("basketball_yolo_seg_") else f"basketball_yolo_seg_{dataset_version}"
    )
    validation = validate_dataset(
        dataset_dir=dataset_dir,
        classes_yaml_path=paths.classes_yaml_path,
        segmentation=segmentation,
    )
    if not validation["valid"]:
        raise ValueError("Dataset validation failed. See validation_report.json for exact errors.")

    run_dir = _next_run_dir(paths.training_runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_examples").mkdir(parents=True, exist_ok=True)
    (run_dir / "failure_cases").mkdir(parents=True, exist_ok=True)

    args = {
        "data": str(dataset_dir / "data.yaml"),
        "epochs": int(epochs),
        "imgsz": int(settings.default_imgsz),
        "batch": settings.default_batch,
        "device": settings.default_device if torch.cuda.is_available() else "cpu",
        "workers": int(settings.default_workers),
        "amp": bool(settings.default_mixed_precision),
        "task": "segment" if segmentation else "detect",
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), encoding="utf-8")

    model_name = "yolo11n-seg.pt" if segmentation else "yolo11n.pt"
    metrics = {}
    try:
        YOLO = _load_yolo()
        model = YOLO(model_name)
        result = model.train(
            data=args["data"],
            epochs=args["epochs"],
            imgsz=args["imgsz"],
            batch=args["batch"],
            device=args["device"],
            workers=args["workers"],
            amp=args["amp"],
            project=str(paths.training_runs_root),
            name=run_dir.name,
            exist_ok=True,
            verbose=True,
        )
        metrics = {
            "mAP50": float(getattr(result.box, "map50", 0.0)) if hasattr(result, "box") else 0.0,
            "mAP50-95": float(getattr(result.box, "map", 0.0)) if hasattr(result, "box") else 0.0,
            "precision": float(getattr(result.box, "mp", 0.0)) if hasattr(result, "box") else 0.0,
            "recall": float(getattr(result.box, "mr", 0.0)) if hasattr(result, "box") else 0.0,
        }
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            torch.cuda.empty_cache()
            args["imgsz"] = 640
            args["batch"] = 4
            args["oom_fallback_applied"] = True
            YOLO = _load_yolo()
            model = YOLO(model_name)
            model.train(
                data=args["data"],
                epochs=args["epochs"],
                imgsz=args["imgsz"],
                batch=args["batch"],
                device=args["device"],
                workers=max(1, args["workers"] // 2),
                amp=False,
                project=str(paths.training_runs_root),
                name=run_dir.name,
                exist_ok=True,
                verbose=True,
            )
            metrics = {"warning": "OOM fallback applied (imgsz=640, batch=4, amp=False)"}
        else:
            metrics = {"warning": f"Training skipped due to runtime error: {exc}"}
    except Exception as exc:  # pragma: no cover - dependency-driven
        metrics = {"warning": f"Training unavailable: {exc}"}

    # Attempt to copy checkpoints if Ultralytics produced them.
    weights_dir = run_dir / "weights"
    best_source = weights_dir / "best.pt"
    last_source = weights_dir / "last.pt"
    if best_source.is_file():
        shutil.copy2(best_source, run_dir / "best.pt")
    if last_source.is_file():
        shutil.copy2(last_source, run_dir / "last.pt")
    confusion = run_dir / "confusion_matrix.png"
    if not confusion.is_file() and (run_dir / "confusion_matrix_normalized.png").is_file():
        shutil.copy2(run_dir / "confusion_matrix_normalized.png", confusion)

    review_queue = read_json(paths.review_queue_path, default={})
    corrected_rows = read_json(paths.corrected_predictions_dir / "corrected_labels.json", default=[])
    rejected_count = len([row for row in corrected_rows if row.get("review_decision") == "reject"])

    metrics_payload = {
        "metrics": metrics,
        "per_class_confidence": {},
        "per_class_false_positives": {},
        "per_class_false_negatives": {},
        "reviewed_frames": len(review_queue.get("items", [])),
        "corrected_objects": len(corrected_rows),
        "rejected_objects": rejected_count,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    return {"run_dir": str(run_dir), "metrics_path": str(run_dir / "metrics.json")}


def run_inference(paths: PipelinePaths, weights: Path, video: Path) -> dict[str, Any]:
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    out_dir = paths.outputs_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / f"{video.stem}_inference.mp4"
    try:
        YOLO = _load_yolo()
        model = YOLO(str(weights))
        model.predict(source=str(video), save=True, project=str(out_dir), name=video.stem, exist_ok=True)
    except Exception as exc:
        write_json(out_dir / "inference_warning.json", {"warning": str(exc)})
    return {"output_dir": str(out_dir), "output_video_hint": str(output_video)}
