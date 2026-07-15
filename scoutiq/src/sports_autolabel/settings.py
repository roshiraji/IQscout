"""Project path and runtime settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelinePaths:
    root: Path
    config_dir: Path
    data_dir: Path
    outputs_dir: Path
    raw_videos_dir: Path
    frames_dir: Path
    raw_predictions_dir: Path
    corrected_predictions_dir: Path
    yolo_det_labels_dir: Path
    yolo_seg_labels_dir: Path
    debug_frames_dir: Path
    review_queue_path: Path
    final_videos_dir: Path
    tracking_dir: Path
    datasets_root: Path
    training_runs_root: Path
    classes_yaml_path: Path
    court_roi_path: Path
    classes_summary_path: Path

    def as_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


@dataclass(slots=True)
class RuntimeSettings:
    default_fps: float = 2.0
    default_review_percent: int = 20
    default_imgsz: int = 960
    default_batch: str = "auto"
    default_device: str = "0"
    default_workers: int = 4
    default_mixed_precision: bool = True


def build_paths(project_root: Path) -> PipelinePaths:
    config_dir = project_root / "config"
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    return PipelinePaths(
        root=project_root,
        config_dir=config_dir,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        raw_videos_dir=data_dir / "raw_videos",
        frames_dir=data_dir / "frames",
        raw_predictions_dir=data_dir / "autolabel" / "raw_predictions",
        corrected_predictions_dir=data_dir / "autolabel" / "corrected_predictions",
        yolo_det_labels_dir=data_dir / "labels" / "yolo_det",
        yolo_seg_labels_dir=data_dir / "labels" / "yolo_seg",
        debug_frames_dir=outputs_dir / "debug_frames",
        review_queue_path=outputs_dir / "review_queue.json",
        final_videos_dir=outputs_dir / "final_videos",
        tracking_dir=outputs_dir / "tracking",
        datasets_root=data_dir / "datasets",
        training_runs_root=outputs_dir / "training_runs",
        classes_yaml_path=config_dir / "classes.yaml",
        court_roi_path=config_dir / "court_roi.json",
        classes_summary_path=config_dir / "classes_summary.json",
    )


def ensure_base_dirs(paths: PipelinePaths) -> None:
    required = [
        paths.config_dir,
        paths.data_dir,
        paths.outputs_dir,
        paths.raw_videos_dir,
        paths.frames_dir,
        paths.raw_predictions_dir,
        paths.corrected_predictions_dir,
        paths.yolo_det_labels_dir,
        paths.yolo_seg_labels_dir,
        paths.debug_frames_dir,
        paths.final_videos_dir,
        paths.tracking_dir,
        paths.datasets_root,
        paths.training_runs_root,
    ]
    for folder in required:
        folder.mkdir(parents=True, exist_ok=True)
