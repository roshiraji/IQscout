"""Load segmentation lab settings from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _SCOUTIQ_ROOT / "configs" / "segmentation_lab.yaml"


@dataclass
class SegmentationLabConfig:
    fine_tune_clip: Path
    yolo_model: str
    yolo_imgsz: int
    yolo_conf: float
    sam_model: str
    frame_stride: int
    max_frames: int
    output_dir: Path
    use_snapshot: bool = False
    snapshot_json: Path | None = None

    @property
    def scoutiq_root(self) -> Path:
        return _SCOUTIQ_ROOT


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path | None = None) -> SegmentationLabConfig:
    path = config_path or _DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation lab config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.resolve().parents[1]

    snapshot_json = raw.get("snapshot_json")
    return SegmentationLabConfig(
        fine_tune_clip=_resolve_path(root, raw.get("fine_tune_clip", "")),
        yolo_model=str(raw.get("yolo_model", "yolo26s.pt")),
        yolo_imgsz=int(raw.get("yolo_imgsz", 1280)),
        yolo_conf=float(raw.get("yolo_conf", 0.25)),
        sam_model=str(raw.get("sam_model", "vit_b")),
        frame_stride=max(1, int(raw.get("frame_stride", 30))),
        max_frames=max(1, int(raw.get("max_frames", 10))),
        output_dir=_resolve_path(root, raw.get("output_dir", "outputs/segmentation_lab")),
        use_snapshot=bool(raw.get("use_snapshot", False)),
        snapshot_json=_resolve_path(root, snapshot_json) if snapshot_json else None,
    )
