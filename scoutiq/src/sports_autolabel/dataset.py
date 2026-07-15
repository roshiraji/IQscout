"""Dataset versioning and build helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .settings import PipelinePaths
from .utils import load_class_map, read_json, write_json


def _split_indices(total: int) -> tuple[int, int, int]:
    train = max(int(total * 0.8), 1)
    val = max(int(total * 0.15), 1) if total > 2 else 1
    test = max(total - train - val, 1) if total > 3 else max(total - train - val, 0)
    while train + val + test > total:
        if train > 1:
            train -= 1
        elif val > 1:
            val -= 1
        else:
            test -= 1
    return train, val, test


def _version_dir(paths: PipelinePaths, version: str) -> Path:
    normalized = version if version.startswith("v") else f"v{version}"
    return paths.datasets_root / f"basketball_yolo_seg_{normalized}"


def build_dataset_version(paths: PipelinePaths, version: str) -> dict:
    dataset_dir = _version_dir(paths, version)
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    image_files = sorted(paths.frames_dir.glob("frame_*.jpg"))
    labeled = [img for img in image_files if (paths.yolo_seg_labels_dir / f"{img.stem}.txt").is_file()]
    if not labeled:
        labeled = [img for img in image_files if (paths.yolo_det_labels_dir / f"{img.stem}.txt").is_file()]
    if not labeled:
        raise FileNotFoundError("No labeled frames found. Run autolabel + export first.")

    train_n, val_n, test_n = _split_indices(len(labeled))
    splits = {
        "train": labeled[:train_n],
        "val": labeled[train_n : train_n + val_n],
        "test": labeled[train_n + val_n : train_n + val_n + test_n],
    }

    source_dir = paths.yolo_seg_labels_dir if any(paths.yolo_seg_labels_dir.glob("*.txt")) else paths.yolo_det_labels_dir
    for split, files in splits.items():
        for image_path in files:
            dest_img = images_root / split / image_path.name
            shutil.copy2(image_path, dest_img)
            src_lbl = source_dir / f"{image_path.stem}.txt"
            if src_lbl.is_file():
                shutil.copy2(src_lbl, labels_root / split / src_lbl.name)

    classes = load_class_map(paths)
    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes,
    }
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    queue = read_json(paths.review_queue_path, default={})
    corrected = read_json(paths.corrected_predictions_dir / "corrected_labels.json", default=[])
    rejected = [row for row in corrected if row.get("review_decision") == "reject"]
    reviewed_frames = sorted({Path(item["frame_path"]).name for item in queue.get("items", [])})

    manifest = {
        "dataset_version": dataset_dir.name,
        "images_total": len(labeled),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "label_source_dir": str(source_dir),
    }
    source_summary = {
        "raw_predictions_dir": str(paths.raw_predictions_dir),
        "corrected_predictions_count": len(corrected),
        "rejected_labels_count": len(rejected),
    }
    write_json(dataset_dir / "dataset_manifest.json", manifest)
    write_json(dataset_dir / "label_source_summary.json", source_summary)
    write_json(dataset_dir / "reviewed_frames.json", {"frames": reviewed_frames})
    write_json(dataset_dir / "rejected_labels.json", rejected)
    return {
        "dataset_dir": str(dataset_dir),
        "images_total": len(labeled),
        "split_counts": manifest["split_counts"],
    }
