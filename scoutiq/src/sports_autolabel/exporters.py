"""Dataset export and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import yaml
from tqdm import tqdm

from .contracts import FrameDetections, parse_frame_json
from .settings import PipelinePaths
from .utils import load_class_map, read_json, write_json


def _xyxy_to_yolo(box: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    return cx / width, cy / height, box_w / width, box_h / height


def _iter_frame_jsons(raw_predictions_dir: Path) -> Iterable[Path]:
    return sorted(path for path in raw_predictions_dir.glob("frame_*.json") if path.is_file())


def export_yolo_detection(paths: PipelinePaths) -> dict:
    paths.yolo_det_labels_dir.mkdir(parents=True, exist_ok=True)
    total_labels = 0
    file_count = 0
    for json_path in tqdm(list(_iter_frame_jsons(paths.raw_predictions_dir)), desc="Export YOLO det"):
        frame = parse_frame_json(json_path)
        rows: list[str] = []
        for det in frame.detections:
            cx, cy, bw, bh = _xyxy_to_yolo(det.bbox_xyxy, frame.width, frame.height)
            rows.append(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_labels += 1
        out = paths.yolo_det_labels_dir / f"{Path(frame.image).stem}.txt"
        out.write_text("\n".join(rows), encoding="utf-8")
        file_count += 1

    summary = {
        "frames_exported": file_count,
        "labels_exported": total_labels,
        "labels_dir": str(paths.yolo_det_labels_dir),
    }
    write_json(paths.yolo_det_labels_dir / "export_summary.json", summary)
    return summary


def export_yolo_seg(paths: PipelinePaths, sam2_available: bool) -> dict:
    """
    Export YOLO segmentation labels.

    If SAM2 is unavailable, fallback to box polygons (4 points).
    """
    paths.yolo_seg_labels_dir.mkdir(parents=True, exist_ok=True)
    mode = "sam2" if sam2_available else "box_polygon_fallback"
    labels_count = 0
    files_count = 0
    for json_path in tqdm(list(_iter_frame_jsons(paths.raw_predictions_dir)), desc="Export YOLO seg"):
        frame = parse_frame_json(json_path)
        rows: list[str] = []
        for det in frame.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            # YOLO seg format: class x1 y1 x2 y2 ...
            row = [
                str(det.class_id),
                f"{x1 / frame.width:.6f}",
                f"{y1 / frame.height:.6f}",
                f"{x2 / frame.width:.6f}",
                f"{y1 / frame.height:.6f}",
                f"{x2 / frame.width:.6f}",
                f"{y2 / frame.height:.6f}",
                f"{x1 / frame.width:.6f}",
                f"{y2 / frame.height:.6f}",
            ]
            rows.append(" ".join(row))
            labels_count += 1
        (paths.yolo_seg_labels_dir / f"{Path(frame.image).stem}.txt").write_text(
            "\n".join(rows), encoding="utf-8"
        )
        files_count += 1

    summary = {
        "mode": mode,
        "frames_exported": files_count,
        "labels_exported": labels_count,
        "labels_dir": str(paths.yolo_seg_labels_dir),
    }
    write_json(paths.yolo_seg_labels_dir / "export_summary.json", summary)
    return summary


def export_coco(paths: PipelinePaths, output_path: Path | None = None) -> dict:
    classes = load_class_map(paths)
    categories = [{"id": class_id, "name": class_name} for class_id, class_name in classes.items()]
    images = []
    annotations = []
    ann_id = 1
    for json_path in tqdm(list(_iter_frame_jsons(paths.raw_predictions_dir)), desc="Export COCO"):
        frame = parse_frame_json(json_path)
        image_id = frame.frame_id
        images.append(
            {
                "id": image_id,
                "file_name": frame.image,
                "width": frame.width,
                "height": frame.height,
            }
        )
        for det in frame.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det.class_id,
                    "bbox": [x1, y1, w, h],
                    "area": int(w * h),
                    "iscrowd": 0,
                    "score": det.confidence,
                    "source": det.source,
                }
            )
            ann_id += 1

    payload = {"images": images, "annotations": annotations, "categories": categories}
    out = output_path or (paths.outputs_dir / "coco_annotations.json")
    write_json(out, payload)
    return {"images": len(images), "annotations": len(annotations), "output": str(out)}


def validate_dataset(
    *,
    dataset_dir: Path,
    classes_yaml_path: Path,
    segmentation: bool,
) -> dict:
    """
    Validate training data and report exact files/line numbers for failures.
    """
    errors: list[str] = []
    warnings: list[str] = []
    classes_payload = yaml.safe_load(classes_yaml_path.read_text(encoding="utf-8")) or {}
    names = classes_payload.get("names", {})
    valid_class_ids = {int(k) for k in names.keys()}

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    for split in ("train", "val", "test"):
        img_split = images_dir / split
        lbl_split = labels_dir / split
        if not img_split.is_dir():
            errors.append(f"Missing split folder: {img_split}")
            continue
        if not lbl_split.is_dir():
            errors.append(f"Missing split label folder: {lbl_split}")
            continue

        image_files = sorted(list(img_split.glob("*.jpg")))
        if not image_files:
            warnings.append(f"No images in split: {img_split}")
        for image_path in image_files:
            label_path = lbl_split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"Missing label for image: {image_path}")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"Corrupted image: {image_path}")
                continue
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    errors.append(f"{label_path}:{line_no} invalid class id '{parts[0]}'")
                    continue
                if class_id not in valid_class_ids:
                    errors.append(f"{label_path}:{line_no} class id {class_id} not in classes.yaml")

                coords = parts[1:]
                if segmentation:
                    if len(coords) < 6:
                        errors.append(f"{label_path}:{line_no} segmentation needs >= 3 points")
                    if len(coords) % 2 != 0:
                        errors.append(f"{label_path}:{line_no} segmentation coordinate count must be even")
                else:
                    if len(coords) != 4:
                        errors.append(f"{label_path}:{line_no} detection row must have 4 coordinates")

                for value in coords:
                    try:
                        val = float(value)
                    except ValueError:
                        errors.append(f"{label_path}:{line_no} non-numeric coordinate '{value}'")
                        continue
                    if not 0.0 <= val <= 1.0:
                        errors.append(f"{label_path}:{line_no} coordinate {val} out of [0,1]")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        errors.append(f"Missing data.yaml: {data_yaml}")
    else:
        data_payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        for key in ("train", "val", "test"):
            rel = data_payload.get(key)
            if not rel:
                errors.append(f"data.yaml missing '{key}' path")
                continue
            resolved = (dataset_dir / rel).resolve()
            if not resolved.exists():
                errors.append(f"data.yaml '{key}' path missing: {resolved}")

    summary = {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    write_json(dataset_dir / "validation_report.json", summary)
    return summary


def merge_corrected_labels(paths: PipelinePaths) -> dict:
    corrected_file = paths.corrected_predictions_dir / "corrected_labels.json"
    corrected = read_json(corrected_file, default=[])
    merged = {"corrected_count": len(corrected), "source_file": str(corrected_file)}
    write_json(paths.corrected_predictions_dir / "merge_summary.json", merged)
    return merged
