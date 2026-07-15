"""COCO instance-segmentation export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_annotation.config import AppConfig
from auto_annotation.segmenter import SegmentedInstance
from auto_annotation.utils import mask_to_polygons, write_json


def export_coco(
    image_records: list[dict[str, Any]],
    cfg: AppConfig,
) -> Path:
    """Write COCO instances.json from accumulated per-image records."""
    cfg.coco_directory.mkdir(parents=True, exist_ok=True)
    out_path = cfg.coco_directory / "instances.json"

    coco: dict[str, Any] = {
        "info": {
            "description": "Auto-annotated with Grounding DINO + SAM",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(cfg.classes)
        ],
    }

    ann_id = 1
    for rec in image_records:
        if rec.get("status") != "ok":
            continue
        coco["images"].append({
            "id": rec["image_id"],
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
        })
        for inst in rec["instances"]:
            seg = inst.get("polygons") or []
            if not seg:
                continue
            # COCO segmentation: list of polygons (flat not required for list-of-lists)
            segmentation = [[coord for pt in poly for coord in pt] for poly in seg]
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": rec["image_id"],
                "category_id": inst["class_id"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": inst["mask_area_px"],
                "segmentation": segmentation,
                "iscrowd": 0,
                "score": inst["confidence"],
                "sam_score": inst["sam_score"],
            })
            ann_id += 1

    write_json(out_path, coco)
    return out_path
