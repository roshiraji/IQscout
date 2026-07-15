"""YOLO segmentation polygon export."""

from __future__ import annotations

from pathlib import Path

from auto_annotation.config import AppConfig
from auto_annotation.utils import normalize_polygon, write_json


def export_yolo_seg(
    image_records: list[dict],
    cfg: AppConfig,
) -> Path:
    labels_dir = cfg.yolo_seg_directory / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    classes_path = cfg.yolo_seg_directory / "classes.txt"
    classes_path.write_text("\n".join(cfg.classes) + "\n", encoding="utf-8")

    exported = 0
    for rec in image_records:
        if rec.get("status") != "ok":
            continue
        w, h = rec["width"], rec["height"]
        lines: list[str] = []
        for inst in rec["instances"]:
            polys = inst.get("polygons") or []
            if not polys:
                continue
            # Use largest polygon
            poly = max(polys, key=lambda p: len(p))
            if len(poly) < 3:
                continue
            norm = normalize_polygon(poly, w, h)
            if len(norm) < 6:
                continue
            coords = " ".join(f"{v:.6f}" for v in norm)
            lines.append(f"{inst['class_id']} {coords}")
        label_path = labels_dir / f"{Path(rec['file_name']).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        exported += 1

    summary = {"labels_exported": exported, "classes_file": str(classes_path)}
    write_json(cfg.yolo_seg_directory / "export_summary.json", summary)
    return cfg.yolo_seg_directory
