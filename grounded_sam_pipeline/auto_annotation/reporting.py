"""Run summaries, review queue, and progress tracking."""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auto_annotation.config import AppConfig
from auto_annotation.utils import read_json, utc_now_iso, write_json

logger = logging.getLogger(__name__)


@dataclass
class ImageReport:
    file_name: str
    width: int = 0
    height: int = 0
    status: str = "pending"  # ok | no_detections | failed
    raw_dino_count: int = 0
    filtered_count: int = 0
    mask_count: int = 0
    classes: list[str] = field(default_factory=list)
    mean_dino_conf: float = 0.0
    mean_sam_score: float = 0.0
    processing_seconds: float = 0.0
    failure_reason: str = ""


def load_progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = read_json(path)
    return set(data.get("completed", []))


def save_progress(path: Path, completed: set[str]) -> None:
    write_json(path, {"completed": sorted(completed), "updated_at": utc_now_iso()})


def append_failure_log(path: Path, file_name: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{file_name}\t{reason}\n")


def build_review_queue(reports: list[ImageReport], image_records: list[dict], cfg: AppConfig) -> list[dict]:
    queue: list[dict] = []
    rec_by_name = {r["file_name"]: r for r in image_records}

    for rep in reports:
        if rep.status == "failed":
            queue.append({"file": rep.file_name, "reason": "processing_failed", "detail": rep.failure_reason})
        elif rep.status == "no_detections":
            queue.append({"file": rep.file_name, "reason": "no_detections", "detail": ""})
        rec = rec_by_name.get(rep.file_name, {})
        for inst in rec.get("instances", []):
            if inst["confidence"] < cfg.box_threshold + 0.05:
                queue.append({"file": rep.file_name, "reason": "low_dino_confidence", "detail": inst["class_name"]})
            if inst["mask_area_fraction"] < cfg.minimum_image_area_percentage:
                queue.append({"file": rep.file_name, "reason": "tiny_mask", "detail": inst["class_name"]})
            if inst["mask_area_fraction"] > cfg.maximum_image_area_percentage * 0.9:
                queue.append({"file": rep.file_name, "reason": "large_mask", "detail": inst["class_name"]})
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            if x1 <= 2 or y1 <= 2:
                queue.append({"file": rep.file_name, "reason": "box_near_boundary", "detail": inst["class_name"]})
    return queue


def write_reports(
    cfg: AppConfig,
    reports: list[ImageReport],
    image_records: list[dict],
    *,
    elapsed_seconds: float,
    env_report: dict,
) -> Path:
    cfg.reports_directory.mkdir(parents=True, exist_ok=True)

    # Per-image CSV
    if cfg.save_summary_csv:
        csv_path = cfg.reports_directory / "image_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "file_name", "width", "height", "status", "raw_dino_count", "filtered_count",
                "mask_count", "classes", "mean_dino_conf", "mean_sam_score", "processing_seconds", "failure_reason",
            ])
            writer.writeheader()
            for r in reports:
                writer.writerow({
                    "file_name": r.file_name,
                    "width": r.width,
                    "height": r.height,
                    "status": r.status,
                    "raw_dino_count": r.raw_dino_count,
                    "filtered_count": r.filtered_count,
                    "mask_count": r.mask_count,
                    "classes": ",".join(r.classes),
                    "mean_dino_conf": round(r.mean_dino_conf, 4),
                    "mean_sam_score": round(r.mean_sam_score, 4),
                    "processing_seconds": round(r.processing_seconds, 3),
                    "failure_reason": r.failure_reason,
                })

    # Class summary
    class_instances: Counter[str] = Counter()
    class_images: defaultdict[str, set] = defaultdict(set)
    dino_sums: defaultdict[str, list] = defaultdict(list)
    sam_sums: defaultdict[str, list] = defaultdict(list)
    area_sums: defaultdict[str, list] = defaultdict(list)

    for rec in image_records:
        if rec.get("status") != "ok":
            continue
        for inst in rec.get("instances", []):
            c = inst["class_name"]
            class_instances[c] += 1
            class_images[c].add(rec["file_name"])
            dino_sums[c].append(inst["confidence"])
            sam_sums[c].append(inst["sam_score"])
            area_sums[c].append(inst["mask_area_fraction"])

    class_rows = []
    for cls in cfg.classes:
        class_rows.append({
            "class_name": cls,
            "instance_count": class_instances.get(cls, 0),
            "image_count": len(class_images.get(cls, set())),
            "avg_dino_conf": round(sum(dino_sums[cls]) / len(dino_sums[cls]), 4) if dino_sums[cls] else 0,
            "avg_sam_score": round(sum(sam_sums[cls]) / len(sam_sums[cls]), 4) if sam_sums[cls] else 0,
            "avg_mask_area_pct": round(100 * sum(area_sums[cls]) / len(area_sums[cls]), 4) if area_sums[cls] else 0,
        })

    class_csv = cfg.reports_directory / "class_counts.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(class_rows[0].keys()) if class_rows else ["class_name"])
        writer.writeheader()
        writer.writerows(class_rows)

    review = build_review_queue(reports, image_records, cfg)
    review_csv = cfg.reports_directory / "review_queue.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "reason", "detail"])
        writer.writeheader()
        writer.writerows(review)

    summary = {
        "generated_at": utc_now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "images_processed": len(reports),
        "images_ok": sum(1 for r in reports if r.status == "ok"),
        "images_no_detections": sum(1 for r in reports if r.status == "no_detections"),
        "images_failed": sum(1 for r in reports if r.status == "failed"),
        "total_instances": sum(r.mask_count for r in reports),
        "class_counts": dict(class_instances),
        "output_directories": {
            "root": str(cfg.output_root),
            "previews": str(cfg.previews_directory),
            "annotations": str(cfg.annotations_directory),
            "reports": str(cfg.reports_directory),
        },
        "environment": env_report,
    }

    summary_path = cfg.reports_directory / "run_summary.json"
    if cfg.save_summary_json:
        write_json(summary_path, summary)

    logger.info(
        "Report: %d images | %d ok | %d no detections | %d failed | %.1fs",
        len(reports), summary["images_ok"], summary["images_no_detections"],
        summary["images_failed"], elapsed_seconds,
    )
    return summary_path
