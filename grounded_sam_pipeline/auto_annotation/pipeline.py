"""End-to-end pipeline orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from auto_annotation.config import AppConfig
from auto_annotation.detector import detect_objects
from auto_annotation.exporters.coco import export_coco
from auto_annotation.exporters.voc import export_voc
from auto_annotation.exporters.yolo_seg import export_yolo_seg
from auto_annotation.model_manager import ModelBundle, load_models
from auto_annotation.prompts import build_class_prompts
from auto_annotation.reporting import ImageReport, append_failure_log, load_progress, save_progress, write_reports
from auto_annotation.segmenter import SegmentedInstance, segment_instances
from auto_annotation.utils import copy_image_to_output, list_images, mask_to_polygons, read_image_bgr
from auto_annotation.visualization import save_individual_masks, save_preview

logger = logging.getLogger(__name__)


def _instances_to_dicts(
    instances: list[SegmentedInstance],
    cfg: AppConfig,
) -> list[dict[str, Any]]:
    rows: list[dict] = []
    for inst in instances:
        polys = mask_to_polygons(
            inst.mask,
            epsilon=cfg.polygon_approximation_epsilon,
            min_area=cfg.min_contour_area_pixels,
        )
        rows.append({
            "bbox_xyxy": inst.bbox_xyxy,
            "confidence": inst.confidence,
            "class_id": inst.class_id,
            "class_name": inst.class_name,
            "prompt": inst.prompt,
            "sam_score": inst.sam_score,
            "mask_area_px": inst.mask_area_px,
            "mask_area_fraction": inst.mask_area_fraction,
            "polygons": polys,
        })
    return rows


def _mock_instances(image_bgr: np.ndarray, cfg: AppConfig) -> list[SegmentedInstance]:
    """Synthetic instances for structure testing without GPU models."""
    h, w = image_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    box = [cx - 40, cy - 60, cx + 40, cy + 60]
    mask = np.zeros((h, w), dtype=bool)
    mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
    cls = cfg.classes[0]
    return [SegmentedInstance(
        bbox_xyxy=box,
        confidence=0.9,
        class_id=0,
        class_name=cls,
        prompt=f"mock {cls}",
        mask=mask,
        sam_score=0.95,
        mask_area_px=int(mask.sum()),
        mask_area_fraction=float(mask.sum()) / (h * w),
    )]


def run_pipeline(
    cfg: AppConfig,
    models: ModelBundle | None,
    *,
    limit: int | None = None,
    single_image: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    mock: bool = False,
    env_report: dict | None = None,
) -> Path:
    prompt_map = build_class_prompts(cfg.classes, cfg.prompt_template, cfg.prompt_overrides)

    if single_image:
        image_paths = [single_image]
    else:
        image_paths = list_images(cfg.images_directory, cfg.extensions)
    if limit:
        image_paths = image_paths[:limit]

    completed = load_progress(cfg.progress_file) if resume and not overwrite else set()
    if overwrite:
        completed = set()

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.previews_directory.mkdir(parents=True, exist_ok=True)
    cfg.masks_directory.mkdir(parents=True, exist_ok=True)
    cfg.images_output_directory.mkdir(parents=True, exist_ok=True)

    failure_log = cfg.reports_directory / "failures.log"
    reports: list[ImageReport] = []
    image_records: list[dict] = []
    start = time.perf_counter()

    for image_id, img_path in enumerate(tqdm(image_paths, desc="Annotating"), start=1):
        file_name = img_path.name
        if resume and file_name in completed and not overwrite:
            continue

        rep = ImageReport(file_name=file_name)
        t0 = time.perf_counter()
        try:
            image_bgr = read_image_bgr(img_path)
            h, w = image_bgr.shape[:2]
            rep.width, rep.height = w, h

            if mock:
                instances = _mock_instances(image_bgr, cfg)
                rep.raw_dino_count = 1
                rep.filtered_count = 1
            else:
                assert models is not None
                dets, raw_count = detect_objects(models.grounding_dino, image_bgr, prompt_map, cfg)
                rep.raw_dino_count = raw_count
                rep.filtered_count = len(dets)
                if not dets:
                    rep.status = "no_detections"
                    reports.append(rep)
                    image_records.append({
                        "image_id": image_id, "file_name": file_name,
                        "width": w, "height": h, "status": "no_detections", "instances": [],
                    })
                    completed.add(file_name)
                    save_progress(cfg.progress_file, completed)
                    continue
                try:
                    instances = segment_instances(models.sam_predictor, image_bgr, dets, cfg)
                except torch.cuda.OutOfMemoryError:
                    if cfg.retry_on_cpu_after_oom and models.device.type == "cuda":
                        logger.warning("CUDA OOM on %s — retrying on CPU", file_name)
                        torch.cuda.empty_cache()
                        cpu_models = load_models(cfg, torch.device("cpu"))
                        instances = segment_instances(cpu_models.sam_predictor, image_bgr, dets, cfg)
                    else:
                        raise

            rep.mask_count = len(instances)
            rep.classes = sorted({i.class_name for i in instances})
            if instances:
                rep.mean_dino_conf = float(np.mean([i.confidence for i in instances]))
                rep.mean_sam_score = float(np.mean([i.sam_score for i in instances]))
            rep.status = "ok" if instances else "no_detections"

            inst_dicts = _instances_to_dicts(instances, cfg)
            copy_image_to_output(img_path, cfg.images_output_directory)

            if cfg.save_previews and instances:
                save_preview(
                    image_bgr, instances,
                    cfg.previews_directory / f"{img_path.stem}_preview.jpg",
                    save_comparison=cfg.save_comparison_previews,
                )
            if cfg.save_individual_masks and instances:
                save_individual_masks(instances, cfg.masks_directory, img_path.stem)

            image_records.append({
                "image_id": image_id,
                "file_name": file_name,
                "width": w,
                "height": h,
                "status": rep.status,
                "instances": inst_dicts,
            })

        except Exception as exc:
            rep.status = "failed"
            rep.failure_reason = str(exc)
            append_failure_log(failure_log, file_name, str(exc))
            logger.exception("Failed on %s: %s", file_name, exc)
            image_records.append({
                "image_id": image_id, "file_name": file_name,
                "width": rep.width, "height": rep.height,
                "status": "failed", "instances": [],
            })

        rep.processing_seconds = time.perf_counter() - t0
        reports.append(rep)
        completed.add(file_name)
        save_progress(cfg.progress_file, completed)

    elapsed = time.perf_counter() - start

    if "coco" in cfg.export_formats:
        export_coco(image_records, cfg)
    if "yolo_segmentation" in cfg.export_formats:
        export_yolo_seg(image_records, cfg)
    if "voc" in cfg.export_formats:
        export_voc(image_records, cfg)

    return write_reports(cfg, reports, image_records, elapsed_seconds=elapsed, env_report=env_report or {})
