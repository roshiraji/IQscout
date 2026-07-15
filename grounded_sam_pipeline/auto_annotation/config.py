"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # grounded_sam_pipeline/
REPO_ROOT = PROJECT_ROOT.parent  # ScoutIQ repo root (shared weights, GroundingDINO)


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p).resolve()


@dataclass
class AppConfig:
    project_root: Path
    images_directory: Path
    extensions: list[str]
    output_root: Path
    previews_directory: Path
    masks_directory: Path
    annotations_directory: Path
    reports_directory: Path
    gdino_config: Path
    gdino_checkpoint: Path
    sam_model_type: str
    sam_checkpoint: Path
    classes: list[str]
    box_threshold: float
    text_threshold: float
    prompt_template: str
    prompt_overrides: dict[str, str]
    nms_iou_threshold: float
    minimum_box_area_pixels: int
    minimum_image_area_percentage: float
    maximum_image_area_percentage: float
    multimask_output: bool
    choose_highest_score: bool
    save_individual_masks: bool
    export_formats: list[str]
    save_previews: bool
    save_comparison_previews: bool
    save_summary_csv: bool
    save_summary_json: bool
    polygon_approximation_epsilon: float
    min_contour_area_pixels: int
    device: str
    retry_on_cpu_after_oom: bool
    log_level: str

    @property
    def images_output_directory(self) -> Path:
        return self.output_root / "images"

    @property
    def coco_directory(self) -> Path:
        return self.annotations_directory / "coco"

    @property
    def yolo_seg_directory(self) -> Path:
        return self.annotations_directory / "yolo_seg"

    @property
    def voc_directory(self) -> Path:
        return self.annotations_directory / "voc"

    @property
    def progress_file(self) -> Path:
        return self.reports_directory / "progress.json"


def load_config(config_path: Path | str, project_root: Path | None = None) -> AppConfig:
    root = project_root or PROJECT_ROOT
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    inp = raw["input"]
    out = raw["output"]
    models = raw["models"]
    det = raw["detection"]
    seg = raw["segmentation"]
    exp = raw["export"]
    pipe = raw.get("pipeline", {})

    return AppConfig(
        project_root=root,
        images_directory=_resolve(inp["images_directory"], root),
        extensions=[e.lower() for e in inp["extensions"]],
        output_root=_resolve(out["root_directory"], root),
        previews_directory=_resolve(out["previews_directory"], root),
        masks_directory=_resolve(out["masks_directory"], root),
        annotations_directory=_resolve(out["annotations_directory"], root),
        reports_directory=_resolve(out["reports_directory"], root),
        gdino_config=_resolve(models["grounding_dino"]["config_path"], root),
        gdino_checkpoint=_resolve(models["grounding_dino"]["checkpoint_path"], root),
        sam_model_type=models["sam"]["model_type"],
        sam_checkpoint=_resolve(models["sam"]["checkpoint_path"], root),
        classes=list(det["classes"]),
        box_threshold=float(det["box_threshold"]),
        text_threshold=float(det["text_threshold"]),
        prompt_template=str(det.get("prompt_template", "all {class_name}s")),
        prompt_overrides=dict(det.get("prompt_overrides", {})),
        nms_iou_threshold=float(det["nms_iou_threshold"]),
        minimum_box_area_pixels=int(det["minimum_box_area_pixels"]),
        minimum_image_area_percentage=float(det["minimum_image_area_percentage"]),
        maximum_image_area_percentage=float(det["maximum_image_area_percentage"]),
        multimask_output=bool(seg.get("multimask_output", True)),
        choose_highest_score=bool(seg.get("choose_highest_score", True)),
        save_individual_masks=bool(seg.get("save_individual_masks", True)),
        export_formats=list(exp.get("formats", ["coco", "yolo_segmentation"])),
        save_previews=bool(exp.get("save_previews", True)),
        save_comparison_previews=bool(exp.get("save_comparison_previews", False)),
        save_summary_csv=bool(exp.get("save_summary_csv", True)),
        save_summary_json=bool(exp.get("save_summary_json", True)),
        polygon_approximation_epsilon=float(exp.get("polygon_approximation_epsilon", 2.0)),
        min_contour_area_pixels=int(exp.get("min_contour_area_pixels", 10)),
        device=str(pipe.get("device", "auto")),
        retry_on_cpu_after_oom=bool(pipe.get("retry_on_cpu_after_oom", True)),
        log_level=str(pipe.get("log_level", "INFO")),
    )


def validate_config(cfg: AppConfig, *, require_models: bool = True, require_input: bool = True) -> list[str]:
    """Return list of validation errors (empty = OK)."""
    errors: list[str] = []
    if require_input and not cfg.images_directory.is_dir():
        errors.append(f"Input directory missing: {cfg.images_directory}")
    if require_models:
        if not cfg.gdino_config.is_file():
            errors.append(f"Grounding DINO config missing: {cfg.gdino_config}")
        if not cfg.gdino_checkpoint.is_file():
            errors.append(f"Grounding DINO checkpoint missing: {cfg.gdino_checkpoint}")
        if not cfg.sam_checkpoint.is_file():
            errors.append(f"SAM checkpoint missing: {cfg.sam_checkpoint}")
        expected = {
            "vit_h": "vit_h",
            "vit_l": "vit_l",
            "vit_b": "vit_b",
        }
        if cfg.sam_model_type not in expected:
            errors.append(f"Unsupported SAM model_type: {cfg.sam_model_type}")
    if not cfg.classes:
        errors.append("At least one detection class is required.")
    for d in [cfg.output_root, cfg.previews_directory, cfg.annotations_directory, cfg.reports_directory]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink()
        except OSError as exc:
            errors.append(f"Cannot write to {d}: {exc}")
    return errors


def apply_cli_overrides(
    cfg: AppConfig,
    *,
    input_dir: str | None = None,
    classes: list[str] | None = None,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    device: str | None = None,
    save_previews: bool | None = None,
) -> AppConfig:
    if input_dir:
        cfg.images_directory = _resolve(input_dir, cfg.project_root)
    if classes:
        cfg.classes = classes
    if box_threshold is not None:
        cfg.box_threshold = box_threshold
    if text_threshold is not None:
        cfg.text_threshold = text_threshold
    if device:
        cfg.device = device
    if save_previews is not None:
        cfg.save_previews = save_previews
    return cfg
