"""
Central YOLO model configuration for the ScoutIQ vision pipeline.

Supports Ultralytics YOLO26 (default) and YOLO11 detection models.
All detection scripts import from here so model and inference settings
can be changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

DEFAULT_YOLO_FAMILY = "YOLO26"
DEFAULT_YOLO_MODEL = "yolo26n.pt"

AVAILABLE_YOLO_FAMILIES = ("YOLO26", "YOLO11")


@dataclass(frozen=True)
class YoloModelOption:
    """One selectable YOLO detection model with UI-friendly metadata."""

    family: str
    weights: str
    size_label: str
    short_label: str
    description: str
    speed_tier: str
    accuracy_tier: str
    params_m: float
    vram_hint: str


def _build_size_options(
    family: str,
    *,
    params: tuple[float, float, float, float, float],
    descriptions: tuple[str, str, str, str, str],
    speed_tiers: tuple[str, str, str, str, str] = ("Fastest", "Fast", "Moderate", "Slow", "Slowest"),
    accuracy_tiers: tuple[str, str, str, str, str] = (
        "Baseline",
        "Good",
        "Better",
        "High",
        "Highest",
    ),
    vram_hints: tuple[str, str, str, str, str] = ("~2 GB", "~3 GB", "~4 GB", "~6 GB", "~8+ GB"),
) -> tuple[YoloModelOption, ...]:
    """Build n/s/m/l/x catalog entries for one YOLO family."""
    sizes = ("Nano", "Small", "Medium", "Large", "Extra Large")
    shorts = ("n", "s", "m", "l", "x")
    return tuple(
        YoloModelOption(
            family=family,
            weights=f"{family.lower()}{short}.pt",
            size_label=size,
            short_label=short,
            description=desc,
            speed_tier=speed,
            accuracy_tier=accuracy,
            params_m=param,
            vram_hint=vram,
        )
        for size, short, desc, speed, accuracy, param, vram in zip(
            sizes, shorts, descriptions, speed_tiers, accuracy_tiers, params, vram_hints
        )
    )


YOLO26_MODEL_CATALOG: tuple[YoloModelOption, ...] = _build_size_options(
    "YOLO26",
    params=(2.4, 9.5, 20.4, 24.8, 55.7),
    descriptions=(
        "Fastest inference and lowest VRAM. Good default for quick iteration.",
        "Balanced speed and accuracy for real-time GPU pipelines.",
        "Stronger distant-player recall; moderate runtime on RTX 3070 Ti.",
        "High accuracy for crowded courts; slower per-frame inference.",
        "Maximum detection quality; best for offline review and tracking prep.",
    ),
)

YOLO11_MODEL_CATALOG: tuple[YoloModelOption, ...] = _build_size_options(
    "YOLO11",
    params=(2.6, 9.4, 20.1, 25.3, 56.9),
    descriptions=(
        "Previous-gen nano model. Fast baseline when comparing against YOLO26.",
        "Solid real-time balance; supports test-time augmentation in quality mode.",
        "Strong mid-size option for busier frames and partial occlusions.",
        "High-accuracy large model for detailed player bounding boxes.",
        "Top-tier YOLO11 accuracy for offline labeling and QA review.",
    ),
)

YOLO_MODEL_CATALOG: tuple[YoloModelOption, ...] = YOLO26_MODEL_CATALOG + YOLO11_MODEL_CATALOG

# Detection models available in the pipeline UI and CLI
AVAILABLE_YOLO_MODELS = [option.weights for option in YOLO_MODEL_CATALOG]

# COCO person class used for basketball player detection in the MVP
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

# Default pipeline mode: favor detection quality over raw speed
QUALITY_MODE = True

# Minimum confidence for inference floor and low-confidence review logging
LOW_CONF_REVIEW_MIN = 0.15

# Allowed inference resolutions (quality mode may use 1280 or 1920)
AVAILABLE_IMGSZ = [640, 1280, 1920]

# Backward-compatible alias (speed-mode primary threshold)
CONFIDENCE_THRESHOLD = 0.35

# Legacy constant — prefer get_model_option().family
YOLO_FAMILY = DEFAULT_YOLO_FAMILY


@dataclass(frozen=True)
class InferenceProfile:
    """YOLO inference settings for one pipeline run."""

    quality_mode: bool
    imgsz: int
    conf: float
    augment: bool
    iou: float

    @property
    def mode_label(self) -> str:
        return "quality" if self.quality_mode else "speed"


def get_inference_profile(
    quality_mode: bool | None = None,
    imgsz: int | None = None,
) -> InferenceProfile:
    """
    Build inference settings from quality/speed mode.

    Quality mode (default): higher resolution, lower conf, TTA, stricter NMS.
    Speed mode: smaller imgsz, higher conf, no augmentation.
    """
    use_quality = QUALITY_MODE if quality_mode is None else quality_mode

    if use_quality:
        resolved_imgsz = imgsz if imgsz is not None else 1280
        if resolved_imgsz not in AVAILABLE_IMGSZ:
            raise ValueError(
                f"Unsupported imgsz {resolved_imgsz}. Choose from: {AVAILABLE_IMGSZ}"
            )
        if resolved_imgsz == 640:
            resolved_imgsz = 1280
        return InferenceProfile(
            quality_mode=True,
            imgsz=resolved_imgsz,
            conf=0.25,
            augment=True,
            iou=0.45,
        )

    return InferenceProfile(
        quality_mode=False,
        imgsz=640,
        conf=0.35,
        augment=False,
        iou=0.50,
    )


def get_models_for_family(family: str) -> list[YoloModelOption]:
    """Return all model options for a YOLO family (YOLO26 or YOLO11)."""
    if family not in AVAILABLE_YOLO_FAMILIES:
        raise ValueError(
            f"Unsupported family '{family}'. Choose from: {', '.join(AVAILABLE_YOLO_FAMILIES)}"
        )
    return [option for option in YOLO_MODEL_CATALOG if option.family == family]


def get_default_model_for_family(family: str) -> str:
    """Return the nano weights filename for a YOLO family."""
    return get_models_for_family(family)[0].weights


def load_yolo_model(model_name: str | None = None) -> YOLO:
    """
    Load a pretrained Ultralytics YOLO detection model.

    Downloads weights on first run (e.g. yolo26n.pt or yolo11n.pt).
    """
    weights = model_name or DEFAULT_YOLO_MODEL
    if weights not in AVAILABLE_YOLO_MODELS:
        raise ValueError(
            f"Unsupported model '{weights}'. Choose from: {', '.join(AVAILABLE_YOLO_MODELS)}"
        )
    return YOLO(weights)


def get_model_option(model_name: str | None = None) -> YoloModelOption:
    """Return catalog metadata for a model weights filename."""
    weights = model_name or DEFAULT_YOLO_MODEL
    for option in YOLO_MODEL_CATALOG:
        if option.weights == weights:
            return option
    raise ValueError(
        f"Unsupported model '{weights}'. Choose from: {', '.join(AVAILABLE_YOLO_MODELS)}"
    )


def model_select_label(model_name: str) -> str:
    """Friendly label for Streamlit/CLI model pickers."""
    option = get_model_option(model_name)
    return f"{option.size_label} ({option.weights})"


def model_display_name(model_name: str | None = None) -> str:
    """Return a readable label for logs and UI."""
    option = get_model_option(model_name)
    return f"{option.family} {option.size_label} ({option.weights})"


def model_summary(model_name: str | None = None) -> str:
    """One-line model description for UI captions."""
    option = get_model_option(model_name)
    return (
        f"{option.description} Speed: {option.speed_tier}. "
        f"Accuracy: {option.accuracy_tier}. Params: {option.params_m}M. "
        f"VRAM hint: {option.vram_hint}."
    )
