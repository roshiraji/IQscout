"""Grounding DINO text-grounding for the fusion pipeline (LocateAnything substitute)."""

from __future__ import annotations

from src.ml_warnings import configure_ml_runtime

configure_ml_runtime()

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SCOUTIQ_ROOT.parent

DEFAULT_CONFIG = REPO_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "weights/groundingdino_swint_ogc.pth"

# Prompts passed to Grounding DINO (caption = comma-joined list).
DEFAULT_CLASS_PROMPTS: list[str] = [
    "all basketball players on the court",
    "all basketball referees",
    "the basketball",
    "basketball hoop",
]
DEFAULT_CLASS_NAMES: list[str] = ["player", "referee", "ball", "hoop"]

_model_cache: object | None = None
_device: str | None = None


@dataclass
class GroundingDinoSettings:
    config_path: Path = DEFAULT_CONFIG
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    class_prompts: list[str] | None = None
    class_names: list[str] | None = None


def grounding_dino_available(settings: GroundingDinoSettings | None = None) -> bool:
    cfg = settings or GroundingDinoSettings()
    return cfg.config_path.is_file() and cfg.checkpoint_path.is_file()


def _get_device() -> str:
    global _device
    if _device is None:
        import torch
        if torch.cuda.is_available():
            # Prefer cuda:0; dual-GPU setups keep GDINO on 0 and SAM2 on 1.
            _device = "cuda:0"
        else:
            _device = "cpu"
    return _device


def set_grounding_dino_device(device: str) -> None:
    """Force device for next model load (clears cache if device changes)."""
    global _device, _model_cache
    if _device != device:
        _model_cache = None
    _device = device


def _get_model(settings: GroundingDinoSettings):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not grounding_dino_available(settings):
        raise FileNotFoundError(
            f"Grounding DINO weights missing. Expected:\n"
            f"  {settings.config_path}\n"
            f"  {settings.checkpoint_path}"
        )
    from groundingdino.util.inference import Model

    device = _get_device()
    logger.info("Loading Grounding DINO from %s (device=%s)", settings.checkpoint_path, device)
    _model_cache = Model(
        model_config_path=str(settings.config_path),
        model_checkpoint_path=str(settings.checkpoint_path),
        device=device,
    )
    logger.info("Grounding DINO ready on %s", device)
    return _model_cache


def detect_with_grounding_dino(
    image_bgr: np.ndarray,
    *,
    frame_id: int,
    timestamp: float,
    image_name: str,
    settings: GroundingDinoSettings | None = None,
) -> list[dict]:
    """Run Grounding DINO and return detections in fusion pipeline format."""
    cfg = settings or GroundingDinoSettings()
    prompts = cfg.class_prompts or DEFAULT_CLASS_PROMPTS
    names = cfg.class_names or DEFAULT_CLASS_NAMES
    if len(prompts) != len(names):
        raise ValueError("class_prompts and class_names must have the same length")

    model = _get_model(cfg)
    h, w = image_bgr.shape[:2]

    detections = model.predict_with_classes(
        image=image_bgr,
        classes=prompts,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    )

    results: list[dict] = []
    if detections is None or len(detections) == 0:
        return results

    for i in range(len(detections)):
        if detections.class_id is None:
            # Single-class prompts sometimes leave class_id unset; treat as class 0.
            cid = 0 if len(names) == 1 else None
        else:
            raw_cid = detections.class_id[i]
            if raw_cid is None:
                cid = 0 if len(names) == 1 else None
            else:
                cid = int(raw_cid)
        if cid is None or cid < 0 or cid >= len(names):
            continue
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i].tolist()]
        class_name = names[cid]
        results.append({
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "image": image_name,
            "width": w,
            "height": h,
            "class_id": cid,
            "class_name": class_name,
            "canonical_class": class_name,
            "prompt": prompts[cid],
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "source": "grounding_dino",
        })
    return results
