"""
Segment Anything (SAM) helpers for ScoutIQ.

Loads checkpoints from scoutiq/models/sam/ and exposes a small API for
box- or point-prompted masks on BGR images (OpenCV convention).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

SAM_MODEL_TYPES = ("vit_b", "vit_l", "vit_h", "default")
SamModelType = Literal["vit_b", "vit_l", "vit_h", "default"]

_CHECKPOINT_NAMES: dict[str, str] = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
    "default": "sam_vit_h_4b8939.pth",
}

_DEFAULT_MODEL_TYPE: SamModelType = "vit_b"
_SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]
_SAM_DIR = _SCOUTIQ_ROOT / "models" / "sam"


def get_sam_checkpoint_path(model_type: SamModelType = _DEFAULT_MODEL_TYPE) -> Path:
    """Return the expected checkpoint path for a SAM variant."""
    filename = _CHECKPOINT_NAMES.get(model_type, _CHECKPOINT_NAMES["vit_b"])
    return _SAM_DIR / filename


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_sam_predictor(model_type: SamModelType = _DEFAULT_MODEL_TYPE) -> SamPredictor:
    """Load SAM once and reuse the predictor (cached per model type)."""
    registry_key = "default" if model_type == "default" else model_type
    checkpoint = get_sam_checkpoint_path(model_type)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            f"Download it with:\n"
            f'  scoutiq\\scripts\\download_sam_checkpoint.bat {model_type}'
        )

    device = _resolve_device()
    sam = sam_model_registry[registry_key](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def predict_mask_from_box(
    image_bgr: np.ndarray,
    box_xyxy: Sequence[float],
    *,
    model_type: SamModelType = _DEFAULT_MODEL_TYPE,
) -> np.ndarray:
    """
    Segment the object inside a pixel box [x1, y1, x2, y2].

    Returns a boolean mask with the same height and width as image_bgr.
    """
    predictor = load_sam_predictor(model_type)
    image_rgb = image_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def predict_mask_from_point(
    image_bgr: np.ndarray,
    point_xy: Tuple[float, float],
    *,
    foreground: bool = True,
    model_type: SamModelType = _DEFAULT_MODEL_TYPE,
) -> np.ndarray:
    """Segment from a single (x, y) click prompt."""
    predictor = load_sam_predictor(model_type)
    image_rgb = image_bgr[:, :, ::-1]
    predictor.set_image(image_rgb)

    point_coords = np.array([point_xy], dtype=np.float32)
    point_labels = np.array([1 if foreground else 0], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)
