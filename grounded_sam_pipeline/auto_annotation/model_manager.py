"""Model loading and optional weight download."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from auto_annotation.config import AppConfig

logger = logging.getLogger(__name__)

WEIGHT_URLS = {
    "groundingdino_swint_ogc.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/"
        "groundingdino_swint_ogc.pth"
    ),
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


@dataclass
class ModelBundle:
    grounding_dino: object
    sam_predictor: object
    device: torch.device


def download_checkpoint(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s (this may take several minutes)", url, dest)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 1024:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed or file corrupt: {dest}")


def ensure_checkpoints(cfg: AppConfig, *, allow_download: bool = False) -> None:
    missing: list[tuple[Path, str]] = []
    for path in (cfg.gdino_checkpoint, cfg.sam_checkpoint):
        if not path.is_file():
            url = WEIGHT_URLS.get(path.name)
            if url:
                missing.append((path, url))
            else:
                raise FileNotFoundError(f"Missing checkpoint with no known URL: {path}")

    if not missing:
        return
    if not allow_download:
        lines = "\n".join(f"  {p}" for p, _ in missing)
        raise FileNotFoundError(
            f"Missing model checkpoints:\n{lines}\n"
            "Download manually or re-run with --download-weights"
        )
    for path, url in missing:
        download_checkpoint(url, path)


def load_models(cfg: AppConfig, device: torch.device) -> ModelBundle:
    from groundingdino.util.inference import Model
    from segment_anything import SamPredictor, sam_model_registry

    logger.info("Loading Grounding DINO from %s", cfg.gdino_checkpoint)
    gdino = Model(
        model_config_path=str(cfg.gdino_config),
        model_checkpoint_path=str(cfg.gdino_checkpoint),
        device=str(device),
    )

    logger.info("Loading SAM %s from %s", cfg.sam_model_type, cfg.sam_checkpoint)
    sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return ModelBundle(grounding_dino=gdino, sam_predictor=predictor, device=device)
