"""Hardware and dependency validation."""

from __future__ import annotations

import importlib.util
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from auto_annotation.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device: torch.device
    device_name: str
    cuda_available: bool
    gpu_memory_gb: float | None


def select_device(preference: str = "auto") -> DeviceInfo:
    cuda_ok = torch.cuda.is_available()
    if preference == "cpu":
        dev = torch.device("cpu")
    elif preference == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA requested but not available. Use --device cpu or install CUDA PyTorch.")
        dev = torch.device("cuda")
    else:
        dev = torch.device("cuda" if cuda_ok else "cpu")

    name = "CPU"
    mem_gb: float | None = None
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem_gb = round(props.total_memory / (1024**3), 2)
    elif preference == "auto":
        logger.warning("CUDA not available — using CPU. Processing will be considerably slower.")

    return DeviceInfo(device=dev, device_name=name, cuda_available=cuda_ok, gpu_memory_gb=mem_gb)


def validate_environment(cfg: AppConfig, *, require_models: bool = True, require_input: bool = True) -> dict:
    """Run startup checks; raise RuntimeError on hard failures."""
    report: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    for pkg in ("cv2", "numpy", "supervision", "PIL", "yaml"):
        if importlib.util.find_spec(pkg if pkg != "cv2" else "cv2") is None:
            raise RuntimeError(f"Required package not installed: {pkg}")

    if importlib.util.find_spec("groundingdino") is None:
        raise RuntimeError(
            "groundingdino not installed. From project root run:\n"
            "  pip install -e GroundingDINO"
        )
    if importlib.util.find_spec("segment_anything") is None:
        raise RuntimeError(
            "segment_anything not installed. From project root run:\n"
            "  pip install -e ."
        )

    import supervision as sv
    report["supervision_version"] = sv.__version__

    from auto_annotation.config import validate_config
    errors = validate_config(cfg, require_models=require_models, require_input=require_input)
    if errors:
        raise RuntimeError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    dev = select_device(cfg.device)
    report["device"] = str(dev.device)
    report["device_name"] = dev.device_name
    report["gpu_memory_gb"] = dev.gpu_memory_gb

    for ckpt in (cfg.gdino_checkpoint, cfg.sam_checkpoint):
        if ckpt.is_file() and ckpt.stat().st_size < 1024:
            raise RuntimeError(f"Checkpoint appears corrupt (too small): {ckpt}")

    logger.info(
        "Environment OK | Python %s | torch %s | device %s (%s)",
        report["python_version"], report["torch_version"], report["device"], report["device_name"],
    )
    return report
