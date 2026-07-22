"""Training loop for lightweight yolo11n-seg → best_player_seg.pt."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

# Windows PyTorch wheels often lack libuv; Ultralytics DDP needs this.
if sys.platform == "win32":
    os.environ.setdefault("USE_LIBUV", "0")

_TCPSTORE_PATCH = """
def _scoutiq_patch_tcpstore():
    import sys
    import torch.distributed as dist
    _Orig = dist.TCPStore
    def _TCPStore(*args, **kwargs):
        kwargs["use_libuv"] = False
        return _Orig(*args, **kwargs)
    dist.TCPStore = _TCPStore
    for _name in (
        "torch.distributed.rendezvous",
        "torch.distributed.elastic.rendezvous.static_tcp_rendezvous",
    ):
        _mod = sys.modules.get(_name)
        if _mod is not None and hasattr(_mod, "TCPStore"):
            _mod.TCPStore = _TCPStore
_scoutiq_patch_tcpstore()
"""


def _install_windows_ddp_patches() -> None:
    """Make Ultralytics multi-GPU DDP work on Windows CUDA wheels without libuv."""
    if sys.platform != "win32":
        return
    from ultralytics.utils import dist as ultra_dist

    if getattr(ultra_dist, "_scoutiq_ddp_patched", False):
        return

    bootstrap = Path(__file__).resolve().parent / "ddp_win_bootstrap.py"
    _orig_generate_ddp_file = ultra_dist.generate_ddp_file
    _orig_generate_ddp_command = ultra_dist.generate_ddp_command

    def generate_ddp_file(trainer):  # type: ignore[no-untyped-def]
        path = _orig_generate_ddp_file(trainer)
        text = Path(path).read_text(encoding="utf-8")
        if "_scoutiq_patch_tcpstore" not in text:
            Path(path).write_text(_TCPSTORE_PATCH + text, encoding="utf-8")
        return path

    def generate_ddp_command(trainer):  # type: ignore[no-untyped-def]
        cmd, file = _orig_generate_ddp_command(trainer)
        # Replace ``python -m torch.distributed.run`` with our libuv-safe bootstrap.
        if len(cmd) >= 3 and cmd[1] == "-m" and "torch.distributed" in cmd[2]:
            cmd = [cmd[0], str(bootstrap), *cmd[3:]]
        return cmd, file

    ultra_dist.generate_ddp_file = generate_ddp_file  # type: ignore[assignment]
    ultra_dist.generate_ddp_command = generate_ddp_command  # type: ignore[assignment]
    # Trainer imports these names at module load — patch there too.
    try:
        from ultralytics.engine import trainer as ultra_trainer

        ultra_trainer.generate_ddp_command = generate_ddp_command  # type: ignore[attr-defined]
        ultra_trainer.ddp_cleanup = ultra_dist.ddp_cleanup
    except Exception:
        pass
    ultra_dist._scoutiq_ddp_patched = True  # type: ignore[attr-defined]
    logger.info("Installed Windows DDP libuv workaround → %s", bootstrap)


def resolve_train_device(cfg_device: Any = None) -> int | str | list[int]:
    """Use [0,1] when two GPUs exist unless config overrides."""
    if cfg_device is not None:
        return cfg_device
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if n >= 2:
        return [0, 1]
    return 0


def train_player_seg(
    *,
    data_yaml: Path,
    runs_root: Path,
    model_name: str = "yolo11n-seg.pt",
    epochs: int = 60,
    imgsz: int = 960,
    batch: int = 16,
    device: Any = None,
    resume_weights: Path | None = None,
    project_name: str = "player_seg",
    workers: int = 8,
    export_name: str = "best_player_seg.pt",
) -> Path:
    from ultralytics import YOLO

    runs_root.mkdir(parents=True, exist_ok=True)
    device = resolve_train_device(device)
    weights = resume_weights if resume_weights and resume_weights.is_file() else model_name
    logger.info("Training %s | device=%s | epochs=%d | batch=%d | imgsz=%d | workers=%d",
                weights, device, epochs, batch, imgsz, workers)

    if sys.platform == "win32":
        os.environ["USE_LIBUV"] = "0"
        if isinstance(device, (list, tuple)) and len(device) > 1:
            _install_windows_ddp_patches()

    model = YOLO(str(weights))
    try:
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(runs_root),
            name=project_name,
            exist_ok=True,
            verbose=True,
        )
    except Exception as exc:
        # Fallback: if multi-GPU DDP fails on Windows, use GPU 0 with half batch.
        if isinstance(device, list) and len(device) > 1:
            logger.warning("Multi-GPU train failed (%s). Falling back to single GPU 0.", exc)
            device = int(device[0])
            model = YOLO(str(weights))
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=max(batch // 2, 8),
                device=device,
                workers=workers,
                project=str(runs_root),
                name=project_name,
                exist_ok=True,
                verbose=True,
            )
        else:
            raise
    best = runs_root / project_name / "weights" / "best.pt"
    if not best.is_file():
        # Ultralytics sometimes nests runs.
        candidates = list(runs_root.glob("**/weights/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Training did not produce weights under {runs_root}")
        best = max(candidates, key=lambda p: p.stat().st_mtime)

    export = runs_root / export_name
    shutil.copy2(best, export)
    last_name = export_name.replace("best_", "last_")
    lasts = sorted(runs_root.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    if lasts:
        shutil.copy2(lasts[-1], runs_root / last_name)
    logger.info("Promoted weights → %s", export)
    return export


def validate_player_seg(weights: Path, data_yaml: Path, imgsz: int = 960, device: Any = None) -> dict[str, float]:
    from ultralytics import YOLO

    device = resolve_train_device(device)
    # Validation on multi-GPU can be flaky — prefer single GPU.
    if isinstance(device, list):
        device = device[0]
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    out: dict[str, float] = {}
    try:
        box = metrics.box
        out["box_map50"] = float(getattr(box, "map50", 0.0) or 0.0)
        out["box_map"] = float(getattr(box, "map", 0.0) or 0.0)
        out["box_precision"] = float(getattr(box, "mp", 0.0) or 0.0)
        out["box_recall"] = float(getattr(box, "mr", 0.0) or 0.0)
    except Exception:
        pass
    try:
        seg = metrics.seg
        out["mask_map50"] = float(getattr(seg, "map50", 0.0) or 0.0)
        out["mask_map"] = float(getattr(seg, "map", 0.0) or 0.0)
    except Exception:
        out.setdefault("mask_map50", 0.0)
    return out


def run_training_from_config(cfg: dict, scoutiq_root: Path) -> Path:
    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (scoutiq_root / p).resolve()

    train_cfg = cfg.get("training") or {}
    out_cfg = cfg.get("output") or {}
    data_yaml = _resolve(cfg["dataset"]["data_yaml"])
    runs_root = _resolve(out_cfg.get("runs_root", "outputs/training_loop"))
    export_name = str(out_cfg.get("weights_name", "best_player_seg.pt"))
    epochs = int(train_cfg.get("epochs", 60))
    imgsz = int(train_cfg.get("imgsz", 960))
    batch = int(train_cfg.get("batch", 16))
    workers = int(train_cfg.get("workers", 8))
    model_name = str(train_cfg.get("model", "yolo11n-seg.pt"))
    device = train_cfg.get("device")  # None → auto dual-GPU
    if isinstance(device, str) and "," in device:
        device = [int(x.strip()) for x in device.split(",")]

    rounds = int(train_cfg.get("max_rounds", 1))
    targets = cfg.get("targets") or {}
    min_box = float(targets.get("min_box_map50", 0.45))
    min_mask = float(targets.get("min_mask_map50", 0.35))

    # Log GPU inventory up front.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %d: %s", i, torch.cuda.get_device_name(i))
    resolved = resolve_train_device(device)
    logger.info(
        "Training will use device=%s batch=%d workers=%d export=%s",
        resolved,
        batch,
        workers,
        export_name,
    )

    history: list[dict] = []
    best_export: Path | None = None
    resume: Path | None = None

    for round_id in range(1, rounds + 1):
        logger.info("=== Seg train round %d/%d ===", round_id, rounds)
        weights = train_player_seg(
            data_yaml=data_yaml,
            runs_root=runs_root,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            resume_weights=resume,
            project_name=f"round_{round_id:03d}",
            workers=workers,
            export_name=export_name,
        )
        metrics = validate_player_seg(weights, data_yaml, imgsz=imgsz, device=device)
        passed = metrics.get("box_map50", 0.0) >= min_box and metrics.get("mask_map50", 0.0) >= min_mask
        record = {"round": round_id, "weights": str(weights), "metrics": metrics, "passed": passed}
        history.append(record)
        (runs_root / f"round_{round_id:03d}_report.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        best_export = weights
        if passed:
            logger.info("Targets met after round %d", round_id)
            break
        resume = weights

    summary = {"history": history, "best_weights": str(best_export) if best_export else None}
    (runs_root / "loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_export or runs_root / export_name
