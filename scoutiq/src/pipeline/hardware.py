"""Probe local hardware and pull max usable resources for ScoutIQ inference.

``device: auto`` (and related auto knobs) resolve at config load time so Analyze,
Confirm, and Clean Isolate run as fast as the user's GPU / CPU allow.

``performance.pull: hard`` (default) uses nearly all cores, FP16 on CUDA,
aggressive cuDNN, and larger isolate working sizes when VRAM allows.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def probe_hardware() -> dict[str, Any]:
    """One-shot inventory of CUDA / VRAM / CPU cores."""
    cpu_count = os.cpu_count() or 4
    info: dict[str, Any] = {
        "cuda": False,
        "device_name": None,
        "vram_gb": 0.0,
        "cuda_device_index": 0,
        "cpu_count": int(cpu_count),
        "torch_available": False,
    }
    try:
        import torch

        info["torch_available"] = True
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "cuda": True,
                    "device_name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "cuda_device_index": idx,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hardware probe (torch) soft-failed: %s", exc)
    return info


def resolve_device(requested: Any, *, prefer_gpu: bool = True) -> str:
    """Map config device hints to an Ultralytics / torch device string."""
    hw = probe_hardware()
    raw = "auto" if requested is None else str(requested).strip().lower()
    if raw in {"", "auto"}:
        if prefer_gpu and hw["cuda"]:
            return str(hw["cuda_device_index"])
        return "cpu"
    if raw in {"gpu", "cuda"}:
        return str(hw["cuda_device_index"]) if hw["cuda"] else "cpu"
    if raw.isdigit():
        return raw if hw["cuda"] else "cpu"
    if raw.startswith("cuda:"):
        return raw if hw["cuda"] else "cpu"
    return raw


def _pull_level(perf: dict[str, Any]) -> str:
    raw = str(perf.get("pull", "hard") or "hard").strip().lower()
    if raw in {"balanced", "safe", "normal"}:
        return "balanced"
    return "hard"


def _resolve_threads(requested: Any, cpu_count: int, *, pull: str) -> int:
    if requested is None or str(requested).strip().lower() in {"", "auto"}:
        if pull == "hard":
            # Use every logical core for Analyze; UI stays responsive via separate process.
            return max(2, int(cpu_count))
        return max(2, int(cpu_count) - 1)
    try:
        return max(1, int(requested))
    except (TypeError, ValueError):
        return max(2, int(cpu_count) - (0 if pull == "hard" else 1))


def apply_runtime_threads(
    torch_threads: int,
    opencv_threads: int,
    *,
    pull: str = "hard",
) -> None:
    """Pin torch + OpenCV thread pools to use available CPU bandwidth."""
    try:
        import torch

        torch.set_num_threads(int(torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            if pull == "hard":
                interop = max(2, min(8, int(torch_threads) // 2))
            else:
                interop = max(1, min(4, int(torch_threads) // 2))
            try:
                torch.set_num_interop_threads(interop)
            except RuntimeError:
                pass
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            if pull == "hard":
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Torch thread setup soft-failed: %s", exc)

    try:
        import cv2

        cv2.setNumThreads(int(opencv_threads))
        if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenCV thread setup soft-failed: %s", exc)

    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(torch_threads)
    if pull == "hard":
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


def boost_process_priority(*, pull: str = "hard") -> bool:
    """Raise Analyze worker priority so the OS schedules GPU/CPU work sooner."""
    if pull != "hard":
        return False
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentProcess()
            # HIGH_PRIORITY_CLASS = 0x00000080
            return bool(kernel32.SetPriorityClass(handle, 0x00000080))
        os.nice(-5)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Process priority boost soft-failed: %s", exc)
        return False


def _auto_half(hw: dict[str, Any], device: str, requested: Any, *, pull: str) -> bool:
    if requested is not None and str(requested).strip().lower() not in {"", "auto"}:
        return str(requested).strip().lower() in {"1", "true", "yes", "on"}
    if str(device).lower() == "cpu" or not hw["cuda"]:
        return False
    return pull == "hard" or float(hw.get("vram_gb") or 0) >= 6


def _auto_imgsz(hw: dict[str, Any], device: str, requested: Any, *, pull: str) -> int:
    if requested is not None and str(requested).strip().lower() not in {"", "auto"}:
        try:
            return max(320, int(requested))
        except (TypeError, ValueError):
            pass
    if str(device).lower() == "cpu" or not hw["cuda"]:
        return 640
    vram = float(hw.get("vram_gb") or 0)
    if pull == "hard":
        if vram >= 16:
            return 1280
        if vram >= 10:
            return 960
        if vram >= 7:
            return 800
        if vram >= 4:
            return 704
        return 640
    return 640


def _auto_ball_every_n(hw: dict[str, Any], device: str) -> int:
    if str(device).lower() == "cpu" or not hw["cuda"]:
        return 2
    return 1


def _auto_hoop_every_n(hw: dict[str, Any], device: str, *, pull: str) -> int:
    if str(device).lower() == "cpu" or not hw["cuda"]:
        return 20
    return 8 if pull == "hard" else 15


def _auto_isolate_mode(hw: dict[str, Any]) -> str:
    if not hw["cuda"]:
        return "temporal"
    try:
        import simple_lama_inpainting  # noqa: F401

        return "hybrid"
    except Exception:  # noqa: BLE001
        return "temporal"


def _auto_ai_working_side(
    hw: dict[str, Any],
    *,
    pull: str,
    prefer: str = "speed",
) -> int | None:
    """Pick LaMa crop size.

    ``prefer=speed`` (default for isolate) uses a smaller working side so each
    frame finishes faster while still saturating the GPU. ``prefer=quality``
    ups the tile size when VRAM allows.
    """
    vram = float(hw.get("vram_gb") or 0)
    prefer = str(prefer or "speed").strip().lower()
    if prefer in {"quality", "hq", "high"}:
        if pull == "hard":
            if vram >= 16:
                return 1280
            if vram >= 10:
                return 1024
            if vram >= 7:
                return 896
            if vram >= 4:
                return 704
            return 576
        if vram >= 16:
            return 1024
        if vram >= 10:
            return 896
        if vram >= 7:
            return 768
        if vram >= 4:
            return 640
        return 512
    # Throughput path — smaller crops = more frames/sec on the same GPU.
    if pull == "hard":
        if vram >= 16:
            return 768
        if vram >= 10:
            return 704
        if vram >= 7:
            return 640
        if vram >= 4:
            return 512
        return 448
    if vram >= 10:
        return 640
    if vram >= 7:
        return 576
    if vram >= 4:
        return 512
    return 448


def _auto_isolate_prep_queue_depth(hw: dict[str, Any], *, pull: str) -> int:
    """How many frames the CPU may prepare ahead of the GPU LaMa call."""
    if pull == "hard":
        # Deep enough that decode+masks stay ahead of LaMa on an 8GB card.
        return max(4, min(12, int(hw.get("cpu_count") or 4) // 2))
    return 3


def _auto_isolate_warp_workers(hw: dict[str, Any], *, pull: str) -> int:
    """CPU workers for parallel neighbor warps during temporal fill."""
    cpus = int(hw.get("cpu_count") or 4)
    if pull == "hard":
        return max(2, min(8, cpus // 2))
    return max(1, min(4, cpus // 3))


def _auto_encode_threads(cpu_count: int, *, pull: str) -> int:
    if pull == "hard":
        return max(2, int(cpu_count))
    return max(2, int(cpu_count) - 1)


def _auto_encode_preset(*, pull: str) -> str:
    return "veryfast" if pull == "hard" else "fast"


def apply_hardware_to_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ``auto`` knobs and apply thread / CUDA runtime settings in-place."""
    out = deepcopy(cfg) if cfg else {}
    hw = probe_hardware()
    perf = dict(out.get("performance") or {})
    mode = str(perf.get("mode", "auto")).strip().lower()
    pull = _pull_level(perf)
    prefer_gpu = mode not in {"cpu", "cpu-only"}

    player_device = resolve_device(out.get("device", "auto"), prefer_gpu=prefer_gpu)
    out["device"] = player_device

    half = _auto_half(hw, player_device, perf.get("half"), pull=pull)
    imgsz = _auto_imgsz(hw, player_device, perf.get("imgsz"), pull=pull)
    encode_threads = _auto_encode_threads(hw["cpu_count"], pull=pull)
    encode_preset = str(perf.get("encode_preset") or _auto_encode_preset(pull=pull))

    ball = dict(out.get("ball") or {})
    ball_device = resolve_device(
        ball.get("device", out.get("device", "auto")),
        prefer_gpu=prefer_gpu,
    )
    ball["device"] = ball_device
    every = ball.get("every_n_frames", "auto")
    if every is None or str(every).strip().lower() in {"", "auto"}:
        ball["every_n_frames"] = _auto_ball_every_n(hw, ball_device)
    else:
        ball["every_n_frames"] = max(1, int(every))

    hoop = dict(ball.get("hoop") or {})
    hoop_every = hoop.get("every_n_frames", "auto")
    if hoop_every is None or str(hoop_every).strip().lower() in {"", "auto"}:
        hoop["every_n_frames"] = _auto_hoop_every_n(hw, ball_device, pull=pull)
    else:
        try:
            hoop["every_n_frames"] = max(1, int(hoop_every))
        except (TypeError, ValueError):
            hoop["every_n_frames"] = _auto_hoop_every_n(hw, ball_device, pull=pull)
    ball["hoop"] = hoop
    out["ball"] = ball

    isolate = dict(out.get("isolate") or {})
    iso_mode = str(isolate.get("mode", "auto")).strip().lower()
    if iso_mode in {"", "auto"}:
        isolate["mode"] = _auto_isolate_mode(hw)
    prefer = str(isolate.get("ai_prefer", "speed") or "speed").strip().lower()
    isolate["ai_prefer"] = prefer if prefer in {"speed", "quality", "hq", "high"} else "speed"
    working = isolate.get("ai_working_side", "auto")
    if working is None or str(working).strip().lower() in {"", "auto"}:
        isolate["ai_working_side"] = _auto_ai_working_side(
            hw, pull=pull, prefer=isolate["ai_prefer"]
        )
    queue_depth = isolate.get("prep_queue_depth", "auto")
    if queue_depth is None or str(queue_depth).strip().lower() in {"", "auto"}:
        isolate["prep_queue_depth"] = _auto_isolate_prep_queue_depth(hw, pull=pull)
    else:
        try:
            isolate["prep_queue_depth"] = max(1, int(queue_depth))
        except (TypeError, ValueError):
            isolate["prep_queue_depth"] = _auto_isolate_prep_queue_depth(hw, pull=pull)
    warp_workers = isolate.get("warp_workers", "auto")
    if warp_workers is None or str(warp_workers).strip().lower() in {"", "auto"}:
        isolate["warp_workers"] = _auto_isolate_warp_workers(hw, pull=pull)
    else:
        try:
            isolate["warp_workers"] = max(1, int(warp_workers))
        except (TypeError, ValueError):
            isolate["warp_workers"] = _auto_isolate_warp_workers(hw, pull=pull)
    if "skip_temporal_when_full_erase" not in isolate:
        # Temporal fill is overwritten when lama_full_erase is on — skip it for speed.
        isolate["skip_temporal_when_full_erase"] = True
    if isolate.get("crop_pad") is None or str(isolate.get("crop_pad")).strip().lower() in {
        "",
        "auto",
    }:
        isolate["crop_pad"] = 32 if isolate["ai_prefer"] == "speed" else 48
    out["isolate"] = isolate

    torch_threads = _resolve_threads(
        perf.get("torch_threads"), hw["cpu_count"], pull=pull
    )
    opencv_threads = _resolve_threads(
        perf.get("opencv_threads"), hw["cpu_count"], pull=pull
    )
    apply_runtime_threads(torch_threads, opencv_threads, pull=pull)
    priority_boosted = boost_process_priority(pull=pull)

    os.environ["SCOUTIQ_ENCODE_THREADS"] = str(encode_threads)
    os.environ["SCOUTIQ_ENCODE_PRESET"] = encode_preset

    out["performance"] = {
        **perf,
        "mode": mode or "auto",
        "pull": pull,
        "torch_threads": torch_threads,
        "opencv_threads": opencv_threads,
        "half": half,
        "imgsz": imgsz,
        "warmup": bool(perf.get("warmup", pull == "hard")),
        "encode_threads": encode_threads,
        "encode_preset": encode_preset,
        "priority_boosted": priority_boosted,
        "resolved_device": player_device,
        "resolved_ball_device": ball_device,
        "resolved_isolate_mode": isolate.get("mode"),
        "hardware": hw,
    }

    logger.info(
        "Resource pull [%s] | device=%s half=%s imgsz=%d ball=%s every_n=%s "
        "hoop_every=%s isolate=%s working_side=%s | cuda=%s (%s, %.1fGB) | "
        "threads torch=%d opencv=%d encode=%d cpus=%d priority=%s",
        pull,
        player_device,
        half,
        imgsz,
        ball_device,
        ball.get("every_n_frames"),
        hoop.get("every_n_frames"),
        isolate.get("mode"),
        isolate.get("ai_working_side"),
        hw["cuda"],
        hw.get("device_name") or "n/a",
        hw.get("vram_gb") or 0.0,
        torch_threads,
        opencv_threads,
        encode_threads,
        hw["cpu_count"],
        priority_boosted,
    )
    return out


def describe_hardware() -> dict[str, Any]:
    """Public snapshot for /api/health and launch splash."""
    hw = probe_hardware()
    return {
        **hw,
        "recommended_device": resolve_device("auto"),
        "recommended_isolate_mode": _auto_isolate_mode(hw),
        "recommended_ball_every_n": _auto_ball_every_n(
            hw, resolve_device("auto")
        ),
        "recommended_ai_working_side": _auto_ai_working_side(
            hw, pull="hard", prefer="speed"
        ),
        "recommended_ai_working_side_quality": _auto_ai_working_side(
            hw, pull="hard", prefer="quality"
        ),
        "recommended_isolate_prep_queue_depth": _auto_isolate_prep_queue_depth(
            hw, pull="hard"
        ),
        "recommended_isolate_warp_workers": _auto_isolate_warp_workers(
            hw, pull="hard"
        ),
        "recommended_half": True if hw["cuda"] else False,
        "recommended_imgsz": _auto_imgsz(
            hw, resolve_device("auto"), "auto", pull="hard"
        ),
        "recommended_pull": "hard",
    }


def warmup_yolo_model(model, *, device: str, imgsz: int, half: bool) -> bool:
    """One dummy forward so the first real frame is not a cold CUDA graph."""
    try:
        import numpy as np

        side = max(int(imgsz), 640)
        dummy = np.zeros((side, side, 3), dtype=np.uint8)
        model.predict(
            dummy,
            imgsz=imgsz,
            conf=0.25,
            verbose=False,
            device=device,
            half=bool(half) and str(device).lower() != "cpu",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("YOLO warmup soft-failed: %s", exc)
        return False
