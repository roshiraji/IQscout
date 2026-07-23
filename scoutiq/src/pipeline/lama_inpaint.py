"""LaMa inpainting for Clean Isolate (AI).

Uses ``simple-lama-inpainting`` on CUDA when available. Speeds up by:
  - cropping to the erase-mask bounding box (skip empty image)
  - optional downscale to ``working_side`` before inference
  - caching the VRAM tile probe across runs
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Prefer largest first — probe stops at the first size that fits VRAM.
_PROBE_SIDES = (1280, 1152, 1024, 896, 768, 640, 512)
_PROBE_CACHE = Path(__file__).resolve().parents[2] / ".scoutiq_runtime" / "lama_tile.json"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def lama_package_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _mask_bbox(mask: np.ndarray, pad: int, h: int, w: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


class LamaInpainter:
    """GPU LaMa with crop + optional downscale for faster isolate."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        working_side: int | None = 768,
        crop_pad: int = 48,
    ) -> None:
        if device.startswith("cuda") and not cuda_available():
            raise RuntimeError("CUDA is required for Clean Isolate (AI) / LaMa.")
        if not lama_package_available():
            raise RuntimeError(
                "simple-lama-inpainting is not installed. "
                "Run: pip install simple-lama-inpainting"
            )
        import torch
        from simple_lama_inpainting import SimpleLama

        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self.device = device
        self.working_side = int(working_side) if working_side else None
        self.crop_pad = max(8, int(crop_pad))
        self._lama = SimpleLama(device=device)
        self.max_side = self._load_or_probe_max_side()
        # Cap tile size by working_side so we never over-allocate for speed mode.
        if self.working_side:
            self.max_side = min(self.max_side, max(self.working_side, 512))
        logger.info(
            "LaMa ready on %s | tile=%d working_side=%s (fast crop path)",
            device,
            self.max_side,
            self.working_side,
        )

    def _load_or_probe_max_side(self) -> int:
        cached = self._read_probe_cache()
        if cached:
            logger.info("LaMa VRAM probe cache hit: %dx%d", cached, cached)
            return cached
        side = self._probe_max_side()
        self._write_probe_cache(side)
        return side

    def _read_probe_cache(self) -> int | None:
        try:
            if not _PROBE_CACHE.is_file():
                return None
            data = json.loads(_PROBE_CACHE.read_text(encoding="utf-8"))
            side = int(data.get("max_side") or 0)
            device_name = str(data.get("device_name") or "")
            import torch

            current = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            )
            if side >= 512 and device_name == current:
                return side
        except Exception:  # noqa: BLE001
            return None
        return None

    def _write_probe_cache(self, side: int) -> None:
        try:
            import torch

            _PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "max_side": int(side),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
            }
            _PROBE_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not cache LaMa probe: %s", exc)

    def _probe_max_side(self) -> int:
        """Find the largest square tile that fits without OOM."""
        import torch
        from PIL import Image

        for side in _PROBE_SIDES:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                img = Image.new("RGB", (side, side), (90, 90, 90))
                mask = Image.new("L", (side, side), 0)
                arr = np.array(mask)
                cv2.circle(arr, (side // 2, side // 2), max(side // 8, 8), 255, -1)
                mask = Image.fromarray(arr, mode="L")
                _ = self._lama(img, mask)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("LaMa VRAM probe OK at %dx%d", side, side)
                return side
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg and "cuda" not in msg:
                    raise
                logger.info("LaMa VRAM probe OOM at %dx%d — trying smaller", side, side)
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        return 512

    def inpaint_bgr(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Inpaint mask==255 regions. Returns BGR uint8 same size as input."""
        if image_bgr is None or mask_u8 is None:
            return image_bgr
        if not mask_u8.any():
            return image_bgr

        h, w = image_bgr.shape[:2]
        bbox = _mask_bbox(mask_u8, self.crop_pad, h, w)
        if bbox is None:
            return image_bgr

        y0, y1, x0, x1 = bbox
        crop_bgr = image_bgr[y0:y1, x0:x1]
        crop_mask = mask_u8[y0:y1, x0:x1]
        filled_crop = self._inpaint_region(crop_bgr, crop_mask)

        out = image_bgr.copy()
        region = crop_mask > 0
        out[y0:y1, x0:x1][region] = filled_crop[region]
        return out

    def _inpaint_region(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Run LaMa on a (possibly downscaled) crop, then restore size."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask_u8 > 0).astype(np.uint8) * 255

        scale = 1.0
        work_rgb, work_mask = rgb, mask
        if self.working_side and max(h, w) > self.working_side:
            scale = self.working_side / float(max(h, w))
            new_w = max(32, int(round(w * scale)))
            new_h = max(32, int(round(h * scale)))
            # Keep even dims for safer network paths.
            new_w -= new_w % 2
            new_h -= new_h % 2
            work_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            work_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if not work_mask.any():
                work_mask = (cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR) > 0).astype(
                    np.uint8
                ) * 255

        if max(work_rgb.shape[0], work_rgb.shape[1]) <= self.max_side:
            out_rgb = self._run_lama(work_rgb, work_mask)
        else:
            out_rgb = self._tiled(work_rgb, work_mask)

        if scale < 1.0:
            out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
                out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _run_lama(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        import torch

        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb, mode="RGB")
        m = Image.fromarray(mask, mode="L")
        try:
            with torch.inference_mode():
                result = self._lama(img, m)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.max_side > 512:
                self.max_side = max(512, int(self.max_side * 0.75))
                logger.warning("LaMa OOM mid-run — reducing max_side to %d", self.max_side)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._tiled(rgb, mask)
            raise
        arr = np.asarray(result.convert("RGB"), dtype=np.uint8)
        if arr.shape[0] != h or arr.shape[1] != w:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return arr

    def _tiled(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlap tiles at max_side; only rewrite masked regions."""
        h, w = rgb.shape[:2]
        side = self.max_side
        overlap = max(32, side // 8)
        step = max(side - overlap, side // 2)
        out = rgb.copy()
        weight = np.zeros((h, w), dtype=np.float32)
        acc = np.zeros((h, w, 3), dtype=np.float32)

        ys = list(range(0, max(h - side, 0) + 1, step))
        xs = list(range(0, max(w - side, 0) + 1, step))
        if not ys or ys[-1] + side < h:
            ys.append(max(0, h - side))
        if not xs or xs[-1] + side < w:
            xs.append(max(0, w - side))

        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + side, h), min(x0 + side, w)
                y0b, x0b = y1 - side, x1 - side
                y0b, x0b = max(0, y0b), max(0, x0b)
                tile_rgb = rgb[y0b:y1, x0b:x1]
                tile_mask = mask[y0b:y1, x0b:x1]
                if not tile_mask.any():
                    continue
                th, tw = tile_rgb.shape[:2]
                if th != side or tw != side:
                    pad_rgb = np.zeros((side, side, 3), dtype=np.uint8)
                    pad_mask = np.zeros((side, side), dtype=np.uint8)
                    pad_rgb[:th, :tw] = tile_rgb
                    pad_mask[:th, :tw] = tile_mask
                    filled = self._run_lama(pad_rgb, pad_mask)[:th, :tw]
                else:
                    filled = self._run_lama(tile_rgb, tile_mask)
                feather = np.ones((th, tw), dtype=np.float32)
                m = min(overlap // 2, th // 4, tw // 4)
                if m > 0:
                    for i in range(m):
                        a = (i + 1) / (m + 1)
                        feather[i, :] *= a
                        feather[th - 1 - i, :] *= a
                        feather[:, i] *= a
                        feather[:, tw - 1 - i] *= a
                region = tile_mask > 0
                if not region.any():
                    continue
                for c in range(3):
                    acc[y0b:y1, x0b:x1, c][region] += (
                        filled[:, :, c][region].astype(np.float32) * feather[region]
                    )
                weight[y0b:y1, x0b:x1][region] += feather[region]

        valid = weight > 1e-6
        if valid.any():
            out[valid] = np.clip(
                acc[valid] / weight[valid][:, None], 0, 255
            ).astype(np.uint8)
        return out


@lru_cache(maxsize=2)
def get_lama_inpainter(
    device: str = "cuda",
    working_side: int | None = 768,
    crop_pad: int = 48,
) -> LamaInpainter:
    return LamaInpainter(
        device=device,
        working_side=working_side,
        crop_pad=crop_pad,
    )


def warmup_lama(isolate_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load LaMa onto the GPU once so the first Confirm is not cold."""
    cfg = isolate_cfg or {}
    if not cuda_available() or not lama_package_available():
        return describe_ai_readiness()
    working = cfg.get("ai_working_side", 768)
    working_side = None if working in (None, 0, "full") else int(working)
    crop_pad = int(cfg.get("crop_pad", 48))
    inpainter = get_lama_inpainter(
        device="cuda",
        working_side=working_side,
        crop_pad=crop_pad,
    )
    if cfg.get("ai_max_side"):
        inpainter.max_side = int(cfg["ai_max_side"])
    info = describe_ai_readiness()
    info["max_side"] = inpainter.max_side
    info["working_side"] = inpainter.working_side
    info["warmed"] = True
    return info


def describe_ai_readiness() -> dict[str, Any]:
    return {
        "cuda": cuda_available(),
        "lama_package": lama_package_available(),
        "ready": cuda_available() and lama_package_available(),
    }
