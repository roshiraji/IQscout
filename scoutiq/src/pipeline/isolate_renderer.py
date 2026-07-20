"""High-quality temporal reconstruction for selected-player isolate previews."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_encoding import encode_for_browser

logger = logging.getLogger(__name__)

DEFAULT_ISOLATE_CFG: dict[str, Any] = {
    "mode": "hybrid",  # hybrid | temporal | ai
    "ai_backend": "lama",  # lama (propainter reserved)
    "erase_dilate": 13,  # stronger non-selected fringe removal
    "keep_dilate": 7,
    "bbox_expand": 0.12,  # expand bbox around polygon to catch shoes/arms
    "ai_max_side": None,  # None = VRAM probe picks max tile
    # Fast GPU path: crop erase region + downscale before LaMa (big speedup on 8GB).
    "ai_working_side": 768,
    "crop_pad": 48,
    "fallback_to_temporal": True,
    "lama_leftover_min_pixels": 64,
    # Anti-blur: only trust temporal when several neighbors agree; LaMa owns the rest.
    "min_temporal_samples": 2,
    "lama_full_erase": True,  # LaMa repaints whole erase (kills smear kept as "filled")
    "hard_composite": True,  # no soft blend of removed players back into the hole
    "orb_features": 700,  # lower = faster temporal align on CPU
}


def _make_lama(cfg: dict[str, Any]):
    from src.pipeline.lama_inpaint import get_lama_inpainter

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
    return inpainter


def _make_orb(cfg: dict[str, Any]):
    nfeatures = int(cfg.get("orb_features", 700))
    return cv2.ORB_create(
        nfeatures=max(200, nfeatures),
        scaleFactor=1.2,
        nlevels=6,
        fastThreshold=15,
    )


class IsolateCancelled(RuntimeError):
    """Raised when a newer Confirm supersedes an isolate render."""


def resolve_isolate_mode(cfg: dict[str, Any] | None = None) -> str:
    """Return effective isolate mode: hybrid | ai | temporal.

    hybrid / ai require CUDA + LaMa; otherwise fall back to temporal when allowed.
    """
    cfg = {**DEFAULT_ISOLATE_CFG, **(cfg or {})}
    requested = str(cfg.get("mode", "hybrid")).lower()
    if requested == "temporal":
        return "temporal"
    if requested not in {"hybrid", "ai"}:
        return "temporal"

    from src.pipeline.lama_inpaint import describe_ai_readiness

    readiness = describe_ai_readiness()
    if readiness["ready"]:
        return requested
    if cfg.get("fallback_to_temporal", True):
        logger.warning(
            "Isolate mode=%s unavailable (cuda=%s lama=%s) — falling back to temporal.",
            requested,
            readiness["cuda"],
            readiness["lama_package"],
        )
        return "temporal"
    raise RuntimeError(
        f"Isolate mode={requested} requires CUDA and simple-lama-inpainting. "
        f"cuda={readiness['cuda']} lama_package={readiness['lama_package']}"
    )


def isolate_mode_label(mode: str) -> str:
    if mode == "hybrid":
        return "Clean Isolate (hybrid: temporal->LaMa)"
    if mode == "ai":
        return "Clean Isolate (AI: LaMa only)"
    return "Isolate (temporal)"


def _group_id(det: dict) -> int:
    value = det.get("person_group_id")
    if value is None:
        value = det.get("track_id", -1)
    return int(value)


def _bbox_ellipse_mask(
    height: int,
    width: int,
    bbox_xyxy: list,
    *,
    expand: float = 0.55,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in (bbox_xyxy or [0, 0, 0, 0])[:4]]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return mask
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (
        max(1, int((x2 - x1) * expand)),
        max(1, int((y2 - y1) * (expand + 0.03))),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _expand_bbox(bbox_xyxy: list, width: int, height: int, frac: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in (bbox_xyxy or [0, 0, 0, 0])[:4]]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_x, pad_y = bw * frac, bh * frac
    return [
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    ]


def _person_mask(
    height: int,
    width: int,
    det: dict,
    *,
    fallback_expand: float = 0.58,
    bbox_expand: float = 0.0,
    boost_bbox: bool = False,
) -> np.ndarray:
    """Person silhouette. For non-selected erase targets, OR polygon with expanded bbox."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = det.get("mask_polygon")
    has_poly = False
    if polygon and len(polygon) >= 3:
        points = np.asarray(
            [[int(point[0]), int(point[1])] for point in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        has_poly = cv2.countNonZero(mask) >= 32

    bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
    if boost_bbox and bbox_expand > 0:
        # Catch shoes / arms the polygon often clips.
        expanded = _expand_bbox(bbox, width, height, bbox_expand)
        ellipse = _bbox_ellipse_mask(
            height, width, expanded, expand=0.52 + bbox_expand
        )
        mask = cv2.bitwise_or(mask, ellipse)
        return mask

    if has_poly:
        return mask
    return _bbox_ellipse_mask(height, width, bbox, expand=fallback_expand)


def _build_masks(
    height: int,
    width: int,
    detections: list[dict],
    selected_group_ids: set[int],
    *,
    erase_dilate: int = 13,
    keep_dilate: int = 7,
    bbox_expand: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all-player occupancy, selected keep mask, erase mask).

    Non-selected masks get polygon + expanded bbox so limbs/shoes are covered.
    erase = dilated non-selected occupancy minus dilated keep (never eats selected).
    """
    occupancy = np.zeros((height, width), dtype=np.uint8)
    keep = np.zeros((height, width), dtype=np.uint8)
    non_selected = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        gid = _group_id(det)
        is_selected = gid in selected_group_ids
        mask = _person_mask(
            height,
            width,
            det,
            bbox_expand=bbox_expand if not is_selected else 0.0,
            boost_bbox=not is_selected,
        )
        if not mask.any():
            continue
        occupancy = cv2.bitwise_or(occupancy, mask)
        if is_selected:
            keep = cv2.bitwise_or(keep, mask)
        else:
            non_selected = cv2.bitwise_or(non_selected, mask)

    e_k = max(3, int(erase_dilate) | 1)
    k_k = max(3, int(keep_dilate) | 1)
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_k, e_k))
    kernel_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_k, k_k))
    # Dilate non-selected more aggressively; keep a bit so selected stay safe.
    non_selected = cv2.dilate(non_selected, kernel_e, iterations=1)
    keep = cv2.dilate(keep, kernel_k, iterations=1)
    occupancy = cv2.bitwise_or(occupancy, non_selected)
    occupancy = cv2.bitwise_or(occupancy, keep)

    erase = cv2.bitwise_and(non_selected, cv2.bitwise_not(keep))
    return occupancy, keep, erase


def _frame_features(
    frame: np.ndarray,
    occupancy: np.ndarray,
    detector,
) -> tuple[list, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature_mask = cv2.bitwise_not(occupancy)
    return detector.detectAndCompute(gray, feature_mask)


def _neighbor_to_target_affine(target: dict, neighbor: dict) -> np.ndarray | None:
    target_desc = target["descriptors"]
    neighbor_desc = neighbor["descriptors"]
    if target_desc is None or neighbor_desc is None:
        return None
    if len(target_desc) < 12 or len(neighbor_desc) < 12:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(neighbor_desc, target_desc, k=2)
    matches = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.76 * second.distance:
            matches.append(first)
    if len(matches) < 10:
        return None

    source = np.float32(
        [neighbor["keypoints"][match.queryIdx].pt for match in matches]
    )
    destination = np.float32(
        [target["keypoints"][match.trainIdx].pt for match in matches]
    )
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1200,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        return None

    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    if not 0.82 <= scale <= 1.22:
        return None
    return affine


def _reconstruct_frame(
    target: dict,
    neighbors: list[dict],
    width: int,
    height: int,
    *,
    use_spatial_fallback: bool = True,
    min_temporal_samples: int = 1,
    hard_composite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal reconstruct erase region.

    Returns (output_bgr, leftover_mask). leftover = erase pixels that did not
    meet ``min_temporal_samples`` clean neighbor hits (LaMa should own these).

    hard_composite=True avoids soft-blending original player pixels back into
    the hole (that blend is a major source of the blurry/ghost look).
    """
    frame = target["frame"]
    erase = target.get("erase")
    keep = target.get("keep")
    if erase is None:
        erase = cv2.bitwise_and(
            target["occupancy"], cv2.bitwise_not(target["keep"])
        )
    leftover = np.zeros((height, width), dtype=np.uint8)
    if not erase.any():
        return frame, leftover

    pixel_sum = np.zeros_like(frame, dtype=np.float32)
    sample_count = np.zeros((height, width), dtype=np.uint8)
    coverage_source = np.full((height, width), 255, dtype=np.uint8)

    for neighbor in neighbors:
        affine = _neighbor_to_target_affine(target, neighbor)
        if affine is None:
            continue
        warped_frame = cv2.warpAffine(
            neighbor["frame"],
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_occupancy = cv2.warpAffine(
            neighbor["occupancy"],
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        coverage = cv2.warpAffine(
            coverage_source,
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (
            (erase > 0)
            & (warped_occupancy == 0)
            & (coverage > 0)
        )
        if not valid.any():
            continue
        pixel_sum[valid] += warped_frame[valid].astype(np.float32)
        sample_count[valid] += 1

    min_hits = max(1, int(min_temporal_samples))
    temporal = sample_count >= min_hits
    leftover = ((erase > 0) & (~temporal)).astype(np.uint8) * 255
    if keep is not None and keep.any():
        leftover = cv2.bitwise_and(leftover, cv2.bitwise_not(keep))

    reconstructed = frame.copy()
    if temporal.any():
        reconstructed[temporal] = np.clip(
            pixel_sum[temporal] / sample_count[temporal][:, None],
            0,
            255,
        ).astype(np.uint8)

    if use_spatial_fallback and leftover.any():
        # Classical fill only for pure temporal mode — hybrid must not use this
        # (NS/Telea streaks are the blurry look).
        spatial = cv2.inpaint(reconstructed, leftover, 3, cv2.INPAINT_TELEA)
        reconstructed[leftover > 0] = spatial[leftover > 0]

    if hard_composite:
        # Hard paste only inside erase; never feather original players back in.
        output = frame.copy()
        erase_bool = erase > 0
        output[erase_bool] = reconstructed[erase_bool]
        if keep is not None and keep.any():
            output[keep > 0] = frame[keep > 0]
        return output, leftover

    # Soft feather (temporal-only path) — can ghost; prefer hard_composite for hybrid.
    alpha = cv2.GaussianBlur(erase, (0, 0), 1.2).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    output = (
        reconstructed.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(output, 0, 255).astype(np.uint8), leftover



def render_temporal_isolate(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    isolate_cfg: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
) -> Path:
    """Render scene-preserving isolate from canonical frames and masks.

    Nearby frames are feature-aligned to the target. Pixels occupied by
    non-selected people are replaced from clean aligned neighbors; Navier-
    Stokes inpaint fills only locations with no clean temporal sample.
    """
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}
    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))
    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = artifact_dir / "tracks.jsonl.gz"
    total = int(canonical.get("source_frame_count") or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radius = max(6, min(18, int(round(fps * 0.5))))
    half = max(3, radius // 2)

    temp_path = artifact_dir / f"focused_isolated.{generation}.tmp.mp4"
    output_path = artifact_dir / "focused_isolated.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create isolate preview: {temp_path}")

    detector = _make_orb(cfg)
    window: dict[int, dict] = {}
    next_output = 0
    last_index = -1

    def process(index: int) -> None:
        target = window[index]
        offsets = (-radius, -half, half, radius)
        neighbors = [
            window[index + offset]
            for offset in offsets
            if index + offset in window
        ]
        out, _leftover = _reconstruct_frame(
            target,
            neighbors,
            width,
            height,
            use_spatial_fallback=True,
            min_temporal_samples=1,
            hard_composite=False,
        )
        writer.write(out)

    try:
        with gzip.open(tracks_path, "rt", encoding="utf-8") as tracks:
            for index, line in enumerate(tracks):
                if cancelled and cancelled():
                    raise IsolateCancelled("A newer selection replaced this isolate job.")
                ok, frame = cap.read()
                if not ok:
                    break
                record = json.loads(line)
                detections = list(record.get("tracks") or [])
                occupancy, keep, erase = _build_masks(
                    height,
                    width,
                    detections,
                    selected_group_ids,
                    erase_dilate=erase_dilate,
                    keep_dilate=keep_dilate,
                    bbox_expand=bbox_expand,
                )
                keypoints, descriptors = _frame_features(
                    frame, occupancy, detector
                )
                window[index] = {
                    "frame": frame,
                    "occupancy": occupancy,
                    "keep": keep,
                    "erase": erase,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                }
                last_index = index

                while next_output + radius <= index:
                    process(next_output)
                    next_output += 1
                    if progress_callback:
                        progress_callback(next_output, max(total, index + 1), "reconstructing")
                    old_index = next_output - radius - 1
                    window.pop(old_index, None)

            while next_output <= last_index:
                if cancelled and cancelled():
                    raise IsolateCancelled("A newer selection replaced this isolate job.")
                process(next_output)
                next_output += 1
                if progress_callback:
                    progress_callback(next_output, max(total, last_index + 1), "reconstructing")
                old_index = next_output - radius - 1
                window.pop(old_index, None)
    finally:
        cap.release()
        writer.release()

    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    if progress_callback:
        progress_callback(max(total, next_output), max(total, next_output), "encoding")
    encode_for_browser(temp_path, quiet=quiet)
    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    temp_path.replace(output_path)
    return output_path


def render_hybrid_isolate(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    isolate_cfg: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
) -> Path:
    """Best path: temporal fill from neighbors, then LaMa only on leftovers."""
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}

    if progress_callback:
        progress_callback(0, 1, "loading_lama")
    inpainter = _make_lama(cfg)

    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))
    leftover_min = int(cfg.get("lama_leftover_min_pixels", 64))
    min_temporal = int(cfg.get("min_temporal_samples", 2))
    lama_full_erase = bool(cfg.get("lama_full_erase", True))
    hard_composite = bool(cfg.get("hard_composite", True))

    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = artifact_dir / "tracks.jsonl.gz"
    total = int(canonical.get("source_frame_count") or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radius = max(6, min(18, int(round(fps * 0.5))))
    half = max(3, radius // 2)

    temp_path = artifact_dir / f"focused_isolated.{generation}.tmp.mp4"
    output_path = artifact_dir / "focused_isolated.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create isolate preview: {temp_path}")

    detector = _make_orb(cfg)
    window: dict[int, dict] = {}
    next_output = 0
    last_index = -1

    def process(index: int) -> None:
        target = window[index]
        offsets = (-radius, -half, half, radius)
        neighbors = [
            window[index + offset]
            for offset in offsets
            if index + offset in window
        ]
        # High-confidence temporal seed only — never NS/Telea, never soft-ghost blend.
        out, leftover = _reconstruct_frame(
            target,
            neighbors,
            width,
            height,
            use_spatial_fallback=False,
            min_temporal_samples=min_temporal,
            hard_composite=hard_composite,
        )
        erase = target.get("erase")
        keep = target.get("keep")
        # Prefer LaMa on the full erase hole so weak temporal smear cannot stick.
        lama_mask = erase if (lama_full_erase and erase is not None) else leftover
        if lama_mask is not None and keep is not None and keep.any():
            lama_mask = cv2.bitwise_and(lama_mask, cv2.bitwise_not(keep))
        if (
            lama_mask is not None
            and lama_mask.any()
            and int(cv2.countNonZero(lama_mask)) >= leftover_min
        ):
            out = inpainter.inpaint_bgr(out, lama_mask)
            if keep is not None and keep.any():
                out = out.copy()
                out[keep > 0] = target["frame"][keep > 0]
        writer.write(out)

    try:
        with gzip.open(tracks_path, "rt", encoding="utf-8") as tracks:
            for index, line in enumerate(tracks):
                if cancelled and cancelled():
                    raise IsolateCancelled("A newer selection replaced this isolate job.")
                ok, frame = cap.read()
                if not ok:
                    break
                record = json.loads(line)
                detections = list(record.get("tracks") or [])
                occupancy, keep, erase = _build_masks(
                    height,
                    width,
                    detections,
                    selected_group_ids,
                    erase_dilate=erase_dilate,
                    keep_dilate=keep_dilate,
                    bbox_expand=bbox_expand,
                )
                keypoints, descriptors = _frame_features(
                    frame, occupancy, detector
                )
                window[index] = {
                    "frame": frame,
                    "occupancy": occupancy,
                    "keep": keep,
                    "erase": erase,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                }
                last_index = index

                while next_output + radius <= index:
                    process(next_output)
                    next_output += 1
                    if progress_callback:
                        progress_callback(
                            next_output,
                            max(total, index + 1),
                            "hybrid_temporal_lama",
                        )
                    old_index = next_output - radius - 1
                    window.pop(old_index, None)

            while next_output <= last_index:
                if cancelled and cancelled():
                    raise IsolateCancelled("A newer selection replaced this isolate job.")
                process(next_output)
                next_output += 1
                if progress_callback:
                    progress_callback(
                        next_output,
                        max(total, last_index + 1),
                        "hybrid_temporal_lama",
                    )
                old_index = next_output - radius - 1
                window.pop(old_index, None)
    finally:
        cap.release()
        writer.release()

    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    if progress_callback:
        progress_callback(max(total, next_output), max(total, next_output), "encoding")
    encode_for_browser(temp_path, quiet=quiet)
    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    temp_path.replace(output_path)
    return output_path


def render_ai_isolate(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    isolate_cfg: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
) -> Path:
    """Clean Isolate (AI): LaMa inpaint on non-selected player masks per frame."""
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}

    if progress_callback:
        progress_callback(0, 1, "loading_lama")
    inpainter = _make_lama(cfg)

    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = artifact_dir / "tracks.jsonl.gz"
    total = int(canonical.get("source_frame_count") or 0)
    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_path = artifact_dir / f"focused_isolated.{generation}.tmp.mp4"
    output_path = artifact_dir / "focused_isolated.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create isolate preview: {temp_path}")

    processed = 0
    try:
        with gzip.open(tracks_path, "rt", encoding="utf-8") as tracks:
            for index, line in enumerate(tracks):
                if cancelled and cancelled():
                    raise IsolateCancelled("A newer selection replaced this isolate job.")
                ok, frame = cap.read()
                if not ok:
                    break
                record = json.loads(line)
                detections = list(record.get("tracks") or [])
                _occ, keep, erase = _build_masks(
                    height,
                    width,
                    detections,
                    selected_group_ids,
                    erase_dilate=erase_dilate,
                    keep_dilate=keep_dilate,
                    bbox_expand=bbox_expand,
                )
                if erase.any():
                    out = inpainter.inpaint_bgr(frame, erase)
                    if keep.any():
                        out = out.copy()
                        out[keep > 0] = frame[keep > 0]
                else:
                    out = frame
                writer.write(out)
                processed = index + 1
                if progress_callback:
                    progress_callback(
                        processed,
                        max(total, processed),
                        "ai_inpainting",
                    )
    finally:
        cap.release()
        writer.release()

    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    if progress_callback:
        progress_callback(max(total, processed), max(total, processed), "encoding")
    encode_for_browser(temp_path, quiet=quiet)
    if cancelled and cancelled():
        temp_path.unlink(missing_ok=True)
        raise IsolateCancelled("A newer selection replaced this isolate job.")
    temp_path.replace(output_path)
    return output_path


def render_isolate(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    isolate_cfg: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
) -> Path:
    """Dispatch hybrid / ai / temporal isolate based on config + GPU availability."""
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}
    mode = resolve_isolate_mode(cfg)
    if mode == "hybrid":
        return render_hybrid_isolate(
            job_dir,
            selected_group_ids,
            generation,
            isolate_cfg=cfg,
            progress_callback=progress_callback,
            cancelled=cancelled,
            quiet=quiet,
        )
    if mode == "ai":
        return render_ai_isolate(
            job_dir,
            selected_group_ids,
            generation,
            isolate_cfg=cfg,
            progress_callback=progress_callback,
            cancelled=cancelled,
            quiet=quiet,
        )
    return render_temporal_isolate(
        job_dir,
        selected_group_ids,
        generation,
        isolate_cfg=cfg,
        progress_callback=progress_callback,
        cancelled=cancelled,
        quiet=quiet,
    )
