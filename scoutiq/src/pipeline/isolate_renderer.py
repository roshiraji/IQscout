"""High-quality temporal reconstruction for selected-player isolate previews.

``ai_backend`` stays "lama" on purpose. ProPainter would be the obvious
upgrade (flow-guided video inpainting, far more temporally consistent than
per-frame LaMa) but its pretrained weights ship under the S-Lab
non-commercial license — not compatible with a commercial product. Instead
the temporal path below was hardened to close most of that gap without a new
model: homography alignment (vs. plain affine) tracks broadcast camera
pans/zooms more accurately, and multi-neighbor *median* blending (vs. mean)
throws out a single misaligned neighbor instead of blurring it in.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import queue
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_encoding import encode_for_browser
from src.pipeline.selection_lock import resolve_tracks_jsonl

logger = logging.getLogger(__name__)

DEFAULT_ISOLATE_CFG: dict[str, Any] = {
    "mode": "hybrid",  # hybrid | temporal | ai
    "ai_backend": "lama",  # lama only — see module docstring re: ProPainter license
    "erase_dilate": 13,  # stronger non-selected fringe removal
    "keep_dilate": 7,
    "bbox_expand": 0.12,  # expand bbox around polygon to catch shoes/arms
    "ai_max_side": None,  # None = VRAM probe picks max tile
    # Fast GPU path: crop erase region + downscale before LaMa (big speedup on 8GB).
    "ai_working_side": 640,
    "ai_prefer": "speed",  # speed | quality — hardware auto resolves working_side
    "crop_pad": 32,
    "fallback_to_temporal": True,
    "lama_leftover_min_pixels": 64,
    # Anti-blur: only trust temporal when several neighbors agree; LaMa owns the rest.
    "min_temporal_samples": 2,
    "lama_full_erase": True,  # LaMa repaints whole erase (kills smear kept as "filled")
    # When True + lama_full_erase, skip the temporal seed — LaMa overwrites it anyway.
    "skip_temporal_when_full_erase": True,
    "hard_composite": True,  # no soft blend of removed players back into the hole
    "orb_features": 700,  # lower = faster temporal align on CPU
    # Homography (perspective) alignment tracks camera pan/zoom far better than a
    # pure similarity affine; falls back to affine when there aren't enough inliers.
    "use_homography": True,
    "homography_min_inliers": 12,
    # Sample more neighbor frames per hole so the median has real outlier
    # rejection instead of averaging in one bad alignment.
    "wide_neighbor_search": True,
    # Soft anti-alias ring *outward* past the erase edge — never inward, so no
    # fragment of the erased person's real pixels leaks back into the hole.
    "edge_feather_px": 2,
    "write_quality_metrics": True,
    # Throughput: CPU prepares frames ahead of GPU LaMa; warps run in parallel.
    "prep_queue_depth": 6,
    "warp_workers": 4,
}


def _make_lama(cfg: dict[str, Any]):
    from src.pipeline.lama_inpaint import get_lama_inpainter

    working = cfg.get("ai_working_side", 640)
    if working in (None, 0, "full", "auto", ""):
        working_side = None
        if str(working).strip().lower() == "auto" or working in (None, ""):
            # Resolve from hardware if config load skipped apply_hardware_to_config.
            try:
                from src.pipeline.hardware import probe_hardware, _auto_ai_working_side

                prefer = str(cfg.get("ai_prefer", "speed") or "speed")
                working_side = _auto_ai_working_side(
                    probe_hardware(), pull="hard", prefer=prefer
                )
            except Exception:  # noqa: BLE001
                working_side = 640
    else:
        working_side = int(working)
    crop_pad = int(cfg.get("crop_pad", 32))
    inpainter = get_lama_inpainter(
        device="cuda",
        working_side=working_side,
        crop_pad=crop_pad,
    )
    if cfg.get("ai_max_side"):
        inpainter.max_side = int(cfg["ai_max_side"])
    return inpainter


def _resolve_worker_count(value: Any, default: int) -> int:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return max(1, int(default))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _boost_isolate_runtime(*, cfg: dict[str, Any]) -> None:
    """Pin threads + raise process priority so isolate saturates the machine."""
    try:
        from src.pipeline.hardware import (
            apply_runtime_threads,
            boost_process_priority,
            probe_hardware,
        )

        hw = probe_hardware()
        cpus = int(hw.get("cpu_count") or os.cpu_count() or 4)
        apply_runtime_threads(cpus, cpus, pull="hard")
        boost_process_priority(pull="hard")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Isolate runtime boost soft-failed: %s", exc)


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
    ``mode: auto`` resolves to hybrid when the GPU stack is ready.
    """
    cfg = {**DEFAULT_ISOLATE_CFG, **(cfg or {})}
    requested = str(cfg.get("mode", "hybrid")).lower()
    if requested in {"", "auto"}:
        from src.pipeline.hardware import describe_hardware

        requested = str(describe_hardware().get("recommended_isolate_mode") or "temporal")
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


def _transform_is_sane(matrix: np.ndarray, width: int, height: int) -> bool:
    """Reject a fitted transform that RANSAC accepted but extrapolates wildly.

    A homography (or affine) with enough inliers on a small local patch of
    matches can still explode outside that patch — this checks the whole
    frame it will actually be used to warp, not just the sample points.
    """
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(warped)):
        return False
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    orig_area = float(width * height)
    if orig_area <= 0 or area <= 0:
        return False
    ratio = area / orig_area
    if not 0.55 <= ratio <= 1.8:
        return False
    diag = math.hypot(width, height)
    shift = float(np.max(np.linalg.norm(warped - corners.reshape(-1, 2), axis=1)))
    return shift <= diag * 0.6


def _matched_points(
    target: dict, neighbor: dict
) -> tuple[np.ndarray, np.ndarray] | None:
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
    return source, destination


def _neighbor_to_target_matrix(
    target: dict,
    neighbor: dict,
    *,
    width: int,
    height: int,
    use_homography: bool = True,
    homography_min_inliers: int = 12,
) -> np.ndarray | None:
    """3x3 matrix mapping neighbor pixels -> target frame, or None.

    Tries a homography first (models the perspective of a panning/zooming
    broadcast camera, not just translation+rotation+scale); falls back to a
    similarity affine when there aren't enough inliers for a stable
    homography. Both paths are sanity-checked against the whole frame, not
    just the matched patch.
    """
    matched = _matched_points(target, neighbor)
    if matched is None:
        return None
    source, destination = matched

    if use_homography and len(source) >= 4:
        homography, inlier_mask = cv2.findHomography(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if (
            homography is not None
            and inlier_mask is not None
            and int(inlier_mask.sum()) >= max(8, int(homography_min_inliers))
            and _transform_is_sane(homography, width, height)
        ):
            return homography

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
    matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)
    if not _transform_is_sane(matrix, width, height):
        return None
    return matrix


def _neighbor_offsets(radius: int, half: int, wide: bool) -> tuple[int, ...]:
    """Frame offsets sampled as temporal-fill candidates for one target frame.

    ``wide`` roughly doubles the candidate count so the median blend (see
    ``_reconstruct_frame``) has real outlier rejection instead of just
    averaging whatever four frames happened to align.
    """
    base = {-radius, -half, half, radius}
    if not wide:
        return tuple(sorted(o for o in base if o != 0))
    step = max(1, half // 2)
    extra = {-radius + step, -half + step, half - step, radius - step, -2, 2}
    offsets = {o for o in (base | extra) if o != 0 and abs(o) <= radius}
    return tuple(sorted(offsets))


def _feather_seam(
    filled: np.ndarray,
    erase_mask: np.ndarray,
    feather_px: int,
    keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Soften the hard erase/background seam right at ``erase_mask``'s edge.

    ``filled`` is the already-composited frame — by the time this runs, the
    erase region no longer holds any of the erased person's real pixels, so
    locally blurring the seam band (a few px on *either* side of the true
    boundary) can only smooth a color/exposure mismatch at the cut; it can
    never reintroduce a fragment of the removed player. (An earlier version
    of this tried to crossfade with the original frame at ring pixels, but
    those pixels are identical to ``filled`` there by construction — a
    no-op. Blurring the composite itself is what actually removes the
    visible cutline.)
    """
    if feather_px <= 0 or not erase_mask.any():
        return filled
    ksize = max(3, feather_px * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(filled, (ksize, ksize), 0)

    dist_out = cv2.distanceTransform(cv2.bitwise_not(erase_mask), cv2.DIST_L2, 3)
    dist_in = cv2.distanceTransform(erase_mask, cv2.DIST_L2, 3)
    dist_to_boundary = np.minimum(dist_out, dist_in)
    alpha = np.clip(1.0 - dist_to_boundary / float(feather_px), 0.0, 1.0)
    if keep_mask is not None and keep_mask.any():
        alpha = alpha * (keep_mask == 0)
    if not alpha.any():
        return filled

    alpha3 = alpha[..., None]
    blended = blurred.astype(np.float32) * alpha3 + filled.astype(np.float32) * (
        1.0 - alpha3
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


class _QualityAccumulator:
    """Cheap per-render telemetry — bookkeeping only, no extra image work."""

    def __init__(self) -> None:
        self.frames_with_erase = 0
        self.frames_with_leftover = 0
        self.erase_pixels_total = 0
        self.leftover_pixels_total = 0
        self.lama_invocations = 0
        self.aligned_neighbor_hits = 0
        self.aligned_neighbor_attempts = 0

    def record_frame(self, erase: np.ndarray, leftover: np.ndarray) -> None:
        erase_px = int(cv2.countNonZero(erase)) if erase is not None else 0
        leftover_px = int(cv2.countNonZero(leftover)) if leftover is not None else 0
        if erase_px:
            self.frames_with_erase += 1
            self.erase_pixels_total += erase_px
        if leftover_px:
            self.frames_with_leftover += 1
            self.leftover_pixels_total += leftover_px

    def record_alignment(self, attempted: int, hit: int) -> None:
        self.aligned_neighbor_attempts += attempted
        self.aligned_neighbor_hits += hit

    def record_lama(self) -> None:
        self.lama_invocations += 1

    def as_dict(self, *, mode: str, cfg: dict[str, Any]) -> dict[str, Any]:
        # For temporal/hybrid this is the share of the hole closed without
        # falling back to spatial inpaint/LaMa; for ai mode LaMa always owns
        # the whole hole so this is always 1.0 (no "leftover" concept there).
        resolved = max(0, self.erase_pixels_total - self.leftover_pixels_total)
        return {
            "mode": mode,
            "frames_with_erase": self.frames_with_erase,
            "frames_with_leftover": self.frames_with_leftover,
            "erase_pixels_total": self.erase_pixels_total,
            "leftover_pixels_total": self.leftover_pixels_total,
            "hole_resolved_without_fallback_fraction": round(
                resolved / self.erase_pixels_total, 4
            )
            if self.erase_pixels_total
            else None,
            "lama_invocations": self.lama_invocations,
            "neighbor_alignment_hit_rate": round(
                self.aligned_neighbor_hits / self.aligned_neighbor_attempts, 4
            )
            if self.aligned_neighbor_attempts
            else None,
            "settings": {
                "use_homography": bool(cfg.get("use_homography", True)),
                "wide_neighbor_search": bool(cfg.get("wide_neighbor_search", True)),
                "edge_feather_px": int(cfg.get("edge_feather_px", 0)),
                "min_temporal_samples": int(cfg.get("min_temporal_samples", 1)),
                "skip_temporal_when_full_erase": bool(
                    cfg.get("skip_temporal_when_full_erase", True)
                ),
                "ai_prefer": str(cfg.get("ai_prefer", "speed")),
                "ai_working_side": cfg.get("ai_working_side"),
                "prep_queue_depth": cfg.get("prep_queue_depth"),
                "warp_workers": cfg.get("warp_workers"),
            },
        }


def _write_quality_metrics(
    artifact_dir: Path, generation: str, accumulator: "_QualityAccumulator", *, mode: str, cfg: dict[str, Any]
) -> None:
    try:
        payload = accumulator.as_dict(mode=mode, cfg=cfg)
        payload["generation"] = generation
        (artifact_dir / "isolate_quality.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a render
        logger.debug("Could not write isolate quality metrics: %s", exc)


def _align_neighbor_crop(
    target: dict,
    neighbor: dict,
    *,
    width: int,
    height: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    erase_bbox: np.ndarray,
    coverage_source: np.ndarray,
    use_homography: bool,
    homography_min_inliers: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Warp one neighbor into the erase bbox; None if alignment fails."""
    matrix = _neighbor_to_target_matrix(
        target,
        neighbor,
        width=width,
        height=height,
        use_homography=use_homography,
        homography_min_inliers=homography_min_inliers,
    )
    if matrix is None:
        return None
    warped_frame = cv2.warpPerspective(
        neighbor["frame"],
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_occupancy = cv2.warpPerspective(
        neighbor["occupancy"],
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    coverage = cv2.warpPerspective(
        coverage_source,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_bbox = (
        erase_bbox
        & (warped_occupancy[y0:y1, x0:x1] == 0)
        & (coverage[y0:y1, x0:x1] > 0)
    )
    if not valid_bbox.any():
        return None
    return warped_frame[y0:y1, x0:x1].astype(np.float32), valid_bbox


def _reconstruct_frame(
    target: dict,
    neighbors: list[dict],
    width: int,
    height: int,
    *,
    use_spatial_fallback: bool = True,
    min_temporal_samples: int = 1,
    hard_composite: bool = False,
    use_homography: bool = True,
    homography_min_inliers: int = 12,
    edge_feather_px: int = 0,
    warp_workers: int = 1,
    quality: "_QualityAccumulator | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal reconstruct erase region.

    Returns (output_bgr, leftover_mask). leftover = erase pixels that did not
    meet ``min_temporal_samples`` clean neighbor hits (LaMa should own these).

    Candidate neighbor pixels are combined with a per-pixel *median* rather
    than a mean — one misaligned or partially-occluded neighbor no longer
    blurs into every pixel it touched, it just gets outvoted.

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
        if quality is not None:
            quality.record_frame(erase, leftover)
        return frame, leftover

    ys, xs = np.where(erase > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    erase_bbox = erase[y0:y1, x0:x1] > 0
    bbox_shape = erase_bbox.shape

    coverage_source = np.full((height, width), 255, dtype=np.uint8)
    candidates: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    align_attempts = len(neighbors)
    align_hits = 0

    workers = max(1, int(warp_workers))
    if workers == 1 or len(neighbors) <= 1:
        aligned = [
            _align_neighbor_crop(
                target,
                neighbor,
                width=width,
                height=height,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                erase_bbox=erase_bbox,
                coverage_source=coverage_source,
                use_homography=use_homography,
                homography_min_inliers=homography_min_inliers,
            )
            for neighbor in neighbors
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(neighbors))) as pool:
            futures = [
                pool.submit(
                    _align_neighbor_crop,
                    target,
                    neighbor,
                    width=width,
                    height=height,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    erase_bbox=erase_bbox,
                    coverage_source=coverage_source,
                    use_homography=use_homography,
                    homography_min_inliers=homography_min_inliers,
                )
                for neighbor in neighbors
            ]
            aligned = [future.result() for future in futures]

    for result in aligned:
        if result is None:
            continue
        crop, valid_bbox = result
        align_hits += 1
        candidates.append(crop)
        candidate_masks.append(valid_bbox)

    if quality is not None:
        quality.record_alignment(align_attempts, align_hits)

    sample_count_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    filled_bbox = frame[y0:y1, x0:x1].copy()
    if candidates:
        stacked_imgs = np.stack(candidates, axis=0)
        stacked_valid = np.stack(candidate_masks, axis=0)
        sample_count_bbox = stacked_valid.sum(axis=0).astype(np.uint8)
        masked = np.where(stacked_valid[..., None], stacked_imgs, np.nan)
        has_sample = sample_count_bbox > 0
        # Pixels with zero valid candidates hit nanmedian's "all-NaN slice"
        # path — expected (has_sample below masks them out before use), so
        # silence the warning instead of computing the median twice.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_bbox = np.nanmedian(masked, axis=0)
        filled_bbox[has_sample] = np.clip(median_bbox[has_sample], 0, 255).astype(
            np.uint8
        )

    min_hits = max(1, int(min_temporal_samples))
    temporal_bbox = sample_count_bbox >= min_hits
    leftover_bbox = erase_bbox & (~temporal_bbox)
    if keep is not None and keep.any():
        leftover_bbox = leftover_bbox & (keep[y0:y1, x0:x1] == 0)
    leftover[y0:y1, x0:x1] = leftover_bbox.astype(np.uint8) * 255

    reconstructed = frame.copy()
    reconstructed[y0:y1, x0:x1][temporal_bbox] = filled_bbox[temporal_bbox]

    if use_spatial_fallback and leftover.any():
        # Classical fill only for pure temporal mode — hybrid must not use this
        # (NS/Telea streaks are the blurry look).
        spatial = cv2.inpaint(reconstructed, leftover, 3, cv2.INPAINT_TELEA)
        reconstructed[leftover > 0] = spatial[leftover > 0]

    if quality is not None:
        quality.record_frame(erase, leftover)

    if hard_composite:
        # Hard paste only inside erase; never feather original players back in.
        output = frame.copy()
        erase_bool = erase > 0
        output[erase_bool] = reconstructed[erase_bool]
        if keep is not None and keep.any():
            output[keep > 0] = frame[keep > 0]
        if edge_feather_px > 0:
            output = _feather_seam(output, erase, edge_feather_px, keep)
        return output, leftover

    # Soft feather (temporal-only path) — can ghost; prefer hard_composite for hybrid.
    alpha = cv2.GaussianBlur(erase, (0, 0), 1.2).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    output = (
        reconstructed.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(output, 0, 255).astype(np.uint8), leftover


def _iter_prepared_lama_frames(
    *,
    video_path: Path,
    tracks_path: Path,
    selected_group_ids: set[int],
    height: int,
    width: int,
    erase_dilate: int,
    keep_dilate: int,
    bbox_expand: float,
    queue_depth: int,
    cancelled: Callable[[], bool] | None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray] | None]:
    """CPU producer: decode + build masks ahead of the GPU consumer.

    Yields ``(index, frame, keep, erase)``. Puts a sentinel ``None`` when done.
    Raises ``IsolateCancelled`` from the consumer side when cancelled mid-stream.
    """
    work: queue.Queue[tuple[int, np.ndarray, np.ndarray, np.ndarray] | None | Exception] = (
        queue.Queue(maxsize=max(1, int(queue_depth)))
    )
    stop = threading.Event()

    def _producer() -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            work.put(ValueError(f"Could not open source video: {video_path}"))
            return
        try:
            with gzip.open(tracks_path, "rt", encoding="utf-8") as tracks:
                for index, line in enumerate(tracks):
                    if stop.is_set() or (cancelled and cancelled()):
                        work.put(IsolateCancelled("A newer selection replaced this isolate job."))
                        return
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
                    work.put((index, frame, keep, erase))
            work.put(None)
        except Exception as exc:  # noqa: BLE001 — surface to consumer
            work.put(exc)
        finally:
            cap.release()

    thread = threading.Thread(target=_producer, name="isolate-prep", daemon=True)
    thread.start()
    try:
        while True:
            item = work.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item
    finally:
        stop.set()
        # Drain so the producer can exit if it is blocked on a full queue.
        while not work.empty():
            try:
                work.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=5.0)


def _render_lama_pipelined(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    isolate_cfg: dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
    mode_label: str = "ai",
) -> Path:
    """LaMa-only isolate with CPU prep pipelined ahead of GPU inference.

    Decode + mask building run on a background thread so the GPU stays busy.
    Used by ``ai`` mode and by ``hybrid`` when ``lama_full_erase`` makes the
    temporal seed a no-op (LaMa overwrites the whole hole anyway).
    """
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}
    _boost_isolate_runtime(cfg=cfg)

    if progress_callback:
        progress_callback(0, 1, "loading_lama")
    inpainter = _make_lama(cfg)

    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = resolve_tracks_jsonl(artifact_dir)
    total = int(canonical.get("source_frame_count") or 0)
    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))
    edge_feather_px = int(cfg.get("edge_feather_px", 0))
    write_quality = bool(cfg.get("write_quality_metrics", True))
    queue_depth = _resolve_worker_count(
        cfg.get("prep_queue_depth"),
        max(4, (os.cpu_count() or 4) // 2),
    )

    cap_probe = cv2.VideoCapture(str(video_path))
    if not cap_probe.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap_probe.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    temp_path = artifact_dir / f"focused_isolated.{generation}.tmp.mp4"
    output_path = artifact_dir / "focused_isolated.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create isolate preview: {temp_path}")

    processed = 0
    quality = _QualityAccumulator()
    try:
        for index, frame, keep, erase in _iter_prepared_lama_frames(
            video_path=video_path,
            tracks_path=tracks_path,
            selected_group_ids=selected_group_ids,
            height=height,
            width=width,
            erase_dilate=erase_dilate,
            keep_dilate=keep_dilate,
            bbox_expand=bbox_expand,
            queue_depth=queue_depth,
            cancelled=cancelled,
        ):
            if cancelled and cancelled():
                raise IsolateCancelled("A newer selection replaced this isolate job.")
            if erase.any():
                out = inpainter.inpaint_bgr(frame, erase)
                quality.record_lama()
                if keep.any():
                    out = out.copy()
                    out[keep > 0] = frame[keep > 0]
                if edge_feather_px > 0:
                    out = _feather_seam(out, erase, edge_feather_px, keep)
            else:
                out = frame
            quality.record_frame(erase, np.zeros_like(erase))
            writer.write(out)
            processed = index + 1
            if progress_callback:
                progress_callback(
                    processed,
                    max(total, processed),
                    "ai_inpainting" if mode_label == "ai" else "hybrid_lama_fast",
                )
    finally:
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
    if write_quality:
        _write_quality_metrics(
            artifact_dir, generation, quality, mode=mode_label, cfg=cfg
        )
    return output_path


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
    _boost_isolate_runtime(cfg=cfg)
    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))
    use_homography = bool(cfg.get("use_homography", True))
    homography_min_inliers = int(cfg.get("homography_min_inliers", 12))
    wide_neighbor_search = bool(cfg.get("wide_neighbor_search", True))
    write_quality = bool(cfg.get("write_quality_metrics", True))
    warp_workers = _resolve_worker_count(
        cfg.get("warp_workers"),
        max(2, (os.cpu_count() or 4) // 2),
    )
    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = resolve_tracks_jsonl(artifact_dir)
    total = int(canonical.get("source_frame_count") or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radius = max(6, min(18, int(round(fps * 0.5))))
    half = max(3, radius // 2)
    offsets = _neighbor_offsets(radius, half, wide_neighbor_search)

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
    quality = _QualityAccumulator()

    def process(index: int) -> None:
        target = window[index]
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
            use_homography=use_homography,
            homography_min_inliers=homography_min_inliers,
            warp_workers=warp_workers,
            quality=quality,
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
    if write_quality:
        _write_quality_metrics(artifact_dir, generation, quality, mode="temporal", cfg=cfg)
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
    """Best path: temporal fill from neighbors, then LaMa only on leftovers.

    When ``lama_full_erase`` is on (default), LaMa overwrites the whole hole —
    the temporal seed is skipped and we run the pipelined LaMa-fast path so
    CPU prep and GPU inference overlap.
    """
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}
    lama_full_erase = bool(cfg.get("lama_full_erase", True))
    skip_temporal = bool(cfg.get("skip_temporal_when_full_erase", True))
    if lama_full_erase and skip_temporal:
        return _render_lama_pipelined(
            job_dir,
            selected_group_ids,
            generation,
            isolate_cfg=cfg,
            progress_callback=progress_callback,
            cancelled=cancelled,
            quiet=quiet,
            mode_label="hybrid",
        )

    _boost_isolate_runtime(cfg=cfg)
    if progress_callback:
        progress_callback(0, 1, "loading_lama")
    inpainter = _make_lama(cfg)

    erase_dilate = int(cfg.get("erase_dilate", 13))
    keep_dilate = int(cfg.get("keep_dilate", 7))
    bbox_expand = float(cfg.get("bbox_expand", 0.12))
    leftover_min = int(cfg.get("lama_leftover_min_pixels", 64))
    min_temporal = int(cfg.get("min_temporal_samples", 2))
    hard_composite = bool(cfg.get("hard_composite", True))
    use_homography = bool(cfg.get("use_homography", True))
    homography_min_inliers = int(cfg.get("homography_min_inliers", 12))
    wide_neighbor_search = bool(cfg.get("wide_neighbor_search", True))
    edge_feather_px = int(cfg.get("edge_feather_px", 0))
    write_quality = bool(cfg.get("write_quality_metrics", True))
    warp_workers = _resolve_worker_count(
        cfg.get("warp_workers"),
        max(2, (os.cpu_count() or 4) // 2),
    )

    artifact_dir = Path(job_dir).resolve()
    canonical = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    video_path = Path(canonical["video_path"])
    tracks_path = resolve_tracks_jsonl(artifact_dir)
    total = int(canonical.get("source_frame_count") or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radius = max(6, min(18, int(round(fps * 0.5))))
    half = max(3, radius // 2)
    offsets = _neighbor_offsets(radius, half, wide_neighbor_search)

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
    quality = _QualityAccumulator()

    def process(index: int) -> None:
        target = window[index]
        neighbors = [
            window[index + offset]
            for offset in offsets
            if index + offset in window
        ]
        out, leftover = _reconstruct_frame(
            target,
            neighbors,
            width,
            height,
            use_spatial_fallback=False,
            min_temporal_samples=min_temporal,
            hard_composite=hard_composite,
            use_homography=use_homography,
            homography_min_inliers=homography_min_inliers,
            edge_feather_px=edge_feather_px,
            warp_workers=warp_workers,
            quality=quality,
        )
        erase = target.get("erase")
        keep = target.get("keep")
        lama_mask = leftover
        if lama_mask is not None and keep is not None and keep.any():
            lama_mask = cv2.bitwise_and(lama_mask, cv2.bitwise_not(keep))
        if (
            lama_mask is not None
            and lama_mask.any()
            and int(cv2.countNonZero(lama_mask)) >= leftover_min
        ):
            out = inpainter.inpaint_bgr(out, lama_mask)
            quality.record_lama()
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
    if write_quality:
        _write_quality_metrics(artifact_dir, generation, quality, mode="hybrid", cfg=cfg)
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
    """Clean Isolate (AI): LaMa inpaint with CPU prep pipelined ahead of GPU."""
    cfg = {**DEFAULT_ISOLATE_CFG, **(isolate_cfg or {})}
    return _render_lama_pipelined(
        job_dir,
        selected_group_ids,
        generation,
        isolate_cfg=cfg,
        progress_callback=progress_callback,
        cancelled=cancelled,
        quiet=quiet,
        mode_label="ai",
    )


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
