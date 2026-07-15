"""High-quality temporal reconstruction for selected-player isolate previews."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from video_encoding import encode_for_browser


class IsolateCancelled(RuntimeError):
    """Raised when a newer Confirm supersedes an isolate render."""


def _group_id(det: dict) -> int:
    value = det.get("person_group_id")
    if value is None:
        value = det.get("track_id", -1)
    return int(value)


def _person_mask(
    height: int,
    width: int,
    det: dict,
    *,
    fallback_expand: float = 0.55,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = det.get("mask_polygon")
    if polygon and len(polygon) >= 3:
        points = np.asarray(
            [[int(point[0]), int(point[1])] for point in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        if cv2.countNonZero(mask) >= 32:
            return mask

    x1, y1, x2, y2 = [
        int(value)
        for value in (det.get("bbox_xyxy") or [0, 0, 0, 0])[:4]
    ]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return mask
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (
        max(1, int((x2 - x1) * fallback_expand)),
        max(1, int((y2 - y1) * (fallback_expand + 0.03))),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _build_masks(
    height: int,
    width: int,
    detections: list[dict],
    selected_group_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (all-player occupancy, selected-player keep mask)."""
    occupancy = np.zeros((height, width), dtype=np.uint8)
    keep = np.zeros((height, width), dtype=np.uint8)
    for det in detections:
        mask = _person_mask(height, width, det)
        if not mask.any():
            continue
        occupancy = cv2.bitwise_or(occupancy, mask)
        if _group_id(det) in selected_group_ids:
            keep = cv2.bitwise_or(keep, mask)

    # Cover mask fringes and compression halos before finding clean background.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    occupancy = cv2.dilate(occupancy, kernel, iterations=1)
    keep = cv2.dilate(keep, kernel, iterations=1)
    return occupancy, keep


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
) -> np.ndarray:
    frame = target["frame"]
    erase = cv2.bitwise_and(
        target["occupancy"], cv2.bitwise_not(target["keep"])
    )
    if not erase.any():
        return frame

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

    # Spatial inpaint is only a fallback for pixels no aligned clean frame saw.
    spatial = cv2.inpaint(frame, erase, 3, cv2.INPAINT_NS)
    reconstructed = spatial
    temporal = sample_count > 0
    if temporal.any():
        reconstructed[temporal] = np.clip(
            pixel_sum[temporal] / sample_count[temporal][:, None],
            0,
            255,
        ).astype(np.uint8)

    # Feather within the dilated removal area to avoid a hard pasted boundary.
    alpha = cv2.GaussianBlur(erase, (0, 0), 1.2).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    output = (
        reconstructed.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def render_temporal_isolate(
    job_dir: str | Path,
    selected_group_ids: set[int],
    generation: str,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    quiet: bool = True,
) -> Path:
    """Render scene-preserving isolate from canonical frames and masks.

    Nearby frames are feature-aligned to the target. Pixels occupied by
    non-selected people are replaced from clean aligned neighbors; Navier-
    Stokes inpaint fills only locations with no clean temporal sample.
    """
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

    detector = cv2.ORB_create(
        nfeatures=1200,
        scaleFactor=1.2,
        nlevels=6,
        fastThreshold=15,
    )
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
        writer.write(_reconstruct_frame(target, neighbors, width, height))

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
                occupancy, keep = _build_masks(
                    height, width, detections, selected_group_ids
                )
                keypoints, descriptors = _frame_features(
                    frame, occupancy, detector
                )
                window[index] = {
                    "frame": frame,
                    "occupancy": occupancy,
                    "keep": keep,
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
