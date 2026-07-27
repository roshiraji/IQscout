"""Camera-motion homographies so trails can be pinned to the floor.

Trajectories are stored in normalized *screen* space, so a broadcast pan drags
every trail sideways even when the player never moved. This module estimates,
for each gameplay frame, the homography that maps that frame back onto an
anchor frame using only background texture (court lines, sideline, stands) with
the tracked players masked out. Warping a point through those matrices puts it
where it belongs on the court instead of where it landed in the frame.

Nothing here re-runs detection: it reads the canonical tracks only to know which
pixels to ignore. Estimation is per gameplay segment — a camera cut has no
common background, so each segment gets its own anchor. When a frame cannot be
matched confidently the previous transform is held, and a run of held frames
starts a fresh anchor rather than letting cumulative drift lie to the UI.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pipeline.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)

CAMERA_MOTION_FILENAME = "camera_motion.json"
COORDINATE_SPACE = "normalized_anchor_frame"

DEFAULT_CAMERA_MOTION_CONFIG: dict[str, Any] = {
    "enabled": True,
    # Flow runs on a downscaled copy; homographies are stored normalized, so
    # the working size only trades accuracy for speed.
    "working_side": 640,
    "max_features": 900,
    "feature_quality": 0.01,
    "min_feature_distance": 8.0,
    # Below this many RANSAC inliers the frame is not trusted.
    "min_inliers": 24,
    "ransac_reproj_px": 2.5,
    # Players are moving foreground: their pixels must not vote on camera motion.
    "player_mask_pad": 14,
    # Consecutive unmatched frames before we stop pretending the anchor holds.
    "max_held_frames": 6,
}


def merge_camera_motion_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CAMERA_MOTION_CONFIG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s for camera motion: %s", path.name, exc)
    return rows


def _boxes_by_frame(job_dir: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Full-resolution player boxes per frame, used only as a feature mask."""
    for name in ("tracks.jsonl.gz", "tracks.jsonl", "locked_tracks.jsonl.gz"):
        rows = _load_jsonl_gz(job_dir / name)
        if rows:
            break
    else:
        return {}

    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for row in rows:
        frame = _as_int(row.get("frame_number") or row.get("frame"))
        if frame is None:
            continue
        frame_boxes: list[tuple[float, float, float, float]] = []
        for det in row.get("tracks") or []:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox_xyxy")
            if not bbox or len(list(bbox)) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in list(bbox)[:4])
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            frame_boxes.append((x1, y1, x2, y2))
        if frame_boxes:
            boxes[frame] = frame_boxes
    return boxes


def _usable_spans(segments_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for seg in (segments_payload or {}).get("segments") or []:
        if not isinstance(seg, dict) or not seg.get("usable_for_analytics"):
            continue
        start = _as_int(seg.get("start_frame"))
        end = _as_int(seg.get("end_frame"))
        if start is None or end is None or end < start:
            continue
        spans.append(
            {
                "segment_id": _as_int(seg.get("segment_id")),
                "start_frame": start,
                "end_frame": end,
            }
        )
    spans.sort(key=lambda s: s["start_frame"])
    return spans


def _background_mask(
    shape: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
    scale: float,
    pad: int,
) -> np.ndarray:
    """255 where features may be sampled (everything but padded player boxes)."""
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        sx1 = int(max(0, x1 * scale - pad))
        sy1 = int(max(0, y1 * scale - pad))
        sx2 = int(min(width, x2 * scale + pad))
        sy2 = int(min(height, y2 * scale + pad))
        if sx2 > sx1 and sy2 > sy1:
            mask[sy1:sy2, sx1:sx2] = 0
    return mask


def _transform_is_sane(matrix: np.ndarray, width: int, height: int) -> bool:
    """Reject warps that fold, mirror, or blow up the frame."""
    if matrix is None or not np.all(np.isfinite(matrix)):
        return False
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(warped)):
        return False
    area = float(cv2.contourArea(warped.astype(np.float32)))
    base = float(width * height)
    if base <= 0 or area <= 0:
        return False
    ratio = area / base
    if ratio < 0.25 or ratio > 4.0:
        return False
    # A pan should not shove the frame more than a couple of frame widths away.
    if np.max(np.abs(warped)) > 4.0 * max(width, height):
        return False
    return True


def _normalize_homography(matrix: np.ndarray, width: int, height: int) -> list[float]:
    """Pixel-space H → normalized-coordinate H, flattened to 8 values."""
    to_norm = np.array(
        [[1.0 / width, 0.0, 0.0], [0.0, 1.0 / height, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    to_pixels = np.array(
        [[float(width), 0.0, 0.0], [0.0, float(height), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = to_norm @ matrix.astype(np.float64) @ to_pixels
    if abs(normalized[2, 2]) > 1e-12:
        normalized = normalized / normalized[2, 2]
    return [round(float(v), 7) for v in normalized.reshape(-1)[:8]]


def _estimate_pair(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_mask: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """Homography mapping ``gray`` coordinates onto ``prev_gray`` coordinates."""
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(cfg["max_features"]),
        qualityLevel=float(cfg["feature_quality"]),
        minDistance=float(cfg["min_feature_distance"]),
        mask=prev_mask,
        blockSize=7,
    )
    if features is None or len(features) < int(cfg["min_inliers"]):
        return None

    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, features, None, winSize=(21, 21), maxLevel=3
    )
    if forward is None:
        return None
    backward, _, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, forward, None, winSize=(21, 21), maxLevel=3
    )
    if backward is None:
        return None

    status = status.reshape(-1).astype(bool)
    # Forward-backward error rejects flow that latched onto the wrong texture.
    fb_error = np.linalg.norm(
        features.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
    )
    keep = status & (fb_error < 1.0)
    if int(keep.sum()) < int(cfg["min_inliers"]):
        return None

    src = forward.reshape(-1, 2)[keep]
    dst = features.reshape(-1, 2)[keep]
    matrix, inlier_mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["ransac_reproj_px"]),
        maxIters=2000,
        confidence=0.995,
    )
    if matrix is None or inlier_mask is None:
        return None
    if int(inlier_mask.sum()) < int(cfg["min_inliers"]):
        return None
    height, width = prev_gray.shape[:2]
    if not _transform_is_sane(matrix, width, height):
        return None
    return matrix


def estimate_camera_motion_for_job(
    job_dir: str | Path,
    *,
    segments_payload: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate per-frame camera homographies and write ``camera_motion.json``.

    Returns the payload that was written. When motion cannot be estimated the
    payload is still written with ``available: false`` so the UI can explain the
    court-anchor toggle instead of silently doing nothing.
    """
    job_dir = Path(job_dir).resolve()
    cfg = merge_camera_motion_config(config)
    analytics_dir = job_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    out_path = analytics_dir / CAMERA_MOTION_FILENAME

    def _unavailable(note: str) -> dict[str, Any]:
        payload = {
            "job_id": job_dir.name,
            "available": False,
            "coordinate_space": COORDINATE_SPACE,
            "anchors": [],
            "frames": [],
            "note": note,
        }
        write_json_atomic(out_path, payload)
        return payload

    if not cfg.get("enabled", True):
        return _unavailable("Camera-motion estimation is disabled in config.")

    summary_path = job_dir / "summary.json"
    if not summary_path.is_file():
        return _unavailable("summary.json is missing; cannot locate the source video.")
    try:
        canonical = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _unavailable(f"Could not read summary.json: {exc}")

    video_path = Path(str(canonical.get("video_path") or ""))
    if not video_path.is_file():
        return _unavailable(f"Source video is unavailable: {video_path}")

    if segments_payload is None:
        from src.analytics.trajectories import SEGMENTS_FILENAME, load_segments_payload

        segments_payload = load_segments_payload(analytics_dir / SEGMENTS_FILENAME)
    spans = _usable_spans(segments_payload)
    if not spans:
        return _unavailable("No usable gameplay segments to anchor against.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _unavailable(f"Could not open source video: {video_path}")

    full_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    full_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    working_side = max(320, int(cfg["working_side"]))
    scale = min(1.0, working_side / float(max(full_width, full_height)))
    work_width = max(2, int(round(full_width * scale)))
    work_height = max(2, int(round(full_height * scale)))
    pad = int(cfg["player_mask_pad"])

    boxes_by_frame = _boxes_by_frame(job_dir)
    span_by_frame: dict[int, dict[str, Any]] = {}
    for span in spans:
        for frame in range(span["start_frame"], span["end_frame"] + 1):
            span_by_frame[frame] = span

    identity = np.eye(3, dtype=np.float64)
    frames_out: list[list[Any]] = []
    anchors: list[dict[str, Any]] = []
    anchor_id = -1
    anchor_ref: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    prev_frame_number: int | None = None
    prev_span_id: Any = None
    held_run = 0
    held_total = 0
    matched_total = 0

    def _open_anchor(frame_number: int, span: dict[str, Any]) -> None:
        nonlocal anchor_id, anchor_ref, held_run
        anchor_id += 1
        anchor_ref = identity.copy()
        held_run = 0
        anchors.append(
            {
                "anchor_id": anchor_id,
                "segment_id": span.get("segment_id"),
                "reference_frame": frame_number,
                "start_frame": frame_number,
                "end_frame": frame_number,
                "frame_count": 0,
                "held_frames": 0,
            }
        )

    def _record(frame_number: int, matrix: np.ndarray) -> None:
        anchors[-1]["end_frame"] = frame_number
        anchors[-1]["frame_count"] += 1
        frames_out.append(
            [frame_number, anchor_id, *_normalize_homography(matrix, work_width, work_height)]
        )

    frame_number = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            span = span_by_frame.get(frame_number)
            if span is None:
                prev_gray = None
                prev_mask = None
                prev_frame_number = None
                prev_span_id = None
                continue

            small = cv2.resize(
                frame, (work_width, work_height), interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mask = _background_mask(
                (work_height, work_width),
                boxes_by_frame.get(frame_number, []),
                scale,
                pad,
            )

            new_span = span.get("segment_id") != prev_span_id
            contiguous = (
                prev_frame_number is not None and frame_number == prev_frame_number + 1
            )
            if prev_gray is None or new_span or not contiguous:
                _open_anchor(frame_number, span)
                _record(frame_number, identity)
            else:
                matrix = _estimate_pair(prev_gray, gray, prev_mask, cfg)
                if matrix is None:
                    held_run += 1
                    held_total += 1
                    if held_run > int(cfg["max_held_frames"]):
                        # Drift would be invisible but real; re-anchor instead.
                        _open_anchor(frame_number, span)
                        _record(frame_number, identity)
                    else:
                        anchors[-1]["held_frames"] += 1
                        _record(frame_number, anchor_ref)  # type: ignore[arg-type]
                else:
                    held_run = 0
                    matched_total += 1
                    anchor_ref = (anchor_ref if anchor_ref is not None else identity) @ matrix
                    if abs(anchor_ref[2, 2]) > 1e-12:
                        anchor_ref = anchor_ref / anchor_ref[2, 2]
                    if not _transform_is_sane(anchor_ref, work_width, work_height):
                        _open_anchor(frame_number, span)
                        _record(frame_number, identity)
                    else:
                        _record(frame_number, anchor_ref)

            prev_gray = gray
            prev_mask = mask
            prev_frame_number = frame_number
            prev_span_id = span.get("segment_id")
    finally:
        cap.release()

    if not frames_out:
        return _unavailable("No gameplay frames were readable from the source video.")

    payload = {
        "job_id": job_dir.name,
        "available": True,
        "coordinate_space": COORDINATE_SPACE,
        "frame_width": full_width,
        "frame_height": full_height,
        "fps": fps,
        "working_width": work_width,
        "working_height": work_height,
        "anchor_count": len(anchors),
        "frame_count": len(frames_out),
        "matched_frames": matched_total,
        "held_frames": held_total,
        "anchors": anchors,
        # [frame_number, anchor_id, h00, h01, h02, h10, h11, h12, h20, h21]
        # maps normalized frame coordinates onto the anchor frame (h22 == 1).
        "frames": frames_out,
        "note": (
            "Background homographies per gameplay segment. Points warped through "
            "these matrices stay on the court while the camera pans; accuracy "
            "degrades on heavy zoom or when the floor is mostly occluded."
        ),
    }
    # One line per-frame matrix would balloon the file; the UI fetches this.
    write_json_atomic(out_path, payload, indent=None)
    logger.info(
        "Camera motion: %d frames across %d anchor(s) (%d held) for %s",
        len(frames_out),
        len(anchors),
        held_total,
        job_dir.name,
    )
    return payload


def load_camera_motion(job_dir: str | Path) -> dict[str, Any] | None:
    path = Path(job_dir).resolve() / "analytics" / CAMERA_MOTION_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read camera_motion.json: %s", exc)
        return None
    return payload if isinstance(payload, dict) else None
