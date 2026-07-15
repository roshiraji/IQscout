"""Lean showcase inference: yolo11n-seg + exclusions + tracker + mask/bbox/ID overlay.

No Grounding DINO. No trajectory chalk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.frame_extractor import extract_all_frames, get_video_info
from src.object_tracker import update_tracker
from src.showcase.exclusions import (
    filter_detections,
    load_exclusion_set,
    resolve_exclusion_path,
)

logger = logging.getLogger(__name__)

# Stable palette for track IDs (BGR).
_ID_PALETTE = [
    (40, 180, 40),
    (220, 160, 40),
    (60, 60, 245),
    (0, 200, 200),
    (200, 100, 255),
    (180, 220, 40),
    (255, 140, 0),
    (100, 100, 255),
]


def _id_color(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (180, 180, 180)
    return _ID_PALETTE[track_id % len(_ID_PALETTE)]


def _mask_from_result(result, index: int, width: int, height: int) -> list[list[float]] | None:
    if result.masks is None:
        return None
    try:
        pts = result.masks.xy[index]
        if pts is None or len(pts) < 3:
            return None
        return [[float(p[0]), float(p[1])] for p in pts]
    except Exception:
        return None


def _smooth_polygon(
    prev: list[list[float]] | None,
    curr: list[list[float]] | None,
    alpha: float = 0.35,
) -> list[list[float]] | None:
    """Light EMA-style smoothing when polygon lengths match; else return curr."""
    if curr is None:
        return prev
    if prev is None or len(prev) != len(curr) or alpha <= 0:
        return curr
    out = []
    for (px, py), (cx, cy) in zip(prev, curr):
        out.append([px * (1 - alpha) + cx * alpha, py * (1 - alpha) + cy * alpha])
    return out


def detect_players_seg(
    model,
    image_bgr: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.70,
) -> list[dict]:
    results = model.predict(image_bgr, conf=conf, iou=iou, verbose=False)
    if not results:
        return []
    r0 = results[0]
    h, w = image_bgr.shape[:2]
    dets: list[dict] = []
    if r0.boxes is None:
        return dets
    for i in range(len(r0.boxes)):
        xyxy = r0.boxes.xyxy[i].tolist()
        score = float(r0.boxes.conf[i])
        cls_id = int(r0.boxes.cls[i]) if r0.boxes.cls is not None else 0
        # Player-only showcase: class 0
        if cls_id != 0:
            continue
        poly = _mask_from_result(r0, i, w, h)
        dets.append({
            "class_id": 0,
            "class_name": "player",
            "canonical_class": "player",
            "confidence": round(score, 4),
            "bbox_xyxy": [int(v) for v in xyxy],
            "mask_polygon": poly,
        })
    return dets


def draw_showcase_frame(
    image: np.ndarray,
    tracked: list[dict],
    *,
    frame_id: int,
    timestamp: float,
    draw_hud: bool = True,
) -> np.ndarray:
    canvas = image.copy()
    for det in tracked:
        color = _id_color(int(det.get("track_id", -1)))
        poly = det.get("mask_polygon")
        if poly and len(poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)
        box = det.get("bbox_xyxy") or []
        if len(box) >= 4:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            tid = det.get("track_id", -1)
            conf = det.get("confidence", 0.0)
            label = f"#{tid} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(canvas, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(canvas, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if draw_hud:
        text = f"frame {frame_id:04d} | t={timestamp:.2f}s | players={len(tracked)}"
        cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run_showcase(
    *,
    video: Path,
    weights: Path,
    output_video: Path,
    frames_dir: Path,
    exclusions_dir: Path,
    clip_id: str | None = None,
    conf: float = 0.25,
    iou: float = 0.70,
    min_track_frames: int = 3,
    mask_smooth_alpha: float = 0.35,
    max_frames: int | None = None,
    write_tracks_json: bool = True,
) -> Path:
    from ultralytics import YOLO

    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Train first: python -m src.showcase train --config configs/showcase_train.yaml"
        )

    info = get_video_info(video)
    frame_paths = extract_all_frames(video, frames_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    from src.showcase.ingest import slugify
    cid = clip_id or slugify(video.name)
    excl_path = resolve_exclusion_path(exclusions_dir, cid)
    exclusions = load_exclusion_set(excl_path, width=info.width, height=info.height, clip_id=cid)

    model = YOLO(str(weights))
    logger.info(
        "Showcase | video=%s | weights=%s | frames=%d | exclusions=%s",
        video.name, weights.name, len(frame_paths), excl_path or "defaults",
    )

    all_tracked: dict[int, list[dict]] = {}
    mask_memory: dict[int, list[list[float]]] = {}
    track_lengths: dict[int, int] = {}
    coverage_hits = 0
    min_players = 1

    for fp in tqdm(frame_paths, desc="Showcase detect+track", unit="frame"):
        try:
            fid = int(fp.stem.split("_")[-1])
        except ValueError:
            fid = 0
        ts = fid / info.fps if info.fps else 0.0
        image = cv2.imread(str(fp))
        if image is None:
            all_tracked[fid] = []
            continue

        dets = detect_players_seg(model, image, conf=conf, iou=iou)
        dets = filter_detections(dets, exclusions, class_names=("player",))
        tracked = update_tracker(dets)

        # Attach / smooth masks by track id.
        for det in tracked:
            tid = int(det.get("track_id", -1))
            if tid >= 0:
                track_lengths[tid] = track_lengths.get(tid, 0) + 1
            poly = det.get("mask_polygon")
            if tid >= 0 and mask_smooth_alpha > 0:
                smoothed = _smooth_polygon(mask_memory.get(tid), poly, alpha=mask_smooth_alpha)
                if smoothed is not None:
                    det["mask_polygon"] = smoothed
                    mask_memory[tid] = smoothed

        # Drop brand-new low-confidence tracks (anti-flicker).
        filtered = []
        for det in tracked:
            tid = int(det.get("track_id", -1))
            age = track_lengths.get(tid, 0)
            if age < min_track_frames and float(det.get("confidence") or 0) < conf + 0.1:
                continue
            filtered.append(det)

        if len(filtered) >= min_players:
            coverage_hits += 1
        all_tracked[fid] = filtered

    # Render
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_video.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {tmp}")

    for fp in tqdm(frame_paths, desc="Showcase render", unit="frame"):
        try:
            fid = int(fp.stem.split("_")[-1])
        except ValueError:
            fid = 0
        ts = fid / info.fps if info.fps else 0.0
        image = cv2.imread(str(fp))
        if image is None:
            image = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        annotated = draw_showcase_frame(image, all_tracked.get(fid, []), frame_id=fid, timestamp=ts)
        writer.write(annotated)
    writer.release()

    # Prefer H.264 when ffmpeg exists.
    from src.render_tracking_overlay import _reencode
    _reencode(tmp, output_video)
    tmp.unlink(missing_ok=True)

    report = {
        "video": str(video),
        "weights": str(weights),
        "output_video": str(output_video),
        "frames": len(frame_paths),
        "frame_coverage": round(coverage_hits / max(len(frame_paths), 1), 4),
        "unique_tracks": len(track_lengths),
        "exclusions": str(excl_path) if excl_path else "defaults",
        "note": "Highlight hard cuts may reset track IDs — expected limitation.",
    }
    report_path = output_video.parent / "showcase_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if write_tracks_json:
        # Compact tracks dump (bbox + id per frame).
        compact = {
            str(fid): [
                {
                    "track_id": d.get("track_id"),
                    "confidence": d.get("confidence"),
                    "bbox_xyxy": d.get("bbox_xyxy"),
                    "has_mask": bool(d.get("mask_polygon")),
                }
                for d in dets
            ]
            for fid, dets in all_tracked.items()
        }
        (output_video.parent / "showcase_tracks.json").write_text(
            json.dumps(compact), encoding="utf-8",
        )

    logger.info("Showcase complete → %s | coverage=%.3f", output_video, report["frame_coverage"])
    return output_video
