#!/usr/bin/env python3
"""Phase 0b — detect vs ByteTrack first-lock for a star-ish crop.

Finds the highest-confidence detection near a seed point (or jersey-color
heuristic), then reports when that body first gets a track_id and how often
the ID switches. Use this when raw detection is already healthy.

Example:
  python scripts/player_presence_track_lock.py \\
    --video "data/input_videos/Caitlin Clark clip.mp4" \\
    --seed-xy 0.78,0.48
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _red_score(frame: np.ndarray, bbox: list[int]) -> float:
    """Fraction of non-skin-ish red pixels in the jersey band of the box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = max(y2 - y1, 1)
    # Upper torso band
    y_a = y1 + int(0.15 * h)
    y_b = y1 + int(0.55 * h)
    crop = frame[max(0, y_a) : max(y_a + 1, y_b), max(0, x1) : max(x1 + 1, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red wraps hue; Fever home reds
    m1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _pick_star(
    frame: np.ndarray,
    dets: list[dict],
    *,
    seed_xy: tuple[float, float] | None,
    prefer_red: bool,
) -> dict | None:
    if not dets:
        return None
    h, w = frame.shape[:2]
    scored = []
    for det in dets:
        cx, cy = _center(det["bbox_xyxy"])
        conf = float(det.get("confidence") or 0.0)
        red = _red_score(frame, det["bbox_xyxy"]) if prefer_red else 0.0
        if seed_xy is not None:
            sx, sy = seed_xy[0] * w, seed_xy[1] * h
            prox = 1.0 / (1.0 + (_dist2((cx, cy), (sx, sy)) ** 0.5) / max(w, 1))
        else:
            prox = 0.0
        score = conf + 0.35 * red + 0.55 * prox
        scored.append((score, det))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def run(
    video: Path,
    *,
    weights: Path,
    conf: float,
    iou: float,
    device: str,
    seed_xy: tuple[float, float] | None,
    prefer_red: bool,
    out: Path,
    previews: Path | None,
) -> dict:
    from ultralytics import YOLO

    from src.object_tracker import ObjectTracker
    from src.pipeline.hardware import resolve_device
    from src.showcase.exclusions import (
        filter_detections,
        load_exclusion_set,
        resolve_exclusion_path,
    )
    from src.showcase.run_showcase import detect_players_seg

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    resolved = resolve_device(device)
    model = YOLO(str(weights))
    try:
        target = f"cuda:{resolved}" if str(resolved).isdigit() else str(resolved)
        model.to(target)
    except Exception:
        pass

    excl_path = resolve_exclusion_path(
        PROJECT_ROOT / "data" / "exclusions", video.stem
    )
    exclusions = load_exclusion_set(
        excl_path, width=width, height=height, clip_id=video.stem
    )
    tracker = ObjectTracker(
        frame_rate=fps,
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.75,
    )

    if previews:
        previews.mkdir(parents=True, exist_ok=True)

    first_detect = None
    first_track = None
    primary_tid = None
    tid_history: list[int] = []
    detect_flags: list[bool] = []
    track_flags: list[bool] = []
    rows: list[dict] = []

    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        raw = detect_players_seg(model, frame, conf=conf, iou=iou)
        kept = filter_detections(raw, exclusions, class_names=("player",))
        tracked = tracker.update(kept)

        star_det = _pick_star(frame, kept, seed_xy=seed_xy, prefer_red=prefer_red)
        has_det = star_det is not None
        detect_flags.append(has_det)
        if has_det and first_detect is None:
            first_detect = frame_id

        matched_tid = None
        if star_det is not None:
            sx, sy = _center(star_det["bbox_xyxy"])
            best = None
            best_d = 1e18
            for t in tracked:
                tx, ty = _center(t["bbox_xyxy"])
                d = _dist2((sx, sy), (tx, ty))
                if d < best_d:
                    best_d = d
                    best = t
            # Require reasonable spatial match (~30% of shorter side)
            thr = (min(width, height) * 0.08) ** 2
            if best is not None and best_d <= thr:
                matched_tid = int(best.get("track_id", -1))

        has_track = matched_tid is not None and matched_tid >= 0
        track_flags.append(has_track)
        if has_track:
            tid_history.append(matched_tid)
            if first_track is None:
                first_track = frame_id
                primary_tid = matched_tid

        rows.append(
            {
                "frame": frame_id,
                "t": round(frame_id / fps, 3),
                "star_detected": has_det,
                "star_tracked": has_track,
                "track_id": matched_tid,
                "star_conf": None
                if star_det is None
                else round(float(star_det["confidence"]), 3),
                "n_kept": len(kept),
                "n_tracked": len(tracked),
            }
        )

        if previews is not None and frame_id in {0, first_track or -1, total // 2, total - 1}:
            canvas = frame.copy()
            for t in tracked:
                x1, y1, x2, y2 = t["bbox_xyxy"]
                color = (0, 255, 0) if t.get("track_id") == matched_tid else (255, 180, 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    canvas,
                    f"id={t.get('track_id')}",
                    (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if seed_xy is not None:
                cv2.circle(
                    canvas,
                    (int(seed_xy[0] * width), int(seed_xy[1] * height)),
                    8,
                    (0, 0, 255),
                    2,
                )
            tag = f"f{frame_id:03d}"
            if frame_id == first_track:
                tag += "_first_track"
            cv2.imwrite(str(previews / f"{tag}.jpg"), canvas)

    cap.release()

    switches = 0
    if tid_history:
        for a, b in zip(tid_history, tid_history[1:]):
            if a != b:
                switches += 1
    counts = Counter(tid_history)
    dominant_tid, dominant_n = (None, 0)
    if counts:
        dominant_tid, dominant_n = counts.most_common(1)[0]

    report = {
        "video": str(video.resolve()),
        "weights": str(weights.resolve()),
        "seed_xy_norm": list(seed_xy) if seed_xy else None,
        "prefer_red": prefer_red,
        "fps": fps,
        "frames": frame_id + 1,
        "milestones": {
            "first_star_detection_frame": first_detect,
            "first_star_track_frame": first_track,
            "detect_to_track_lag_frames": None
            if first_detect is None or first_track is None
            else first_track - first_detect,
            "primary_track_id_at_first_lock": primary_tid,
            "dominant_track_id": dominant_tid,
            "dominant_track_frames": dominant_n,
        },
        "aggregates": {
            "pct_frames_star_detected": round(
                100.0 * sum(detect_flags) / max(len(detect_flags), 1), 2
            ),
            "pct_frames_star_tracked": round(
                100.0 * sum(track_flags) / max(len(track_flags), 1), 2
            ),
            "track_id_switches": switches,
            "unique_track_ids_on_star": len(counts),
        },
        "diagnosis_hints": [],
        "timeline": rows,
    }
    hints = report["diagnosis_hints"]
    lag = report["milestones"]["detect_to_track_lag_frames"]
    if lag is not None and lag > 5:
        hints.append(
            f"ByteTrack lags detection by {lag} frames — activation/matching may be too strict."
        )
    if report["aggregates"]["pct_frames_star_detected"] > 90 and report[
        "aggregates"
    ]["pct_frames_star_tracked"] < 80:
        hints.append(
            "Star is detected often but tracked less — tracker drops / ID churn."
        )
    if switches > 10:
        hints.append(
            f"{switches} ID switches on star body — stitch/appearance merge is the training target."
        )
    if report["aggregates"]["pct_frames_star_detected"] < 85:
        hints.append(
            "Star detection presence is the bottleneck — mine hard frames + retrain."
        )
    if not hints:
        hints.append(
            "Detect+track look healthy for this seed; late UI lock may be selection/identity UX."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="data/input_videos/Caitlin Clark clip.mp4")
    p.add_argument("--weights", default="outputs/training_loop_v2/best_player_seg_v2.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--seed-xy",
        default="0.78,0.48",
        help="Normalized seed for star body at start (x,y). Empty to disable.",
    )
    p.add_argument("--prefer-red", action="store_true", default=True)
    p.add_argument("--no-prefer-red", action="store_false", dest="prefer_red")
    p.add_argument("--out", default="outputs/presence_loop/clark_track_lock.json")
    p.add_argument("--previews", default="outputs/presence_loop/clark_track_previews")
    args = p.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = PROJECT_ROOT / video
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    previews = Path(args.previews) if args.previews else None
    if previews and not previews.is_absolute():
        previews = PROJECT_ROOT / previews

    seed = None
    if args.seed_xy.strip():
        xs, ys = args.seed_xy.split(",")
        seed = (float(xs), float(ys))

    report = run(
        video,
        weights=weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        seed_xy=seed,
        prefer_red=args.prefer_red,
        out=out,
        previews=previews,
    )
    m = report["milestones"]
    a = report["aggregates"]
    print("=== Star track-lock report ===")
    print(f"video: {report['video']}")
    print(
        f"first detect frame={m['first_star_detection_frame']}  "
        f"first track={m['first_star_track_frame']}  lag={m['detect_to_track_lag_frames']}"
    )
    print(
        f"presence detect={a['pct_frames_star_detected']}%  "
        f"tracked={a['pct_frames_star_tracked']}%  "
        f"id_switches={a['track_id_switches']}  unique_ids={a['unique_track_ids_on_star']}"
    )
    print(f"dominant tid={m['dominant_track_id']} ({m['dominant_track_frames']} frames)")
    for h in report["diagnosis_hints"]:
        print(f"  - {h}")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
