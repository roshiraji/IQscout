#!/usr/bin/env python3
"""Phase 0 — diagnose why a visible player is missing until late in a clip.

Runs the live player-seg weights frame-by-frame and reports:
  - raw detector hits vs post-exclusion hits
  - first frame with N detections
  - long empty / weak stretches
  - optional top-of-key ROI occupancy (star stress region)

Example:
  python scripts/player_presence_miss_report.py \\
    --video "data/input_videos/Caitlin Clark clip.mp4" \\
    --out outputs/presence_loop/clark_miss_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _in_key_roi(cx: float, cy: float, w: int, h: int) -> bool:
    """Loose 'top of the key / central paint' band in normalized screen space.

    Tuned for half-court broadcasts; adjust later per-clip if needed.
    """
    nx, ny = cx / max(w, 1), cy / max(h, 1)
    return 0.28 <= nx <= 0.72 and 0.22 <= ny <= 0.62


def _summarize_gaps(flags: list[bool]) -> list[dict]:
    gaps: list[dict] = []
    start = None
    for i, ok in enumerate(flags):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            gaps.append({"start_frame": start, "end_frame": i - 1, "length": i - start})
            start = None
    if start is not None:
        gaps.append(
            {"start_frame": start, "end_frame": len(flags) - 1, "length": len(flags) - start}
        )
    return gaps


def run_miss_report(
    video: Path,
    *,
    weights: Path,
    conf: float,
    iou: float,
    device: str,
    stride: int,
    save_previews: Path | None,
    clip_id: str | None,
) -> dict:
    from ultralytics import YOLO

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

    excl_dir = PROJECT_ROOT / "data" / "exclusions"
    cid = clip_id or video.stem
    excl_path = resolve_exclusion_path(excl_dir, cid)
    exclusions = load_exclusion_set(excl_path, width=width, height=height, clip_id=cid)

    if save_previews:
        save_previews.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    raw_counts: list[int] = []
    kept_counts: list[int] = []
    key_counts: list[int] = []
    first_any = None
    first_key = None
    first_ge3 = None
    first_ge5 = None

    t0 = time.perf_counter()
    frame_id = -1
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % max(stride, 1) != 0:
            continue
        sampled += 1

        raw = detect_players_seg(model, frame, conf=conf, iou=iou)
        kept = filter_detections(raw, exclusions, class_names=("player",))
        key_hits = []
        for det in kept:
            cx, cy = _center(det["bbox_xyxy"])
            if _in_key_roi(cx, cy, width, height):
                key_hits.append(det)

        n_raw, n_kept, n_key = len(raw), len(kept), len(key_hits)
        raw_counts.append(n_raw)
        kept_counts.append(n_kept)
        key_counts.append(n_key)

        if first_any is None and n_kept > 0:
            first_any = frame_id
        if first_key is None and n_key > 0:
            first_key = frame_id
        if first_ge3 is None and n_kept >= 3:
            first_ge3 = frame_id
        if first_ge5 is None and n_kept >= 5:
            first_ge5 = frame_id

        dropped = n_raw - n_kept
        row = {
            "frame": frame_id,
            "t": round(frame_id / fps, 3),
            "raw": n_raw,
            "kept": n_kept,
            "key_roi": n_key,
            "excluded": dropped,
            "max_conf": round(max((d["confidence"] for d in kept), default=0.0), 3),
            "key_max_conf": round(max((d["confidence"] for d in key_hits), default=0.0), 3),
        }
        rows.append(row)

        # Save a few diagnostic frames (early / first-key / mid / late).
        if save_previews is not None:
            want = False
            tag = ""
            if frame_id == 0:
                want, tag = True, "f000_start"
            elif first_key is not None and frame_id == first_key:
                want, tag = True, f"f{frame_id:03d}_first_key"
            elif frame_id == max(total // 2 - (total // 2) % stride, 0):
                want, tag = True, f"f{frame_id:03d}_mid"
            elif frame_id >= total - stride:
                want, tag = True, f"f{frame_id:03d}_end"
            if want:
                canvas = frame.copy()
                for det in kept:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    cx, cy = _center(det["bbox_xyxy"])
                    color = (0, 255, 0) if _in_key_roi(cx, cy, width, height) else (255, 180, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        canvas,
                        f"{det['confidence']:.2f}",
                        (x1, max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                # Draw key ROI outline
                x0, x1b = int(0.28 * width), int(0.72 * width)
                y0, y1b = int(0.22 * height), int(0.62 * height)
                cv2.rectangle(canvas, (x0, y0), (x1b, y1b), (0, 0, 255), 1)
                cv2.putText(
                    canvas,
                    f"raw={n_raw} kept={n_kept} key={n_key} excl={dropped}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(save_previews / f"{tag}.jpg"), canvas)

    cap.release()
    elapsed = round(time.perf_counter() - t0, 2)

    empty_kept = [c == 0 for c in kept_counts]
    empty_key = [c == 0 for c in key_counts]
    weak_kept = [c < 3 for c in kept_counts]

    report = {
        "video": str(video.resolve()),
        "weights": str(weights.resolve()),
        "device": resolved,
        "conf": conf,
        "iou": iou,
        "stride": stride,
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": total,
        "sampled_frames": sampled,
        "runtime_seconds": elapsed,
        "exclusion_file": str(excl_path) if excl_path else None,
        "milestones": {
            "first_any_detection_frame": first_any,
            "first_key_roi_detection_frame": first_key,
            "first_3plus_players_frame": first_ge3,
            "first_5plus_players_frame": first_ge5,
            "first_any_detection_sec": None
            if first_any is None
            else round(first_any / fps, 2),
            "first_key_roi_detection_sec": None
            if first_key is None
            else round(first_key / fps, 2),
        },
        "aggregates": {
            "mean_raw": round(float(np.mean(raw_counts) if raw_counts else 0), 3),
            "mean_kept": round(float(np.mean(kept_counts) if kept_counts else 0), 3),
            "mean_key_roi": round(float(np.mean(key_counts) if key_counts else 0), 3),
            "pct_frames_empty_kept": round(
                100.0 * sum(empty_kept) / max(len(empty_kept), 1), 2
            ),
            "pct_frames_empty_key_roi": round(
                100.0 * sum(empty_key) / max(len(empty_key), 1), 2
            ),
            "pct_frames_under_3_players": round(
                100.0 * sum(weak_kept) / max(len(weak_kept), 1), 2
            ),
            "total_excluded_boxes": int(sum(r["excluded"] for r in rows)),
        },
        "gaps": {
            "empty_detection": _summarize_gaps([not e for e in empty_kept]),
            "empty_key_roi": _summarize_gaps([not e for e in empty_key]),
            "under_3_players": _summarize_gaps([not w for w in weak_kept]),
        },
        "diagnosis_hints": [],
        "timeline": rows,
    }

    hints = report["diagnosis_hints"]
    if first_any is not None and first_any > int(0.15 * total):
        hints.append(
            f"Late first detection (frame {first_any}) — detector likely missing early players."
        )
    if first_key is not None and first_any is not None and first_key > first_any + int(0.5 * fps):
        hints.append(
            f"Key-ROI (top-of-key band) locks {first_key - first_any} frames after other players — "
            "star/focal region under-detected."
        )
    if first_key is None:
        hints.append(
            "No detections ever landed in the key ROI — star at top of key may be systematically missed."
        )
    if report["aggregates"]["total_excluded_boxes"] > 0.25 * sum(raw_counts):
        hints.append(
            "Exclusions drop a large share of raw boxes — check court ROI / scorebug masks."
        )
    if report["aggregates"]["pct_frames_empty_kept"] > 10:
        hints.append(
            "Many frames have zero kept players — prioritize hard-frame mining + lower-conf rescue."
        )
    if not hints:
        hints.append(
            "Detector fires early; if UI still shows late Clark lock, inspect tracker/stitch next."
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="data/input_videos/Caitlin Clark clip.mp4",
        help="Seed clip path (relative to scoutiq/ or absolute)",
    )
    parser.add_argument(
        "--weights",
        default="outputs/training_loop_v2/best_player_seg_v2.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=1, help="1 = every frame")
    parser.add_argument(
        "--out",
        default="outputs/presence_loop/clark_miss_report.json",
    )
    parser.add_argument(
        "--previews",
        default="outputs/presence_loop/clark_previews",
    )
    parser.add_argument("--clip-id", default=None)
    args = parser.parse_args()

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

    report = run_miss_report(
        video,
        weights=weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stride=max(1, args.stride),
        save_previews=previews,
        clip_id=args.clip_id,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["milestones"]
    a = report["aggregates"]
    print("=== Player presence miss report ===")
    print(f"video:     {report['video']}")
    print(f"weights:   {report['weights']}")
    print(f"device:    {report['device']}  conf={report['conf']}")
    print(f"frames:    {report['sampled_frames']} sampled / {report['source_frame_count']} total")
    print(f"first any: {m['first_any_detection_frame']} @ {m['first_any_detection_sec']}s")
    print(f"first key: {m['first_key_roi_detection_frame']} @ {m['first_key_roi_detection_sec']}s")
    print(f"first 3+:  {m['first_3plus_players_frame']}")
    print(f"first 5+:  {m['first_5plus_players_frame']}")
    print(f"mean kept: {a['mean_kept']}  empty%={a['pct_frames_empty_kept']}  <3%={a['pct_frames_under_3_players']}")
    print(f"key empty%:{a['pct_frames_empty_key_roi']}  excluded boxes={a['total_excluded_boxes']}")
    print("hints:")
    for h in report["diagnosis_hints"]:
        print(f"  - {h}")
    print(f"wrote: {out}")
    if previews:
        print(f"previews: {previews}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
