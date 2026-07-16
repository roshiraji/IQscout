"""Build top-K keyframe crops for each tracklet."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _bbox_area_norm(xyxy: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, width * height))


def _blur_score(image) -> float:
    if image.size == 0:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(1.0 - min(variance / 500.0, 1.0), 0.0, 1.0))


def _is_large_enough_for_jersey(
    xyxy: list[float],
    *,
    min_height: int = 140,
    min_width: int = 55,
) -> bool:
    x1, y1, x2, y2 = xyxy
    return (y2 - y1) >= min_height and (x2 - x1) >= min_width


def _score(meta: dict) -> float:
    return (
        0.15 * meta["det_conf"]
        + 0.50 * meta["bbox_area_norm"]
        + 0.30 * (1.0 - meta["blur_score"])
        + 0.05 * meta["temporal_bonus"]
    )


def build_tracklet_crops(
    *,
    video: Path,
    tracks_json: Path,
    output_dir: Path,
    top_k: int = 24,
    min_spacing_frames: int = 4,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracklets_dir = output_dir / "tracklets"
    tracklets_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(tracks_json.read_text(encoding="utf-8"))
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_track[int(row["track_id"])].append(row)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # map frame -> entries so we decode each frame once
    frame_entries: dict[int, list[dict]] = defaultdict(list)
    for tid, rows in by_track.items():
        for row in rows:
            frame_entries[int(row["frame_idx"])].append({"track_id": tid, "row": row})

    chosen: dict[int, list[dict]] = defaultdict(list)
    per_track_candidates: dict[int, list[dict]] = defaultdict(list)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for item in frame_entries.get(frame_idx, []):
            row = item["row"]
            tid = item["track_id"]
            x1, y1, x2, y2 = [int(v) for v in row["bbox_xyxy"]]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            if not _is_large_enough_for_jersey([x1, y1, x2, y2]):
                continue
            crop = frame[y1:y2, x1:x2]
            meta = {
                "track_id": tid,
                "frame_idx": frame_idx,
                "timestamp_seconds": row["timestamp_seconds"],
                "bbox_xyxy": [x1, y1, x2, y2],
                "det_conf": float(row["det_conf"]),
                "bbox_area_norm": _bbox_area_norm([x1, y1, x2, y2], width, height),
                "blur_score": _blur_score(crop),
                "temporal_bonus": 1.0,
            }
            per_track_candidates[tid].append(meta)
        frame_idx += 1
    cap.release()

    # select top-K with temporal spacing
    for tid, candidates in per_track_candidates.items():
        candidates.sort(key=_score, reverse=True)
        selected: list[dict] = []
        for meta in candidates:
            if any(abs(meta["frame_idx"] - s["frame_idx"]) < min_spacing_frames for s in selected):
                continue
            selected.append(meta)
            if len(selected) >= top_k:
                break
        chosen[tid] = selected

    # second decode pass writes selected crops
    cap = cv2.VideoCapture(str(video))
    write_lookup: dict[int, list[dict]] = defaultdict(list)
    for tid, rows in chosen.items():
        for row in rows:
            write_lookup[row["frame_idx"]].append(row)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for meta in write_lookup.get(frame_idx, []):
            tid = meta["track_id"]
            tdir = tracklets_dir / f"track_{tid:06d}"
            tdir.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = meta["bbox_xyxy"]
            crop = frame[y1:y2, x1:x2]
            crop_path = tdir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            meta["crop_path"] = str(crop_path.resolve())
        frame_idx += 1
    cap.release()

    manifest = []
    for tid, rows in chosen.items():
        manifest.append(
            {
                "track_id": tid,
                "num_keyframes": len(rows),
                "keyframes": rows,
            }
        )
    out_path = output_dir / "tracklet_keyframes.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "video": str(video.resolve()),
        "tracklets": len(manifest),
        "keyframes_total": sum(m["num_keyframes"] for m in manifest),
        "manifest": str(out_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build top-K keyframe crops from tracks.json.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--min-spacing", type=int, default=4)
    args = parser.parse_args()
    summary = build_tracklet_crops(
        video=Path(args.video),
        tracks_json=Path(args.tracks),
        output_dir=Path(args.output),
        top_k=args.top_k,
        min_spacing_frames=args.min_spacing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
