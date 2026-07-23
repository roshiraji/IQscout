"""Build top-K keyframe crops for each tracklet.

Default path searches each person's full track with cheap OCR-opportunity
scores, then keeps only the best temporally diverse candidates for OCR.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DEFAULT_JERSEY_OCR = {
    "max_candidates_per_person": 5,
    "min_spacing_frames": 15,
    "min_bbox_height": 100,
    "min_bbox_width": 45,
    "min_sharpness": 25.0,
    "min_contrast": 18.0,
}


def _bbox_area_norm(xyxy: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, width * height))


def _blur_score(image) -> float:
    """Legacy 0–1 blur score (1 = blurrier). Kept for older callers."""
    if image is None or image.size == 0:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(1.0 - min(variance / 500.0, 1.0), 0.0, 1.0))


def _crop_quality(image) -> dict[str, float]:
    """Cheap OCR-opportunity metrics (no OCR)."""
    if image is None or image.size == 0:
        return {
            "sharpness": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
            "blur_score": 1.0,
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "blur_score": float(np.clip(1.0 - min(sharpness / 500.0, 1.0), 0.0, 1.0)),
    }


def _is_large_enough_for_jersey(
    xyxy: list[float],
    *,
    min_height: int = 100,
    min_width: int = 45,
) -> bool:
    x1, y1, x2, y2 = xyxy
    return (y2 - y1) >= min_height and (x2 - x1) >= min_width


def _person_aspect_score(xyxy: list[float]) -> float:
    """Prefer upright person boxes (taller than wide) for jersey OCR."""
    x1, y1, x2, y2 = xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    ratio = height / width
    # Ideal roughly 1.6–2.8 for standing players.
    if 1.4 <= ratio <= 3.2:
        return 1.0
    if ratio < 1.0:
        return 0.2
    return float(np.clip(1.0 - abs(ratio - 2.0) / 3.0, 0.15, 1.0))


def _brightness_score(brightness: float) -> float:
    # Prefer mid-range exposure for digit readability.
    return float(np.clip(1.0 - abs(brightness - 128.0) / 128.0, 0.0, 1.0))


def _opportunity_score(meta: dict) -> float:
    """Rank crops for jersey OCR without running OCR."""
    sharpness_norm = float(np.clip(meta.get("sharpness", 0.0) / 500.0, 0.0, 1.0))
    contrast_norm = float(np.clip(meta.get("contrast", 0.0) / 80.0, 0.0, 1.0))
    return (
        0.12 * float(meta.get("det_conf", 0.0))
        + 0.28 * float(meta.get("bbox_area_norm", 0.0))
        + 0.28 * sharpness_norm
        + 0.16 * contrast_norm
        + 0.08 * float(meta.get("aspect_score", 0.0))
        + 0.08 * float(meta.get("brightness_score", 0.0))
    )


def _score(meta: dict) -> float:
    """Backward-compatible alias used by selection."""
    if "opportunity_score" in meta:
        return float(meta["opportunity_score"])
    return _opportunity_score(meta)


def build_tracklet_crops(
    *,
    video: Path,
    tracks_json: Path,
    output_dir: Path,
    top_k: int = 48,
    min_spacing_frames: int = 3,
    exclude_track_ids: set[int] | None = None,
    jersey_ocr: dict | None = None,
) -> dict:
    """Search each track for best jersey OCR opportunities, write top-K crops."""
    settings = {**DEFAULT_JERSEY_OCR, **(jersey_ocr or {})}
    max_candidates = int(
        settings.get("max_candidates_per_person")
        or top_k
        or DEFAULT_JERSEY_OCR["max_candidates_per_person"]
    )
    spacing = int(
        settings.get("min_spacing_frames")
        or min_spacing_frames
        or DEFAULT_JERSEY_OCR["min_spacing_frames"]
    )
    min_height = int(settings.get("min_bbox_height", 100))
    min_width = int(settings.get("min_bbox_width", 45))
    min_sharpness = float(settings.get("min_sharpness", 25.0))
    min_contrast = float(settings.get("min_contrast", 18.0))
    exclude = {int(v) for v in (exclude_track_ids or set())}

    output_dir.mkdir(parents=True, exist_ok=True)
    tracklets_dir = output_dir / "tracklets"
    tracklets_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(tracks_json.read_text(encoding="utf-8"))
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        tid = int(row["track_id"])
        if tid in exclude:
            continue
        by_track[tid].append(row)

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
    rejected = 0
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
            if not _is_large_enough_for_jersey(
                [x1, y1, x2, y2],
                min_height=min_height,
                min_width=min_width,
            ):
                rejected += 1
                continue
            crop = frame[y1:y2, x1:x2]
            quality = _crop_quality(crop)
            if quality["sharpness"] < min_sharpness or quality["contrast"] < min_contrast:
                rejected += 1
                continue
            meta = {
                "track_id": tid,
                "frame_idx": frame_idx,
                "timestamp_seconds": row["timestamp_seconds"],
                "bbox_xyxy": [x1, y1, x2, y2],
                "det_conf": float(row["det_conf"]),
                "bbox_area_norm": _bbox_area_norm([x1, y1, x2, y2], width, height),
                "blur_score": quality["blur_score"],
                "sharpness": round(quality["sharpness"], 3),
                "contrast": round(quality["contrast"], 3),
                "brightness": round(quality["brightness"], 3),
                "aspect_score": round(_person_aspect_score([x1, y1, x2, y2]), 4),
                "brightness_score": round(
                    _brightness_score(quality["brightness"]), 4
                ),
                "temporal_bonus": 1.0,
            }
            meta["opportunity_score"] = round(_opportunity_score(meta), 6)
            per_track_candidates[tid].append(meta)
        frame_idx += 1
    cap.release()

    # select top-K with temporal spacing (best opportunities first)
    for tid, candidates in per_track_candidates.items():
        candidates.sort(key=_score, reverse=True)
        selected: list[dict] = []
        for meta in candidates:
            if any(
                abs(meta["frame_idx"] - s["frame_idx"]) < spacing for s in selected
            ):
                continue
            selected.append(meta)
            if len(selected) >= max_candidates:
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
                "selection_method": "best_frame_opportunity",
                "keyframes": rows,
            }
        )
    out_path = output_dir / "tracklet_keyframes.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Optional debug: opportunity summary for identity folder consumers.
    summary_path = output_dir / "opportunity_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "max_candidates_per_person": max_candidates,
                "min_spacing_frames": spacing,
                "candidates_kept": sum(len(v) for v in chosen.values()),
                "candidates_rejected_early": rejected,
                "tracks": {
                    str(tid): {
                        "kept": len(rows),
                        "top_opportunity": (
                            rows[0]["opportunity_score"] if rows else None
                        ),
                    }
                    for tid, rows in chosen.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "video": str(video.resolve()),
        "tracklets": len(manifest),
        "keyframes_total": sum(m["num_keyframes"] for m in manifest),
        "candidates_rejected_early": rejected,
        "manifest": str(out_path.resolve()),
        "opportunity_summary": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build best-frame jersey OCR crops from tracks.json."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--min-spacing", type=int, default=3)
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
