"""Lightweight ball tracker with hard scene-cut resets + short-gap coasting.

Short-gap coasting (WS1): when a live track misses a few frames, the tracker
emits a predicted box from last velocity (``interpolated=True``, decayed
confidence) so possession does not drop out on a 1-5 frame detection miss.
A separate post-pass (``interpolate_ball_track_gaps``) does true linear
interpolation between two confident detections when the whole file is known.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 1.0) * max(ay2 - ay1, 1.0)
    area_b = max(bx2 - bx1, 1.0) * max(by2 - by1, 1.0)
    return inter / (area_a + area_b - inter)


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class _BallTrack:
    track_id: int
    box: list[float]
    missed: int = 0
    age: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    last_conf: float = 0.5
    prev_box: list[float] | None = field(default=None)


class BallTracker:
    """Single-class IoU tracker; call reset() on hard scene cuts.

    Args:
        iou_threshold: min IoU to attach a detection to an existing track.
        max_missed: drop a track after this many consecutive missed frames.
        coast_max_frames: emit a coasted (predicted) box for up to this many
            missed frames so short gaps still yield a ball position.
        coast_conf_decay: multiply confidence by this per coasted frame.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.15,
        max_missed: int = 12,
        coast_max_frames: int = 5,
        coast_conf_decay: float = 0.7,
    ) -> None:
        self._iou_thr = iou_threshold
        self._max_missed = max_missed
        self._coast_max = coast_max_frames
        self._coast_decay = coast_conf_decay
        self._tracks: list[_BallTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        """Hard scene-cut reset: drop all tracks, do not reuse IDs across cuts."""
        self._tracks = []

    def update(self, detections: list[dict]) -> list[dict]:
        for track in self._tracks:
            track.missed += 1

        results: list[dict] = []
        matched_ids: set[int] = set()
        for det in detections:
            box = [float(v) for v in (det.get("bbox_xyxy") or [])[:4]]
            if len(box) < 4:
                continue
            best = None
            best_iou = 0.0
            for track in self._tracks:
                if track.missed > self._max_missed or track.track_id in matched_ids:
                    continue
                iou = _iou(box, track.box)
                if iou > best_iou and iou >= self._iou_thr:
                    best_iou = iou
                    best = track
            conf = float(det.get("confidence") or 0.0)
            if best is not None:
                pcx, pcy = _center(best.box)
                ncx, ncy = _center(box)
                best.velocity = (ncx - pcx, ncy - pcy)
                best.prev_box = best.box
                best.box = box
                best.missed = 0
                best.age += 1
                best.last_conf = conf
                tid = best.track_id
                matched_ids.add(tid)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(
                    _BallTrack(track_id=tid, box=box, age=1, last_conf=conf)
                )
                matched_ids.add(tid)
            row = dict(det)
            row["track_id"] = tid
            row["bbox_xyxy"] = box
            row["interpolated"] = False
            results.append(row)

        # Coast unmatched-but-alive tracks over short gaps.
        for track in self._tracks:
            if track.track_id in matched_ids:
                continue
            if track.missed > self._coast_max:
                continue
            vx, vy = track.velocity
            x1, y1, x2, y2 = track.box
            coasted = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]
            track.box = coasted
            decayed = round(track.last_conf * (self._coast_decay ** track.missed), 4)
            results.append(
                {
                    "bbox_xyxy": coasted,
                    "confidence": decayed,
                    "class_id": 0,
                    "class_name": "ball",
                    "source": "coast",
                    "track_id": track.track_id,
                    "interpolated": True,
                }
            )

        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]
        # Prefer real detections first, then coasted, cap to 2 balls.
        results.sort(key=lambda r: (r.get("interpolated", False), -float(r.get("confidence") or 0)))
        return results[:2]


def interpolate_ball_track_gaps(
    tracks_path: Path,
    *,
    max_gap: int = 5,
    min_conf: float = 0.2,
) -> int:
    """Post-pass true linear interpolation of ball_tracks.jsonl.gz.

    Fills frames that have no real ball between two confident detections of the
    same track within ``max_gap`` frames. Rewrites the file in place. Returns
    the number of frames filled.
    """
    tracks_path = Path(tracks_path)
    if not tracks_path.is_file():
        return 0
    frames: list[dict] = []
    with gzip.open(tracks_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            frames.append(json.loads(line))
    if not frames:
        return 0

    # Index best real ball per frame.
    def _best_real(balls: list[dict]) -> dict | None:
        real = [b for b in balls if not b.get("interpolated")]
        if not real:
            return None
        return max(real, key=lambda b: float(b.get("confidence") or 0.0))

    filled = 0
    i = 0
    n = len(frames)
    while i < n:
        balls_i = frames[i].get("balls") or []
        anchor = _best_real(balls_i)
        if anchor is None or float(anchor.get("confidence") or 0) < min_conf:
            i += 1
            continue
        # Find next real anchor within max_gap.
        j = i + 1
        while j < n and j - i <= max_gap + 1:
            balls_j = frames[j].get("balls") or []
            nxt = _best_real(balls_j)
            if nxt is not None and float(nxt.get("confidence") or 0) >= min_conf:
                break
            j += 1
        if j >= n or j - i <= 1 or j - i > max_gap + 1:
            i += 1
            continue
        nxt = _best_real(frames[j].get("balls") or [])
        if nxt is None:
            i += 1
            continue
        a = anchor["bbox_xyxy"]
        b = nxt["bbox_xyxy"]
        span = j - i
        for k in range(i + 1, j):
            if frames[k].get("balls"):
                continue  # already has something (coasted etc.)
            t = (k - i) / span
            box = [round(a[m] + (b[m] - a[m]) * t, 2) for m in range(4)]
            conf = round(min(float(anchor["confidence"]), float(nxt["confidence"])) * 0.6, 4)
            frames[k]["balls"] = [
                {
                    "track_id": anchor.get("track_id"),
                    "confidence": conf,
                    "bbox_xyxy": box,
                    "source": "interp",
                    "interpolated": True,
                }
            ]
            filled += 1
        i = j

    with gzip.open(tracks_path, "wt", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
    return filled
