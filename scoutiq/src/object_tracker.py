"""Object tracker — ByteTrack via supervision with simple IoU fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_bytetrack: object | None = None
_bytetrack_failed = False


def _get_bytetrack():
    global _bytetrack, _bytetrack_failed
    if _bytetrack is not None:
        return _bytetrack
    if _bytetrack_failed:
        return None
    try:
        import warnings
        import supervision as sv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _bytetrack = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
                minimum_consecutive_frames=1,
            )
        logger.info("ByteTrack initialized via supervision.")
    except Exception as exc:
        logger.warning("ByteTrack unavailable (%s) — using IoU fallback tracker.", exc)
        _bytetrack_failed = True
    return _bytetrack


# --------------------------------------------------------------------------- #
# ByteTrack path                                                               #
# --------------------------------------------------------------------------- #

def _detections_to_sv(detections: list[dict]) -> "supervision.Detections":
    import supervision as sv
    if not detections:
        return sv.Detections.empty()
    boxes = np.array([d["bbox_xyxy"][:4] for d in detections], dtype=np.float32)
    confs = np.array([d.get("confidence", 0.5) for d in detections], dtype=np.float32)
    class_ids = np.array([d.get("class_id", 0) for d in detections], dtype=int)
    return sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)


def track_with_bytetrack(detections: list[dict]) -> list[dict]:
    tracker = _get_bytetrack()
    if tracker is None:
        return []

    import supervision as sv
    sv_dets = _detections_to_sv(detections)
    if len(sv_dets) == 0:
        return []

    try:
        tracked = tracker.update_with_detections(sv_dets)
    except Exception as exc:
        logger.warning("ByteTrack update failed: %s", exc)
        return []

    results: list[dict] = []
    for i in range(len(tracked)):
        track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
        box = tracked.xyxy[i].tolist()
        cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5

        # Match back to original detection for mask/class_name.
        src = _find_closest_detection(box, detections, cls_id)
        base = src.copy() if src else {}
        base.update({
            "track_id": track_id,
            "bbox_xyxy": [int(v) for v in box],
            "class_id": cls_id,
            "confidence": round(conf, 4),
        })
        results.append(base)
    return results


def _find_closest_detection(box: list, detections: list[dict], cls_id: int) -> dict | None:
    best_iou = 0.0
    best = None
    for det in detections:
        if det.get("class_id") != cls_id:
            continue
        iou = _iou(box, det["bbox_xyxy"][:4])
        if iou > best_iou:
            best_iou = iou
            best = det
    return best


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)
    area_b = max(bx2 - bx1, 1) * max(by2 - by1, 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# IoU fallback tracker                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class _Track:
    track_id: int
    class_id: int
    box: list
    age: int = 0
    missed: int = 0


class IouTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 15) -> None:
        self._iou_thr = iou_threshold
        self._max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        for track in self._tracks:
            track.missed += 1

        results: list[dict] = []
        matched_tracks: set[int] = set()

        for det in detections:
            box = det["bbox_xyxy"][:4]
            cls_id = det.get("class_id", 0)
            best_track = None
            best_iou = 0.0
            for track in self._tracks:
                if track.class_id != cls_id or track.missed > self._max_missed:
                    continue
                iou = _iou(box, track.box)
                if iou > best_iou and iou >= self._iou_thr:
                    best_iou = iou
                    best_track = track

            if best_track is not None:
                best_track.box = box
                best_track.age += 1
                best_track.missed = 0
                matched_tracks.add(id(best_track))
                tid = best_track.track_id
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(_Track(track_id=tid, class_id=cls_id, box=box))

            r = det.copy()
            r["track_id"] = tid
            results.append(r)

        # Prune lost tracks.
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]
        return results


_iou_tracker = IouTracker()


# --------------------------------------------------------------------------- #
# Public interface                                                              #
# --------------------------------------------------------------------------- #

def update_tracker(detections: list[dict]) -> list[dict]:
    """Run ByteTrack; fall back to IoU tracker if unavailable."""
    bt = _get_bytetrack()
    if bt is not None:
        tracked = track_with_bytetrack(detections)
        if tracked or not detections:
            return tracked
        logger.debug("ByteTrack returned empty; using IoU fallback.")
    return _iou_tracker.update(detections)
