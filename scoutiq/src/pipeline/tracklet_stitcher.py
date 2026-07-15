"""Post-pass tracklet stitching for stable person_group_id continuity.

Keeps ByteTrack raw IDs for debugging, then merges adjacent tracklets that
likely belong to the same person using temporal gap, spatial proximity, and
jersey-color histograms (optional OSNet / torchreid when installed).
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    track_id: int
    first_frame: int
    last_frame: int
    detections: int = 0
    confidence_sum: float = 0.0
    centers: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[list[int]] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    color_hist: np.ndarray | None = None
    appearance: np.ndarray | None = None

    @property
    def mean_confidence(self) -> float:
        return float(self.confidence_sum) / max(self.detections, 1)

    def start_center(self) -> tuple[float, float]:
        return self.centers[0] if self.centers else (0.0, 0.0)

    def end_center(self) -> tuple[float, float]:
        return self.centers[-1] if self.centers else (0.0, 0.0)


def _hsv_hist(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(32 * 32, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def _hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_tracklets(frames: list[dict]) -> dict[int, Tracklet]:
    tracklets: dict[int, Tracklet] = {}
    for frame in frames:
        fid = int(frame["frame_number"])
        for det in frame.get("tracks") or []:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            box = [int(v) for v in (det.get("bbox_xyxy") or [0, 0, 0, 0])[:4]]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            tl = tracklets.get(tid)
            if tl is None:
                tl = Tracklet(track_id=tid, first_frame=fid, last_frame=fid)
                tracklets[tid] = tl
            tl.last_frame = fid
            tl.detections += 1
            tl.confidence_sum += float(det.get("confidence") or 0.0)
            tl.centers.append((cx, cy))
            tl.boxes.append(box)
            tl.frame_ids.append(fid)
    return tracklets


def _sample_appearance(
    video: Path,
    tracklets: dict[int, Tracklet],
    *,
    keyframe_samples: int,
    use_osnet: bool,
    device: str,
) -> str:
    """Fill color_hist (+ optional OSNet) from a few keyframe crops per tracklet."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        logger.warning("Could not open video for appearance sampling: %s", video)
        return "none"

    osnet = None
    appearance_backend = "color_hist"
    if use_osnet:
        try:
            # Optional soft dependency; color hist remains the default path.
            import torch
            from torchreid.utils import FeatureExtractor  # type: ignore

            target = f"cuda:{device}" if str(device).isdigit() else str(device)
            if target.startswith("cuda") and not torch.cuda.is_available():
                target = "cpu"
            osnet = FeatureExtractor(
                model_name="osnet_x0_25",
                model_path="",
                device=target,
            )
            appearance_backend = "osnet_x0_25+color_hist"
            logger.info("OSNet-x0.25 available for stitch appearance on %s", target)
        except Exception as exc:  # noqa: BLE001
            logger.info("OSNet unavailable (%s) — stitching with jersey color only.", exc)
            osnet = None
            appearance_backend = "color_hist"

    try:
        for tl in tracklets.values():
            if not tl.frame_ids:
                continue
            n = max(1, min(keyframe_samples, len(tl.frame_ids)))
            idxs = np.linspace(0, len(tl.frame_ids) - 1, n, dtype=int)
            hists = []
            embeds = []
            for i in idxs:
                fid = tl.frame_ids[i]
                box = tl.boxes[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                hists.append(_hsv_hist(crop))
                if osnet is not None:
                    try:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        feat = osnet([rgb])
                        vec = np.asarray(feat).reshape(-1).astype(np.float32)
                        embeds.append(vec)
                    except Exception:
                        pass
            if hists:
                tl.color_hist = np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32)
            if embeds:
                tl.appearance = np.mean(np.stack(embeds, axis=0), axis=0).astype(np.float32)
    finally:
        cap.release()
    return appearance_backend


def _merge_score(
    a: Tracklet,
    b: Tracklet,
    *,
    fps: float,
    frame_diag: float,
    max_gap_seconds: float,
    max_center_distance_ratio: float,
    color_weight: float,
    appearance_weight: float,
    spatial_weight: float,
) -> float | None:
    """Return merge score in [0,1] or None if hard constraints fail."""
    if a.track_id == b.track_id:
        return None
    # Enforce temporal order without overlap.
    if a.last_frame >= b.first_frame and b.last_frame >= a.first_frame:
        return None
    early, late = (a, b) if a.last_frame <= b.first_frame else (b, a)
    gap_frames = late.first_frame - early.last_frame - 1
    if gap_frames < 0:
        return None
    gap_seconds = gap_frames / max(fps, 1e-6)
    if gap_seconds > max_gap_seconds:
        return None

    ex, ey = early.end_center()
    lx, ly = late.start_center()
    dist = float(np.hypot(ex - lx, ey - ly))
    max_dist = max_center_distance_ratio * frame_diag
    if dist > max_dist:
        return None

    spatial = 1.0 - min(dist / max(max_dist, 1e-6), 1.0)
    color = max(_hist_similarity(early.color_hist, late.color_hist), 0.0)
    appearance = max(_cosine(early.appearance, late.appearance), 0.0)
    if early.appearance is None or late.appearance is None:
        # Redistribute appearance weight into color when OSNet is unavailable.
        w_color = color_weight + appearance_weight
        w_app = 0.0
    else:
        w_color = color_weight
        w_app = appearance_weight
    w_spatial = spatial_weight
    denom = max(w_color + w_app + w_spatial, 1e-6)
    score = (w_color * color + w_app * appearance + w_spatial * spatial) / denom
    # Soft penalty for larger gaps.
    score *= 1.0 - 0.25 * (gap_seconds / max(max_gap_seconds, 1e-6))
    return float(score)


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stitch_tracklets(
    tracks_jsonl: Path,
    video: Path,
    *,
    fps: float,
    width: int,
    height: int,
    stitch_cfg: dict | None = None,
    device: str = "0",
) -> dict:
    """Merge raw tracklets into stable person groups.

    Returns payload written by callers to person_groups.json.
    """
    cfg = {
        "enabled": True,
        "max_gap_seconds": 1.5,
        "max_center_distance_ratio": 0.18,
        "color_weight": 0.65,
        "appearance_weight": 0.35,
        "spatial_weight": 0.25,
        "min_merge_score": 0.55,
        "min_group_frames": 8,
        "keyframe_samples": 3,
        "use_osnet": True,
        **(stitch_cfg or {}),
    }
    if not cfg.get("enabled", True):
        return {
            "enabled": False,
            "person_groups": [],
            "track_to_group": {},
            "unique_raw_tracks": 0,
            "unique_person_groups": 0,
        }

    frames: list[dict] = []
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frames.append(json.loads(line))

    tracklets = _build_tracklets(frames)
    raw_ids = sorted(tracklets)
    if not raw_ids:
        return {
            "enabled": True,
            "person_groups": [],
            "track_to_group": {},
            "unique_raw_tracks": 0,
            "unique_person_groups": 0,
            "average_id_switches_per_person_pre": 0.0,
            "average_id_switches_per_person_post": 0.0,
        }

    appearance_backend = _sample_appearance(
        video,
        tracklets,
        keyframe_samples=int(cfg["keyframe_samples"]),
        use_osnet=bool(cfg.get("use_osnet", True)),
        device=device,
    )

    frame_diag = float(np.hypot(width, height))
    candidates: list[tuple[float, int, int]] = []
    tids = list(tracklets)
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tracklets[tids[i]], tracklets[tids[j]]
            score = _merge_score(
                a,
                b,
                fps=fps,
                frame_diag=frame_diag,
                max_gap_seconds=float(cfg["max_gap_seconds"]),
                max_center_distance_ratio=float(cfg["max_center_distance_ratio"]),
                color_weight=float(cfg["color_weight"]),
                appearance_weight=float(cfg["appearance_weight"]),
                spatial_weight=float(cfg["spatial_weight"]),
            )
            if score is not None and score >= float(cfg["min_merge_score"]):
                candidates.append((score, a.track_id, b.track_id))

    candidates.sort(reverse=True)
    uf = _UnionFind(raw_ids)
    for _score, a, b in candidates:
        # Reject if merge would create temporal overlap in the growing set.
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        members_a = [tid for tid in raw_ids if uf.find(tid) == ra]
        members_b = [tid for tid in raw_ids if uf.find(tid) == rb]
        spans_a = [(tracklets[t].first_frame, tracklets[t].last_frame) for t in members_a]
        spans_b = [(tracklets[t].first_frame, tracklets[t].last_frame) for t in members_b]
        overlap = False
        for s1, e1 in spans_a:
            for s2, e2 in spans_b:
                if s1 <= e2 and s2 <= e1:
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            continue
        uf.union(a, b)

    roots: dict[int, list[int]] = {}
    for tid in raw_ids:
        roots.setdefault(uf.find(tid), []).append(tid)

    # Sort groups by total detections (stable selling-unit ranking).
    group_members = sorted(
        roots.values(),
        key=lambda members: (
            -sum(tracklets[t].detections for t in members),
            min(members),
        ),
    )
    person_groups = []
    track_to_group: dict[int, int] = {}
    min_group_frames = int(cfg["min_group_frames"])
    group_id = 0
    for members in group_members:
        members = sorted(members)
        first_f = min(tracklets[t].first_frame for t in members)
        last_f = max(tracklets[t].last_frame for t in members)
        total_det = sum(tracklets[t].detections for t in members)
        span = last_f - first_f + 1
        if total_det < min_group_frames and len(members) == 1:
            # Keep short singles for completeness; snapshot prefers mature ones.
            pass
        group_id += 1
        for tid in members:
            track_to_group[tid] = group_id
        person_groups.append(
            {
                "person_group_id": group_id,
                "label": f"Person #{group_id}",
                "member_track_ids": members,
                "first_frame": first_f,
                "last_frame": last_f,
                "detections": total_det,
                "span_frames": span,
                "mean_confidence": round(
                    sum(tracklets[t].confidence_sum for t in members)
                    / max(total_det, 1),
                    4,
                ),
                "raw_tracklet_count": len(members),
            }
        )

    # Pre-stitch: each physical run of tracklets is a potential ID switch.
    # Approximate switches as (raw tracklets - person groups) / person groups.
    unique_groups = len(person_groups)
    switches_pre = max(len(raw_ids) - unique_groups, 0)
    avg_switches_pre = switches_pre / max(unique_groups, 1)
    avg_switches_post = 0.0  # after stitch, each group is one identity

    payload = {
        "enabled": True,
        "appearance_backend": appearance_backend,
        "config": {
            "max_gap_seconds": cfg["max_gap_seconds"],
            "min_merge_score": cfg["min_merge_score"],
            "min_group_frames": cfg["min_group_frames"],
        },
        "unique_raw_tracks": len(raw_ids),
        "unique_person_groups": unique_groups,
        "average_id_switches_per_person_pre": round(avg_switches_pre, 4),
        "average_id_switches_per_person_post": round(avg_switches_post, 4),
        "merges_accepted": len(raw_ids) - unique_groups,
        "person_groups": person_groups,
        "track_to_group": {str(k): v for k, v in track_to_group.items()},
    }
    logger.info(
        "Stitch complete: raw=%d groups=%d merges=%d backend=%s",
        len(raw_ids),
        unique_groups,
        payload["merges_accepted"],
        appearance_backend,
    )
    return payload


def annotate_tracks_jsonl(
    tracks_jsonl: Path,
    track_to_group: dict[int, int],
) -> None:
    """Rewrite tracks.jsonl.gz adding person_group_id on each detection."""
    frames: list[dict] = []
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frames.append(json.loads(line))
    with gzip.open(tracks_jsonl, "wt", encoding="utf-8") as handle:
        for frame in frames:
            for det in frame.get("tracks") or []:
                tid = int(det.get("track_id", -1))
                gid = track_to_group.get(tid)
                if gid is not None:
                    det["person_group_id"] = gid
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")


def group_frames_with_selected(
    tracks_jsonl: Path,
    selected_group_ids: set[int],
) -> tuple[int, int]:
    """Return (frames_with_selected, total_frames) for retention metric."""
    total = 0
    with_sel = 0
    with gzip.open(tracks_jsonl, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            total += 1
            for det in frame.get("tracks") or []:
                gid = det.get("person_group_id")
                if gid is None:
                    continue
                if int(gid) in selected_group_ids:
                    with_sel += 1
                    break
    return with_sel, total
