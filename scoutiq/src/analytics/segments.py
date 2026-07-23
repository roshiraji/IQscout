"""Camera-consistent gameplay segmentation from existing job artifacts.

Uses hoop-side persistence, lightweight scene-change signals, and closeup
classification. Only ``wide_gameplay`` segments are marked
``usable_for_analytics``.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.analytics.trajectories import (
    COORDINATE_SPACE,
    iter_track_frames,
    resolve_frame_size,
)
from src.pipeline.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)

SEGMENTS_FILENAME = "game_segments.json"

CAMERA_WIDE = "wide_gameplay"
CAMERA_TRANSITION = "transition"
CAMERA_UNKNOWN = "unknown"
CAMERA_CLOSEUP = "closeup"

DEFAULT_SEGMENT_CONFIG: dict[str, Any] = {
    "hoop_left_max": 0.35,
    "hoop_right_min": 0.65,
    "hoop_side_window": 8,
    "hoop_side_min_votes": 5,
    "transition_pad_frames": 2,
    "track_loss_ratio": 0.45,
    "track_loss_min_prev": 4,
    "bbox_scale_ratio": 2.5,
    "probe_video": False,
    "pixel_cut_score": 45.0,
    "pixel_probe_width": 64,
    "pixel_probe_height": 36,
    # Closeup (Phase 4)
    "closeup_max_players": 2,
    "closeup_min_area_ratio": 0.20,
    "closeup_track_drop_ratio": 0.45,
    "closeup_track_drop_min_prev": 4,
    "closeup_feet_max_y_norm": 0.88,
    "closeup_persist_window": 5,
    "closeup_persist_min_votes": 3,
}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def merge_segment_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge YAML/analytics.segments overrides onto defaults."""
    cfg = dict(DEFAULT_SEGMENT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg

def classify_hoop_side(
    hoop_x_normalized: float,
    *,
    left_max: float = 0.35,
    right_min: float = 0.65,
) -> str:
    """Classify hoop horizontal position into left / center / right."""
    if hoop_x_normalized < left_max:
        return "left"
    if hoop_x_normalized > right_min:
        return "right"
    return "center"


def hoop_center_x_normalized(bbox_xyxy: Any, frame_width: int) -> float | None:
    if frame_width <= 0 or bbox_xyxy is None:
        return None
    try:
        coords = list(bbox_xyxy)[:4]
    except TypeError:
        return None
    if len(coords) < 4:
        return None
    x1 = _as_float(coords[0])
    x2 = _as_float(coords[2])
    if x1 is None or x2 is None or x2 <= x1:
        return None
    return ((x1 + x2) / 2.0) / float(frame_width)


def _bbox_area(bbox_xyxy: Any) -> float | None:
    try:
        coords = list(bbox_xyxy)[:4]
    except TypeError:
        return None
    if len(coords) < 4:
        return None
    x1, y1, x2, y2 = (_as_float(v) for v in coords)
    if None in (x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
        return None
    return float((x2 - x1) * (y2 - y1))


def load_quality_cut_frames(quality_path: Path) -> list[int]:
    if not quality_path.is_file():
        return []
    try:
        payload = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cuts = payload.get("cut_frames") or []
    out: list[int] = []
    for value in cuts:
        frame = _as_int(value)
        if frame is not None:
            out.append(frame)
    return sorted(set(out))


def iter_hoop_frames(hoop_path: Path) -> Iterator[dict[str, Any]]:
    if not hoop_path.is_file():
        return
    opener = gzip.open if str(hoop_path).endswith(".gz") else open
    with opener(hoop_path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_hoop_side_observations(
    hoop_path: Path,
    *,
    frame_width: int,
    left_max: float,
    right_min: float,
) -> list[tuple[int, str, float]]:
    """Return sorted ``(frame, side, x_norm)`` for frames with a hoop bbox."""
    observations: list[tuple[int, str, float]] = []
    for row in iter_hoop_frames(hoop_path):
        frame = _as_int(row.get("frame_number"))
        if frame is None:
            continue
        hoops = row.get("hoops") or []
        if not isinstance(hoops, list) or not hoops:
            continue
        best = None
        best_conf = -1.0
        for hoop in hoops:
            if not isinstance(hoop, dict):
                continue
            conf = _as_float(hoop.get("confidence")) or 0.0
            if conf >= best_conf:
                best = hoop
                best_conf = conf
        if best is None:
            continue
        x_norm = hoop_center_x_normalized(best.get("bbox_xyxy"), frame_width)
        if x_norm is None:
            continue
        side = classify_hoop_side(x_norm, left_max=left_max, right_min=right_min)
        observations.append((frame, side, float(x_norm)))
    observations.sort(key=lambda item: item[0])
    return observations


def stabilize_hoop_sides(
    observations: Sequence[tuple[int, str, float]],
    *,
    frame_start: int,
    frame_end: int,
    window: int = 8,
    min_votes: int = 5,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Persist hoop side; emit left↔right flip events with transition spans.

    A candidate side must appear in at least ``min_votes`` of the next
    ``window`` analyzed hoop frames before the stable side switches.
    """
    window = max(1, int(window))
    min_votes = max(1, min(int(min_votes), window))

    side_by_frame: dict[int, str] = {
        f: "unknown" for f in range(frame_start, frame_end + 1)
    }
    flip_events: list[dict[str, Any]] = []
    if not observations or frame_end < frame_start:
        return side_by_frame, flip_events

    sides_only = [(frame, side) for frame, side, _ in observations]
    stable: str | None = None
    stable_from_idx = 0

    def _window_vote(start_idx: int, candidate: str) -> bool:
        chunk = sides_only[start_idx : start_idx + window]
        if len(chunk) < min_votes:
            return False
        return sum(1 for _, side in chunk if side == candidate) >= min_votes

    # Establish initial stable side.
    for idx in range(len(sides_only)):
        candidate = sides_only[idx][1]
        if _window_vote(idx, candidate):
            stable = candidate
            stable_from_idx = idx
            break

    if stable is None:
        # Not enough persistence — forward-fill raw majority where possible.
        for frame, side, _ in observations:
            if frame_start <= frame <= frame_end:
                side_by_frame[frame] = side
        last = "unknown"
        for frame in range(frame_start, frame_end + 1):
            if side_by_frame[frame] != "unknown":
                last = side_by_frame[frame]
            else:
                side_by_frame[frame] = last
        return side_by_frame, flip_events

    # Paint initial stable backward/forward until first flip confirmation.
    first_frame = sides_only[stable_from_idx][0]
    for frame in range(frame_start, frame_end + 1):
        if frame >= first_frame:
            side_by_frame[frame] = stable

    last_lr_side = stable if stable in {"left", "right"} else None
    last_lr_end_frame = first_frame if last_lr_side else None

    idx = stable_from_idx + 1
    while idx < len(sides_only):
        frame, side = sides_only[idx]
        if side == stable:
            idx += 1
            continue
        if not _window_vote(idx, side):
            idx += 1
            continue

        # Confirmation index: earliest end of a min_votes run inside the window.
        confirm_idx = idx
        votes = 0
        for j in range(idx, min(len(sides_only), idx + window)):
            if sides_only[j][1] == side:
                votes += 1
                if votes >= min_votes:
                    confirm_idx = j
                    break
        confirm_frame = sides_only[confirm_idx][0]
        old_side = stable
        new_side = side
        transition_start = frame
        transition_end = confirm_frame

        # Direct left↔right, or opposite LR side after an intermediate center.
        opposite_lr = (
            new_side in {"left", "right"}
            and last_lr_side in {"left", "right"}
            and new_side != last_lr_side
        )
        direct_lr = {old_side, new_side} == {"left", "right"}
        if direct_lr or opposite_lr:
            flip_events.append(
                {
                    "from_side": (last_lr_side if opposite_lr else old_side),
                    "to_side": new_side,
                    "transition_start_frame": transition_start,
                    "transition_end_frame": transition_end,
                    "confirmed_frame": confirm_frame,
                }
            )

        stable = new_side
        if stable in {"left", "right"}:
            last_lr_side = stable
            last_lr_end_frame = confirm_frame
        elif old_side in {"left", "right"}:
            last_lr_end_frame = frame

        for paint in range(max(frame_start, confirm_frame), frame_end + 1):
            side_by_frame[paint] = stable
        idx = confirm_idx + 1

    # Forward-fill any leading unknowns before first observation.
    first_known = next(
        (f for f in range(frame_start, frame_end + 1) if side_by_frame[f] != "unknown"),
        None,
    )
    if first_known is not None:
        fill = side_by_frame[first_known]
        for frame in range(frame_start, first_known):
            side_by_frame[frame] = fill

    return side_by_frame, flip_events


def detect_track_layout_cuts(
    track_frames: Iterable[dict[str, Any]],
    *,
    track_loss_ratio: float = 0.45,
    track_loss_min_prev: int = 4,
    bbox_scale_ratio: float = 2.5,
) -> list[int]:
    """Detect likely cuts from sudden track-count / bbox-scale changes."""
    cuts: list[int] = []
    prev_count: int | None = None
    prev_median_area: float | None = None

    for row in track_frames:
        frame = _as_int(row.get("frame_number"))
        if frame is None:
            continue
        tracks = row.get("tracks") or []
        if not isinstance(tracks, list):
            tracks = []
        count = len(tracks)
        areas = []
        for det in tracks:
            if isinstance(det, dict):
                area = _bbox_area(det.get("bbox_xyxy"))
                if area is not None:
                    areas.append(area)
        median_area = None
        if areas:
            areas.sort()
            median_area = areas[len(areas) // 2]

        if (
            prev_count is not None
            and prev_count >= track_loss_min_prev
            and count <= prev_count * track_loss_ratio
        ):
            cuts.append(frame)

        if (
            prev_median_area is not None
            and median_area is not None
            and prev_median_area > 0
            and median_area > 0
            and (
                median_area >= prev_median_area * bbox_scale_ratio
                or prev_median_area >= median_area * bbox_scale_ratio
            )
        ):
            cuts.append(frame)

        prev_count = count
        if median_area is not None:
            prev_median_area = median_area

    return sorted(set(cuts))


def probe_video_pixel_cuts(
    video_path: Path,
    *,
    score_threshold: float = 45.0,
    probe_width: int = 64,
    probe_height: int = 36,
    max_frame: int | None = None,
) -> list[int]:
    """Lightweight grayscale mean-abs probe (same idea as tracking quality cuts)."""
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover
        logger.warning("OpenCV/numpy unavailable; skipping video pixel cut probe")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for cut probe: %s", video_path)
        return []

    cuts: list[int] = []
    previous: Any = None
    frame_id = -1
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frame_id += 1
            if max_frame is not None and frame_id > max_frame:
                break
            probe = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                (int(probe_width), int(probe_height)),
            )
            if previous is not None:
                score = float(
                    np.mean(
                        np.abs(
                            probe.astype(np.float32) - previous.astype(np.float32)
                        )
                    )
                )
                if score >= score_threshold:
                    cuts.append(frame_id)
            previous = probe
    finally:
        cap.release()
    return cuts


def collect_cut_frames(
    *,
    quality_cuts: Sequence[int],
    track_cuts: Sequence[int],
    video_cuts: Sequence[int] | None = None,
) -> list[int]:
    merged = set(int(c) for c in quality_cuts)
    merged.update(int(c) for c in track_cuts)
    if video_cuts:
        merged.update(int(c) for c in video_cuts)
    return sorted(merged)


def build_transition_frames(
    *,
    cut_frames: Sequence[int],
    hoop_flip_events: Sequence[dict[str, Any]],
    frame_start: int,
    frame_end: int,
    pad_frames: int = 2,
) -> set[int]:
    """Frames marked unusable around hard cuts and left↔right hoop flips."""
    transition: set[int] = set()
    pad = max(0, int(pad_frames))
    for cut in cut_frames:
        for frame in range(cut - pad, cut + pad + 1):
            if frame_start <= frame <= frame_end:
                transition.add(frame)

    for event in hoop_flip_events:
        start = _as_int(event.get("transition_start_frame"))
        end = _as_int(event.get("transition_end_frame"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        for frame in range(start, end + 1):
            if frame_start <= frame <= frame_end:
                transition.add(frame)
    return transition


def _largest_track_metrics(
    tracks: Sequence[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float | None]:
    """Return (largest_area_ratio, largest_bbox_y2_normalized)."""
    frame_area = float(frame_width) * float(frame_height)
    if frame_area <= 0:
        return 0.0, None
    best_area = 0.0
    best_y2_norm: float | None = None
    for det in tracks:
        if not isinstance(det, dict):
            continue
        area = _bbox_area(det.get("bbox_xyxy"))
        if area is None:
            continue
        ratio = area / frame_area
        if ratio >= best_area:
            best_area = ratio
            try:
                y2 = _as_float(list(det.get("bbox_xyxy") or [])[3])
            except (TypeError, IndexError):
                y2 = None
            best_y2_norm = (y2 / float(frame_height)) if y2 is not None else None
    return best_area, best_y2_norm


def is_closeup_candidate(
    *,
    player_count: int,
    largest_area_ratio: float,
    hoop_absent: bool,
    tracks_dropped_sharply: bool,
    feet_likely_missing: bool,
    max_players: int = 2,
    min_area_ratio: float = 0.20,
) -> bool:
    """Starting closeup rule with optional feet-missing support.

    Core rule:
      player_count <= max_players
      AND largest_area_ratio >= min_area_ratio
      AND (hoop_absent OR tracks_dropped_sharply OR feet_likely_missing)
    """
    if player_count > max_players:
        return False
    if largest_area_ratio < min_area_ratio:
        return False
    return bool(hoop_absent or tracks_dropped_sharply or feet_likely_missing)


def detect_closeup_candidate_frames(
    track_frames: Iterable[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    hoop_present_frames: set[int] | None = None,
    hoop_available: bool = False,
    max_players: int = 2,
    min_area_ratio: float = 0.20,
    track_drop_ratio: float = 0.45,
    track_drop_min_prev: int = 4,
    feet_max_y_norm: float = 0.88,
) -> tuple[set[int], dict[str, int]]:
    """Return raw per-frame closeup candidates (no persistence yet)."""
    hoop_present_frames = hoop_present_frames or set()
    candidates: set[int] = set()
    stats = {
        "frames_evaluated": 0,
        "candidate_frames": 0,
        "triggered_hoop_absent": 0,
        "triggered_track_drop": 0,
        "triggered_feet_missing": 0,
    }
    prev_count: int | None = None

    for row in track_frames:
        frame = _as_int(row.get("frame_number"))
        if frame is None:
            continue
        stats["frames_evaluated"] += 1
        tracks = row.get("tracks") or []
        if not isinstance(tracks, list):
            tracks = []
        count = len(tracks)
        largest_ratio, y2_norm = _largest_track_metrics(
            tracks, frame_width=frame_width, frame_height=frame_height
        )
        dropped = (
            prev_count is not None
            and prev_count >= track_drop_min_prev
            and count <= prev_count * track_drop_ratio
        )
        # If hoop tracking was never produced, do not treat every frame as
        # "hoop absent" alone — rely on layout / feet signals instead.
        if hoop_available:
            hoop_absent = frame not in hoop_present_frames
        else:
            hoop_absent = False
        feet_missing = (
            y2_norm is not None
            and largest_ratio >= min_area_ratio
            and y2_norm < feet_max_y_norm
        )
        if is_closeup_candidate(
            player_count=count,
            largest_area_ratio=largest_ratio,
            hoop_absent=hoop_absent,
            tracks_dropped_sharply=dropped,
            feet_likely_missing=feet_missing,
            max_players=max_players,
            min_area_ratio=min_area_ratio,
        ):
            candidates.add(frame)
            stats["candidate_frames"] += 1
            if hoop_absent:
                stats["triggered_hoop_absent"] += 1
            if dropped:
                stats["triggered_track_drop"] += 1
            if feet_missing:
                stats["triggered_feet_missing"] += 1
        prev_count = count

    return candidates, stats


def persist_frame_flags(
    candidate_frames: set[int],
    *,
    frame_start: int,
    frame_end: int,
    window: int = 5,
    min_votes: int = 3,
) -> set[int]:
    """Confirm flags only when they persist across a short frame window."""
    window = max(1, int(window))
    min_votes = max(1, min(int(min_votes), window))
    confirmed: set[int] = set()
    if frame_end < frame_start:
        return confirmed
    for frame in range(frame_start, frame_end + 1):
        votes = 0
        for i in range(frame - window + 1, frame + 1):
            if i in candidate_frames:
                votes += 1
        if votes >= min_votes:
            confirmed.add(frame)
    return confirmed


def build_game_segments(
    *,
    frame_start: int,
    frame_end: int,
    frame_times: dict[int, float],
    hoop_side_by_frame: dict[int, str],
    transition_frames: set[int],
    closeup_frames: set[int] | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    """Merge contiguous frames sharing camera_state + hoop_side.

    Priority: transition > closeup > wide_gameplay.
    """
    if frame_end < frame_start:
        return []
    closeup_frames = closeup_frames or set()

    def _time_for(frame: int) -> float:
        if frame in frame_times:
            return float(frame_times[frame])
        if fps and fps > 0:
            return float(frame) / float(fps)
        return 0.0

    def _state_for(frame: int) -> str:
        if frame in transition_frames:
            return CAMERA_TRANSITION
        if frame in closeup_frames:
            return CAMERA_CLOSEUP
        return CAMERA_WIDE

    segments: list[dict[str, Any]] = []
    seg_start = frame_start
    cur_state = _state_for(frame_start)
    cur_side = hoop_side_by_frame.get(frame_start, "unknown")

    for frame in range(frame_start + 1, frame_end + 1):
        state = _state_for(frame)
        side = hoop_side_by_frame.get(frame, "unknown")
        if state == cur_state and side == cur_side:
            continue
        segments.append(
            {
                "segment_id": len(segments) + 1,
                "start_frame": seg_start,
                "end_frame": frame - 1,
                "start_time": round(_time_for(seg_start), 6),
                "end_time": round(_time_for(frame - 1), 6),
                "camera_state": cur_state,
                "hoop_side": cur_side,
                "usable_for_analytics": cur_state == CAMERA_WIDE,
            }
        )
        seg_start = frame
        cur_state = state
        cur_side = side

    segments.append(
        {
            "segment_id": len(segments) + 1,
            "start_frame": seg_start,
            "end_frame": frame_end,
            "start_time": round(_time_for(seg_start), 6),
            "end_time": round(_time_for(frame_end), 6),
            "camera_state": cur_state,
            "hoop_side": cur_side,
            "usable_for_analytics": cur_state == CAMERA_WIDE,
        }
    )
    return segments


def _load_track_timeline(
    tracks_path: Path,
) -> tuple[list[dict[str, Any]], dict[int, float], int, int]:
    frames = list(iter_track_frames(tracks_path))
    frame_times: dict[int, float] = {}
    numbers: list[int] = []
    for row in frames:
        frame = _as_int(row.get("frame_number"))
        if frame is None:
            continue
        numbers.append(frame)
        ts = _as_float(row.get("timestamp_seconds"))
        if ts is not None:
            frame_times[frame] = ts
    if not numbers:
        raise FileNotFoundError(f"No track frames found in {tracks_path}")
    return frames, frame_times, min(numbers), max(numbers)


def _estimate_fps(frame_times: dict[int, float], frame_start: int, frame_end: int) -> float | None:
    if frame_end <= frame_start:
        return None
    t0 = frame_times.get(frame_start)
    t1 = frame_times.get(frame_end)
    if t0 is None or t1 is None or t1 <= t0:
        return None
    return float(frame_end - frame_start) / float(t1 - t0)


def generate_segments_for_job(
    job_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    """Generate ``analytics/game_segments.json`` for a tracking job.

    Reuses ``tracks.jsonl.gz``, ``quality.json``, and optional ``hoop_tracks.jsonl.gz``.
    Does not re-run YOLO / ByteTrack. Hoop absence soft-fails with reduced confidence.
    """
    job_dir = Path(job_dir).resolve()
    if config is None:
        try:
            from src.pipeline.player_track_pipeline import load_pipeline_config

            pipeline_cfg = load_pipeline_config()
            config = (pipeline_cfg.get("analytics") or {}).get("segments") or {}
        except Exception:  # noqa: BLE001
            config = {}
    cfg = merge_segment_config(config)

    tracks_path = job_dir / "tracks.jsonl.gz"
    if not tracks_path.is_file():
        # Allow uncompressed for tests / alternate exports.
        alt = job_dir / "tracks.jsonl"
        if alt.is_file():
            tracks_path = alt
        else:
            raise FileNotFoundError(f"Tracking output missing: expected {job_dir / 'tracks.jsonl.gz'}")

    width, height = resolve_frame_size(
        job_dir, frame_width=frame_width, frame_height=frame_height
    )
    track_frames, frame_times, frame_start, frame_end = _load_track_timeline(tracks_path)
    fps = _estimate_fps(frame_times, frame_start, frame_end)

    quality_cuts = load_quality_cut_frames(job_dir / "quality.json")
    track_cuts = detect_track_layout_cuts(
        track_frames,
        track_loss_ratio=float(cfg["track_loss_ratio"]),
        track_loss_min_prev=int(cfg["track_loss_min_prev"]),
        bbox_scale_ratio=float(cfg["bbox_scale_ratio"]),
    )

    video_cuts: list[int] = []
    signals_used = ["quality_cut_frames", "track_layout"]
    probe_video = bool(cfg.get("probe_video"))
    if probe_video or not quality_cuts:
        video_path = None
        summary_path = job_dir / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                candidate = summary.get("video_path")
                if candidate and Path(candidate).is_file():
                    video_path = Path(candidate)
            except (OSError, json.JSONDecodeError):
                video_path = None
        if video_path is not None and (probe_video or not quality_cuts):
            logger.info("Probing video pixel cuts for job %s", job_dir.name)
            video_cuts = probe_video_pixel_cuts(
                video_path,
                score_threshold=float(cfg["pixel_cut_score"]),
                probe_width=int(cfg["pixel_probe_width"]),
                probe_height=int(cfg["pixel_probe_height"]),
                max_frame=frame_end,
            )
            if video_cuts:
                signals_used.append("video_pixel_probe")

    cut_frames = collect_cut_frames(
        quality_cuts=quality_cuts,
        track_cuts=track_cuts,
        video_cuts=video_cuts,
    )

    hoop_path = job_dir / "hoop_tracks.jsonl.gz"
    hoop_available = hoop_path.is_file()
    hoop_observations: list[tuple[int, str, float]] = []
    if hoop_available:
        hoop_observations = load_hoop_side_observations(
            hoop_path,
            frame_width=width,
            left_max=float(cfg["hoop_left_max"]),
            right_min=float(cfg["hoop_right_min"]),
        )
        if hoop_observations:
            signals_used.append("hoop_side")

    hoop_side_by_frame, flip_events = stabilize_hoop_sides(
        hoop_observations,
        frame_start=frame_start,
        frame_end=frame_end,
        window=int(cfg["hoop_side_window"]),
        min_votes=int(cfg["hoop_side_min_votes"]),
    )

    transition_frames = build_transition_frames(
        cut_frames=cut_frames,
        hoop_flip_events=flip_events,
        frame_start=frame_start,
        frame_end=frame_end,
        pad_frames=int(cfg["transition_pad_frames"]),
    )

    hoop_present_frames = {frame for frame, _, _ in hoop_observations}
    closeup_candidates, closeup_raw_stats = detect_closeup_candidate_frames(
        track_frames,
        frame_width=width,
        frame_height=height,
        hoop_present_frames=hoop_present_frames,
        hoop_available=hoop_available and bool(hoop_observations),
        max_players=int(cfg["closeup_max_players"]),
        min_area_ratio=float(cfg["closeup_min_area_ratio"]),
        track_drop_ratio=float(cfg["closeup_track_drop_ratio"]),
        track_drop_min_prev=int(cfg["closeup_track_drop_min_prev"]),
        feet_max_y_norm=float(cfg["closeup_feet_max_y_norm"]),
    )
    closeup_frames = persist_frame_flags(
        closeup_candidates,
        frame_start=frame_start,
        frame_end=frame_end,
        window=int(cfg["closeup_persist_window"]),
        min_votes=int(cfg["closeup_persist_min_votes"]),
    )
    if closeup_candidates:
        signals_used.append("closeup")

    segments = build_game_segments(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_times=frame_times,
        hoop_side_by_frame=hoop_side_by_frame,
        transition_frames=transition_frames,
        closeup_frames=closeup_frames,
        fps=fps,
    )

    usable = [s for s in segments if s.get("usable_for_analytics")]
    confidence = "high" if hoop_available and hoop_observations else "reduced"
    if not hoop_available:
        logger.warning(
            "Hoop tracks missing for job %s; segmenting with scene/player signals only",
            job_dir.name,
        )

    payload = {
        "job_id": job_dir.name,
        "coordinate_space": COORDINATE_SPACE,
        "frame_width": width,
        "frame_height": height,
        "frame_range": {"start": frame_start, "end": frame_end},
        "fps_estimate": round(fps, 6) if fps else None,
        "confidence": confidence,
        "hoop_available": hoop_available,
        "hoop_observations": len(hoop_observations),
        "signals_used": signals_used,
        "cut_frames": cut_frames,
        "hoop_orientation_flips": flip_events,
        "closeup_detection": {
            "enabled": True,
            "raw_candidate_frames": sorted(closeup_candidates),
            "confirmed_frames": sorted(closeup_frames),
            "stats": {
                **closeup_raw_stats,
                "confirmed_frame_count": len(closeup_frames),
            },
        },
        "segments": segments,
        "summary": {
            "segment_count": len(segments),
            "usable_segment_count": len(usable),
            "transition_segment_count": sum(
                1 for s in segments if s["camera_state"] == CAMERA_TRANSITION
            ),
            "closeup_segment_count": sum(
                1 for s in segments if s["camera_state"] == CAMERA_CLOSEUP
            ),
            "transition_frame_count": len(transition_frames),
            "closeup_frame_count": len(closeup_frames),
            "cut_frame_count": len(cut_frames),
            "hoop_flip_count": len(flip_events),
        },
    }

    out_path = job_dir / "analytics" / SEGMENTS_FILENAME
    write_json_atomic(out_path, payload)
    logger.info(
        "Wrote %s segments (%s usable, %s closeup) → %s [confidence=%s]",
        len(segments),
        len(usable),
        payload["summary"]["closeup_segment_count"],
        out_path,
        confidence,
    )
    return payload
