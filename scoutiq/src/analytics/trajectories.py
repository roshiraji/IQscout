"""Build per-player screen-space trajectories from existing job tracking output.

Reads ``locked_tracks.jsonl.gz`` when Confirm produced it (SAM2 + ReID
identity-locked bodies), else falls back to raw ``tracks.jsonl.gz``, plus
``person_groups.json`` (no YOLO / ByteTrack re-run).

Contact point prefers the segmentation-mask stance point (mean of the lowest
slice of the player's mask — an approximation of foot/ground contact) and
falls back to bbox bottom-center (``x = (x1+x2)/2``, ``y = y2``) when no mask
is present on a detection. Coordinates are normalized by frame width/height
(screen space, not court units).

Phase 5: annotate points with gameplay segments and split path fragments so
lines are not connected across cuts, hoop-side changes, closeups, or long gaps.
Small gaps (a handful of missed-detection frames) are linearly interpolated so
the drawn trail does not visibly teleport; interpolated points are flagged
so callers can render them distinctly from real detections.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.pipeline.atomic_json import write_json_atomic
from src.pipeline.geometry_points import stance_point_from_mask

logger = logging.getLogger(__name__)

COORDINATE_SPACE = "normalized screen-space coordinates"
TRAJECTORIES_FILENAME = "trajectories.jsonl.gz"
TRAJECTORIES_ANALYTICS_FILENAME = "trajectories_analytics.jsonl.gz"
SUMMARY_FILENAME = "trajectory_summary.json"
SEGMENTS_FILENAME = "game_segments.json"
LOCKED_TRACKS_FILENAME = "locked_tracks.jsonl.gz"
RAW_TRACKS_FILENAME = "tracks.jsonl.gz"

DEFAULT_TRAJECTORY_CONFIG: dict[str, Any] = {
    "max_gap_frames": 15,
    # Missed-detection gaps at or below this many frames get linearly
    # interpolated instead of drawn as a straight jump across the whole gap.
    "max_interpolate_gap_frames": 6,
    # Centered moving-average window (in detections) used to cancel
    # per-frame box/mask localization jitter before drawing the trail.
    "smoothing_window_frames": 5,
}


def resolve_tracks_path(job_dir: Path, tracks_filename: str | None = None) -> Path:
    """Pick the most accurate tracks source available for a job.

    Confirm's SAM 2 / ReID selection lock writes ``locked_tracks.jsonl.gz``
    with each selected player re-associated to the correct body frame by
    frame (surviving contact/occlusion swaps). When present it is a strictly
    better trajectory source than the raw canonical detections, so prefer it
    unless the caller explicitly names a file.
    """
    job_dir = Path(job_dir)
    if tracks_filename:
        return job_dir / tracks_filename
    locked = job_dir / LOCKED_TRACKS_FILENAME
    if locked.is_file():
        return locked
    return job_dir / RAW_TRACKS_FILENAME


class TrajectoryInputError(FileNotFoundError):
    """Raised when required tracking artifacts are missing."""


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


def merge_trajectory_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_TRAJECTORY_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def bbox_bottom_center(bbox_xyxy: Any) -> tuple[float, float] | None:
    """Return bottom-center contact point, or None if the bbox is invalid."""
    if bbox_xyxy is None:
        return None
    try:
        coords = list(bbox_xyxy)[:4]
    except TypeError:
        return None
    if len(coords) < 4:
        return None
    x1 = _as_float(coords[0])
    y1 = _as_float(coords[1])
    x2 = _as_float(coords[2])
    y2 = _as_float(coords[3])
    if None in (x1, y1, x2, y2):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, float(y2))


def load_team_lookup(person_groups_path: Path) -> dict[int, dict[str, Any]]:
    """Map ``person_group_id`` → team metadata from ``person_groups.json``."""
    payload = json.loads(person_groups_path.read_text(encoding="utf-8"))
    groups: Iterable[dict[str, Any]]
    if isinstance(payload, list):
        groups = payload
    elif isinstance(payload, dict):
        groups = payload.get("person_groups") or []
    else:
        groups = []

    lookup: dict[int, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = _as_int(group.get("person_group_id"))
        if gid is None:
            continue
        team_raw = group.get("team_id")
        team_id = _as_int(team_raw) if team_raw is not None else None
        jersey = group.get("jersey_color_name")
        # Infer team sides from jersey when Analyze left team_id unset — otherwise
        # the Events Red/Blue filters silently keep or drop the wrong trails.
        if team_id is None and jersey:
            name = str(jersey).strip().lower()
            if name == "red":
                team_id = 0
            elif name == "blue":
                team_id = 1
        lookup[gid] = {
            "team_id": team_id,
            "jersey_color_name": str(jersey).lower() if jersey else None,
            "team_label": group.get("team_label"),
            "role": group.get("role") or "player",
        }
    return lookup


def resolve_frame_size(
    job_dir: Path,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[int, int]:
    """Resolve video frame size from overrides, then ``snapshot.json``."""
    if frame_width and frame_height and frame_width > 0 and frame_height > 0:
        return int(frame_width), int(frame_height)

    snapshot_path = job_dir / "snapshot.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        width = _as_int(snapshot.get("frame_width"))
        height = _as_int(snapshot.get("frame_height"))
        if width and height and width > 0 and height > 0:
            return width, height

    raise TrajectoryInputError(
        f"Frame width/height unavailable for job at {job_dir}. "
        "Expected snapshot.json with frame_width and frame_height."
    )


def iter_track_frames(tracks_path: Path) -> Iterator[dict[str, Any]]:
    """Yield frame objects from ``tracks.jsonl.gz`` or plain ``.jsonl``."""
    compressed = str(tracks_path).endswith(".gz")
    opener = gzip.open if compressed else open

    with opener(tracks_path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSON on %s line %s: %s",
                    tracks_path.name,
                    line_number,
                    exc,
                )
                continue
            if isinstance(payload, dict):
                yield payload


def default_segment_meta() -> dict[str, Any]:
    return {
        "segment_id": None,
        "camera_state": "unknown",
        "hoop_side": "unknown",
        "usable_for_analytics": False,
    }


def build_frame_segment_index(
    segments: Iterable[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Map each frame number to segment metadata."""
    index: dict[int, dict[str, Any]] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = _as_int(seg.get("start_frame"))
        end = _as_int(seg.get("end_frame"))
        if start is None or end is None or end < start:
            continue
        meta = {
            "segment_id": _as_int(seg.get("segment_id")),
            "camera_state": str(seg.get("camera_state") or "unknown"),
            "hoop_side": str(seg.get("hoop_side") or "unknown"),
            "usable_for_analytics": bool(seg.get("usable_for_analytics")),
        }
        for frame in range(start, end + 1):
            index[frame] = meta
    return index


def load_segments_payload(segments_path: Path) -> dict[str, Any] | None:
    if not segments_path.is_file():
        return None
    try:
        payload = json.loads(segments_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def annotate_points_with_segments(
    points: list[dict[str, Any]],
    frame_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach segment fields to each trajectory point (mutates copies)."""
    annotated: list[dict[str, Any]] = []
    for point in points:
        row = dict(point)
        frame = _as_int(row.get("frame"))
        meta = (
            frame_index.get(frame, default_segment_meta())
            if frame is not None
            else default_segment_meta()
        )
        row["segment_id"] = meta.get("segment_id")
        row["camera_state"] = meta.get("camera_state")
        row["hoop_side"] = meta.get("hoop_side")
        row["usable_for_analytics"] = bool(meta.get("usable_for_analytics"))
        annotated.append(row)
    return annotated


def _smooth_fragment_points(
    points: list[dict[str, Any]],
    *,
    window: int,
    max_neighbor_gap_frames: int,
) -> list[dict[str, Any]]:
    """Low-pass filter per-frame localization jitter with a centered moving average.

    A real player cannot reverse direction every single frame; a wobbling
    detection box or a segmentation mask that gains/loses a few edge pixels
    can. Averaging over a small centered window cancels that zero-mean noise
    while keeping the path's real shape and timing intact. Neighbors more
    than ``max_neighbor_gap_frames`` away in time are excluded from a given
    point's window so a real gap in detections never gets smeared across.
    """
    n = len(points)
    if n < 3 or window <= 1:
        return points

    half = max(1, window // 2)
    smoothed: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        frame = _as_int(point.get("frame"))
        chunk = [point]
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if j == i:
                continue
            neighbor = points[j]
            neighbor_frame = _as_int(neighbor.get("frame"))
            if (
                frame is not None
                and neighbor_frame is not None
                and abs(neighbor_frame - frame) > max_neighbor_gap_frames
            ):
                continue
            chunk.append(neighbor)
        if len(chunk) < 2:
            smoothed.append(point)
            continue
        row = dict(point)
        row["x"] = round(sum(p["x"] for p in chunk) / len(chunk), 3)
        row["y"] = round(sum(p["y"] for p in chunk) / len(chunk), 3)
        row["x_normalized"] = round(
            sum(p["x_normalized"] for p in chunk) / len(chunk), 6
        )
        row["y_normalized"] = round(
            sum(p["y_normalized"] for p in chunk) / len(chunk), 6
        )
        smoothed.append(row)
    return smoothed


def _interpolate_fragment_gaps(
    points: list[dict[str, Any]],
    *,
    max_interpolate_gap_frames: int,
) -> list[dict[str, Any]]:
    """Fill short missed-detection gaps with linearly interpolated points.

    A fragment already guarantees no segment break, so a small frame gap
    inside it means the detector/tracker simply missed a few frames — not a
    cut or a real disappearance. Bridging those with interpolation keeps the
    drawn trail following the player's actual path instead of jumping in a
    straight line across the whole gap. Interpolated points are tagged so
    the UI can render them (e.g. dashed) distinctly from real detections.
    """
    if len(points) < 2 or max_interpolate_gap_frames <= 0:
        return points

    filled: list[dict[str, Any]] = [points[0]]
    for prev, nxt in zip(points, points[1:]):
        prev_frame = _as_int(prev.get("frame"))
        next_frame = _as_int(nxt.get("frame"))
        if (
            prev_frame is not None
            and next_frame is not None
            and 1 < (next_frame - prev_frame) <= max_interpolate_gap_frames
        ):
            gap = next_frame - prev_frame
            prev_time = float(prev.get("time_seconds") or 0.0)
            next_time = float(nxt.get("time_seconds") or 0.0)
            for step in range(1, gap):
                t = step / gap
                filled.append(
                    {
                        "frame": prev_frame + step,
                        "time_seconds": round(
                            prev_time + t * (next_time - prev_time), 6
                        ),
                        "x": round(prev["x"] + t * (nxt["x"] - prev["x"]), 3),
                        "y": round(prev["y"] + t * (nxt["y"] - prev["y"]), 3),
                        "x_normalized": round(
                            prev["x_normalized"]
                            + t * (nxt["x_normalized"] - prev["x_normalized"]),
                            6,
                        ),
                        "y_normalized": round(
                            prev["y_normalized"]
                            + t * (nxt["y_normalized"] - prev["y_normalized"]),
                            6,
                        ),
                        "point_source": "interpolated",
                        "interpolated": True,
                        "segment_id": prev.get("segment_id"),
                        "camera_state": prev.get("camera_state"),
                        "hoop_side": prev.get("hoop_side"),
                        "usable_for_analytics": bool(prev.get("usable_for_analytics"))
                        and bool(nxt.get("usable_for_analytics")),
                    }
                )
        filled.append(nxt)
    return filled


def split_path_fragments(
    points: list[dict[str, Any]],
    *,
    max_gap_frames: int = 15,
    analytics_safe_only: bool = False,
    max_interpolate_gap_frames: int = 6,
    smoothing_window_frames: int = 5,
) -> list[dict[str, Any]]:
    """Split points into path fragments.

    Breaks when:
      - ``segment_id`` changes (covers cuts, hoop-side changes, closeups)
      - frame gap exceeds ``max_gap_frames``
    Within a fragment, real detections are first smoothed with a centered
    moving average (``smoothing_window_frames``, see
    ``_smooth_fragment_points``) to remove per-frame localization jitter,
    then gaps up to ``max_interpolate_gap_frames`` are linearly interpolated
    between the smoothed points (see ``_interpolate_fragment_gaps``); larger
    gaps within the fragment are still drawn as a straight connector.
    """
    max_gap = max(1, int(max_gap_frames))
    ordered = sorted(points, key=lambda p: (p.get("frame", -1), p.get("time_seconds", 0.0)))
    if analytics_safe_only:
        ordered = [p for p in ordered if p.get("usable_for_analytics")]

    fragments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        first = current[0]
        last = current[-1]
        detected_count = sum(1 for p in current if not p.get("interpolated"))
        smoothed = _smooth_fragment_points(
            current,
            window=smoothing_window_frames,
            max_neighbor_gap_frames=max_interpolate_gap_frames,
        )
        filled = _interpolate_fragment_gaps(
            smoothed, max_interpolate_gap_frames=max_interpolate_gap_frames
        )
        fragments.append(
            {
                "fragment_id": len(fragments) + 1,
                "segment_id": first.get("segment_id"),
                "camera_state": first.get("camera_state"),
                "hoop_side": first.get("hoop_side"),
                "usable_for_analytics": bool(first.get("usable_for_analytics")),
                "start_frame": first.get("frame"),
                "end_frame": last.get("frame"),
                "start_time": first.get("time_seconds"),
                "end_time": last.get("time_seconds"),
                "point_count": len(filled),
                "detected_point_count": detected_count,
                "points": filled,
            }
        )
        current = []

    prev: dict[str, Any] | None = None
    for point in ordered:
        if not current:
            current = [point]
            prev = point
            continue
        assert prev is not None
        prev_frame = _as_int(prev.get("frame"))
        frame = _as_int(point.get("frame"))
        gap_break = (
            prev_frame is not None
            and frame is not None
            and (frame - prev_frame) > max_gap
        )
        segment_break = point.get("segment_id") != prev.get("segment_id")
        if gap_break or segment_break:
            _flush()
            current = [point]
        else:
            current.append(point)
        prev = point

    _flush()
    return fragments


def build_trajectories(
    track_frames: Iterable[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    team_lookup: dict[int, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group per-frame contact points by ``person_group_id``.

    Points use the mask-based stance point when a detection carries a mask
    polygon, else bbox bottom-center. Gap interpolation happens later, per
    fragment, in ``split_path_fragments``.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    team_lookup = team_lookup or {}
    points_by_gid: dict[int, list[dict[str, Any]]] = {}

    frames_seen = 0
    detections_seen = 0
    points_written = 0
    missing_person_group_id = 0
    invalid_bbox = 0
    mask_points_used = 0
    min_frame: int | None = None
    max_frame: int | None = None
    min_time: float | None = None
    max_time: float | None = None

    for frame_obj in track_frames:
        frames_seen += 1
        frame_number = _as_int(frame_obj.get("frame_number"))
        if frame_number is None:
            continue
        time_seconds = _as_float(frame_obj.get("timestamp_seconds"))
        if time_seconds is None:
            time_seconds = 0.0

        if min_frame is None or frame_number < min_frame:
            min_frame = frame_number
        if max_frame is None or frame_number > max_frame:
            max_frame = frame_number
        if min_time is None or time_seconds < min_time:
            min_time = time_seconds
        if max_time is None or time_seconds > max_time:
            max_time = time_seconds

        tracks = frame_obj.get("tracks") or []
        if not isinstance(tracks, list):
            continue

        for det in tracks:
            if not isinstance(det, dict):
                continue
            detections_seen += 1
            gid = _as_int(det.get("person_group_id"))
            if gid is None:
                missing_person_group_id += 1
                continue
            contact = bbox_bottom_center(det.get("bbox_xyxy"))
            if contact is None:
                invalid_bbox += 1
                continue
            point_source = "bbox_bottom"
            mask_polygon = det.get("mask_polygon")
            if mask_polygon:
                sx, sy, point_source = stance_point_from_mask(
                    mask_polygon, det.get("bbox_xyxy")
                )
                if point_source == "mask_bottom":
                    contact = (sx, sy)
            if point_source == "mask_bottom":
                mask_points_used += 1
            x, y = contact
            points_by_gid.setdefault(gid, []).append(
                {
                    "frame": frame_number,
                    "time_seconds": round(float(time_seconds), 6),
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "x_normalized": round(x / float(frame_width), 6),
                    "y_normalized": round(y / float(frame_height), 6),
                    "point_source": point_source,
                    "interpolated": False,
                }
            )
            points_written += 1

    trajectories: list[dict[str, Any]] = []
    for gid in sorted(points_by_gid):
        points = points_by_gid[gid]
        points.sort(key=lambda p: (p["frame"], p["time_seconds"]))
        meta = team_lookup.get(gid)
        if isinstance(meta, dict):
            team_id = meta.get("team_id")
            jersey = meta.get("jersey_color_name")
            team_label = meta.get("team_label")
            role = meta.get("role")
        else:
            # Legacy callers may still pass ``{gid: team_id}``.
            team_id = meta
            jersey = None
            team_label = None
            role = None
        trajectories.append(
            {
                "person_group_id": gid,
                "team_id": team_id,
                "jersey_color_name": jersey,
                "team_label": team_label,
                "role": role,
                "points": points,
            }
        )

    duration_seconds = 0.0
    if min_time is not None and max_time is not None:
        duration_seconds = max(0.0, float(max_time) - float(min_time))

    summary = {
        "tracked_players": len(trajectories),
        "trajectory_points": points_written,
        "frame_range": {
            "start": min_frame,
            "end": max_frame,
        },
        "duration_seconds": round(duration_seconds, 6),
        "coordinate_space": COORDINATE_SPACE,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "missing_data": {
            "frames_seen": frames_seen,
            "detections_seen": detections_seen,
            "missing_person_group_id": missing_person_group_id,
            "invalid_bbox": invalid_bbox,
            "skipped_detections": missing_person_group_id + invalid_bbox,
        },
        "mask_stance_points": mask_points_used,
        "bbox_fallback_points": max(0, points_written - mask_points_used),
    }
    return trajectories, summary


def attach_segments_and_fragments(
    trajectories: list[dict[str, Any]],
    *,
    frame_index: dict[int, dict[str, Any]],
    max_gap_frames: int = 15,
    max_interpolate_gap_frames: int = 6,
    smoothing_window_frames: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Annotate raw points, build all fragments, and analytics-safe player rows."""
    raw_out: list[dict[str, Any]] = []
    analytics_out: list[dict[str, Any]] = []

    total_raw_points = 0
    total_analytics_safe_points = 0
    total_path_fragments = 0
    analytics_path_fragments = 0
    excluded_closeup_points = 0
    excluded_transition_points = 0
    excluded_closeup_frames: set[int] = set()
    excluded_transition_frames: set[int] = set()

    for traj in trajectories:
        points = annotate_points_with_segments(list(traj.get("points") or []), frame_index)
        total_raw_points += len(points)

        for point in points:
            state = point.get("camera_state")
            frame = _as_int(point.get("frame"))
            if state == "closeup":
                excluded_closeup_points += 1
                if frame is not None:
                    excluded_closeup_frames.add(frame)
            elif state == "transition":
                excluded_transition_points += 1
                if frame is not None:
                    excluded_transition_frames.add(frame)
            if point.get("usable_for_analytics"):
                total_analytics_safe_points += 1

        all_fragments = split_path_fragments(
            points,
            max_gap_frames=max_gap_frames,
            analytics_safe_only=False,
            max_interpolate_gap_frames=max_interpolate_gap_frames,
            smoothing_window_frames=smoothing_window_frames,
        )
        safe_fragments = split_path_fragments(
            points,
            max_gap_frames=max_gap_frames,
            analytics_safe_only=True,
            max_interpolate_gap_frames=max_interpolate_gap_frames,
            smoothing_window_frames=smoothing_window_frames,
        )
        total_path_fragments += len(all_fragments)
        analytics_path_fragments += len(safe_fragments)

        raw_out.append(
            {
                "person_group_id": traj.get("person_group_id"),
                "team_id": traj.get("team_id"),
                "jersey_color_name": traj.get("jersey_color_name"),
                "team_label": traj.get("team_label"),
                "role": traj.get("role"),
                "points": points,
                "path_fragments": all_fragments,
                "analytics_safe_point_count": sum(
                    1 for p in points if p.get("usable_for_analytics")
                ),
                "path_fragment_count": len(all_fragments),
                "analytics_safe_fragment_count": len(safe_fragments),
            }
        )
        if safe_fragments:
            analytics_out.append(
                {
                    "person_group_id": traj.get("person_group_id"),
                    "team_id": traj.get("team_id"),
                    "jersey_color_name": traj.get("jersey_color_name"),
                    "team_label": traj.get("team_label"),
                    "role": traj.get("role"),
                    "path_fragments": safe_fragments,
                    "point_count": sum(f["point_count"] for f in safe_fragments),
                    "path_fragment_count": len(safe_fragments),
                }
            )

    stats = {
        "total_raw_points": total_raw_points,
        "total_analytics_safe_points": total_analytics_safe_points,
        "excluded_closeup_points": excluded_closeup_points,
        "excluded_transition_points": excluded_transition_points,
        "excluded_closeup_frames": len(excluded_closeup_frames),
        "excluded_transition_frames": len(excluded_transition_frames),
        "path_fragment_count": total_path_fragments,
        "analytics_safe_fragment_count": analytics_path_fragments,
        "max_gap_frames": int(max_gap_frames),
    }
    return raw_out, analytics_out, stats


def write_trajectories_jsonl_gz(
    trajectories: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write one trajectory object per line (gzip-compressed JSONL)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for traj in trajectories:
            handle.write(json.dumps(traj, separators=(",", ":")))
            handle.write("\n")
    return output_path


def _ensure_segments_for_job(job_dir: Path) -> dict[str, Any]:
    """Load ``game_segments.json``, generating it when missing."""
    segments_path = job_dir / "analytics" / SEGMENTS_FILENAME
    payload = load_segments_payload(segments_path)
    if payload is not None and isinstance(payload.get("segments"), list):
        return payload

    # Lazy import avoids circular import with segments.py at module load.
    from src.analytics.segments import generate_segments_for_job

    logger.info("game_segments.json missing for %s — generating", job_dir.name)
    return generate_segments_for_job(job_dir)


def generate_trajectories_for_job(
    job_dir: str | Path,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    tracks_filename: str | None = None,
    person_groups_filename: str = "person_groups.json",
    ensure_segments: bool = True,
    trajectory_config: dict[str, Any] | None = None,
    segments_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate job-local trajectory artifacts under ``analytics/``.

    Writes:
      - ``analytics/trajectories.jsonl.gz`` (raw + annotated + all fragments)
      - ``analytics/trajectories_analytics.jsonl.gz`` (usable fragments only)
      - ``analytics/trajectory_summary.json``

    ``tracks_filename`` defaults to whichever source is more accurate for
    this job — see ``resolve_tracks_path``.
    """
    job_dir = Path(job_dir).resolve()
    tracks_path = resolve_tracks_path(job_dir, tracks_filename)
    person_groups_path = job_dir / person_groups_filename

    if not tracks_path.is_file():
        raise TrajectoryInputError(
            f"Tracking output missing: expected {tracks_path}"
        )

    if trajectory_config is None:
        try:
            from src.pipeline.player_track_pipeline import load_pipeline_config

            pipeline_cfg = load_pipeline_config()
            trajectory_config = (pipeline_cfg.get("analytics") or {}).get(
                "trajectories"
            ) or {}
        except Exception:  # noqa: BLE001
            trajectory_config = {}
    traj_cfg = merge_trajectory_config(trajectory_config)
    max_gap_frames = int(traj_cfg.get("max_gap_frames", 15))
    max_interpolate_gap_frames = int(
        traj_cfg.get("max_interpolate_gap_frames", 6)
    )
    smoothing_window_frames = int(traj_cfg.get("smoothing_window_frames", 5))

    width, height = resolve_frame_size(
        job_dir, frame_width=frame_width, frame_height=frame_height
    )

    team_lookup: dict[int, Any] = {}
    if person_groups_path.is_file():
        team_lookup = load_team_lookup(person_groups_path)
    else:
        logger.warning(
            "person_groups.json missing at %s; team_id will be null",
            person_groups_path,
        )

    if segments_payload is None:
        if ensure_segments:
            segments_payload = _ensure_segments_for_job(job_dir)
        else:
            segments_payload = load_segments_payload(
                job_dir / "analytics" / SEGMENTS_FILENAME
            ) or {"segments": []}

    frame_index = build_frame_segment_index(segments_payload.get("segments") or [])

    logger.info(
        "Building trajectories for job %s (%sx%s) from %s",
        job_dir.name,
        width,
        height,
        tracks_path.name,
    )
    trajectories, summary = build_trajectories(
        iter_track_frames(tracks_path),
        frame_width=width,
        frame_height=height,
        team_lookup=team_lookup,
    )
    raw_out, analytics_out, filter_stats = attach_segments_and_fragments(
        trajectories,
        frame_index=frame_index,
        max_gap_frames=max_gap_frames,
        max_interpolate_gap_frames=max_interpolate_gap_frames,
        smoothing_window_frames=smoothing_window_frames,
    )

    analytics_dir = job_dir / "analytics"
    trajectories_path = analytics_dir / TRAJECTORIES_FILENAME
    analytics_path = analytics_dir / TRAJECTORIES_ANALYTICS_FILENAME
    summary_path = analytics_dir / SUMMARY_FILENAME

    write_trajectories_jsonl_gz(raw_out, trajectories_path)
    write_trajectories_jsonl_gz(analytics_out, analytics_path)

    job_id = job_dir.name
    summary_payload = {
        "job_id": job_id,
        **summary,
        "total_raw_points": filter_stats["total_raw_points"],
        "total_analytics_safe_points": filter_stats["total_analytics_safe_points"],
        "excluded_closeup_frames": filter_stats["excluded_closeup_frames"],
        "excluded_transition_frames": filter_stats["excluded_transition_frames"],
        "excluded_closeup_points": filter_stats["excluded_closeup_points"],
        "excluded_transition_points": filter_stats["excluded_transition_points"],
        "path_fragment_count": filter_stats["path_fragment_count"],
        "analytics_safe_fragment_count": filter_stats["analytics_safe_fragment_count"],
        "max_gap_frames": filter_stats["max_gap_frames"],
        "max_interpolate_gap_frames": max_interpolate_gap_frames,
        "smoothing_window_frames": smoothing_window_frames,
        "tracks_source": tracks_path.name,
        "identity_locked": tracks_path.name == LOCKED_TRACKS_FILENAME,
        "segments_confidence": segments_payload.get("confidence"),
        "artifacts": {
            "trajectories": f"analytics/{TRAJECTORIES_FILENAME}",
            "trajectories_analytics": f"analytics/{TRAJECTORIES_ANALYTICS_FILENAME}",
            "summary": f"analytics/{SUMMARY_FILENAME}",
            "segments": f"analytics/{SEGMENTS_FILENAME}",
        },
    }
    write_json_atomic(summary_path, summary_payload)

    logger.info(
        "Wrote %s players / raw=%s safe=%s fragments=%s → %s",
        summary_payload["tracked_players"],
        summary_payload["total_raw_points"],
        summary_payload["total_analytics_safe_points"],
        summary_payload["path_fragment_count"],
        trajectories_path,
    )
    return summary_payload
