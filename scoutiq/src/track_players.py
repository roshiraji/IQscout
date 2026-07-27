"""
Player tracking for user-selected targets.

After the GUI snapshot step, the user picks Player_# boxes. This module:
1. Runs YOLO person detection on every frame
2. Associates detections with ByteTrack IDs
3. Locks selected boxes → track IDs at the snapshot frame
4. Exports per-frame positions for the full clip (including before snapshot)

Output feeds future ScoutIQ metrics (speed, spacing, screens, etc.).
No spacing or tactical scoring in this step yet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from player_snapshot_selector import DEFAULT_MODEL, resolve_device, validate_video
from video_encoding import encode_for_browser

# scoutiq/ on path so the trained seg-model helpers (src.*) import cleanly
# whether run via the API or directly from the CLI.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.object_tracker import update_tracker  # noqa: E402
from src.pipeline.coach_insights import (  # noqa: E402
    compute_coach_insights,
    write_pair_distances_csv,
)
from src.pipeline.geometry_points import stance_point_from_mask  # noqa: E402
from src.pipeline.play_outcome import load_play_outcome  # noqa: E402
from src.pipeline.player_track_pipeline import (  # noqa: E402
    _apply_device,
    _resolve,
    _slugify,
    load_pipeline_config,
)
from src.pipeline.selection_lock import (  # noqa: E402
    build_locked_frame_records,
    lock_selected_bodies,
    write_locked_tracks_jsonl,
)
from src.pipeline.jersey_team_colors import enrich_tracks_with_appearance  # noqa: E402
from src.pipeline.sam2_video_tracker import (  # noqa: E402
    Sam2Unavailable,
    is_enabled as sam2_video_enabled,
    propagate_selected_players,
)
from src.pipeline.selected_player_reid import SelectedPlayerReId  # noqa: E402
from src.showcase.exclusions import (  # noqa: E402
    filter_detections,
    load_exclusion_set,
    resolve_exclusion_path,
)
from src.showcase.run_showcase import _smooth_polygon, detect_players_seg  # noqa: E402

TRACKING_DIR = Path("outputs/tracking")
TRACKING_VIDEOS_DIR = Path("outputs/tracking_videos")
SELECTIONS_DIR = Path("outputs/tracking_selections")

CONFIDENCE_THRESHOLD = 0.35
MIN_MATCH_IOU = 0.30

# Distinct BGR colors per tracked player for the preview video
TRACK_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 255, 0),
    (180, 180, 255),
]


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_tracking_selection(selection_path: str | Path) -> dict:
    """Load a saved GUI tracking-selection JSON file."""
    path = Path(selection_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking selection not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _match_selected_to_tracks(
    selected_players: list[dict],
    tracked: list[dict],
) -> dict[int, str]:
    """
    At the snapshot frame, map ByteTrack IDs to user labels (Player_2, etc.)
    using IoU matching between selected boxes and tracked seg detections.
    """
    tracked = [d for d in tracked if int(d.get("track_id", -1)) >= 0]
    if not tracked:
        return {}

    n_selected = len(selected_players)
    n_detections = len(tracked)
    iou_matrix = np.zeros((n_selected, n_detections), dtype=np.float32)

    for row_idx, player in enumerate(selected_players):
        selected_box = [player["x1"], player["y1"], player["x2"], player["y2"]]
        for col_idx, det in enumerate(tracked):
            iou_matrix[row_idx, col_idx] = _box_iou(selected_box, det["bbox_xyxy"][:4])

    cost_matrix = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    label_by_track_id: dict[int, str] = {}
    for row_idx, col_idx in zip(row_indices, col_indices):
        if iou_matrix[row_idx, col_idx] < MIN_MATCH_IOU:
            continue
        track_id = int(tracked[col_idx]["track_id"])
        label_by_track_id[track_id] = selected_players[row_idx]["label"]

    return label_by_track_id


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Stable color per Player_# label."""
    try:
        digits = "".join(char for char in label if char.isdigit())
        number = int(digits)
    except ValueError:
        number = 1
    return TRACK_COLORS[(number - 1) % len(TRACK_COLORS)]


def _write_tracking_csv(path: Path, rows: list[dict]) -> None:
    """Write per-frame tracked positions."""
    fieldnames = [
        "frame_number",
        "timestamp_seconds",
        "player_label",
        "track_id",
        "person_group_id",
        "team_id",
        "team_label",
        "jersey_color_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "stance_x",
        "stance_y",
        "position_source",
        "box_width",
        "box_height",
        "has_mask",
        "selection_locked",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


MASK_ALPHA = 0.35


def _draw_tracked_frame(
    frame,
    tracked: list[dict],
    label_by_id: dict[int, str],
    color_by_id: dict[int, tuple[int, int, int]] | None = None,
    *,
    draw_style: str = "full",
) -> None:
    """Draw only the user-selected tracked players.

    draw_style:
      - ``full``: jersey mask + bbox + label (default Confirm preview)
      - ``masks``: jersey-colored segmentation only (no boxes / text)
    """
    color_by_id = color_by_id or {}
    masks_only = draw_style == "masks"
    for det in tracked:
        draw_id = det.get("person_group_id")
        if draw_id is None:
            draw_id = det.get("track_id", -1)
        draw_id = int(draw_id)
        label = label_by_id.get(draw_id)
        if label is None:
            continue

        color = color_by_id.get(draw_id) or _color_for_label(label)
        confidence = float(det.get("confidence") or 0.0)

        poly = det.get("mask_polygon")
        if poly and len(poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
            if not masks_only:
                cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
        elif masks_only:
            # Soft ellipse fill when polygon is missing so masks-only still shows them.
            h, w = frame.shape[:2]
            soft = _soft_bbox_cutout_mask(h, w, det.get("bbox_xyxy") or [0, 0, 0, 0])
            if soft.any():
                overlay = frame.copy()
                overlay[soft > 0] = color
                cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)

        if masks_only:
            continue

        box = det.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _soft_bbox_cutout_mask(
    height: int,
    width: int,
    bbox_xyxy: list,
    *,
    expand: float = 0.48,
) -> np.ndarray:
    """Elliptical soft mask when segmentation polygon is missing/weak."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in (bbox_xyxy or [0, 0, 0, 0])[:4]]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return mask
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ax = max(1, int((x2 - x1) * expand))
    ay = max(1, int((y2 - y1) * (expand + 0.03)))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _isolation_mask_for_det(
    height: int,
    width: int,
    det: dict,
    *,
    expand: float = 0.48,
) -> np.ndarray:
    """Binary person mask: prefer polygon, else soft bbox ellipse."""
    poly = det.get("mask_polygon")
    if poly and len(poly) >= 3:
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        if int(mask.sum()) >= 32:
            return mask
    return _soft_bbox_cutout_mask(
        height, width, det.get("bbox_xyxy") or [0, 0, 0, 0], expand=expand
    )


def _draw_isolated_frame(
    frame,
    all_dets: list[dict],
    selected_group_ids: set[int],
    label_by_id: dict[int, str] | None = None,
    color_by_id: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Keep court/crowd/scene; erase non-selected players via their masks.

    Selected players stay as natural pixels. Non-selected regions are inpainted
    so the background continues underneath. Missing / weak mask_polygon falls
    back to an elliptical soft bbox. Thin team-color outline marks selected.
    """
    label_by_id = label_by_id or {}
    color_by_id = color_by_id or {}
    canvas = frame.copy()
    height, width = frame.shape[:2]
    erase = np.zeros((height, width), dtype=np.uint8)
    keep = np.zeros((height, width), dtype=np.uint8)

    for det in all_dets:
        gid = det.get("person_group_id")
        if gid is None:
            gid = det.get("track_id", -1)
        gid = int(gid)
        # Slightly larger soft bbox when erasing so fringe/jersey edges go away.
        expand = 0.48 if gid in selected_group_ids else 0.55
        cut = _isolation_mask_for_det(height, width, det, expand=expand)
        if not cut.any():
            continue
        if gid in selected_group_ids:
            keep = cv2.bitwise_or(keep, cut)
        else:
            erase = cv2.bitwise_or(erase, cut)

    # Never wipe pixels that belong to a selected player (occlusion overlap).
    if erase.any():
        erase = cv2.bitwise_and(erase, cv2.bitwise_not(keep))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erase = cv2.dilate(erase, kernel, iterations=1)
        erase = cv2.bitwise_and(erase, cv2.bitwise_not(keep))
        if erase.any():
            canvas = cv2.inpaint(canvas, erase, 3, cv2.INPAINT_TELEA)

    # Minimal selected highlight — outline only, no black cutout.
    for det in all_dets:
        gid = det.get("person_group_id")
        if gid is None:
            gid = det.get("track_id", -1)
        gid = int(gid)
        if gid not in selected_group_ids:
            continue
        label = label_by_id.get(gid)
        color = color_by_id.get(gid) or (
            _color_for_label(label) if label else (0, 255, 255)
        )
        cut = _isolation_mask_for_det(height, width, det, expand=0.48)
        if cut.any():
            contours, _ = cv2.findContours(
                cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 1, cv2.LINE_AA)
    return canvas


def _resolve_selected_person_groups(
    artifact_dir: Path,
    selected_track_ids: list[int] | None,
    selected_person_group_ids: list[int] | None,
    selected_players: list[dict] | None,
) -> tuple[set[int], dict[int, int]]:
    """Map selection to stable person_group_ids (track_id fallback supported).

    Snapshot boxes use person_group_id as the selectable unit and often mirror
    that value into track_id. Never treat those mirrored IDs as raw ByteTrack
    members through track_to_group — that incorrectly pulls in other groups.
    """
    track_to_group: dict[int, int] = {}
    groups_path = artifact_dir / "person_groups.json"
    if groups_path.is_file():
        payload = json.loads(groups_path.read_text(encoding="utf-8"))
        track_to_group = {
            int(k): int(v) for k, v in (payload.get("track_to_group") or {}).items()
        }

    selected: set[int] = set()
    for gid in selected_person_group_ids or []:
        selected.add(int(gid))
    for player in selected_players or []:
        if player.get("person_group_id") is not None:
            selected.add(int(player["person_group_id"]))

    # Prefer explicit person groups; only resolve raw track_ids as a fallback.
    if selected:
        return selected, track_to_group

    for player in selected_players or []:
        if player.get("track_id") is not None:
            tid = int(player["track_id"])
            selected.add(int(track_to_group.get(tid, tid)))
    for tid in selected_track_ids or []:
        tid_i = int(tid)
        selected.add(int(track_to_group.get(tid_i, tid_i)))
    return selected, track_to_group


def _selection_seed_boxes(
    selected_players: list[dict] | None,
    selected_group_ids: set[int],
    frame_records: list[dict],
    snapshot_frame: int,
) -> dict[int, list[float]]:
    """Clicked box per selected person, falling back to the snapshot detection.

    SAM 2 is prompted with exactly what the coach drew a box around; anything
    else reintroduces the "it started on the wrong player" failure at frame one.
    """
    seeds: dict[int, list[float]] = {}
    for player in selected_players or []:
        gid = player.get("person_group_id")
        if gid is None:
            gid = player.get("track_id")
        if gid is None or int(gid) not in selected_group_ids:
            continue
        if player.get("x1") is None:
            continue
        seeds[int(gid)] = [
            float(player["x1"]),
            float(player["y1"]),
            float(player["x2"]),
            float(player["y2"]),
        ]

    missing = selected_group_ids - set(seeds)
    if not missing:
        return seeds

    snapshot_record = None
    for record in frame_records:
        if int(record.get("frame_number", -1)) == int(snapshot_frame):
            snapshot_record = record
            break
    for det in (snapshot_record or {}).get("tracks") or []:
        gid = det.get("person_group_id")
        if gid is None:
            gid = det.get("track_id")
        if gid is None or int(gid) not in missing:
            continue
        box = det.get("bbox_xyxy") or []
        if len(box) >= 4:
            seeds[int(gid)] = [float(v) for v in box[:4]]
    return seeds


def render_selected_tracks(
    job_dir: str | Path,
    selected_track_ids: list[int] | None = None,
    *,
    selected_person_group_ids: list[int] | None = None,
    selected_players: list[dict] | None = None,
    quiet: bool = False,
    progress_cb: Callable[..., None] | None = None,
) -> dict:
    """Render selected person groups from canonical artifacts (no YOLO/ByteTrack).

    progress_cb(stage: str, progress: float, detail: str | None) is optional;
    progress is 0..1 for the Confirm UI.
    """
    def _progress(stage: str, progress: float, detail: str | None = None) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, max(0.0, min(1.0, float(progress))), detail)
        except Exception:  # noqa: BLE001
            pass

    artifact_dir = Path(job_dir).resolve()
    summary_path = artifact_dir / "summary.json"
    tracks_path = artifact_dir / "tracks.jsonl.gz"
    if not summary_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError("Canonical tracking artifacts are missing for this job.")

    _progress("loading_selection", 0.04, "Locking your selected players…")
    canonical = json.loads(summary_path.read_text(encoding="utf-8"))
    video_path = Path(canonical["video_path"])
    selected_group_ids, _track_to_group = _resolve_selected_person_groups(
        artifact_dir,
        selected_track_ids,
        selected_person_group_ids,
        selected_players,
    )
    if not selected_group_ids:
        raise ValueError("Select at least one person group (or track ID).")
    if len(selected_group_ids) < 2:
        raise ValueError(
            "Select at least two person groups for coach insights "
            "(pairwise metrics require a primary pair)."
        )

    _progress(
        "loading_tracks",
        0.10,
        f"Loading tracks for {len(selected_group_ids)} players…",
    )

    label_by_group_id = {
        gid: f"Person #{gid}" for gid in sorted(selected_group_ids)
    }
    color_by_group_id: dict[int, tuple[int, int, int]] = {}
    team_meta_by_group: dict[int, dict] = {}
    groups_path = artifact_dir / "person_groups.json"
    if groups_path.is_file():
        try:
            for group in (
                json.loads(groups_path.read_text(encoding="utf-8")).get("person_groups")
                or []
            ):
                gid = int(group["person_group_id"])
                # Load jersey meta for *all* groups so lock scoring can read
                # color on handoff targets that no longer carry the selected gid.
                team_meta_by_group[gid] = {
                    "team_id": group.get("team_id"),
                    "team_label": group.get("team_label"),
                    "jersey_color_name": group.get("jersey_color_name"),
                    "role": group.get("role") or "player",
                }
                if gid not in selected_group_ids:
                    continue
                bgr = group.get("overlay_color_bgr")
                if bgr and len(bgr) >= 3:
                    color_by_group_id[gid] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            pass
    for player in selected_players or []:
        gid = player.get("person_group_id")
        if gid is None and player.get("track_id") is not None:
            gid = player["track_id"]
        if gid is None:
            continue
        gid = int(gid)
        if gid in selected_group_ids:
            label_by_group_id[gid] = (
                player.get("player_name") or player.get("label") or f"Person #{gid}"
            )
            bgr = player.get("overlay_color_bgr")
            if bgr and len(bgr) >= 3:
                color_by_group_id[gid] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            team_meta_by_group[gid] = {
                "team_id": player.get("team_id"),
                "team_label": player.get("team_label"),
                "jersey_color_name": player.get("jersey_color_name"),
                "role": player.get("role") or "player",
            }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = artifact_dir / "focused_players.mp4"
    tmp_video = artifact_dir / "focused_players.tmp.mp4"
    output_masks_video = artifact_dir / "focused_masks.mp4"
    tmp_masks_video = artifact_dir / "focused_masks.tmp.mp4"
    output_csv = artifact_dir / "selected_tracks.csv"
    output_summary = artifact_dir / "selection_summary.json"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    masks_writer = cv2.VideoWriter(str(tmp_masks_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create focused preview: {tmp_video}")
    if not masks_writer.isOpened():
        writer.release()
        cap.release()
        raise RuntimeError(f"Could not create masks preview: {tmp_masks_video}")

    rows: list[dict] = []
    frames_rendered = 0
    frames_with_selected = 0
    start = time.perf_counter()
    expected_frames = max(int(canonical.get("source_frame_count") or 0), 1)
    last_progress_emit = 0.0

    # Load the full canonical track timeline, then lock onto the bodies clicked
    # in the snapshot (prevents ByteTrack ID handoffs).
    _progress("loading_tracks", 0.12, "Locking selected bodies across the clip…")
    frame_records: list[dict] = []
    with gzip.open(tracks_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame_records.append(json.loads(line))

    # Prefer the snapshot the user actually clicked (snapshot.json), not a
    # stale summary value — wrong frame + click boxes = wrong bodies at t=0.
    snapshot_frame = None
    snap_path = artifact_dir / "snapshot.json"
    if snap_path.is_file():
        try:
            snap_payload = json.loads(snap_path.read_text(encoding="utf-8"))
            if snap_payload.get("snapshot_frame_number") is not None:
                snapshot_frame = int(snap_payload["snapshot_frame_number"])
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            snapshot_frame = None
    if snapshot_frame is None:
        for player in selected_players or []:
            if player.get("snapshot_frame_number") is not None:
                snapshot_frame = int(player["snapshot_frame_number"])
                break
    if snapshot_frame is None and canonical.get("snapshot_frame_number") is not None:
        snapshot_frame = int(canonical["snapshot_frame_number"])
    snapshot_frame = int(snapshot_frame or 0)

    reid = SelectedPlayerReId()
    reid_embeddings = 0
    locked: dict[int, dict[int, dict]] | None = None
    lock_backend = "osnet_reid+spatial"
    sam2_stats: dict | None = None
    sam2_error: str | None = None

    # Primary path: SAM 2 video propagation carries one mask memory per clicked
    # body, so a defender pressed against a target cannot inherit the label.
    seeds = _selection_seed_boxes(
        selected_players, selected_group_ids, frame_records, snapshot_frame
    )
    if sam2_video_enabled() and len(seeds) == len(selected_group_ids):
        _progress("loading_tracks", 0.13, "Locking selected players with SAM 2…")

        def _sam2_progress(fraction: float, detail: str) -> None:
            _progress("loading_tracks", 0.13 + 0.34 * float(fraction), detail)

        try:
            locked, sam2_stats = propagate_selected_players(
                video_path=video_path,
                frame_records=frame_records,
                snapshot_frame_number=snapshot_frame,
                seeds=seeds,
                labels=label_by_group_id,
                reid=reid,
                progress_cb=_sam2_progress,
            )
            lock_backend = "sam2_video+osnet_reid"
        except Sam2Unavailable as exc:
            sam2_error = str(exc)
            locked = None
            print(f"[track_players] SAM 2 video unavailable: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - Confirm must still produce a clip
            sam2_error = f"{type(exc).__name__}: {exc}"
            locked = None
            print(f"[track_players] SAM 2 video propagation failed: {exc}", flush=True)
    elif sam2_video_enabled():
        sam2_error = "Missing a clicked box for at least one selected player."

    if locked is None:
        # Fallback: deep-ReID features on every candidate body, then the
        # spatial/appearance lock decides ownership frame by frame.
        _progress("loading_tracks", 0.16, "Building player identity references…")
        appearance_cap = cv2.VideoCapture(str(video_path))
        if appearance_cap.isOpened():
            try:
                for index, frame_record in enumerate(frame_records):
                    ok, image = appearance_cap.read()
                    if not ok:
                        break
                    tracks = list(frame_record.get("tracks") or [])
                    enrich_tracks_with_appearance(image, tracks)
                    reid_embeddings += reid.embed_detections(image, tracks)
                    if index and index % 30 == 0:
                        _progress(
                            "loading_tracks",
                            0.16 + 0.20 * min(index / max(len(frame_records), 1), 1.0),
                            "Reading deep player identity features…",
                        )
            finally:
                appearance_cap.release()

        locked = lock_selected_bodies(
            frame_records,
            snapshot_frame_number=snapshot_frame,
            selected_group_ids=selected_group_ids,
            selected_players=selected_players,
            label_by_group_id=label_by_group_id,
            team_meta_by_group=team_meta_by_group,
            frame_width=width,
            frame_height=height,
        )

    locked_records = build_locked_frame_records(
        frame_records, locked, selected_group_ids
    )
    locked_path = artifact_dir / "locked_tracks.jsonl.gz"
    write_locked_tracks_jsonl(locked_path, locked_records)

    try:
        for frame_record in locked_records:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number = int(frame_record["frame_number"])
            timestamp = float(frame_record.get("timestamp_seconds") or 0.0)
            tracked = []
            for det in (locked.get(frame_number) or {}).values():
                enriched = dict(det)
                enriched["person_group_id"] = int(det["person_group_id"])
                tracked.append(enriched)
            if tracked:
                frames_with_selected += 1
            for det in tracked:
                track_id = int(det.get("track_id", -1))
                gid = int(det["person_group_id"])
                x1, y1, x2, y2 = [
                    int(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])[:4]
                ]
                meta = team_meta_by_group.get(gid) or {}
                stance_x, stance_y, position_source = stance_point_from_mask(
                    det.get("mask_polygon"),
                    [x1, y1, x2, y2],
                )
                rows.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp,
                        "player_label": label_by_group_id[gid],
                        "track_id": track_id,
                        "person_group_id": gid,
                        "team_id": meta.get("team_id"),
                        "team_label": meta.get("team_label"),
                        "jersey_color_name": meta.get("jersey_color_name"),
                        "confidence": round(
                            float(det.get("confidence") or 0.0), 4
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": round((x1 + x2) / 2, 2),
                        "center_y": round((y1 + y2) / 2, 2),
                        "stance_x": round(stance_x, 2),
                        "stance_y": round(stance_y, 2),
                        "position_source": position_source,
                        "box_width": x2 - x1,
                        "box_height": y2 - y1,
                        "has_mask": bool(det.get("mask_polygon")),
                        "selection_locked": True,
                    }
                )
            annotated = frame.copy()
            _draw_tracked_frame(
                annotated, tracked, label_by_group_id, color_by_group_id
            )
            writer.write(annotated)
            masks_frame = frame.copy()
            _draw_tracked_frame(
                masks_frame,
                tracked,
                label_by_group_id,
                color_by_group_id,
                draw_style="masks",
            )
            masks_writer.write(masks_frame)
            frames_rendered += 1
            frac = frames_rendered / max(expected_frames, 1)
            progress_val = 0.50 + 0.20 * min(frac, 1.0)
            if (
                progress_val - last_progress_emit >= 0.03
                or frames_rendered >= expected_frames
            ):
                last_progress_emit = progress_val
                _progress(
                    "rendering_preview",
                    progress_val,
                    f"Building focused + segmentation previews "
                    f"({frames_rendered} / {expected_frames})…",
                )
    finally:
        cap.release()
        writer.release()
        masks_writer.release()

    if not rows:
        tmp_video.unlink(missing_ok=True)
        tmp_masks_video.unlink(missing_ok=True)
        raise RuntimeError("Selected person groups had no positions in canonical data.")

    _progress("encoding_videos", 0.72, "Encoding browser-ready previews…")
    tmp_video.replace(output_video)
    tmp_masks_video.replace(output_masks_video)
    encode_for_browser(output_video, quiet=quiet)
    _progress("encoding_videos", 0.80, "Encoding segmentation-only preview…")
    encode_for_browser(output_masks_video, quiet=quiet)
    _write_tracking_csv(output_csv, rows)
    runtime = round(time.perf_counter() - start, 2)
    source_frame_count = int(
        canonical.get("source_frame_count") or frames_rendered or 1
    )
    frame_coverage = round(frames_with_selected / max(frames_rendered, 1), 4)
    selected_player_retention = round(
        frames_with_selected / max(source_frame_count, 1), 4
    )
    quality_canon = canonical.get("quality") or {}

    primary_pair = tuple(sorted(selected_group_ids)[:2])
    possession_payload: dict = {}
    possession_path = artifact_dir / "possession.json"
    if possession_path.is_file():
        try:
            possession_payload = json.loads(possession_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            possession_payload = {}
    _progress("coach_insights", 0.88, "Computing Matchup Outcomes…")
    play_outcome_payload = load_play_outcome(artifact_dir)
    coach_insights = compute_coach_insights(
        rows,
        source_frame_count=source_frame_count,
        frame_width=width,
        frame_height=height,
        label_by_group=label_by_group_id,
        team_meta_by_group=team_meta_by_group,
        primary_pair=primary_pair,
        possession=possession_payload,
        play_outcome=play_outcome_payload,
    )
    coach_insights_path = artifact_dir / "coach_insights.json"
    coach_insights_path.write_text(
        json.dumps(coach_insights, indent=2), encoding="utf-8"
    )
    pair_distances_path = artifact_dir / "pair_distances.csv"
    write_pair_distances_csv(
        pair_distances_path,
        coach_insights.get("pair_distance_series") or [],
    )
    coach_narrative = coach_insights.get("coach_narrative") or ""

    summary = {
        "job_id": canonical.get("job_id"),
        "video_path": str(video_path.resolve()),
        "snapshot_frame_number": canonical.get("snapshot_frame_number"),
        "timestamp_seconds": canonical.get("timestamp_seconds"),
        "frame_index_base": 0,
        "source_frame_count": source_frame_count,
        "frames_tracked": frames_rendered,
        "tracked_frame_start": 0,
        "tracked_frame_end": max(frames_rendered - 1, 0),
        "frames_with_selected": frames_with_selected,
        "frame_coverage": frame_coverage,
        "selected_player_retention": selected_player_retention,
        "total_positions": len(rows),
        "selected_labels": list(label_by_group_id.values()),
        "selected_person_group_ids": sorted(selected_group_ids),
        "selection_lock": lock_backend,
        "selection_reid_backend": reid.backend,
        "selection_reid_embeddings": reid_embeddings,
        "selection_sam2": sam2_stats,
        "selection_sam2_error": sam2_error,
        "locked_tracks_jsonl": str(locked_path.resolve()),
        "track_mapping": {
            label: gid for gid, label in label_by_group_id.items()
        },
        "unique_raw_tracks": quality_canon.get("unique_raw_tracks"),
        "unique_person_groups": quality_canon.get("unique_person_groups"),
        "unmatched_labels": [],
        "tracks_csv": str(output_csv.resolve()),
        "pair_distances_csv": str(pair_distances_path.resolve()),
        "tracking_video": str(output_video.resolve()),
        "tracking_masks_video": str(output_masks_video.resolve()),
        "isolated_tracking_video": None,
        "isolation_available": False,
        "isolation_status": "pending",
        "coach_insights": coach_insights,
        "coach_narrative": coach_narrative,
        "coach_insights_json": str(coach_insights_path.resolve()),
        "possession": possession_payload,
        "play_outcome": play_outcome_payload,
        "offense_team_id": possession_payload.get("offense_team_id"),
        "defense_team_id": possession_payload.get("defense_team_id"),
        "quality": {
            "complete_clip": frames_rendered == source_frame_count,
            "frame_coverage": frame_coverage,
            "selected_player_retention": selected_player_retention,
            "unique_raw_tracks": quality_canon.get("unique_raw_tracks"),
            "unique_person_groups": quality_canon.get("unique_person_groups"),
            "average_id_switches_per_person_pre": quality_canon.get(
                "average_id_switches_per_person_pre"
            ),
            "average_id_switches_per_person_post": quality_canon.get(
                "average_id_switches_per_person_post"
            ),
            "canonical_fragmentation_warning": bool(
                quality_canon.get("fragmentation_warning")
            ),
            "gate": quality_canon.get("gate"),
            "note": "Selection reused stitched person groups; no inference was rerun.",
        },
        "total_runtime_seconds": runtime,
        "device": canonical.get("device"),
        "model": canonical.get("model"),
        "weights": canonical.get("weights"),
        "canonical_reuse": True,
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(output_summary.resolve())
    _progress("ready", 0.96, "Focused preview ready — queuing isolate…")
    return summary


def run_player_tracking(
    selection_path: str | Path | None = None,
    selection_data: dict | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    *,
    quiet: bool = False,
) -> dict:
    """
    Track user-selected players across the full clip.

    Snapshot frame only locks selected boxes → track IDs. Overlays/CSV cover
    every frame those IDs appear (including before the snapshot).

    Returns paths and summary stats for API / CLI use.
    """
    if selection_data is None:
        if selection_path is None:
            raise ValueError("Provide selection_path or selection_data.")
        selection_data = load_tracking_selection(selection_path)

    video_path = Path(selection_data["video_path"])
    metadata = validate_video(video_path)

    selected_players = selection_data.get("selected_players", [])
    if not selected_players:
        raise ValueError("No selected players found in tracking selection.")

    anchor_frame = int(selection_data["snapshot_frame_number"])
    if anchor_frame < 1 or anchor_frame > metadata.frame_count:
        raise ValueError(f"Snapshot frame {anchor_frame} is outside the video range.")

    cfg = load_pipeline_config()
    weights = _resolve(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained player-seg weights not found: {weights}. Train best_player_seg.pt first."
        )
    conf = float(cfg["conf"])
    iou = float(cfg["iou"])
    inference_device = str(cfg.get("device", "0")) if device is None else device
    mask_smooth_alpha = float(cfg.get("mask_smooth_alpha", 0.35))

    model = YOLO(str(weights))
    _apply_device(model, inference_device)

    clip_id = _slugify(video_path.name)
    excl_path = resolve_exclusion_path(_resolve(cfg["exclusions_dir"]), clip_id)
    exclusions = load_exclusion_set(
        excl_path, width=metadata.width, height=metadata.height, clip_id=clip_id
    )
    if not quiet:
        print(f"[track_players] weights={weights} device={inference_device} "
              f"conf={conf} iou={iou} exclusions={excl_path.name if excl_path else 'defaults'}")

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    tracks_csv_path = TRACKING_DIR / f"{stem}_tracks.csv"
    summary_json_path = TRACKING_DIR / f"{stem}_tracking_summary.json"
    tracking_video_path = TRACKING_VIDEOS_DIR / f"{stem}_tracked_players.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    writer = cv2.VideoWriter(
        str(tracking_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create tracking video: {tracking_video_path}")

    label_by_track_id: dict[int, str] = {}
    track_rows: list[dict] = []
    frames_exported = 0
    mask_memory: dict[int, list[list[float]]] = {}
    pipeline_start = time.perf_counter()
    # Buffer pre-snapshot frames so we can paint selected IDs from clip start
    # once the snapshot lock is known (ByteTrack IDs are stable across the pass).
    pending_frames: list[tuple[object, list[dict], int, float]] = []

    def _export_frame(
        frame_bgr,
        tracked_dets: list[dict],
        frame_number: int,
        timestamp_seconds: float,
    ) -> None:
        nonlocal frames_exported
        for det in tracked_dets:
            track_id = int(det.get("track_id", -1))
            label = label_by_track_id.get(track_id)
            if label is None:
                continue

            box = det.get("bbox_xyxy") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            confidence = round(float(det.get("confidence") or 0.0), 4)
            center_x = round((x1 + x2) / 2, 2)
            center_y = round((y1 + y2) / 2, 2)
            stance_x, stance_y, position_source = stance_point_from_mask(
                det.get("mask_polygon"),
                [x1, y1, x2, y2],
            )

            track_rows.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "player_label": label,
                    "track_id": track_id,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "stance_x": round(stance_x, 2),
                    "stance_y": round(stance_y, 2),
                    "position_source": position_source,
                    "box_width": x2 - x1,
                    "box_height": y2 - y1,
                    "has_mask": bool(det.get("mask_polygon")),
                }
            )

        frames_exported += 1
        annotated = frame_bgr.copy()
        _draw_tracked_frame(annotated, tracked_dets, label_by_track_id)
        writer.write(annotated)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp_seconds = round((frame_number - 1) / metadata.fps, 4)

        dets = detect_players_seg(model, frame, conf=conf, iou=iou)
        dets = filter_detections(dets, exclusions, class_names=("player",))
        tracked = update_tracker(dets)

        # Smooth masks per track id to reduce jitter.
        if mask_smooth_alpha > 0:
            for det in tracked:
                tid = int(det.get("track_id", -1))
                if tid < 0:
                    continue
                smoothed = _smooth_polygon(
                    mask_memory.get(tid), det.get("mask_polygon"), alpha=mask_smooth_alpha
                )
                if smoothed is not None:
                    det["mask_polygon"] = smoothed
                    mask_memory[tid] = smoothed

        if frame_number == anchor_frame:
            label_by_track_id = _match_selected_to_tracks(selected_players, tracked)
            if not label_by_track_id:
                cap.release()
                writer.release()
                raise RuntimeError(
                    "Could not match any selected players to tracker IDs at the snapshot frame."
                )
            for pending_frame, pending_tracked, pending_fn, pending_ts in pending_frames:
                _export_frame(pending_frame, pending_tracked, pending_fn, pending_ts)
            pending_frames.clear()
            _export_frame(frame, tracked, frame_number, timestamp_seconds)
        elif not label_by_track_id:
            pending_frames.append(
                (frame.copy(), list(tracked), frame_number, timestamp_seconds)
            )
        else:
            _export_frame(frame, tracked, frame_number, timestamp_seconds)

    cap.release()
    writer.release()
    encode_for_browser(tracking_video_path, quiet=quiet)
    total_runtime = round(time.perf_counter() - pipeline_start, 2)

    if not track_rows:
        raise RuntimeError("Tracking finished without exporting any player positions.")

    _write_tracking_csv(tracks_csv_path, track_rows)

    unmatched_labels = [
        player["label"]
        for player in selected_players
        if player["label"] not in label_by_track_id.values()
    ]

    summary = {
        "video_path": str(video_path.resolve()),
        "snapshot_frame_number": anchor_frame,
        "timestamp_seconds": selection_data.get("timestamp_seconds"),
        "selected_labels": [player["label"] for player in selected_players],
        "track_mapping": {label: track_id for track_id, label in label_by_track_id.items()},
        "frames_tracked": frames_exported,
        "total_positions": len(track_rows),
        "unmatched_labels": unmatched_labels,
        "tracks_csv": str(tracks_csv_path.resolve()),
        "tracking_video": str(tracking_video_path.resolve()),
        "total_runtime_seconds": total_runtime,
        "device": inference_device,
        "model": weights.name,
        "weights": str(weights.resolve()),
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json_path.resolve())

    if not quiet:
        print(
            f"Tracked {len(label_by_track_id)} player(s) across full clip "
            f"(locked at snapshot frame {anchor_frame})."
        )
        print(f"  Positions exported: {len(track_rows)}")
        print(f"  Tracks CSV:         {tracks_csv_path.resolve()}")
        print(f"  Tracking video:     {tracking_video_path.resolve()}")
        print(f"  Summary JSON:       {summary_json_path.resolve()}")
        if unmatched_labels:
            print(f"  Warning: could not match {unmatched_labels} at anchor frame.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track user-selected players from a saved GUI selection JSON."
    )
    parser.add_argument(
        "selection_path",
        help="Path to outputs/tracking_selections/<clip>_tracking_selection.json",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO weights filename (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    try:
        run_player_tracking(selection_path=args.selection_path, model_name=args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("Error: CUDA out of memory during tracking. Retry with a smaller model.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
