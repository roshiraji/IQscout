"""Camera-motion homographies used to pin trajectory trails to the court."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from src.analytics.camera_motion import (
    CAMERA_MOTION_FILENAME,
    estimate_camera_motion_for_job,
    load_camera_motion,
)


def _apply(row: list[float], x: float, y: float) -> tuple[float, float]:
    """Warp a normalized point through a stored frame row."""
    h = [*row[2:10], 1.0]
    w = h[6] * x + h[7] * y + h[8]
    return ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)


def _write_panning_clip(
    job_dir: Path,
    *,
    width: int = 320,
    height: int = 180,
    frames: int = 24,
    shift_per_frame: int = 3,
) -> None:
    """A textured court that slides left under a fixed camera-style pan."""
    rng = np.random.default_rng(7)
    canvas_width = width + frames * shift_per_frame + 40
    board = np.full((height, canvas_width, 3), 30, dtype=np.uint8)
    # Non-repeating blobs: a periodic grid would let flow lock onto the wrong
    # copy of the pattern, which is a property of the fixture, not the code.
    for _ in range(900):
        cx = int(rng.integers(0, canvas_width))
        cy = int(rng.integers(0, height))
        color = tuple(int(v) for v in rng.integers(80, 255, size=3))
        cv2.circle(board, (cx, cy), int(rng.integers(2, 6)), color, -1)

    video_path = job_dir / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    track_rows = []
    for index in range(frames):
        offset = index * shift_per_frame
        frame = board[:, offset : offset + width].copy()
        # A moving foreground body that must not vote on camera motion.
        px = 40 + index * 6
        cv2.rectangle(frame, (px, 60), (px + 24, 130), (0, 0, 255), -1)
        writer.write(frame)
        track_rows.append(
            {
                "frame_number": index,
                "timestamp_seconds": round(index / 30.0, 6),
                "tracks": [{"bbox_xyxy": [px, 60, px + 24, 130]}],
            }
        )
    writer.release()

    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in track_rows:
            handle.write(json.dumps(row) + "\n")

    (job_dir / "summary.json").write_text(
        json.dumps({"video_path": str(video_path)}), encoding="utf-8"
    )
    (job_dir / "snapshot.json").write_text(
        json.dumps({"frame_width": width, "frame_height": height}), encoding="utf-8"
    )


def _segments(frames: int) -> dict:
    return {
        "segments": [
            {
                "segment_id": 1,
                "start_frame": 0,
                "end_frame": frames - 1,
                "camera_state": "wide_gameplay",
                "usable_for_analytics": True,
            }
        ]
    }


def test_camera_motion_soft_fails_without_video(tmp_path: Path):
    payload = estimate_camera_motion_for_job(tmp_path)
    assert payload["available"] is False
    assert payload["frames"] == []
    assert (tmp_path / "analytics" / CAMERA_MOTION_FILENAME).is_file()


def test_camera_motion_cancels_a_pan(tmp_path: Path):
    frames = 24
    shift = 3
    width = 320
    _write_panning_clip(tmp_path, frames=frames, shift_per_frame=shift, width=width)

    payload = estimate_camera_motion_for_job(
        tmp_path, segments_payload=_segments(frames)
    )
    assert payload["available"] is True
    assert payload["anchor_count"] == 1
    assert payload["frame_count"] == frames

    rows = {int(row[0]): row for row in payload["frames"]}
    assert rows[0][1] == 0

    # A spot fixed to the board drifts left in screen space as the camera pans;
    # warping it back onto the anchor frame should hold it still.
    anchored = []
    for index in (0, 8, 16, 23):
        board_x = 0.5 * width  # where the spot sat in frame 0
        screen_x = (board_x - index * shift) / width
        anchored.append(_apply(rows[index], screen_x, 0.5)[0])

    assert max(anchored) - min(anchored) < 0.02
    assert abs(anchored[0] - 0.5) < 0.01


def test_camera_motion_round_trips_from_disk(tmp_path: Path):
    frames = 12
    _write_panning_clip(tmp_path, frames=frames)
    written = estimate_camera_motion_for_job(
        tmp_path, segments_payload=_segments(frames)
    )
    loaded = load_camera_motion(tmp_path)
    assert loaded is not None
    assert loaded["frame_count"] == written["frame_count"]
    assert loaded["anchors"][0]["reference_frame"] == 0


def test_camera_motion_skips_non_gameplay_frames(tmp_path: Path):
    frames = 18
    _write_panning_clip(tmp_path, frames=frames)
    segments = {
        "segments": [
            {
                "segment_id": 1,
                "start_frame": 0,
                "end_frame": 5,
                "camera_state": "closeup",
                "usable_for_analytics": False,
            },
            {
                "segment_id": 2,
                "start_frame": 6,
                "end_frame": frames - 1,
                "camera_state": "wide_gameplay",
                "usable_for_analytics": True,
            },
        ]
    }
    payload = estimate_camera_motion_for_job(tmp_path, segments_payload=segments)
    stored = {int(row[0]) for row in payload["frames"]}
    assert stored == set(range(6, frames))
    assert payload["anchors"][0]["reference_frame"] == 6
