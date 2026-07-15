"""Draw detection overlays on frames and assemble the final annotated MP4."""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from collections import deque as Deque  # noqa: F401 (unused but kept for future use)

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Visual config.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "ball": (0, 140, 255),          # orange
    "player": (40, 180, 40),        # green (NBA trained detector)
    "player_red": (60, 60, 245),    # red (BGR)
    "player_black_blue": (220, 160, 40),  # blue (BGR)
    "referee": (200, 200, 200),     # light gray
    "hoop": (255, 200, 0),          # cyan-ish
}
MASK_ALPHA = 0.35
TRAIL_BALL_FRAMES = 15
TRAIL_PLAYER_FRAMES = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICK = 1


def _color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, (180, 180, 180))


def _draw_mask(canvas: np.ndarray, polygon: list[list[float]], color: tuple, alpha: float) -> None:
    if not polygon or len(polygon) < 3:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in polygon], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)


def _draw_box(canvas: np.ndarray, box: list, color: tuple, label: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    text_y = max(y1 - 4, th + 2)
    cv2.rectangle(canvas, (x1, text_y - th - 2), (x1 + tw + 4, text_y + 2), color, -1)
    cv2.putText(canvas, label, (x1 + 2, text_y), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)


def _draw_trail(
    canvas: np.ndarray,
    trail: list[tuple[int, int]],
    color: tuple,
    *,
    ball: bool = False,
) -> None:
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha_frac = i / len(trail)
        faded = tuple(int(c * alpha_frac) for c in color)
        thickness = 3 if ball else 2
        radius = max(2, int(4 * alpha_frac)) if ball else 0
        cv2.line(canvas, trail[i - 1], trail[i], faded, thickness, cv2.LINE_AA)
        if ball and i == len(trail) - 1:
            cv2.circle(canvas, trail[i], radius + 1, color, -1)


def draw_frame(
    image: np.ndarray,
    tracked_dets: list[dict],
    frame_id: int,
    timestamp: float,
    ball_trails: dict[int, list],
    player_trails: dict[int, list],
) -> np.ndarray:
    canvas = image.copy()

    # 1) Masks (drawn first, behind boxes).
    for det in tracked_dets:
        poly = det.get("mask_polygon")
        color = _color(det.get("class_name", ""))
        if poly:
            _draw_mask(canvas, poly, color, MASK_ALPHA)

    # 2) Trails.
    for tid, trail in player_trails.items():
        # Get most recent class for color.
        color = _color("player_black_blue")
        for det in tracked_dets:
            if det.get("track_id") == tid:
                color = _color(det.get("class_name", ""))
                break
        _draw_trail(canvas, trail, color, ball=False)

    for tid, trail in ball_trails.items():
        _draw_trail(canvas, trail, CLASS_COLORS["ball"], ball=True)

    # 3) Boxes + labels.
    for det in tracked_dets:
        box = det.get("bbox_xyxy", [])
        if not box or len(box) < 4:
            continue
        class_name = det.get("class_name", "unknown")
        color = _color(class_name)
        track_id = det.get("track_id", -1)
        conf = det.get("confidence", 0.0)
        label = f"{class_name} #{track_id} {conf:.2f}"
        _draw_box(canvas, box, color, label)

    # 4) HUD.
    fps_str = f"frame {frame_id:04d} | t={timestamp:.2f}s"
    cv2.putText(canvas, fps_str, (8, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, fps_str, (8, 22), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def update_trails(
    tracked_dets: list[dict],
    ball_trails: dict[int, list],
    player_trails: dict[int, list],
) -> None:
    seen_tids: set[int] = set()
    for det in tracked_dets:
        tid = det.get("track_id", -1)
        if tid < 0:
            continue
        seen_tids.add(tid)
        box = det.get("bbox_xyxy", [])
        if not box:
            continue
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        class_name = det.get("class_name", "")
        if class_name == "ball":
            ball_trails.setdefault(tid, []).append((cx, cy))
            if len(ball_trails[tid]) > TRAIL_BALL_FRAMES:
                ball_trails[tid].pop(0)
        else:
            player_trails.setdefault(tid, []).append((cx, cy))
            if len(player_trails[tid]) > TRAIL_PLAYER_FRAMES:
                player_trails[tid].pop(0)


def render_video(
    frame_paths: list[Path],
    detections_by_frame: dict[int, list[dict]],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {tmp_path}")

    ball_trails: dict[int, list] = {}
    player_trails: dict[int, list] = {}

    from tqdm import tqdm
    for frame_path in tqdm(frame_paths, desc="Rendering overlay", unit="frame"):
        # Derive frame_id from filename.
        try:
            frame_id = int(frame_path.stem.split("_")[-1])
        except ValueError:
            frame_id = 0
        timestamp = frame_id / fps

        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning("Could not read frame: %s", frame_path)
            image = np.zeros((height, width, 3), dtype=np.uint8)
        elif image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))

        tracked_dets = detections_by_frame.get(frame_id, [])
        update_trails(tracked_dets, ball_trails, player_trails)
        annotated = draw_frame(image, tracked_dets, frame_id, timestamp, ball_trails, player_trails)
        writer.write(annotated)

    writer.release()

    # Re-encode to H.264 for browser/player compatibility.
    _reencode(tmp_path, output_path)
    tmp_path.unlink(missing_ok=True)
    logger.info("Saved final video: %s", output_path)
    return output_path


def _reencode(src: Path, dst: Path) -> None:
    """Re-encode mp4v → H.264 for broad player compatibility."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "fast",
             "-crf", "22", "-movflags", "+faststart", str(dst)],
            capture_output=True, timeout=300,
        )
        if result.returncode == 0:
            return
        logger.debug("ffmpeg re-encode failed (rc=%d) — keeping mp4v output.", result.returncode)
    except FileNotFoundError:
        logger.debug("ffmpeg not found — keeping mp4v output.")
    except Exception as exc:
        logger.debug("ffmpeg error: %s — keeping mp4v output.", exc)
    # Fallback: just rename the raw file.
    import shutil
    shutil.copy2(src, dst)
