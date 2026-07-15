"""
Player Snapshot Selection — find one clear frame with visible players.

Samples a basketball MP4, runs YOLO on every Nth frame, picks the frame with
the most person detections, draws labeled boxes (Player_1, Player_2, …), and
saves a still image plus JSON for a future GUI where the user picks who to track.

No tracking, spacing metrics, or player identification in this step.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from yolo_config import PERSON_CLASS_ID

# scoutiq/ on path so the trained seg-model helpers (src.*) import cleanly
# whether run via the API or directly from the CLI.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.player_track_pipeline import (  # noqa: E402
    _apply_device,
    _resolve,
    _slugify,
    load_pipeline_config,
)
from src.showcase.exclusions import (  # noqa: E402
    filter_detections,
    load_exclusion_set,
    resolve_exclusion_path,
)
from src.showcase.run_showcase import detect_players_seg  # noqa: E402

# --- MVP settings ---
DEFAULT_MODEL = "yolov8n.pt"  # legacy fallback; live path uses best_player_seg.pt
CONFIDENCE_THRESHOLD = 0.35
SAMPLE_INTERVAL = 15
MIN_PREFERRED_PLAYERS = 5
SNAPSHOT_DIR = Path("outputs/player_snapshots")


@dataclass
class VideoMetadata:
    """Basic video properties read through OpenCV."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class PlayerDetection:
    """One detected player box on the snapshot frame."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: float
    center_y: float


@dataclass
class FrameCandidate:
    """A sampled frame and how many players YOLO found on it."""

    frame_number: int
    timestamp_seconds: float
    frame: object
    players: list[PlayerDetection]


def validate_video(video_path: Path) -> VideoMetadata:
    """Confirm the file exists, is MP4, and OpenCV can read its metadata."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Unsupported file type '{video_path.suffix}'. Only .mp4 is supported.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, _ = cap.read()
        if not ret:
            raise ValueError(f"Video is unreadable or has no frames: {video_path}")

        if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError(
                "Video metadata is invalid (FPS, width, height, or frame count is zero)."
            )

        return VideoMetadata(
            path=video_path,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
    finally:
        cap.release()


def resolve_device() -> str:
    """Use GPU when available, otherwise CPU."""
    return "0" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str) -> YOLO:
    """Load Ultralytics YOLO weights (downloads on first run)."""
    return YOLO(model_name)


def parse_person_detections(result, confidence_threshold: float) -> list[PlayerDetection]:
    """Extract person boxes above the confidence threshold from one YOLO result."""
    detections: list[PlayerDetection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id != PERSON_CLASS_ID:
            continue
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = round((x1 + x2) / 2, 2)
        center_y = round((y1 + y2) / 2, 2)

        detections.append(
            PlayerDetection(
                label="",
                confidence=round(confidence, 4),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=center_x,
                center_y=center_y,
            )
        )

    # Left-to-right order gives stable Player_1, Player_2 labels for the GUI.
    detections.sort(key=lambda d: d.center_x)
    for index, det in enumerate(detections, start=1):
        det.label = f"Player_{index}"

    return detections


def _players_from_seg(detections: list[dict]) -> list[PlayerDetection]:
    """Convert player-seg detection dicts into labeled PlayerDetection boxes."""
    players: list[PlayerDetection] = []
    for det in detections:
        box = det.get("bbox_xyxy") or []
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        players.append(
            PlayerDetection(
                label="",
                confidence=round(float(det.get("confidence") or 0.0), 4),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=round((x1 + x2) / 2, 2),
                center_y=round((y1 + y2) / 2, 2),
            )
        )

    # Left-to-right order gives stable Player_1, Player_2 labels for the GUI.
    players.sort(key=lambda d: d.center_x)
    for index, player in enumerate(players, start=1):
        player.label = f"Player_{index}"
    return players


def run_seg_on_frame(model: YOLO, frame, exclusions, conf: float, iou: float) -> list[PlayerDetection]:
    """Detect on-court players on one BGR frame with best_player_seg.pt + exclusions."""
    dets = detect_players_seg(model, frame, conf=conf, iou=iou)
    dets = filter_detections(dets, exclusions, class_names=("player",))
    return _players_from_seg(dets)


def sample_frame_candidates(
    metadata: VideoMetadata,
    model: YOLO,
    exclusions,
    conf: float,
    iou: float,
) -> list[FrameCandidate]:
    """
    Sample every SAMPLE_INTERVAL frames (skipping frame 0 when possible),
    run the seg model on each, and collect detection counts.
    """
    cap = cv2.VideoCapture(str(metadata.path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {metadata.path}")

    candidates: list[FrameCandidate] = []

    try:
        # Start at frame 15 to avoid the very first frame when possible.
        frame_indices = list(range(SAMPLE_INTERVAL, metadata.frame_count, SAMPLE_INTERVAL))

        # Fallback: if the clip is very short, still try later frames first.
        if not frame_indices:
            frame_indices = [0]

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_number = frame_index + 1
            timestamp_seconds = round(frame_index / metadata.fps, 2)
            players = run_seg_on_frame(model, frame, exclusions, conf, iou)

            candidates.append(
                FrameCandidate(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    frame=frame.copy(),
                    players=players,
                )
            )
    finally:
        cap.release()

    return candidates


def select_best_candidate(candidates: list[FrameCandidate]) -> FrameCandidate | None:
    """
    Pick the frame with the most player detections.

    Prefer frames with at least MIN_PREFERRED_PLAYERS. Avoid frame 1 when tied.
    """
    with_players = [c for c in candidates if c.players]
    if not with_players:
        return None

    preferred = [c for c in with_players if len(c.players) >= MIN_PREFERRED_PLAYERS]
    pool = preferred if preferred else with_players

    max_count = max(len(c.players) for c in pool)
    top = [c for c in pool if len(c.players) == max_count]

    non_first = [c for c in top if c.frame_number > 1]
    return non_first[0] if non_first else top[0]


def draw_labeled_snapshot(frame, players: list[PlayerDetection]):
    """Draw bounding boxes and Player_N labels on the selected frame."""
    annotated = frame.copy()

    for player in players:
        label = f"{player.label} {player.confidence:.2f}"
        cv2.rectangle(
            annotated,
            (player.x1, player.y1),
            (player.x2, player.y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            label,
            (player.x1, max(player.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_snapshot_outputs(
    video_path: Path,
    candidate: FrameCandidate,
    video_metadata: VideoMetadata,
) -> tuple[Path, Path, dict]:
    """Write the labeled JPG and player metadata JSON. Returns paths and payload dict."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    image_path = SNAPSHOT_DIR / f"{stem}_snapshot_labeled.jpg"
    json_path = SNAPSHOT_DIR / f"{stem}_snapshot_players.json"

    labeled_frame = draw_labeled_snapshot(candidate.frame, candidate.players)
    if not cv2.imwrite(str(image_path), labeled_frame):
        raise RuntimeError(f"Failed to save snapshot image: {image_path}")

    payload = {
        "video_path": str(video_path.resolve()),
        "snapshot_frame_number": candidate.frame_number,
        "timestamp_seconds": candidate.timestamp_seconds,
        "frame_width": video_metadata.width,
        "frame_height": video_metadata.height,
        "players": [
            {
                "label": player.label,
                "confidence": player.confidence,
                "x1": player.x1,
                "y1": player.y1,
                "x2": player.x2,
                "y2": player.y2,
                "center_x": player.center_x,
                "center_y": player.center_y,
            }
            for player in candidate.players
        ],
    }

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    return image_path, json_path, payload


def run_snapshot_selection(
    video_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    *,
    quiet: bool = False,
) -> dict:
    """
    Full pipeline: validate → sample → select → save.

    Returns a dict with snapshot metadata and output paths for API / GUI use.
    """
    path = Path(video_path)
    metadata = validate_video(path)

    if not quiet:
        print(f"Video: {metadata.path.name}")
        print(f"  FPS: {metadata.fps}  Resolution: {metadata.width}x{metadata.height}")
        print(f"  Frames: {metadata.frame_count}")

    cfg = load_pipeline_config()
    weights = _resolve(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained player-seg weights not found: {weights}. Train best_player_seg.pt first."
        )
    conf = float(cfg["conf"])
    iou = float(cfg["iou"])
    device = str(cfg.get("device", "0"))

    if not quiet:
        print(f"Loading player-seg model: {weights.name} (device: {device})")
    model = YOLO(str(weights))
    _apply_device(model, device)

    clip_id = _slugify(path.name)
    excl_path = resolve_exclusion_path(_resolve(cfg["exclusions_dir"]), clip_id)
    exclusions = load_exclusion_set(
        excl_path, width=metadata.width, height=metadata.height, clip_id=clip_id
    )

    if not quiet:
        print(f"Sampling every {SAMPLE_INTERVAL} frames (conf >= {conf}, exclusions="
              f"{excl_path.name if excl_path else 'defaults'})...")
    candidates = sample_frame_candidates(metadata, model, exclusions, conf, iou)

    if not candidates:
        raise RuntimeError("No frames could be sampled from the video.")

    best = select_best_candidate(candidates)
    if best is None:
        raise RuntimeError("No players detected in any sampled frame.")

    image_path, json_path, payload = save_snapshot_outputs(path, best, metadata)

    if not quiet:
        print("\nSnapshot selected:")
        print(f"  Frame number:     {best.frame_number}")
        print(f"  Timestamp:        {best.timestamp_seconds}s")
        print(f"  Players detected: {len(best.players)}")
        print(f"  Image saved:      {image_path.resolve()}")
        print(f"  JSON saved:       {json_path.resolve()}")

    return {
        **payload,
        "image_path": str(image_path.resolve()),
        "json_path": str(json_path.resolve()),
        "image_filename": image_path.name,
        "json_filename": json_path.name,
        "player_count": len(best.players),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a clear player snapshot frame and save labeled boxes + JSON."
    )
    parser.add_argument(
        "video_path",
        help="Path to input .mp4 (e.g. data/input_videos/caitlin_clark_clip.mp4)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO weights filename (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    try:
        run_snapshot_selection(args.video_path, model_name=args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
