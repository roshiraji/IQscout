"""Run YOLO + BoT-SORT and serialize per-frame tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = SCOUTIQ_ROOT / "outputs" / "nba_training_loop" / "best_player_detector.pt"
DEFAULT_TRACKING_CONFIG = SCOUTIQ_ROOT / "configs" / "nba_player_tracking.yaml"


def _load_tracking_config(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def run_tracking(
    *,
    video: Path,
    weights: str,
    output_dir: Path,
    conf: float = 0.20,
    iou: float = 0.70,
    tracker: str = "botsort.yaml",
    save_video: bool = False,
    player_class_id: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.json"
    summary_path = output_dir / "tracking_summary.json"

    model = YOLO(weights)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    video_out: Path | None = None
    if save_video:
        video_out = output_dir / f"{video.stem}_tracked.mp4"
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    records: list[dict] = []
    frame_idx = 0
    player_detections = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou, verbose=False)[0]
        if writer is not None:
            writer.write(result.plot())
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, det_conf, cls, tid in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.id.cpu().numpy(),
            ):
                cls_id = int(cls)
                if cls_id == player_class_id:
                    player_detections += 1
                records.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_seconds": round(frame_idx / fps, 4),
                        "track_id": int(tid),
                        "bbox_xyxy": [float(x) for x in xyxy],
                        "det_conf": float(det_conf),
                        "cls": cls_id,
                    }
                )
        frame_idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    tracks_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "video": str(video.resolve()),
        "weights": str(Path(weights).resolve()) if Path(weights).exists() else weights,
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "frames_processed": frame_idx,
        "detections_serialized": len(records),
        "player_detections": player_detections,
        "tracks_json": str(tracks_path.resolve()),
        "tracked_video": str(video_out.resolve()) if video_out else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BoT-SORT tracking and write tracks.json.")
    parser.add_argument("--config", default=str(DEFAULT_TRACKING_CONFIG), help="YAML with video/weights defaults")
    parser.add_argument("--video")
    parser.add_argument("--weights")
    parser.add_argument("--output", help="Output run directory")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--tracker")
    parser.add_argument("--save-video", action="store_true", help="Write annotated MP4 with track IDs")
    parser.add_argument("--player-class-id", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_tracking_config(Path(args.config) if args.config else None)
    video = _resolve(args.video or cfg.get("video", ""), SCOUTIQ_ROOT)
    weights_path = args.weights or cfg.get("weights", str(DEFAULT_WEIGHTS))
    weights = str(_resolve(weights_path, SCOUTIQ_ROOT) if not Path(weights_path).is_absolute() else Path(weights_path))
    if not Path(weights).is_file():
        weights = weights_path

    output = Path(args.output) if args.output else _resolve(
        cfg.get("output_dir", f"outputs/player_tracking/{video.stem}"),
        SCOUTIQ_ROOT,
    )
    conf = args.conf if args.conf is not None else float(cfg.get("conf", 0.25))
    iou = args.iou if args.iou is not None else float(cfg.get("iou", 0.70))
    tracker = args.tracker or str(cfg.get("tracker", "botsort.yaml"))
    save_video = args.save_video or bool(cfg.get("save_video", True))
    player_class_id = int(cfg.get("player_class_id", args.player_class_id))

    summary = run_tracking(
        video=video,
        weights=weights,
        output_dir=output,
        conf=conf,
        iou=iou,
        tracker=tracker,
        save_video=save_video,
        player_class_id=player_class_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
