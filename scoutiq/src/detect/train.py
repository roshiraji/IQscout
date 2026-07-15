"""Thin wrapper around Ultralytics YOLO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_train(
    *,
    data: Path,
    model: str = "yolo26s.pt",
    epochs: int = 80,
    imgsz: int = 960,
    project: str = "outputs/yolo",
    name: str = "player_detector_v1",
) -> None:
    detector = YOLO(model)
    detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        task="detect",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WNBA player detector with Ultralytics.")
    parser.add_argument("--config", required=True, help="Path to data YAML")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--project", default="outputs/yolo")
    parser.add_argument("--name", default="player_detector_v1")
    args = parser.parse_args()
    run_train(
        data=Path(args.config),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
