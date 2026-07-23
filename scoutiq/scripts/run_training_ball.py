"""Orchestrate ball autolabel → dataset → train → HSV comparison notes.

Usage (from scoutiq/):
  ..\\.venv\\Scripts\\python.exe scripts\\run_training_ball.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

SCOUTIQ_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUNS_ROOT = SCOUTIQ_ROOT / "outputs" / "training_loop_ball"
DATASET_YAML = SCOUTIQ_ROOT / "data" / "datasets" / "wnba_ball_det" / "data.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_ball")


def run_stage(name: str, args: list[str]) -> None:
    logger.info("=== STAGE %s ===", name)
    t0 = time.time()
    result = subprocess.run([PYTHON, *args], cwd=str(SCOUTIQ_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}")
    logger.info("=== STAGE %s done in %.1f min ===", name, dt / 60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-autolabel", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sam-refine", action="store_true", help="Tighten ball boxes with SAM")
    parser.add_argument("--force", action="store_true", help="Re-label even if labels exist")
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not args.skip_autolabel:
        al = ["-m", "src.ball", "autolabel", "--manifest", "data/frames/manifest.csv"]
        if args.smoke:
            al += ["--max-frames", str(args.max_frames or 24)]
        elif args.max_frames:
            al += ["--max-frames", str(args.max_frames)]
        if args.sam_refine:
            al.append("--sam-refine")
        if args.force:
            al.append("--force")
        run_stage("autolabel", al)
        run_stage(
            "build-dataset",
            [
                "-m",
                "src.ball",
                "build-dataset",
                "--autolabel",
                "data/auto_labels_ball",
                "--dataset",
                "data/datasets/wnba_ball_det",
            ],
        )

    if not args.skip_train:
        tr = ["-m", "src.ball", "train", "--config", "configs/showcase_ball_train.yaml"]
        if args.smoke:
            tr.append("--smoke")
        run_stage("train", tr)

    report = {
        "elapsed_minutes": round((time.time() - started) / 60, 1),
        "dataset_yaml": str(DATASET_YAML) if DATASET_YAML.is_file() else None,
        "best_ball_det": str(RUNS_ROOT / "best_ball_det.pt")
        if (RUNS_ROOT / "best_ball_det.pt").is_file()
        else None,
        "note": (
            "Player weights untouched (best_player_seg_v2.pt). "
            "Enable live path via ball.weights in player_tracking_mvp.yaml."
        ),
    }
    if (RUNS_ROOT / "best_ball_det.pt").is_file() and DATASET_YAML.is_file():
        sys.path.insert(0, str(SCOUTIQ_ROOT))
        from src.ball.train import validate_ball_det

        report["val_metrics"] = validate_ball_det(
            RUNS_ROOT / "best_ball_det.pt", DATASET_YAML, imgsz=960, device=0
        )

    (RUNS_ROOT / "ball_orchestrator_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Ball training orchestration complete → %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
