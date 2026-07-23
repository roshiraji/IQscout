"""Orchestrate hoop+referee seg: autolabel → dataset → train.

Usage (from scoutiq/):
  ..\\.venv\\Scripts\\python.exe -u scripts\\run_hoop_ref_seg_training.py
  ..\\.venv\\Scripts\\python.exe -u scripts\\run_hoop_ref_seg_training.py --smoke
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
RUNS_ROOT = SCOUTIQ_ROOT / "outputs" / "training_loop_hoop_ref"
DATASET_YAML = SCOUTIQ_ROOT / "data" / "datasets" / "wnba_hoop_ref_seg" / "data.yaml"
CONFIG = "configs/showcase_hoop_ref_seg_train.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_hoop_ref")


def run_stage(name: str, args: list[str]) -> None:
    logger.info("=== STAGE %s ===", name)
    t0 = time.time()
    # Unbuffered child so epoch/batch updates stream live.
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONUNBUFFERED": "1"}
    result = subprocess.run([PYTHON, "-u", *args], cwd=str(SCOUTIQ_ROOT), env=env)
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
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not args.skip_autolabel:
        al = [
            "-m",
            "src.ball",
            "hoop-ref-autolabel",
            "--manifest",
            "data/frames/manifest.csv",
            "--output",
            "data/auto_labels_hoop_ref",
            "--stride",
            str(args.stride),
        ]
        if args.smoke:
            al += ["--max-frames", str(args.max_frames or 24)]
        elif args.max_frames:
            al += ["--max-frames", str(args.max_frames)]
        if args.force:
            al.append("--force")
        run_stage("autolabel", al)
        run_stage(
            "build-dataset",
            [
                "-m",
                "src.ball",
                "hoop-ref-build-dataset",
                "--autolabel",
                "data/auto_labels_hoop_ref",
                "--dataset",
                "data/datasets/wnba_hoop_ref_seg",
            ],
        )

    if not args.skip_train:
        tr = ["-m", "src.ball", "hoop-ref-train", "--config", CONFIG]
        if args.smoke:
            tr.append("--smoke")
        run_stage("train", tr)

    report: dict = {
        "elapsed_minutes": round((time.time() - started) / 60, 1),
        "dataset_yaml": str(DATASET_YAML) if DATASET_YAML.is_file() else None,
        "best_hoop_ref_seg": str(RUNS_ROOT / "best_hoop_ref_seg.pt")
        if (RUNS_ROOT / "best_hoop_ref_seg.pt").is_file()
        else None,
        "batch": 12,
        "note": (
            "Player weights untouched. Live conf target 0.80–0.90 after "
            "val mAP gates pass. Enable via player_tracking_mvp.yaml when ready."
        ),
    }
    if (RUNS_ROOT / "best_hoop_ref_seg.pt").is_file() and DATASET_YAML.is_file():
        sys.path.insert(0, str(SCOUTIQ_ROOT))
        from src.showcase.train_loop import validate_player_seg

        report["val_metrics"] = validate_player_seg(
            RUNS_ROOT / "best_hoop_ref_seg.pt", DATASET_YAML, imgsz=960, device=0
        )

    (RUNS_ROOT / "hoop_ref_orchestrator_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Hoop+ref seg orchestration complete → %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
