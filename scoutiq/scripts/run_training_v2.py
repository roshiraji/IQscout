"""End-to-end v2 training orchestrator: autolabel -> dataset -> train -> compare.

Runs every stage of the multi-game training loop and writes a final report.
Each stage is idempotent/resumable:
  - autolabel skips frames that already have labels (skip_existing)
  - build-dataset rebuilds deterministically from autolabel output (seed fixed)
  - training resumes from the last checkpoint if interrupted (Ultralytics)

Usage (from scoutiq/):
  ..\\.venv\\Scripts\\python.exe scripts\\run_training_v2.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCOUTIQ_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
INGEST_CFG = "configs/showcase_ingest_v2.yaml"
TRAIN_CFG = "configs/showcase_train_v2.yaml"
RUNS_ROOT = SCOUTIQ_ROOT / "outputs" / "training_loop_v2"
DATASET_YAML = SCOUTIQ_ROOT / "data" / "datasets" / "nba_player_seg_v2" / "data.yaml"
PRODUCTION_WEIGHTS = SCOUTIQ_ROOT / "outputs" / "training_loop" / "best_player_seg.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_v2")


def run_stage(name: str, args: list[str]) -> None:
    logger.info("=== STAGE %s ===", name)
    t0 = time.time()
    result = subprocess.run([PYTHON, *args], cwd=str(SCOUTIQ_ROOT))
    dt = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}")
    logger.info("=== STAGE %s done in %.1f min ===", name, dt / 60)


def validate_weights(weights: Path, tag: str) -> dict:
    """Validate a weights file on the v2 val split (single GPU)."""
    if not weights.is_file():
        logger.warning("Weights missing for %s: %s", tag, weights)
        return {}
    sys.path.insert(0, str(SCOUTIQ_ROOT))
    from src.showcase.train_loop import validate_player_seg

    logger.info("Validating %s (%s) on v2 val split", tag, weights.name)
    return validate_player_seg(weights, DATASET_YAML, imgsz=960, device=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify all stages")
    parser.add_argument("--skip-autolabel", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not args.skip_autolabel:
        al_args = ["-m", "src.showcase", "autolabel", "--config", INGEST_CFG]
        if args.smoke:
            al_args += ["--max-frames", "12"]
        run_stage("autolabel", al_args)

    run_stage("build-dataset", ["-m", "src.showcase", "build-dataset", "--config", INGEST_CFG])

    if not args.skip_train and not args.smoke:
        run_stage("train", ["-m", "src.showcase", "train", "--config", TRAIN_CFG])

    # Promote v2 weights under an explicit name; never touch production weights.
    v2_generic = RUNS_ROOT / "best_player_seg.pt"
    v2_named = RUNS_ROOT / "best_player_seg_v2.pt"
    if v2_generic.is_file():
        shutil.copy2(v2_generic, v2_named)
        # last.pt from the most recent round, for resume/debug.
        lasts = sorted(RUNS_ROOT.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if lasts:
            shutil.copy2(lasts[-1], RUNS_ROOT / "last_player_seg_v2.pt")
        logger.info("Promoted v2 weights -> %s", v2_named)

    comparison: dict = {"elapsed_minutes": round((time.time() - started) / 60, 1)}
    if v2_named.is_file():
        comparison["v2"] = validate_weights(v2_named, "v2")
        comparison["production_v1"] = validate_weights(PRODUCTION_WEIGHTS, "production_v1")
        comparison["v2_weights"] = str(v2_named)
        comparison["production_weights"] = str(PRODUCTION_WEIGHTS)

    report_path = RUNS_ROOT / "v2_comparison_report.json"
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("Report -> %s", report_path)
    logger.info("ALL STAGES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
