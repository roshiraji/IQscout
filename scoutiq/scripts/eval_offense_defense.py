"""Offense/Defense evaluation harness (WS5).

Compares predicted O/D (from possession.json windows, or recomputed) against a
small hand-labeled ground-truth file, and reports possession + O/D accuracy.

Ground-truth JSON format (see data/eval/od_ground_truth.template.json):
{
  "clips": [
    {
      "job_id": "outputs/jobs/<id>",          # dir with possession.json
      "windows": [
        {"start_frame": 30, "end_frame": 120, "offense_team_id": 0},
        {"start_frame": 121, "end_frame": 210, "offense_team_id": 1}
      ]
    }
  ]
}

Usage (from scoutiq/):
  ..\\.venv\\Scripts\\python.exe scripts\\eval_offense_defense.py \
      --ground-truth data\\eval\\od_ground_truth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCOUTIQ = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (SCOUTIQ / path)


def _load_pred_windows(job_dir: Path) -> list[dict]:
    poss = job_dir / "possession.json"
    if not poss.is_file():
        return []
    try:
        data = json.loads(poss.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("windows") or []


def _pred_offense_at(frame: int, windows: list[dict]) -> int | None:
    for w in windows:
        if w.get("start_frame") is None or w.get("end_frame") is None:
            continue
        if int(w["start_frame"]) <= frame <= int(w["end_frame"]):
            return w.get("offense_team_id")
    return None


def evaluate(gt_path: Path) -> dict:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    clips_report = []
    total_windows = 0
    correct_windows = 0
    resolved_windows = 0
    for clip in gt.get("clips") or []:
        job_dir = _resolve(clip["job_id"])
        pred_windows = _load_pred_windows(job_dir)
        clip_correct = 0
        clip_total = 0
        clip_resolved = 0
        details = []
        for gw in clip.get("windows") or []:
            mid = (int(gw["start_frame"]) + int(gw["end_frame"])) // 2
            pred = _pred_offense_at(mid, pred_windows)
            truth = gw.get("offense_team_id")
            clip_total += 1
            is_resolved = pred is not None
            is_correct = is_resolved and int(pred) == int(truth)
            if is_resolved:
                clip_resolved += 1
            if is_correct:
                clip_correct += 1
            details.append(
                {
                    "start_frame": gw["start_frame"],
                    "end_frame": gw["end_frame"],
                    "truth": truth,
                    "pred": pred,
                    "correct": is_correct,
                    "resolved": is_resolved,
                }
            )
        clips_report.append(
            {
                "job_id": clip["job_id"],
                "windows": clip_total,
                "resolved": clip_resolved,
                "correct": clip_correct,
                "accuracy_resolved": round(clip_correct / clip_resolved, 4) if clip_resolved else None,
                "coverage": round(clip_resolved / clip_total, 4) if clip_total else None,
                "details": details,
            }
        )
        total_windows += clip_total
        correct_windows += clip_correct
        resolved_windows += clip_resolved

    report = {
        "total_windows": total_windows,
        "resolved_windows": resolved_windows,
        "correct_windows": correct_windows,
        "od_accuracy_on_resolved": round(correct_windows / resolved_windows, 4)
        if resolved_windows
        else None,
        "resolution_coverage": round(resolved_windows / total_windows, 4)
        if total_windows
        else None,
        "clips": clips_report,
        "note": (
            "od_accuracy_on_resolved = correct / windows where model committed. "
            "resolution_coverage = how often model committed (rest → override)."
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", default="outputs/training_loop_ball/od_eval_report.json")
    args = parser.parse_args()

    gt_path = _resolve(args.ground_truth)
    if not gt_path.is_file():
        print(f"Ground-truth not found: {gt_path}")
        print("Copy data/eval/od_ground_truth.template.json and fill it in.")
        return 1
    report = evaluate(gt_path)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))
    print(f"Full report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
