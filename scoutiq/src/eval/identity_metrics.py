"""Lightweight identity metrics for resolved tracklets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_resolutions(resolved_json: Path, truth_json: Path) -> dict:
    resolved = json.loads(resolved_json.read_text(encoding="utf-8")).get("resolved_tracklets", [])
    truth = {row["tracklet_id"]: row["player_id"] for row in json.loads(truth_json.read_text(encoding="utf-8"))}
    total = len([r for r in resolved if r["tracklet_id"] in truth])
    correct = 0
    covered = 0
    for row in resolved:
        tid = row["tracklet_id"]
        if tid not in truth:
            continue
        if row.get("status") == "resolved":
            covered += 1
        if row.get("resolved_player_id") == truth[tid]:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    coverage = (covered / total) if total else 0.0
    return {"total": total, "correct": correct, "accuracy_top1": round(accuracy, 4), "coverage": round(coverage, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple identity metrics.")
    parser.add_argument("--resolved", required=True)
    parser.add_argument("--truth", required=True)
    args = parser.parse_args()
    print(json.dumps(score_resolutions(Path(args.resolved), Path(args.truth)), indent=2))


if __name__ == "__main__":
    main()
