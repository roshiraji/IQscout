#!/usr/bin/env python3
"""Presence / star-continuity eval harness (Phase 1 scaffold).

Consumes miss-report + track-lock JSON (and optional manual star masks later)
and scores the training-loop gates:

  - presence_recall: fraction of frames with >=1 kept player (clip-level floor)
  - star_detect_presence: fraction of frames where star body is detected
  - star_track_presence: fraction of frames where star has a track_id
  - star_id_stability: 1 - (switches / max(star_tracked_frames-1, 1))

Gate example (not enforced yet):
  star_detect_presence >= 0.98 AND star_track_presence >= 0.98
  AND star_id_stability >= 0.90

Usage:
  python scripts/eval_player_presence.py \\
    --miss outputs/presence_loop/clark_miss_report.json \\
    --track outputs/presence_loop/clark_track_lock.json \\
    --out outputs/presence_loop/clark_presence_score.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_from_reports(miss: dict | None, track: dict | None) -> dict:
    out: dict = {
        "gates": {
            "star_detect_presence_min": 0.98,
            "star_track_presence_min": 0.98,
            "star_id_stability_min": 0.90,
            "clip_presence_recall_min": 0.98,
        },
        "metrics": {},
        "passed": {},
        "sources": {},
    }

    if miss:
        a = miss.get("aggregates") or {}
        empty = float(a.get("pct_frames_empty_kept") or 0.0) / 100.0
        presence_recall = 1.0 - empty
        out["metrics"]["clip_presence_recall"] = round(presence_recall, 4)
        out["metrics"]["mean_kept_players"] = a.get("mean_kept")
        out["sources"]["miss_report"] = miss.get("video")
        out["passed"]["clip_presence_recall"] = (
            presence_recall >= out["gates"]["clip_presence_recall_min"]
        )

    if track:
        a = track.get("aggregates") or {}
        m = track.get("milestones") or {}
        det_p = float(a.get("pct_frames_star_detected") or 0.0) / 100.0
        trk_p = float(a.get("pct_frames_star_tracked") or 0.0) / 100.0
        switches = int(a.get("track_id_switches") or 0)
        tracked_frames = max(int(round(trk_p * int(track.get("frames") or 0))), 0)
        stability = 1.0 - (switches / max(tracked_frames - 1, 1))
        out["metrics"]["star_detect_presence"] = round(det_p, 4)
        out["metrics"]["star_track_presence"] = round(trk_p, 4)
        out["metrics"]["star_id_stability"] = round(max(0.0, min(1.0, stability)), 4)
        out["metrics"]["detect_to_track_lag_frames"] = m.get(
            "detect_to_track_lag_frames"
        )
        out["metrics"]["unique_track_ids_on_star"] = a.get("unique_track_ids_on_star")
        out["sources"]["track_lock"] = track.get("video")
        out["passed"]["star_detect_presence"] = (
            det_p >= out["gates"]["star_detect_presence_min"]
        )
        out["passed"]["star_track_presence"] = (
            trk_p >= out["gates"]["star_track_presence_min"]
        )
        out["passed"]["star_id_stability"] = (
            out["metrics"]["star_id_stability"] >= out["gates"]["star_id_stability_min"]
        )

    out["all_gates_passed"] = bool(out["passed"]) and all(out["passed"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miss", default="")
    ap.add_argument("--track", default="")
    ap.add_argument("--out", default="outputs/presence_loop/presence_score.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    miss = None
    track = None
    if args.miss:
        p = Path(args.miss)
        if not p.is_absolute():
            p = root / p
        miss = json.loads(p.read_text(encoding="utf-8"))
    if args.track:
        p = Path(args.track)
        if not p.is_absolute():
            p = root / p
        track = json.loads(p.read_text(encoding="utf-8"))

    score = score_from_reports(miss, track)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")

    print("=== Presence eval ===")
    print(json.dumps(score["metrics"], indent=2))
    print("passed:", score["passed"])
    print("all_gates_passed:", score["all_gates_passed"])
    print(f"wrote: {out}")
    return 0 if score.get("all_gates_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
