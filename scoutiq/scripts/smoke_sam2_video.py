"""Smoke check: SAM 2 video propagation on a finished tracking job.

Usage:
    python scripts/smoke_sam2_video.py <job_id_or_dir> [gid gid ...]

Prints per-player lock coverage plus the propagation stats so a regression in
the Confirm identity path is visible without running the whole API.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.sam2_video_tracker import propagate_selected_players  # noqa: E402
from src.pipeline.selected_player_reid import SelectedPlayerReId  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    target = Path(sys.argv[1])
    job_dir = target if target.is_dir() else ROOT / "outputs" / "jobs" / sys.argv[1]
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    snapshot = summary.get("snapshot") or {}
    players = list(snapshot.get("players") or [])

    wanted = {int(value) for value in sys.argv[2:]}
    if wanted:
        players = [p for p in players if int(p["person_group_id"]) in wanted]
    else:
        players = players[:2]
    if len(players) < 2:
        print("Need at least two snapshot players to test contact behaviour.")
        return 2

    frame_records = []
    with gzip.open(job_dir / "tracks.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            frame_records.append(json.loads(line))

    seeds = {
        int(p["person_group_id"]): [
            float(p["x1"]),
            float(p["y1"]),
            float(p["x2"]),
            float(p["y2"]),
        ]
        for p in players
    }
    labels = {
        int(p["person_group_id"]): str(p.get("label") or p["person_group_id"])
        for p in players
    }

    locked, stats = propagate_selected_players(
        video_path=summary["video_path"],
        frame_records=frame_records,
        snapshot_frame_number=int(summary.get("snapshot_frame_number") or 0),
        seeds=seeds,
        labels=labels,
        reid=SelectedPlayerReId(),
        progress_cb=lambda pct, detail: print(f"  {pct:5.0%} {detail}"),
    )

    print(json.dumps(stats, indent=2))
    print(f"frames with any lock: {len(locked)} / {len(frame_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
