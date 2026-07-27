"""Smoke check: run the Confirm render for two snapshot players on a job.

Usage:
    python scripts/smoke_confirm_render.py <job_id_or_dir> [gid gid ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.track_players import render_selected_tracks  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    target = Path(sys.argv[1])
    job_dir = target if target.is_dir() else ROOT / "outputs" / "jobs" / sys.argv[1]
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    players = list((summary.get("snapshot") or {}).get("players") or [])

    wanted = {int(value) for value in sys.argv[2:]}
    if wanted:
        players = [p for p in players if int(p["person_group_id"]) in wanted]
    else:
        players = players[:2]

    result = render_selected_tracks(
        job_dir,
        selected_person_group_ids=[int(p["person_group_id"]) for p in players],
        selected_players=players,
        quiet=True,
        progress_cb=lambda stage, pct, detail: print(f"  {pct:5.0%} [{stage}] {detail}"),
    )
    for key in (
        "selection_lock",
        "selection_sam2",
        "selection_sam2_error",
        "frames_with_selected",
        "frame_coverage",
        "selected_player_retention",
    ):
        print(f"{key}: {json.dumps(result.get(key))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
