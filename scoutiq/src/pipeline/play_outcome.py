"""Manual play-outcome tag storage (make / miss / foul_drawn / turnover / no_shot).

We don't auto-detect shot results yet, so a coach/user tags what actually
happened on the clip; Matchup Outcomes blends that tag with tracked process
metrics (separation, contest quality) to grade offense and defense
independently — see ``src.pipeline.coach_insights.OUTCOME_OFFENSE_POINTS``
and ``OUTCOME_DEFENSE_ADJUST``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PLAY_OUTCOME_TAGS = ("make", "miss", "foul_drawn", "turnover", "no_shot")
PLAY_OUTCOME_FILENAME = "play_outcome.json"


def load_play_outcome(artifact_dir: str | Path) -> dict[str, Any] | None:
    """Read the current play-outcome tag for a job, or None if unset/invalid."""
    path = Path(artifact_dir) / PLAY_OUTCOME_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("tag") not in PLAY_OUTCOME_TAGS:
        return None
    return payload


def write_play_outcome(
    artifact_dir: str | Path,
    tag: str | None,
    *,
    note: str | None = None,
    set_by: str = "user",
) -> dict[str, Any] | None:
    """Persist a play-outcome tag, or clear it when ``tag`` is None.

    Returns the stored payload, or None when clearing.
    """
    path = Path(artifact_dir) / PLAY_OUTCOME_FILENAME
    if tag is None:
        path.unlink(missing_ok=True)
        return None
    if tag not in PLAY_OUTCOME_TAGS:
        raise ValueError(
            f"Unknown play outcome tag {tag!r}; expected one of {PLAY_OUTCOME_TAGS}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tag": tag, "note": note, "set_by": set_by}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
