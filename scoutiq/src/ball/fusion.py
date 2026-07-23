"""Multi-signal offense/defense fusion over possession windows (WS2 + WS4).

Instead of one clip-wide vote, segment the clip into windows (scene cuts +
sustained possession flips) and score each window's offense team from several
signals, none of which alone decides:

  - ball possession  (nearest-player team when the ball is held)
  - attacking direction (which team's centroid is closer to the attacked hoop)
  - team spread (offense tends to be more spread than a collapsing defense)
  - possession continuity (offense rarely flips every frame → smoothing)

Every window gets its own offense/defense + confidence + timestamps. Low
confidence stays unresolved (never invents O/D); manual override still wins.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.ball.possession import assign_frame_possession


def _centroid(rows: list[dict], team_id: int) -> tuple[float, float] | None:
    pts = []
    for r in rows:
        try:
            if int(r.get("team_id")) != team_id:
                continue
        except (TypeError, ValueError):
            continue
        box = r.get("bbox_xyxy") or []
        if len(box) >= 4:
            pts.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _spread(rows: list[dict], team_id: int, diag: float) -> float | None:
    """Mean pairwise distance (norm) among a team's players — spacing proxy."""
    pts = []
    for r in rows:
        try:
            if int(r.get("team_id")) != team_id:
                continue
        except (TypeError, ValueError):
            continue
        box = r.get("bbox_xyxy") or []
        if len(box) >= 4:
            pts.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    if len(pts) < 2:
        return None
    total = 0.0
    n = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            total += math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            n += 1
    return (total / n) / max(diag, 1.0)


def _attacked_hoop_x(ball_xy, hoops: list[dict], persons: list[dict]):
    """Pick the hoop the play is oriented toward (ball-nearest, else uppermost)."""
    if not hoops:
        return None
    cxs = []
    for hp in hoops:
        b = hp.get("bbox_xyxy") or []
        if len(b) >= 4:
            cxs.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, float(hp.get("confidence") or 0)))
    if not cxs:
        return None
    if ball_xy is not None and len(cxs) > 1:
        # Attacked hoop = the one most players are advancing toward → proxy by
        # the hoop closest to the ball's horizontal position.
        return min(cxs, key=lambda c: abs(c[0] - ball_xy[0]))
    # Highest-confidence hoop otherwise.
    return max(cxs, key=lambda c: c[2])


def _frame_offense_signals(
    ball: dict | None,
    persons: list[dict],
    hoops: list[dict],
    *,
    diag: float,
    max_dist_norm: float,
    teams: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Per-frame evidence for each team being on offense."""
    poss = assign_frame_possession(ball, persons, max_dist_norm=max_dist_norm, frame_diagonal=diag)
    ball_xy = None
    if ball and len(ball.get("bbox_xyxy") or []) >= 4:
        b = ball["bbox_xyxy"]
        ball_xy = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    hoop = _attacked_hoop_x(ball_xy, hoops, persons)
    direction_team = None
    if hoop is not None:
        hx, hy = hoop[0], hoop[1]
        c0 = _centroid(persons, teams[0])
        c1 = _centroid(persons, teams[1])
        d0 = math.hypot(c0[0] - hx, c0[1] - hy) if c0 else None
        d1 = math.hypot(c1[0] - hx, c1[1] - hy) if c1 else None
        if d0 is not None and d1 is not None:
            direction_team = teams[0] if d0 < d1 else teams[1]

    spread_team = None
    s0 = _spread(persons, teams[0], diag)
    s1 = _spread(persons, teams[1], diag)
    if s0 is not None and s1 is not None and abs(s0 - s1) > 0.02:
        spread_team = teams[0] if s0 > s1 else teams[1]

    return {
        "held_team": poss.get("team_id") if poss.get("state") == "held" else None,
        "held_conf": float(poss.get("confidence") or 0.0) if poss.get("state") == "held" else 0.0,
        "state": poss.get("state"),
        "direction_team": direction_team,
        "spread_team": spread_team,
        "person_group_id": poss.get("person_group_id"),
        "jersey_color_name": poss.get("jersey_color_name"),
    }


def _smooth_votes(votes: list[int | None], window: int = 9) -> list[int | None]:
    """Rolling majority to remove single-frame possession flicker."""
    out: list[int | None] = []
    half = max(window // 2, 1)
    for i in range(len(votes)):
        lo = max(0, i - half)
        hi = min(len(votes), i + half + 1)
        c = Counter(v for v in votes[lo:hi] if v is not None)
        out.append(c.most_common(1)[0][0] if c else None)
    return out


def compute_possession_windows(
    frame_records: list[dict],
    *,
    cut_frames: list[int],
    fps: float,
    diag: float,
    weights: dict[str, float] | None = None,
    max_dist_norm: float = 0.12,
    min_window_frames: int = 8,
    min_confidence: float = 0.45,
    margin: float = 0.12,
    teams: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Segment into windows and fuse signals → per-window offense/defense.

    frame_records: [{"frame_number", "ball": dict|None, "persons": [...],
                     "hoops": [...]}], one per scored frame (in order).
    """
    w = weights or {
        "possession": 0.55,
        "direction": 0.25,
        "spread": 0.08,
        "continuity": 0.12,
    }
    cut_set = set(int(c) for c in (cut_frames or []))

    signals: list[dict] = []
    for rec in frame_records:
        signals.append(
            _frame_offense_signals(
                rec.get("ball"),
                rec.get("persons") or [],
                rec.get("hoops") or [],
                diag=diag,
                max_dist_norm=max_dist_norm,
                teams=teams,
            )
        )
    held_votes = [s["held_team"] for s in signals]
    smoothed = _smooth_votes(held_votes, window=max(int(round(fps / 3)) | 1, 5))

    # Boundaries: scene cuts OR sustained smoothed-vote flips.
    boundaries = [0]
    for i in range(1, len(frame_records)):
        fid = int(frame_records[i].get("frame_number", i))
        if fid in cut_set:
            boundaries.append(i)
        elif smoothed[i] is not None and smoothed[i - 1] is not None and smoothed[i] != smoothed[i - 1]:
            boundaries.append(i)
    boundaries.append(len(frame_records))
    # Dedual + sort
    boundaries = sorted(set(boundaries))

    windows: list[dict] = []
    for bi in range(len(boundaries) - 1):
        lo, hi = boundaries[bi], boundaries[bi + 1]
        if hi - lo < min_window_frames:
            # Merge tiny window into previous if possible.
            if windows:
                windows[-1]["_hi"] = hi
                continue
        windows.append({"_lo": lo, "_hi": hi})

    results: list[dict] = []
    for win in windows:
        lo, hi = win["_lo"], win["_hi"]
        seg = signals[lo:hi]
        n = max(hi - lo, 1)
        # Possession component
        held = [s for s in seg if s["held_team"] is not None]
        held_counts: Counter = Counter()
        held_conf: dict[int, float] = {}
        for s in held:
            t = s["held_team"]
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t == 2:
                continue
            held_counts[t] += 1
            held_conf[t] = held_conf.get(t, 0.0) + s["held_conf"]
        dir_counts = Counter(s["direction_team"] for s in seg if s["direction_team"] is not None)
        spread_counts = Counter(s["spread_team"] for s in seg if s["spread_team"] is not None)

        scores: dict[int, float] = {}
        for t in teams:
            poss_c = held_counts.get(t, 0) / n
            poss_q = (held_conf.get(t, 0.0) / held_counts[t]) if held_counts.get(t) else 0.0
            dir_c = dir_counts.get(t, 0) / n
            spread_c = spread_counts.get(t, 0) / n
            scores[t] = (
                w["possession"] * (0.5 * poss_c + 0.5 * poss_q)
                + w["direction"] * dir_c
                + w["spread"] * spread_c
            )
        # Continuity: reward the smoothed majority of this window.
        seg_smooth = [v for v in smoothed[lo:hi] if v is not None]
        if seg_smooth:
            maj = Counter(seg_smooth).most_common(1)[0][0]
            scores[maj] = scores.get(maj, 0.0) + w["continuity"]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        offense_team_id = None
        defense_team_id = None
        source = "unresolved"
        conf = 0.0
        if ranked:
            top_team, top_score = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            conf = round(float(top_score), 4)
            if top_score >= min_confidence and (top_score - second) >= margin and held_counts:
                offense_team_id = int(top_team)
                defense_team_id = teams[1] if offense_team_id == teams[0] else teams[0]
                source = "fused_signals"
            else:
                source = "unresolved_low_confidence"

        results.append(
            {
                "start_frame": int(frame_records[lo].get("frame_number", lo)),
                "end_frame": int(frame_records[hi - 1].get("frame_number", hi - 1)),
                "start_seconds": round(lo / fps, 2) if fps else None,
                "end_seconds": round((hi - 1) / fps, 2) if fps else None,
                "frames": hi - lo,
                "offense_team_id": offense_team_id,
                "defense_team_id": defense_team_id,
                "source": source,
                "confidence": conf,
                "held_frames": len(held),
                "team_held_counts": {str(k): int(v) for k, v in held_counts.items()},
                "component_scores": {str(k): round(v, 4) for k, v in scores.items()},
            }
        )

    # Clip-level rollup: dominant confident offense across windows (by frames).
    frames_by_team: Counter = Counter()
    for r in results:
        if r["offense_team_id"] is not None:
            frames_by_team[r["offense_team_id"]] += r["frames"]
    clip_offense = None
    clip_defense = None
    clip_conf = 0.0
    clip_source = "unresolved"
    total_conf_frames = sum(frames_by_team.values())
    if frames_by_team and total_conf_frames > 0:
        clip_offense = frames_by_team.most_common(1)[0][0]
        clip_defense = teams[1] if clip_offense == teams[0] else teams[0]
        clip_conf = round(frames_by_team[clip_offense] / total_conf_frames, 4)
        clip_source = "fused_windows"

    return {
        "windows": results,
        "offense_team_id": clip_offense,
        "defense_team_id": clip_defense,
        "source": clip_source,
        "confidence": clip_conf,
        "n_windows": len(results),
        "weights": w,
        "note": (
            "Per-window offense/defense fused from possession + direction + "
            "spread + continuity. Clip value = window frames majority."
        ),
    }
