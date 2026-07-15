"""Coach insights from selected-player track rows (pixel / normalized space).

Pure functions over per-frame centers — no ball, OCR, or court calibration.
Proxies are labeled as such; never invent player names.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any

# Normalized distance thresholds (fraction of frame diagonal).
CLOSE_GAP = 0.08
NEAR_GAP = 0.15
PRESSURE_GAP = 0.20

# Closing speed (diagonal fractions per second) for a "closeout burst".
CLOSEOUT_SPEED = 0.35
CLOSEOUT_MIN_GAP = 0.18

# High-effort: instantaneous speed above this (diag fractions / sec).
HIGH_EFFORT_SPEED = 0.25

# Chase: frames with gap under PRESSURE_GAP and clear leader motion.
CHASE_GAP = 0.18


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    w = idx - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _label_for(gid: int, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    base = meta.get("player_label") or f"Person #{gid}"
    color = meta.get("jersey_color_name")
    if color and str(color).lower() not in str(base).lower():
        return f"{base} ({str(color).title()})"
    return str(base)


def _frame_diagonal(width: int, height: int) -> float:
    return float(math.hypot(max(width, 1), max(height, 1)))


def build_series_by_group(
    rows: list[dict],
) -> dict[int, list[dict]]:
    """Group selected track rows by person_group_id, sorted by frame."""
    series: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        gid = row.get("person_group_id")
        if gid is None:
            continue
        series[int(gid)].append(row)
    for gid in series:
        series[gid].sort(
            key=lambda r: (
                int(r.get("frame_number") or 0),
                float(r.get("timestamp_seconds") or 0.0),
            )
        )
    return dict(series)


def compute_single_player_metrics(
    series: list[dict],
    *,
    source_frame_count: int,
    frame_diagonal: float,
    person_group_id: int,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Movement intensity + coverage trust for one locked identity."""
    meta = meta or {}
    diag = max(frame_diagonal, 1.0)
    frames = {int(r.get("frame_number") or 0) for r in series}
    visible = len(frames)
    src = max(int(source_frame_count), 1)
    coverage = visible / src

    speeds: list[float] = []
    path = 0.0
    high_effort_dt = 0.0
    total_dt = 0.0
    longest_run = 0
    run = 0
    prev_f: int | None = None
    prev: dict | None = None

    for row in series:
        fid = int(row.get("frame_number") or 0)
        if prev_f is None or fid == prev_f + 1:
            run += 1
        else:
            longest_run = max(longest_run, run)
            run = 1
        longest_run = max(longest_run, run)
        prev_f = fid

        if prev is not None:
            dx = _safe_float(row.get("center_x")) - _safe_float(prev.get("center_x"))
            dy = _safe_float(row.get("center_y")) - _safe_float(prev.get("center_y"))
            dist = math.hypot(dx, dy) / diag
            path += dist
            dt = max(
                _safe_float(row.get("timestamp_seconds"))
                - _safe_float(prev.get("timestamp_seconds")),
                1e-3,
            )
            speed = dist / dt
            speeds.append(speed)
            total_dt += dt
            if speed >= HIGH_EFFORT_SPEED:
                high_effort_dt += dt
        prev = row

    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    # Approximate seconds for longest run from median frame dt.
    dts: list[float] = []
    for i in range(1, len(series)):
        dt = _safe_float(series[i].get("timestamp_seconds")) - _safe_float(
            series[i - 1].get("timestamp_seconds")
        )
        if 0 < dt < 1.0:
            dts.append(dt)
    median_dt = _percentile(dts, 50) or (1.0 / 30.0)

    return {
        "person_group_id": person_group_id,
        "label": _label_for(person_group_id, meta),
        "team_id": meta.get("team_id"),
        "team_label": meta.get("team_label"),
        "jersey_color_name": meta.get("jersey_color_name"),
        "coverage_trust": {
            "frames_visible": visible,
            "source_frame_count": src,
            "coverage": _round(coverage),
            "longest_continuous_frames": longest_run,
            "longest_continuous_seconds": _round(longest_run * float(median_dt)),
        },
        "movement_intensity": {
            "path_length_norm": _round(path),
            "mean_speed_norm": _round(mean_speed),
            "p90_speed_norm": _round(_percentile(speeds, 90)),
            "high_effort_fraction": _round(
                high_effort_dt / total_dt if total_dt > 0 else 0.0
            ),
            "note": "Speeds are frame-diagonal fractions per second (clip proxies).",
        },
    }


def _align_pair(
    series_a: list[dict],
    series_b: list[dict],
) -> list[tuple[dict, dict]]:
    by_b = {int(r.get("frame_number") or 0): r for r in series_b}
    aligned: list[tuple[dict, dict]] = []
    for row_a in series_a:
        fid = int(row_a.get("frame_number") or 0)
        row_b = by_b.get(fid)
        if row_b is not None:
            aligned.append((row_a, row_b))
    return aligned


def _distances(aligned: list[tuple[dict, dict]], diag: float) -> list[float]:
    dists: list[float] = []
    for a, b in aligned:
        dx = _safe_float(a.get("center_x")) - _safe_float(b.get("center_x"))
        dy = _safe_float(a.get("center_y")) - _safe_float(b.get("center_y"))
        dists.append(math.hypot(dx, dy) / max(diag, 1.0))
    return dists


def compute_pairwise_metrics(
    series_a: list[dict],
    series_b: list[dict],
    *,
    gid_a: int,
    gid_b: int,
    meta_a: dict[str, Any] | None,
    meta_b: dict[str, Any] | None,
    source_frame_count: int,
    frame_diagonal: float,
) -> dict[str, Any]:
    """Pairwise separation, closeouts, chase / sticky / screen / cut alignment."""
    meta_a = meta_a or {}
    meta_b = meta_b or {}
    diag = max(frame_diagonal, 1.0)
    src = max(int(source_frame_count), 1)
    aligned = _align_pair(series_a, series_b)
    both_visible = len(aligned)
    dists = _distances(aligned, diag)

    mean_gap = sum(dists) / len(dists) if dists else None
    pct_close = (
        sum(1 for d in dists if d <= CLOSE_GAP) / len(dists) if dists else 0.0
    )
    pct_near = (
        sum(1 for d in dists if d <= NEAR_GAP) / len(dists) if dists else 0.0
    )

    # Closeout bursts: successive frames where gap shrinks fast while still near.
    closeouts = 0
    max_close_rate = 0.0
    for i in range(1, len(aligned)):
        a0, b0 = aligned[i - 1]
        a1, b1 = aligned[i]
        d0 = dists[i - 1]
        d1 = dists[i]
        dt = max(
            _safe_float(a1.get("timestamp_seconds"))
            - _safe_float(a0.get("timestamp_seconds")),
            1e-3,
        )
        closing = (d0 - d1) / dt
        if closing > max_close_rate:
            max_close_rate = closing
        if (
            closing >= CLOSEOUT_SPEED
            and d0 <= CLOSEOUT_MIN_GAP
            and d1 < d0
        ):
            closeouts += 1

    team_a = meta_a.get("team_id")
    team_b = meta_b.get("team_id")
    same_team = (
        team_a is not None
        and team_b is not None
        and int(team_a) == int(team_b)
        and int(team_a) != 2  # officials
    )
    opp_team = (
        team_a is not None
        and team_b is not None
        and int(team_a) != int(team_b)
        and int(team_a) != 2
        and int(team_b) != 2
    )

    chase: dict[str, Any] = {
        "applicable": bool(opp_team),
        "leader_person_group_id": None,
        "chaser_person_group_id": None,
        "chase_frame_fraction": None,
        "mean_gap_when_close": None,
    }
    sticky: dict[str, Any] = {
        "applicable": bool(opp_team),
        "note": "With only two selected players, each is the other's only selected opponent when both visible.",
        "sticky_fraction": _round(both_visible / src) if opp_team else None,
    }

    if opp_team and len(aligned) >= 3:
        a_leads = 0
        b_leads = 0
        close_gaps: list[float] = []
        for i in range(1, len(aligned)):
            if dists[i] > CHASE_GAP:
                continue
            a0, b0 = aligned[i - 1]
            a1, b1 = aligned[i]
            va = (
                _safe_float(a1.get("center_x")) - _safe_float(a0.get("center_x")),
                _safe_float(a1.get("center_y")) - _safe_float(a0.get("center_y")),
            )
            vb = (
                _safe_float(b1.get("center_x")) - _safe_float(b0.get("center_x")),
                _safe_float(b1.get("center_y")) - _safe_float(b0.get("center_y")),
            )
            # Who is ahead along A's motion (relative displacement of B from A).
            rel = (
                _safe_float(b1.get("center_x")) - _safe_float(a1.get("center_x")),
                _safe_float(b1.get("center_y")) - _safe_float(a1.get("center_y")),
            )
            speed_a = math.hypot(*va)
            speed_b = math.hypot(*vb)
            close_gaps.append(dists[i])
            if speed_a < 1e-3 and speed_b < 1e-3:
                continue
            # Leader is the faster mover when gap is small (pressure proxy).
            if speed_a >= speed_b * 1.1:
                a_leads += 1
            elif speed_b >= speed_a * 1.1:
                b_leads += 1
        total_lead = a_leads + b_leads
        if total_lead > 0:
            if a_leads >= b_leads:
                chase["leader_person_group_id"] = gid_a
                chase["chaser_person_group_id"] = gid_b
                chase["chase_frame_fraction"] = _round(a_leads / total_lead)
            else:
                chase["leader_person_group_id"] = gid_b
                chase["chaser_person_group_id"] = gid_a
                chase["chase_frame_fraction"] = _round(b_leads / total_lead)
        if close_gaps:
            chase["mean_gap_when_close"] = _round(
                sum(close_gaps) / len(close_gaps)
            )

    screen: dict[str, Any] = {
        "applicable": bool(same_team),
        "score": None,
        "soft_label": None,
        "min_gap": _round(min(dists)) if dists else None,
        "note": "Soft proxy only — not a labeled play call.",
    }
    if same_team and len(dists) >= 5:
        min_i = min(range(len(dists)), key=lambda i: dists[i])
        left = dists[: min_i + 1]
        right = dists[min_i:]
        approached = len(left) >= 2 and left[0] - left[-1] > 0.03
        split = len(right) >= 2 and right[-1] - right[0] > 0.03
        merged = dists[min_i] <= NEAR_GAP
        score = (
            (0.4 if approached else 0.0)
            + (0.3 if merged else 0.0)
            + (0.3 if split else 0.0)
        )
        screen["score"] = _round(score)
        if score >= 0.7:
            screen["soft_label"] = "approach_merge_split"
        elif score >= 0.4:
            screen["soft_label"] = "partial_screen_shape"
        else:
            screen["soft_label"] = "no_clear_screen_shape"

    # Cut timing: high-effort window overlap.
    def _effort_mask(series: list[dict]) -> set[int]:
        out: set[int] = set()
        prev = None
        for row in series:
            if prev is not None:
                dx = _safe_float(row.get("center_x")) - _safe_float(
                    prev.get("center_x")
                )
                dy = _safe_float(row.get("center_y")) - _safe_float(
                    prev.get("center_y")
                )
                dt = max(
                    _safe_float(row.get("timestamp_seconds"))
                    - _safe_float(prev.get("timestamp_seconds")),
                    1e-3,
                )
                speed = math.hypot(dx, dy) / diag / dt
                if speed >= HIGH_EFFORT_SPEED:
                    out.add(int(row.get("frame_number") or 0))
            prev = row
        return out

    effort_a = _effort_mask(series_a)
    effort_b = _effort_mask(series_b)
    union = effort_a | effort_b
    inter = effort_a & effort_b
    cut_align = len(inter) / len(union) if union else 0.0

    return {
        "person_group_ids": [gid_a, gid_b],
        "labels": [_label_for(gid_a, meta_a), _label_for(gid_b, meta_b)],
        "relationship": (
            "same_team" if same_team else "opponent" if opp_team else "unknown"
        ),
        "coverage_trust": {
            "frames_both_visible": both_visible,
            "source_frame_count": src,
            "both_visible_fraction": _round(both_visible / src),
        },
        "pairwise_separation": {
            "mean_gap_norm": _round(mean_gap),
            "p50_gap_norm": _round(_percentile(dists, 50)),
            "pct_frames_within_close": _round(pct_close),
            "pct_frames_within_near": _round(pct_near),
            "close_threshold": CLOSE_GAP,
            "near_threshold": NEAR_GAP,
        },
        "closeout_bursts": {
            "count": closeouts,
            "max_close_rate_norm": _round(max_close_rate),
            "threshold_speed": CLOSEOUT_SPEED,
        },
        "chase_role": chase,
        "sticky_matchup": sticky,
        "screen_shape_proxy": screen,
        "cut_timing_alignment": {
            "high_effort_overlap": _round(cut_align),
            "frames_a_high_effort": len(effort_a),
            "frames_b_high_effort": len(effort_b),
            "frames_both_high_effort": len(inter),
        },
    }


def build_coach_narrative(insights: dict[str, Any]) -> str:
    """Plain-language coach blurb; proxies labeled; Person # / color only."""
    players = insights.get("players") or []
    pairs = insights.get("pairs") or []
    if len(players) < 2 or not pairs:
        return (
            "Select at least two players to generate pairwise coach insights "
            "from this clip."
        )

    primary = pairs[0]
    labels = primary.get("labels") or ["Player A", "Player B"]
    a_lab, b_lab = labels[0], labels[1]
    sep = primary.get("pairwise_separation") or {}
    cov = primary.get("coverage_trust") or {}
    closes = primary.get("closeout_bursts") or {}
    rel = primary.get("relationship") or "unknown"

    both_pct = round(
        float(cov.get("both_visible_fraction") or 0.0) * 100
    )
    near_pct = round(float(sep.get("pct_frames_within_near") or 0.0) * 100)
    mean_gap = sep.get("mean_gap_norm")
    mean_gap_txt = (
        f"{mean_gap:.3f} of frame diagonal" if mean_gap is not None else "n/a"
    )
    close_n = int(closes.get("count") or 0)

    parts = [
        f"{a_lab} and {b_lab} were both visible in {both_pct}% of clip frames "
        f"(coverage trust). Mean gap was {mean_gap_txt}; they were within the "
        f"near band ({int(NEAR_GAP * 100)}% diagonal) on {near_pct}% of "
        f"shared frames, with {close_n} closeout-speed burst(s) (pixel proxy)."
    ]

    if rel == "opponent":
        chase = primary.get("chase_role") or {}
        leader = chase.get("leader_person_group_id")
        chaser = chase.get("chaser_person_group_id")
        if leader is not None and chaser is not None:
            lead_lab = next(
                (
                    p.get("label")
                    for p in players
                    if p.get("person_group_id") == leader
                ),
                f"Person #{leader}",
            )
            chase_lab = next(
                (
                    p.get("label")
                    for p in players
                    if p.get("person_group_id") == chaser
                ),
                f"Person #{chaser}",
            )
            frac = chase.get("chase_frame_fraction")
            frac_txt = f"{round(float(frac) * 100)}%" if frac is not None else "—"
            parts.append(
                f"Opposite teams: when close, {lead_lab} looked like the "
                f"leader and {chase_lab} the chaser on {frac_txt} of contested "
                f"frames (pressure proxy — not on-ball proof)."
            )
        sticky = primary.get("sticky_matchup") or {}
        if sticky.get("sticky_fraction") is not None:
            parts.append(
                f"Sticky matchup share of the clip: "
                f"{round(float(sticky['sticky_fraction']) * 100)}% "
                f"(only selected opponent)."
            )
    elif rel == "same_team":
        screen = primary.get("screen_shape_proxy") or {}
        label = screen.get("soft_label") or "no_clear_screen_shape"
        parts.append(
            f"Same team: off-ball shape score suggests {label.replace('_', ' ')} "
            f"(soft proxy, not a called play)."
        )
        cut = primary.get("cut_timing_alignment") or {}
        ov = cut.get("high_effort_overlap")
        if ov is not None:
            parts.append(
                f"High-effort window overlap (cut-timing proxy): "
                f"{round(float(ov) * 100)}%."
            )

    # Intensity one-liners
    inten_bits = []
    for p in players[:2]:
        mi = (p.get("movement_intensity") or {})
        he = mi.get("high_effort_fraction")
        path = mi.get("path_length_norm")
        if he is not None and path is not None:
            inten_bits.append(
                f"{p.get('label')} high-effort {round(float(he) * 100)}%, "
                f"path {path:.2f} diag"
            )
    if inten_bits:
        parts.append("Intensity: " + "; ".join(inten_bits) + ".")

    parts.append(
        "All distances/speeds are normalized to the frame — not court feet."
    )
    return " ".join(parts)


def compute_coach_insights(
    rows: list[dict],
    *,
    source_frame_count: int,
    frame_width: int,
    frame_height: int,
    label_by_group: dict[int, str] | None = None,
    team_meta_by_group: dict[int, dict] | None = None,
    primary_pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build full coach_insights payload from selected track rows."""
    label_by_group = label_by_group or {}
    team_meta_by_group = team_meta_by_group or {}
    series_map = build_series_by_group(rows)
    gids = sorted(series_map)
    diag = _frame_diagonal(frame_width, frame_height)

    players: list[dict] = []
    for gid in gids:
        meta = dict(team_meta_by_group.get(gid) or {})
        if gid in label_by_group:
            meta["player_label"] = label_by_group[gid]
        players.append(
            compute_single_player_metrics(
                series_map[gid],
                source_frame_count=source_frame_count,
                frame_diagonal=diag,
                person_group_id=gid,
                meta=meta,
            )
        )

    pairs: list[dict] = []
    for ga, gb in combinations(gids, 2):
        ma = dict(team_meta_by_group.get(ga) or {})
        mb = dict(team_meta_by_group.get(gb) or {})
        if ga in label_by_group:
            ma["player_label"] = label_by_group[ga]
        if gb in label_by_group:
            mb["player_label"] = label_by_group[gb]
        pairs.append(
            compute_pairwise_metrics(
                series_map[ga],
                series_map[gb],
                gid_a=ga,
                gid_b=gb,
                meta_a=ma,
                meta_b=mb,
                source_frame_count=source_frame_count,
                frame_diagonal=diag,
            )
        )

    # Prefer primary pair first in list for UI.
    if primary_pair and pairs:
        pa, pb = sorted(primary_pair)
        pairs.sort(
            key=lambda p: 0
            if sorted(p.get("person_group_ids") or []) == [pa, pb]
            else 1
        )
    elif pairs:
        # Default: first two selected IDs
        prefer = sorted(gids)[:2]
        pairs.sort(
            key=lambda p: 0
            if sorted(p.get("person_group_ids") or []) == prefer
            else 1
        )

    payload = {
        "version": 1,
        "units": {
            "distance": "fraction_of_frame_diagonal",
            "speed": "diagonal_fractions_per_second",
            "note": "Clip proxies — not calibrated court feet.",
        },
        "non_goals": [
            "ball_tracking",
            "shot_make_miss",
            "named_identity",
            "calibrated_feet_per_second",
            "help_rotation_without_ball",
        ],
        "frame_width": frame_width,
        "frame_height": frame_height,
        "source_frame_count": source_frame_count,
        "selected_person_group_ids": gids,
        "players": players,
        "pairs": pairs,
        "glance": {},
    }

    if pairs:
        primary = pairs[0]
        sep = primary.get("pairwise_separation") or {}
        cov = primary.get("coverage_trust") or {}
        closes = primary.get("closeout_bursts") or {}
        glance_players = []
        for p in players:
            if p["person_group_id"] in (primary.get("person_group_ids") or []):
                glance_players.append(
                    {
                        "person_group_id": p["person_group_id"],
                        "label": p["label"],
                        "high_effort_fraction": (
                            p.get("movement_intensity") or {}
                        ).get("high_effort_fraction"),
                        "coverage": (p.get("coverage_trust") or {}).get(
                            "coverage"
                        ),
                    }
                )
        payload["glance"] = {
            "both_visible_pct": _round(
                float(cov.get("both_visible_fraction") or 0.0) * 100, 1
            ),
            "mean_gap_norm": sep.get("mean_gap_norm"),
            "closeout_count": closes.get("count"),
            "relationship": primary.get("relationship"),
            "players": glance_players,
            "pair_person_group_ids": primary.get("person_group_ids"),
        }

    payload["coach_narrative"] = build_coach_narrative(payload)
    return payload
