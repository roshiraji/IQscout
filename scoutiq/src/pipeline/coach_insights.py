"""Coach insights from selected-player track rows (pixel / normalized space).

Pure functions over per-frame stance / center points — no ball, OCR, or court
calibration. Prefers mask-bottom stance when present; falls back to bbox.
Proxies are labeled as such; never invent player names.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
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

# Recovery: after leaving near band, how long until back under NEAR_GAP.
RECOVERY_OPEN_GAP = NEAR_GAP
RECOVERY_MAX_SECONDS = 4.0

# Scorecard composite weights (clip proxies — not defensive rating).
# Stay-close defense emphasizes time attached over camera coverage.
PRESSURE_W_PRESSURE = 0.45
PRESSURE_W_CLOSE = 0.20
PRESSURE_W_GAP = 0.20
PRESSURE_W_CLOSEOUT = 0.15
PRESSURE_W_RECOVERY = 0.15

# Create-space offense: getting away from the matchup.
SPACE_W_OPEN = 0.40
SPACE_W_BURST = 0.30
SPACE_W_MAX = 0.30

OFFENSE_W_SCREEN = 0.40
OFFENSE_W_CUT = 0.30
OFFENSE_W_SPACING = 0.30

# Separation burst: gap opens quickly (blow-by / get-away proxy).
SEPARATION_SPEED = 0.30
SEPARATION_MIN_GAP = 0.10


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


def _pos(row: dict[str, Any]) -> tuple[float, float]:
    """Prefer stance (mask/bbox bottom); fall back to bbox midpoint."""
    if row.get("stance_x") is not None and row.get("stance_y") is not None:
        return _safe_float(row.get("stance_x")), _safe_float(row.get("stance_y"))
    return _safe_float(row.get("center_x")), _safe_float(row.get("center_y"))


def _bbox_center(row: dict[str, Any]) -> tuple[float, float]:
    if all(k in row for k in ("x1", "y1", "x2", "y2")):
        return (
            (_safe_float(row["x1"]) + _safe_float(row["x2"])) / 2.0,
            (_safe_float(row["y1"]) + _safe_float(row["y2"])) / 2.0,
        )
    return _safe_float(row.get("center_x")), _safe_float(row.get("center_y"))


def _pair_source(a: dict[str, Any], b: dict[str, Any]) -> str:
    sa = str(a.get("position_source") or "")
    sb = str(b.get("position_source") or "")
    if sa == "mask_bottom" and sb == "mask_bottom":
        return "mask_bottom"
    if "mask_bottom" in (sa, sb):
        return "mixed"
    return "bbox_bottom"


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
            px, py = _pos(prev)
            cx, cy = _pos(row)
            dist = math.hypot(cx - px, cy - py) / diag
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
            "note": (
                "Speeds use mask-bottom stance when available "
                "(else bbox); frame-diagonal fractions per second."
            ),
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


def _distances(
    aligned: list[tuple[dict, dict]],
    diag: float,
    *,
    use_bbox_center: bool = False,
) -> list[float]:
    dists: list[float] = []
    for a, b in aligned:
        if use_bbox_center:
            ax, ay = _bbox_center(a)
            bx, by = _bbox_center(b)
        else:
            ax, ay = _pos(a)
            bx, by = _pos(b)
        dists.append(math.hypot(ax - bx, ay - by) / max(diag, 1.0))
    return dists


def _gap_band(gap: float) -> str:
    if gap <= CLOSE_GAP:
        return "close"
    if gap <= NEAR_GAP:
        return "near"
    if gap <= PRESSURE_GAP:
        return "pressure"
    return "far"


def _closing_speeds(
    aligned: list[tuple[dict, dict]],
    dists: list[float],
) -> list[float | None]:
    """Per-frame closing speed (diag fractions / sec). Index 0 is None."""
    out: list[float | None] = [None]
    for i in range(1, len(aligned)):
        a0, _b0 = aligned[i - 1]
        a1, _b1 = aligned[i]
        dt = max(
            _safe_float(a1.get("timestamp_seconds"))
            - _safe_float(a0.get("timestamp_seconds")),
            1e-3,
        )
        out.append((dists[i - 1] - dists[i]) / dt)
    return out


def _recovery_lags(
    aligned: list[tuple[dict, dict]],
    dists: list[float],
) -> list[float]:
    """Seconds from leaving near-band to returning under NEAR_GAP."""
    lags: list[float] = []
    open_since_t: float | None = None
    was_near = False
    for i, (a, _b) in enumerate(aligned):
        gap = dists[i]
        t = _safe_float(a.get("timestamp_seconds"))
        if gap <= RECOVERY_OPEN_GAP:
            if open_since_t is not None:
                lag = t - open_since_t
                if 0.0 < lag <= RECOVERY_MAX_SECONDS:
                    lags.append(lag)
                open_since_t = None
            was_near = True
        elif was_near and open_since_t is None:
            open_since_t = t
            was_near = False
    return lags


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _infer_relationship(
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
) -> str:
    """Opponent / same_team / unknown from team_id, else jersey color name."""
    team_a = meta_a.get("team_id")
    team_b = meta_b.get("team_id")
    try:
        if team_a is not None and team_b is not None:
            ta, tb = int(team_a), int(team_b)
            if ta == 2 or tb == 2:
                return "unknown"
            if ta == tb:
                return "same_team"
            return "opponent"
    except (TypeError, ValueError):
        pass

    color_a = str(meta_a.get("jersey_color_name") or "").strip().lower()
    color_b = str(meta_b.get("jersey_color_name") or "").strip().lower()
    if not color_a or not color_b:
        return "unknown"
    if color_a in {"official", "referee", "unknown", "neutral"}:
        return "unknown"
    if color_b in {"official", "referee", "unknown", "neutral"}:
        return "unknown"
    if color_a == color_b:
        return "same_team"
    return "opponent"


def _build_matchup_scorecard(
    *,
    relationship: str,
    pct_near: float,
    pct_close: float,
    mean_gap: float | None,
    max_gap: float | None,
    closeouts: int,
    separation_bursts: int,
    recovery_mean: float | None,
    screen: dict[str, Any],
    cut_overlap: float,
    shared_seconds: float = 1.0,
    defender_label: str | None = None,
    attacker_label: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Always emit numeric performance scores when distance data exists."""
    shared = max(float(shared_seconds), 0.5)

    closeout_rate = closeouts / shared
    closeout_component = _clamp01(closeout_rate / 0.5)
    sep_rate = separation_bursts / shared
    sep_component = _clamp01(sep_rate / 0.5)

    gap_component = 0.0
    if mean_gap is not None:
        gap_component = _clamp01(1.0 - (mean_gap / max(PRESSURE_GAP, 1e-6)))

    recovery_component = 0.0
    if recovery_mean is not None:
        recovery_component = _clamp01(recovery_mean / RECOVERY_MAX_SECONDS)

    pct_open = _clamp01(1.0 - float(pct_near))
    max_gap_component = 0.0
    if max_gap is not None:
        max_gap_component = _clamp01(max_gap / max(PRESSURE_GAP * 2.0, 1e-6))

    stay_close_score = _clamp01(
        PRESSURE_W_PRESSURE * pct_near
        + PRESSURE_W_CLOSE * pct_close
        + PRESSURE_W_GAP * gap_component
        + PRESSURE_W_CLOSEOUT * closeout_component
        - PRESSURE_W_RECOVERY * recovery_component
    )
    create_space_score = _clamp01(
        SPACE_W_OPEN * pct_open
        + SPACE_W_BURST * sep_component
        + SPACE_W_MAX * max_gap_component
    )

    screen_score = float(screen.get("score") or 0.0)
    spacing = 0.0
    if mean_gap is not None:
        spacing = _clamp01(1.0 - abs(mean_gap - 0.22) / 0.22)
    off_ball_score = _clamp01(
        OFFENSE_W_SCREEN * screen_score
        + OFFENSE_W_CUT * float(cut_overlap)
        + OFFENSE_W_SPACING * spacing
    )

    # Fallback: if pick/cut signals are flat, still report a spacing-based score.
    if off_ball_score < 0.05 and mean_gap is not None:
        off_ball_score = _clamp01(0.55 * spacing + 0.45 * float(cut_overlap))

    def_lab = defender_label or label_a or "Player A"
    att_lab = attacker_label or label_b or "Player B"
    close_pct = round(pct_near * 100)
    open_pct = round(pct_open * 100)

    if relationship == "same_team":
        mode = "offense_spacing"
        pressure_index = stay_close_score
        offense_index = off_ball_score
        headline = (
            "Off-ball action score — pick / cut / spacing on this clip."
        )
        outcome_summary = (
            f"Off-ball action {round(off_ball_score * 100)}/100 "
            f"(screen {round(screen_score * 100)}, "
            f"cut {round(float(cut_overlap) * 100)}, "
            f"spacing {round(spacing * 100)})."
        )
    else:
        # Opponent OR unknown team: always score the gap battle.
        mode = "matchup_outcomes"
        pressure_index = stay_close_score
        offense_index = create_space_score
        headline = (
            "Matchup outcome scores — Stay Close vs Create Space."
        )
        outcome_summary = (
            f"{def_lab} Stay Close {round(stay_close_score * 100)}/100 "
            f"(attached {close_pct}% of contested time). "
            f"{att_lab} Create Space {round(create_space_score * 100)}/100 "
            f"(separated {open_pct}% of contested time)."
        )

    return {
        "mode": mode,
        "headline": headline,
        "outcome_summary": outcome_summary,
        "pressure_index": _round(pressure_index),
        "offense_index": _round(offense_index),
        "stay_close_score": _round(stay_close_score),
        "create_space_score": _round(create_space_score),
        "off_ball_score": _round(off_ball_score),
        "primary_score": _round(
            off_ball_score if relationship == "same_team" else stay_close_score
        ),
        "secondary_score": _round(
            None if relationship == "same_team" else create_space_score
        ),
        "defender_label": def_lab,
        "attacker_label": att_lab,
        "components": {
            "time_attached": _round(pct_near),
            "time_very_tight": _round(pct_close),
            "gap_control": _round(gap_component),
            "closeout_activity": _round(closeout_component),
            "recovery_penalty": _round(recovery_component),
            "time_separated": _round(pct_open),
            "separation_bursts": _round(sep_component),
            "max_gap_stretch": _round(max_gap_component),
            "screen_score": _round(screen_score),
            "cut_overlap": _round(cut_overlap),
            "spacing_quality": _round(spacing),
        },
        "disclaimer": (
            "Numeric clip scores from stance distance. "
            "Create Space = got away, not points scored."
        ),
    }


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
    possession: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pairwise separation, closeouts, chase / sticky / screen / cut alignment."""
    meta_a = meta_a or {}
    meta_b = meta_b or {}
    possession = possession or {}
    diag = max(frame_diagonal, 1.0)
    src = max(int(source_frame_count), 1)
    aligned = _align_pair(series_a, series_b)
    both_visible = len(aligned)
    dists = _distances(aligned, diag)
    bbox_dists = _distances(aligned, diag, use_bbox_center=True)
    mask_frames = sum(
        1 for a, b in aligned if _pair_source(a, b) == "mask_bottom"
    )
    position_source_mix = mask_frames / len(aligned) if aligned else 0.0

    mean_gap = sum(dists) / len(dists) if dists else None
    mean_gap_bbox = (
        sum(bbox_dists) / len(bbox_dists) if bbox_dists else None
    )
    min_gap = min(dists) if dists else None
    max_gap = max(dists) if dists else None
    p10_gap = _percentile(dists, 10)
    pct_close = (
        sum(1 for d in dists if d <= CLOSE_GAP) / len(dists) if dists else 0.0
    )
    pct_near = (
        sum(1 for d in dists if d <= NEAR_GAP) / len(dists) if dists else 0.0
    )
    pct_pressure = (
        sum(1 for d in dists if d <= PRESSURE_GAP) / len(dists) if dists else 0.0
    )

    closing = _closing_speeds(aligned, dists)
    recovery_list = _recovery_lags(aligned, dists)
    mean_recovery = (
        sum(recovery_list) / len(recovery_list) if recovery_list else None
    )

    # Closeout bursts: gap shrinks fast. Separation bursts: gap opens fast.
    closeouts = 0
    separation_bursts = 0
    max_close_rate = 0.0
    max_sep_rate = 0.0
    for i in range(1, len(aligned)):
        rate = closing[i]
        if rate is None:
            continue
        if rate > max_close_rate:
            max_close_rate = rate
        if (-rate) > max_sep_rate:
            max_sep_rate = -rate
        if (
            rate >= CLOSEOUT_SPEED
            and dists[i - 1] <= CLOSEOUT_MIN_GAP
            and dists[i] < dists[i - 1]
        ):
            closeouts += 1
        if (
            rate <= -SEPARATION_SPEED
            and dists[i - 1] >= SEPARATION_MIN_GAP
            and dists[i] > dists[i - 1]
        ):
            separation_bursts += 1

    relationship = _infer_relationship(meta_a, meta_b)
    same_team = relationship == "same_team"
    opp_team = relationship == "opponent"

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
            ax0, ay0 = _pos(a0)
            ax1, ay1 = _pos(a1)
            bx0, by0 = _pos(b0)
            bx1, by1 = _pos(b1)
            va = (ax1 - ax0, ay1 - ay0)
            vb = (bx1 - bx0, by1 - by0)
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
                px, py = _pos(prev)
                cx, cy = _pos(row)
                dt = max(
                    _safe_float(row.get("timestamp_seconds"))
                    - _safe_float(prev.get("timestamp_seconds")),
                    1e-3,
                )
                speed = math.hypot(cx - px, cy - py) / diag / dt
                if speed >= HIGH_EFFORT_SPEED:
                    out.add(int(row.get("frame_number") or 0))
            prev = row
        return out

    effort_a = _effort_mask(series_a)
    effort_b = _effort_mask(series_b)
    union = effort_a | effort_b
    inter = effort_a & effort_b
    cut_align = len(inter) / len(union) if union else 0.0

    shared_seconds = both_visible / 30.0
    if len(aligned) >= 2:
        t0 = _safe_float(aligned[0][0].get("timestamp_seconds"))
        t1 = _safe_float(aligned[-1][0].get("timestamp_seconds"))
        if t1 > t0:
            shared_seconds = t1 - t0

    label_a = _label_for(gid_a, meta_a)
    label_b = _label_for(gid_b, meta_b)
    defender_label = label_a
    attacker_label = label_b
    role_source = "geometry_proxy"
    offense_team_id = possession.get("offense_team_id")
    defense_team_id = possession.get("defense_team_id")
    roles_confident = (
        offense_team_id is not None
        and str(possession.get("source") or "")
        in {"ball_possession", "fused_windows", "manual_override"}
        and float(possession.get("confidence") or 0.0) >= 0.45
    ) or str(possession.get("source") or "") == "manual_override"

    if roles_confident and opp_team:
        team_a = meta_a.get("team_id")
        team_b = meta_b.get("team_id")
        try:
            team_a = int(team_a) if team_a is not None else None
            team_b = int(team_b) if team_b is not None else None
            offense_team_id = int(offense_team_id)
            if defense_team_id is not None:
                defense_team_id = int(defense_team_id)
        except (TypeError, ValueError):
            team_a = team_b = None
        if team_a is not None and team_b is not None and team_a != team_b:
            if team_a == offense_team_id:
                attacker_label = label_a
                defender_label = label_b
                role_source = str(possession.get("source") or "ball_possession")
            elif team_b == offense_team_id:
                attacker_label = label_b
                defender_label = label_a
                role_source = str(possession.get("source") or "ball_possession")
    elif opp_team or relationship == "unknown":
        if chase.get("chaser_person_group_id") == gid_a:
            defender_label = label_a
            attacker_label = label_b
        elif chase.get("chaser_person_group_id") == gid_b:
            defender_label = label_b
            attacker_label = label_a

    scorecard = _build_matchup_scorecard(
        relationship=relationship,
        pct_near=pct_near,
        pct_close=pct_close,
        mean_gap=mean_gap,
        max_gap=max_gap,
        closeouts=closeouts,
        separation_bursts=separation_bursts,
        recovery_mean=mean_recovery,
        screen=screen,
        cut_overlap=cut_align,
        shared_seconds=shared_seconds,
        defender_label=defender_label,
        attacker_label=attacker_label,
        label_a=label_a,
        label_b=label_b,
    )
    scorecard["role_source"] = role_source
    scorecard["roles_confident"] = bool(roles_confident and role_source != "geometry_proxy")
    scorecard["offense_team_id"] = offense_team_id if roles_confident else None
    scorecard["defense_team_id"] = defense_team_id if roles_confident else None
    if not scorecard["roles_confident"]:
        scorecard["role_note"] = (
            "Offense/defense roles use geometry proxies until ball possession "
            "is confident — or set offense team manually."
        )
    else:
        scorecard["role_note"] = (
            f"Roles from {role_source}: Create Space = offense, Stay Close = defense."
        )

    return {
        "person_group_ids": [gid_a, gid_b],
        "labels": [_label_for(gid_a, meta_a), _label_for(gid_b, meta_b)],
        "relationship": relationship,
        "coverage_trust": {
            "frames_both_visible": both_visible,
            "source_frame_count": src,
            "both_visible_fraction": _round(both_visible / src),
        },
        "pairwise_separation": {
            "mean_gap_norm": _round(mean_gap),
            "mean_gap_bbox_norm": _round(mean_gap_bbox),
            "min_gap_norm": _round(min_gap),
            "max_gap_norm": _round(max_gap),
            "p10_gap_norm": _round(p10_gap),
            "p50_gap_norm": _round(_percentile(dists, 50)),
            "pct_frames_within_close": _round(pct_close),
            "pct_frames_within_near": _round(pct_near),
            "pct_frames_within_pressure": _round(pct_pressure),
            "close_threshold": CLOSE_GAP,
            "near_threshold": NEAR_GAP,
            "pressure_threshold": PRESSURE_GAP,
            "position_source_mix": _round(position_source_mix),
            "note": (
                "Primary gaps use mask-bottom stance when available; "
                "mean_gap_bbox_norm is bbox-midpoint for comparison."
            ),
        },
        "recovery": {
            "event_count": len(recovery_list),
            "mean_lag_seconds": _round(mean_recovery),
            "p50_lag_seconds": _round(_percentile(recovery_list, 50)),
            "note": (
                "Time from leaving near-band to returning under near-band "
                f"(capped at {RECOVERY_MAX_SECONDS:.0f}s)."
            ),
        },
        "closeout_bursts": {
            "count": closeouts,
            "max_close_rate_norm": _round(max_close_rate),
            "threshold_speed": CLOSEOUT_SPEED,
        },
        "separation_bursts": {
            "count": separation_bursts,
            "max_open_rate_norm": _round(max_sep_rate),
            "threshold_speed": SEPARATION_SPEED,
            "note": "Fast gap-opening windows — get-away / blow-by proxy.",
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
        "matchup_scorecard": scorecard,
        "possession_roles": {
            "offense_team_id": scorecard.get("offense_team_id"),
            "defense_team_id": scorecard.get("defense_team_id"),
            "role_source": scorecard.get("role_source"),
            "roles_confident": scorecard.get("roles_confident"),
        },
    }


def build_coach_narrative(insights: dict[str, Any]) -> str:
    """Performance-first coach blurb — outcomes, not camera coverage."""
    players = insights.get("players") or []
    pairs = insights.get("pairs") or []
    if len(players) < 2 or not pairs:
        return (
            "Select at least two players to score the matchup on this clip."
        )

    primary = pairs[0]
    labels = primary.get("labels") or ["Player A", "Player B"]
    a_lab, b_lab = labels[0], labels[1]
    sep = primary.get("pairwise_separation") or {}
    closes = primary.get("closeout_bursts") or {}
    seps = primary.get("separation_bursts") or {}
    scorecard = primary.get("matchup_scorecard") or {}
    rel = primary.get("relationship") or "unknown"

    near_pct = round(float(sep.get("pct_frames_within_near") or 0.0) * 100)
    open_pct = max(0, 100 - near_pct)
    close_n = int(closes.get("count") or 0)
    sep_n = int(seps.get("count") or 0)

    if scorecard.get("outcome_summary"):
        parts = [str(scorecard["outcome_summary"])]
    else:
        parts = [f"{a_lab} vs {b_lab} matchup on this clip."]

    if rel == "opponent":
        stay = scorecard.get("stay_close_score")
        create = scorecard.get("create_space_score")
        def_lab = scorecard.get("defender_label") or "Stay-close side"
        att_lab = scorecard.get("attacker_label") or "Create-space side"
        roles_confident = bool(scorecard.get("roles_confident"))
        if stay is not None and create is not None:
            if roles_confident:
                parts = [
                    f"{def_lab} (defense) Stay Close {round(float(stay) * 100)}/100 - "
                    f"attached for {near_pct}% of contested time with {close_n} "
                    f"closeout burst(s).",
                    f"{att_lab} (offense) Create Space {round(float(create) * 100)}/100 - "
                    f"separated for {open_pct}% of contested time with {sep_n} "
                    f"get-away burst(s).",
                ]
            else:
                parts = [
                    f"{def_lab} earned Stay Close {round(float(stay) * 100)}/100 - "
                    f"attached for {near_pct}% of contested time with {close_n} "
                    f"closeout burst(s).",
                    f"{att_lab} earned Create Space {round(float(create) * 100)}/100 - "
                    f"separated for {open_pct}% of contested time with {sep_n} "
                    f"get-away burst(s).",
                    "Offense/defense not locked — set offense team or wait for ball possession.",
                ]
        if stay is not None and create is not None and roles_confident:
            if float(stay) >= float(create) + 0.08:
                parts.append(
                    "Edge: defense held the matchup tighter on this clip."
                )
            elif float(create) >= float(stay) + 0.08:
                parts.append(
                    "Edge: offense created more separation on this clip."
                )
            else:
                parts.append("Edge: contested — neither side clearly won the gap.")
        elif stay is not None and create is not None:
            parts.append("Edge: contested — roles use geometry until possession is confident.")
    elif rel == "same_team":
        screen = primary.get("screen_shape_proxy") or {}
        label = (screen.get("soft_label") or "no_clear_screen_shape").replace(
            "_", " "
        )
        idx = scorecard.get("offense_index")
        if idx is not None:
            parts.append(
                f"Off-ball action {round(float(idx) * 100)}/100 "
                f"(screen shape: {label})."
            )
        cut = primary.get("cut_timing_alignment") or {}
        ov = cut.get("high_effort_overlap")
        if ov is not None:
            parts.append(
                f"Cut-timing overlap {round(float(ov) * 100)}%."
            )

    # Intensity one-liners (effort, not coverage)
    inten_bits = []
    for p in players[:2]:
        mi = p.get("movement_intensity") or {}
        he = mi.get("high_effort_fraction")
        if he is not None:
            inten_bits.append(
                f"{p.get('label')} high-effort {round(float(he) * 100)}%"
            )
    if inten_bits:
        parts.append("Work rate: " + "; ".join(inten_bits) + ".")

    parts.append(
        "Scores use stance distance only — create-space means got away, "
        "not points scored (no ball/shot model yet)."
    )
    return " ".join(parts)


def _series_stride(source_frame_count: int) -> int:
    return 1 if int(source_frame_count) <= 400 else 2


def build_pair_distance_series(
    series_map: dict[int, list[dict]],
    *,
    frame_diagonal: float,
    source_frame_count: int,
    pair_ids: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Per-frame pairwise gaps for performance / offline analysis."""
    diag = max(frame_diagonal, 1.0)
    stride = _series_stride(source_frame_count)
    gids = sorted(series_map)
    pairs = pair_ids or list(combinations(gids, 2))
    out: list[dict[str, Any]] = []
    for ga, gb in pairs:
        aligned = _align_pair(series_map[ga], series_map[gb])
        dists = _distances(aligned, diag)
        closing = _closing_speeds(aligned, dists)
        frames: list[dict[str, Any]] = []
        for idx, (a, b) in enumerate(aligned):
            if idx % stride != 0:
                continue
            gap = dists[idx]
            frames.append(
                {
                    "frame_number": int(a.get("frame_number") or 0),
                    "t": _round(_safe_float(a.get("timestamp_seconds")), 4),
                    "gap_norm": _round(gap),
                    "gap_band": _gap_band(gap),
                    "closing_speed_norm": _round(closing[idx]),
                    "source": _pair_source(a, b),
                }
            )
        out.append(
            {
                "person_group_a": ga,
                "person_group_b": gb,
                "stride": stride,
                "frames": frames,
            }
        )
    return out


def write_pair_distances_csv(
    path: Path,
    pair_distance_series: list[dict[str, Any]],
) -> Path:
    """Flat CSV of pairwise gaps (one row per series frame sample)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_number",
        "timestamp_seconds",
        "person_group_a",
        "person_group_b",
        "gap_norm",
        "gap_band",
        "closing_speed_norm",
        "position_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pair_distance_series:
            ga = pair.get("person_group_a")
            gb = pair.get("person_group_b")
            for frame in pair.get("frames") or []:
                writer.writerow(
                    {
                        "frame_number": frame.get("frame_number"),
                        "timestamp_seconds": frame.get("t"),
                        "person_group_a": ga,
                        "person_group_b": gb,
                        "gap_norm": frame.get("gap_norm"),
                        "gap_band": frame.get("gap_band"),
                        "closing_speed_norm": frame.get("closing_speed_norm"),
                        "position_source": frame.get("source"),
                    }
                )
    return path


def compute_coach_insights(
    rows: list[dict],
    *,
    source_frame_count: int,
    frame_width: int,
    frame_height: int,
    label_by_group: dict[int, str] | None = None,
    team_meta_by_group: dict[int, dict] | None = None,
    primary_pair: tuple[int, int] | None = None,
    possession: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full coach_insights payload from selected track rows."""
    label_by_group = label_by_group or {}
    team_meta_by_group = team_meta_by_group or {}
    possession = possession or {}
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
                possession=possession,
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

    pair_distance_series = build_pair_distance_series(
        series_map,
        frame_diagonal=diag,
        source_frame_count=source_frame_count,
    )

    payload = {
        "version": 3,
        "units": {
            "distance": "fraction_of_frame_diagonal",
            "speed": "diagonal_fractions_per_second",
            "position": "mask_bottom_stance_preferred",
            "note": (
                "Clip proxies — stance from segmentation bottom when available; "
                "not calibrated court feet."
            ),
        },
        "non_goals": [
            "shot_make_miss",
            "named_identity",
            "calibrated_feet_per_second",
            "help_rotation_without_ball",
            "true_defensive_rating",
        ],
        "frame_width": frame_width,
        "frame_height": frame_height,
        "source_frame_count": source_frame_count,
        "selected_person_group_ids": gids,
        "players": players,
        "pairs": pairs,
        "pair_distance_series": pair_distance_series,
        "possession": {
            "offense_team_id": possession.get("offense_team_id"),
            "defense_team_id": possession.get("defense_team_id"),
            "source": possession.get("source"),
            "confidence": possession.get("confidence"),
            "held_fraction": possession.get("held_fraction"),
            "n_windows": possession.get("n_windows"),
            "windows": possession.get("windows"),
        },
        "glance": {},
    }

    if pairs:
        primary = pairs[0]
        sep = primary.get("pairwise_separation") or {}
        closes = primary.get("closeout_bursts") or {}
        recovery = primary.get("recovery") or {}
        scorecard = primary.get("matchup_scorecard") or {}
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
                    }
                )
        seps = primary.get("separation_bursts") or {}
        payload["glance"] = {
            "mean_gap_norm": sep.get("mean_gap_norm"),
            "min_gap_norm": sep.get("min_gap_norm"),
            "p10_gap_norm": sep.get("p10_gap_norm"),
            "max_gap_norm": sep.get("max_gap_norm"),
            "time_attached_pct": _round(
                float(sep.get("pct_frames_within_near") or 0.0) * 100, 1
            ),
            "time_separated_pct": _round(
                max(0.0, 100.0 - float(sep.get("pct_frames_within_near") or 0.0) * 100),
                1,
            ),
            "closeout_count": closes.get("count"),
            "separation_burst_count": seps.get("count"),
            "recovery_mean_seconds": recovery.get("mean_lag_seconds"),
            "stay_close_score": scorecard.get("stay_close_score"),
            "create_space_score": scorecard.get("create_space_score"),
            "off_ball_score": scorecard.get("off_ball_score"),
            "primary_score": scorecard.get("primary_score"),
            "secondary_score": scorecard.get("secondary_score"),
            "pressure_index": scorecard.get("pressure_index"),
            "offense_index": scorecard.get("offense_index"),
            "scorecard_mode": scorecard.get("mode"),
            "outcome_summary": scorecard.get("outcome_summary"),
            "defender_label": scorecard.get("defender_label"),
            "attacker_label": scorecard.get("attacker_label"),
            "relationship": primary.get("relationship"),
            "roles_confident": scorecard.get("roles_confident"),
            "role_source": scorecard.get("role_source"),
            "offense_team_id": scorecard.get("offense_team_id"),
            "defense_team_id": scorecard.get("defense_team_id"),
            "players": glance_players,
            "pair_person_group_ids": primary.get("person_group_ids"),
        }

    payload["coach_narrative"] = build_coach_narrative(payload)
    return payload
