"""Jersey-color team assignment for snapshot selection and metrics.

Splits court players on a red ↔ dark/blue axis (typical WNBA home red vs
away navy/black), isolates striped/grey officials as white, and does not
invent roster names.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.resolver.team_normalizer import roster_team_from_jersey_color

logger = logging.getLogger(__name__)

DISPLAY_BGR = {
    "red": (50, 50, 230),
    "blue": (230, 140, 40),  # away navy/black shown as vivid blue for UX
    "black": (45, 45, 45),
    "white": (235, 235, 235),
}

TEAM_RED = 0
TEAM_BLUE = 1
TEAM_OFFICIAL = 2


def _torso_crop(
    image: np.ndarray,
    box: list[int],
    mask_polygon: list | None = None,
) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    bh, bw = y2 - y1, x2 - x1
    ty1 = y1 + int(bh * 0.18)
    ty2 = y1 + int(bh * 0.52)
    tx1 = x1 + int(bw * 0.20)
    tx2 = x2 - int(bw * 0.20)
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    region = image[ty1:ty2, tx1:tx2].copy()
    if mask_polygon and len(mask_polygon) >= 3:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array([[int(p[0]), int(p[1])] for p in mask_polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        mask_roi = mask[ty1:ty2, tx1:tx2]
        if int((mask_roi > 0).sum()) >= 40:
            region = cv2.bitwise_and(region, region, mask=mask_roi)
            ys, xs = np.where(mask_roi > 0)
            if len(xs) >= 20:
                region = region[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return region


def _jersey_signals(crop_bgr: np.ndarray) -> dict[str, float] | None:
    """Per-crop jersey signals robust to dark navy / bright red uniforms."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    nonzero = np.any(crop_bgr > 0, axis=2)
    if int(nonzero.sum()) < 30:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(crop_bgr)
    h, s, v = cv2.split(hsv)

    court = (h >= 35) & (h <= 95) & (s >= 45) & (v >= 40)
    keep = nonzero & ~court
    if int(keep.sum()) < 25:
        keep = nonzero
    if int(keep.sum()) < 20:
        return None

    hk, sk, vk = h[keep], s[keep], v[keep]
    rk, gk, bk = r[keep], g[keep], b[keep]

    red = ((hk <= 8) | (hk >= 170)) & (sk >= 90) & (vk >= 70) & (vk <= 245)
    red |= (
        (rk.astype(np.int16) - np.maximum(gk, bk).astype(np.int16) > 55)
        & (rk >= 110)
        & (sk >= 70)
    )

    dark = (vk <= 105) & (sk <= 160) & ~(((hk >= 35) & (hk <= 95) & (sk >= 50)))
    v_med = float(np.median(vk))
    rel_dark = vk <= max(60.0, v_med * 0.85)
    dark = dark | (
        rel_dark
        & (sk <= 120)
        & (rk.astype(np.int16) < gk.astype(np.int16) + 25)
    )

    blue = (hk >= 95) & (hk <= 130) & (sk >= 45) & (vk >= 35)
    gray = (sk <= 50) & (vk >= 55) & (vk <= 200)
    whiteish = (sk <= 45) & (vk >= 185)

    red_frac = float(red.mean())
    dark_frac = float(dark.mean())
    blue_frac = float(blue.mean())
    gray_frac = float(gray.mean())
    white_frac = float(whiteish.mean())

    gray_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    col_means = gray_img.mean(axis=0)
    stripe = 0.0
    if col_means.size >= 8:
        diffs = np.abs(np.diff(col_means.astype(np.float32)))
        # Referee shirts alternate light/dark vertically across columns.
        stripe = float(np.percentile(diffs, 85) / 32.0)
        # Count strong column-to-column swings (stripe frequency).
        strong = float(np.mean(diffs > 18.0))
        stripe = max(stripe, strong * 2.2)
    stripe = min(stripe, 4.0)

    away_frac = max(dark_frac, blue_frac)
    axis = (red_frac * 1.15) - (away_frac * 1.35)

    return {
        "red_frac": red_frac,
        "dark_frac": dark_frac,
        "blue_frac": blue_frac,
        "gray_frac": gray_frac,
        "white_frac": white_frac,
        "stripe": stripe,
        "axis": float(axis),
        "mean_v": float(vk.mean()) / 255.0,
        "mean_s": float(sk.mean()) / 255.0,
        "keep_frac": float(keep.sum()) / float(max(h.size, 1)),
    }


def _court_official_prior(box: list[int], frame_w: int, frame_h: int) -> float:
    """Bias official vs player from broadcast geometry (not jersey alone).

    Near-sideline / close-to-camera bodies (bottom of frame) are often refs.
    Bodies up-court on the wood are almost always players.
    Returns roughly [-0.5, +0.7] (positive → more likely official).
    """
    if frame_w <= 1 or frame_h <= 1:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx = 0.5 * (x1 + x2) / frame_w
    cy = 0.5 * (y1 + y2) / frame_h
    feet = y2 / frame_h
    height_frac = max(y2 - y1, 1.0) / frame_h
    prior = 0.0
    # Near camera sideline (bottom strip) — classic ref standing zone.
    if feet >= 0.90:
        prior += 0.45
    elif feet >= 0.82:
        prior += 0.25
    # Tall close-up bodies on the near edge.
    if height_frac >= 0.28 and feet >= 0.78:
        prior += 0.20
    # Far / mid-court wood — players.
    if cy <= 0.55 and feet <= 0.78:
        prior -= 0.35
    if 0.28 <= cx <= 0.72 and cy <= 0.62:
        prior -= 0.15
    return float(max(-0.5, min(0.7, prior)))


def _is_official(
    sig: dict[str, float],
    *,
    court_prior: float = 0.0,
) -> bool:
    """Striped / grey / black officials — never assign clear team jerseys here."""
    neutral = sig["gray_frac"] + sig["white_frac"]
    strong_red = sig["red_frac"]
    strong_blue = sig["blue_frac"]
    strong_team = max(strong_red, strong_blue)

    # Hard veto: clear team color on torso → always a player.
    if strong_red >= 0.18:
        return False
    if strong_blue >= 0.22 and sig["mean_s"] >= 0.22:
        return False
    if abs(sig["axis"]) >= 0.45 and strong_team >= 0.15:
        return False

    score = 0.0
    # Classic zebra shirt: column stripes + grey/white fabric, little team color.
    if sig["stripe"] >= 0.55 and neutral >= 0.30 and strong_team < 0.18:
        score += 0.85
    elif sig["stripe"] >= 0.75 and neutral >= 0.22 and strong_team < 0.22:
        score += 0.70
    elif sig["stripe"] >= 1.15 and neutral >= 0.28 and strong_red < 0.22:
        score += 0.65
    # Solid grey official shirts (not navy players).
    if (
        sig["gray_frac"] >= 0.45
        and strong_red < 0.10
        and strong_blue < 0.18
        and sig["dark_frac"] < 0.55
        and sig["mean_s"] <= 0.28
    ):
        score += 0.55
    # Near-sideline black/grey refs (common when stripes are washed out).
    if (
        court_prior >= 0.20
        and strong_team < 0.15
        and sig["mean_s"] <= 0.32
        and (sig["dark_frac"] + neutral) >= 0.45
    ):
        score += 0.50 + 0.25 * min(court_prior, 0.7)
    # Weak stripe alone is not enough on-court (jersey folds / numbers).
    if sig["stripe"] >= 0.45 and sig["stripe"] < 0.55 and neutral >= 0.35 and strong_team < 0.12:
        score += 0.25

    score += 0.35 * court_prior
    return score >= 0.55


def jersey_color_from_detection(
    image_bgr: np.ndarray,
    det: dict,
) -> str | None:
    """Classify one detection's torso as red / blue / official (or None).

    Used by Confirm spatial lock so close offense/defense pairs stay on the
    *body color*, not the ByteTrack group id (which hands off under contact).
    """
    box = det.get("bbox_xyxy") or [0, 0, 0, 0]
    crop = _torso_crop(image_bgr, box, det.get("mask_polygon"))
    if crop is None:
        return None
    sig = _jersey_signals(crop)
    if sig is None:
        return None
    frame_h, frame_w = image_bgr.shape[:2]
    prior = _court_official_prior(box, frame_w, frame_h)
    if _is_official(sig, court_prior=prior):
        return "official"
    if sig["red_frac"] >= 0.16 and sig["axis"] >= -0.05:
        return "red"
    if sig["axis"] < 0 or max(sig["dark_frac"], sig["blue_frac"]) >= 0.14:
        return "blue"
    if sig["red_frac"] >= 0.12:
        return "red"
    return None


def enrich_tracks_with_appearance(image_bgr: np.ndarray, tracks: list[dict]) -> None:
    """Stamp jersey_color_name / role onto in-memory track dets (mutates)."""
    for det in tracks:
        name = jersey_color_from_detection(image_bgr, det)
        if not name:
            continue
        det["jersey_color_name"] = name
        if name == "official":
            det["role"] = "official"


def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _balanced_two_team_labels(axes: np.ndarray) -> np.ndarray:
    """Split players on red↔dark axis with near-equal team sizes."""
    n = int(axes.size)
    labels = np.zeros(n, dtype=np.int32)
    if n == 0:
        return labels
    if n == 1:
        labels[0] = TEAM_RED if float(axes[0]) >= 0 else TEAM_BLUE
        return labels

    order = np.argsort(axes)  # low (dark/blue) → high (red)
    n_blue = n // 2
    labels[order[:n_blue]] = TEAM_BLUE
    labels[order[n_blue:]] = TEAM_RED

    natural_red = int(np.sum(axes >= 0))
    natural_blue = n - natural_red
    bal_nat = 1.0 - abs(natural_red - natural_blue) / n
    # Only keep natural split when already ≈balanced (avoids 12-red / 1-blue).
    if bal_nat >= 0.78 and min(natural_red, natural_blue) >= 3:
        return np.where(axes >= 0, TEAM_RED, TEAM_BLUE).astype(np.int32)
    return labels


def assign_jersey_teams(
    image_bgr: np.ndarray,
    detections: list[dict],
    *,
    game_context: dict | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Assign team red / team blue / official (white) from jersey crops."""
    del seed  # deterministic from signals; kept for API compatibility
    frame_h, frame_w = image_bgr.shape[:2]
    samples: list[tuple[int, dict[str, float], float]] = []
    for det in detections:
        gid = det.get("person_group_id")
        if gid is None:
            continue
        gid = int(gid)
        box = det.get("bbox_xyxy") or [0, 0, 0, 0]
        crop = _torso_crop(
            image_bgr,
            box,
            det.get("mask_polygon"),
        )
        sig = _jersey_signals(crop) if crop is not None else None
        if sig is not None:
            prior = _court_official_prior(box, frame_w, frame_h)
            samples.append((gid, sig, prior))

    if not samples:
        return {"teams": [], "group_to_team": {}, "separation_score": 0.0}

    officials = [
        (g, s, p) for g, s, p in samples if _is_official(s, court_prior=p)
    ]
    # Broadcast usually shows ≤3 officials. Extra hits are almost always
    # mislabeled players — keep the strongest zebra/grey/sideline scores.
    if len(officials) > 3:

        def _off_strength(item: tuple[int, dict[str, float], float]) -> float:
            _g, s, p = item
            return (
                s["stripe"]
                + s["gray_frac"]
                + s["white_frac"]
                + 0.5 * p
                - max(s["red_frac"], s["blue_frac"])
            )

        officials = sorted(officials, key=_off_strength, reverse=True)[:3]

    official_gids = {g for g, _s, _p in officials}
    players = [(g, s, p) for g, s, p in samples if g not in official_gids]

    home = (game_context or {}).get("home_team") or None
    away = (game_context or {}).get("away_team") or None
    red_label = f"Team Red ({home})" if home else "Team Red"
    blue_label = f"Team Blue ({away})" if away else "Team Blue"
    red_roster_id = roster_team_from_jersey_color("red", game_context)
    blue_roster_id = roster_team_from_jersey_color("blue", game_context)

    group_to_team: dict[int, dict] = {}
    red_ids: list[int] = []
    blue_ids: list[int] = []
    official_ids: list[int] = []

    white_bgr = DISPLAY_BGR["white"]
    for gid, _sig, _prior in officials:
        official_ids.append(gid)
        group_to_team[gid] = {
            "team_id": TEAM_OFFICIAL,
            "team_label": "Official",
            "jersey_color_name": "white",
            "overlay_color_bgr": [int(v) for v in white_bgr],
            "overlay_color_hex": _bgr_to_hex(white_bgr),
            "role": "official",
        }

    if players:
        gids = [g for g, _s, _p in players]
        axes = np.array([s["axis"] for _g, s, _p in players], dtype=np.float32)
        labels = _balanced_two_team_labels(axes)
        red_bgr = DISPLAY_BGR["red"]
        blue_bgr = DISPLAY_BGR["blue"]
        for i, gid in enumerate(gids):
            if int(labels[i]) == TEAM_RED:
                red_ids.append(gid)
                group_to_team[gid] = {
                    "team_id": TEAM_RED,
                    "roster_team_id": red_roster_id,
                    "team_label": red_label,
                    "jersey_color_name": "red",
                    "overlay_color_bgr": [int(v) for v in red_bgr],
                    "overlay_color_hex": _bgr_to_hex(red_bgr),
                    "role": "player",
                }
            else:
                blue_ids.append(gid)
                group_to_team[gid] = {
                    "team_id": TEAM_BLUE,
                    "roster_team_id": blue_roster_id,
                    "team_label": blue_label,
                    "jersey_color_name": "blue",
                    "overlay_color_bgr": [int(v) for v in blue_bgr],
                    "overlay_color_hex": _bgr_to_hex(blue_bgr),
                    "role": "player",
                }

    teams = [
        {
            "team_id": TEAM_RED,
            "roster_team_id": red_roster_id,
            "jersey_color_name": "red",
            "label": red_label,
            "overlay_color_bgr": [int(v) for v in DISPLAY_BGR["red"]],
            "overlay_color_hex": _bgr_to_hex(DISPLAY_BGR["red"]),
            "member_group_ids": sorted(red_ids),
            "role": "player",
        },
        {
            "team_id": TEAM_BLUE,
            "roster_team_id": blue_roster_id,
            "jersey_color_name": "blue",
            "label": blue_label,
            "overlay_color_bgr": [int(v) for v in DISPLAY_BGR["blue"]],
            "overlay_color_hex": _bgr_to_hex(DISPLAY_BGR["blue"]),
            "member_group_ids": sorted(blue_ids),
            "role": "player",
        },
    ]
    if official_ids:
        teams.append(
            {
                "team_id": TEAM_OFFICIAL,
                "jersey_color_name": "white",
                "label": "Officials",
                "overlay_color_bgr": [int(v) for v in white_bgr],
                "overlay_color_hex": _bgr_to_hex(white_bgr),
                "member_group_ids": sorted(official_ids),
                "role": "official",
            }
        )

    n_players = max(len(red_ids) + len(blue_ids), 1)
    balance = 1.0 - abs(len(red_ids) - len(blue_ids)) / n_players
    axis_sep = 0.0
    if players:
        red_ax = [s["axis"] for g, s, _p in players if g in red_ids]
        blue_ax = [s["axis"] for g, s, _p in players if g in blue_ids]
        if red_ax and blue_ax:
            axis_sep = abs(float(np.mean(red_ax)) - float(np.mean(blue_ax)))
    separation = axis_sep + 1.25 * balance + 0.35 * float(len(official_ids) > 0)

    logger.info(
        "Jersey teams: red=%d blue=%d officials=%d sep=%.3f",
        len(red_ids),
        len(blue_ids),
        len(official_ids),
        separation,
    )
    return {
        "teams": teams,
        "group_to_team": group_to_team,
        "separation_score": round(separation, 4),
    }


def pick_dense_frame_candidates(
    frame_scores: list[tuple[int, float, list[dict]]],
    *,
    top_fraction: float = 0.80,
    max_candidate_frames: int | None = None,
    max_candidates: int = 12,
    seed: int = 0,
) -> list[tuple[int, float, list[dict]]]:
    """Keep near-best multi-player frames for jersey sampling.

    Prefers *earliest* dense frames so Select/snapshot anchors near clip start
    instead of randomly landing mid-clip (which felt like tracking "starting late").
    ``seed`` is retained for API compatibility; earliest-first ordering is stable.
    """
    if not frame_scores:
        return []
    _ = seed  # API compat; deterministic earliest-first (no shuffle).
    limit = int(max_candidate_frames or max_candidates or 12)
    best = max(score for _, score, _ in frame_scores)
    floor = best * float(top_fraction)
    cands = [item for item in frame_scores if item[1] >= floor]
    # Earliest first among dense frames; break ties by higher score.
    cands.sort(key=lambda item: (item[0], -item[1]))
    return cands[: max(limit, 1)]
