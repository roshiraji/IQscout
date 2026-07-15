"""Jersey-color team assignment for snapshot selection and metrics.

Splits court players on a red ↔ dark/blue axis (typical WNBA home red vs
away navy/black), isolates striped/grey officials as white, and does not
invent roster names.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import cv2
import numpy as np

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


def _is_official(sig: dict[str, float]) -> bool:
    """Striped / grey officials — never assign to team red/blue."""
    neutral = sig["gray_frac"] + sig["white_frac"]
    strong_team_color = max(sig["red_frac"], sig["blue_frac"])
    # Classic zebra shirt: column stripes + grey/white fabric, little team color.
    if sig["stripe"] >= 0.45 and neutral >= 0.28 and strong_team_color < 0.22:
        return True
    if sig["stripe"] >= 0.75 and neutral >= 0.20 and strong_team_color < 0.25:
        return True
    if sig["stripe"] >= 1.15 and neutral >= 0.30 and sig["red_frac"] < 0.30:
        return True
    # Solid grey official shirts.
    if (
        sig["gray_frac"] >= 0.40
        and sig["red_frac"] < 0.12
        and max(sig["dark_frac"], sig["blue_frac"]) < 0.40
    ):
        return True
    return False


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
    samples: list[tuple[int, dict[str, float]]] = []
    for det in detections:
        gid = det.get("person_group_id")
        if gid is None:
            continue
        gid = int(gid)
        crop = _torso_crop(
            image_bgr,
            det.get("bbox_xyxy") or [0, 0, 0, 0],
            det.get("mask_polygon"),
        )
        sig = _jersey_signals(crop) if crop is not None else None
        if sig is not None:
            samples.append((gid, sig))

    if not samples:
        return {"teams": [], "group_to_team": {}, "separation_score": 0.0}

    officials = [(g, s) for g, s in samples if _is_official(s)]
    players = [(g, s) for g, s in samples if not _is_official(s)]

    home = (game_context or {}).get("home_team") or None
    away = (game_context or {}).get("away_team") or None
    red_label = f"Team Red ({home})" if home else "Team Red"
    blue_label = f"Team Blue ({away})" if away else "Team Blue"

    group_to_team: dict[int, dict] = {}
    red_ids: list[int] = []
    blue_ids: list[int] = []
    official_ids: list[int] = []

    white_bgr = DISPLAY_BGR["white"]
    for gid, _sig in officials:
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
        gids = [g for g, _ in players]
        axes = np.array([s["axis"] for _, s in players], dtype=np.float32)
        labels = _balanced_two_team_labels(axes)
        red_bgr = DISPLAY_BGR["red"]
        blue_bgr = DISPLAY_BGR["blue"]
        for i, gid in enumerate(gids):
            if int(labels[i]) == TEAM_RED:
                red_ids.append(gid)
                group_to_team[gid] = {
                    "team_id": TEAM_RED,
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
                    "team_label": blue_label,
                    "jersey_color_name": "blue",
                    "overlay_color_bgr": [int(v) for v in blue_bgr],
                    "overlay_color_hex": _bgr_to_hex(blue_bgr),
                    "role": "player",
                }

    teams = [
        {
            "team_id": TEAM_RED,
            "jersey_color_name": "red",
            "label": red_label,
            "overlay_color_bgr": [int(v) for v in DISPLAY_BGR["red"]],
            "overlay_color_hex": _bgr_to_hex(DISPLAY_BGR["red"]),
            "member_group_ids": sorted(red_ids),
            "role": "player",
        },
        {
            "team_id": TEAM_BLUE,
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
        red_ax = [s["axis"] for g, s in players if g in red_ids]
        blue_ax = [s["axis"] for g, s in players if g in blue_ids]
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
    top_fraction: float = 0.92,
    max_candidate_frames: int | None = None,
    max_candidates: int = 12,
    seed: int = 0,
) -> list[tuple[int, float, list[dict]]]:
    """Keep near-best multi-player frames for jersey sampling / random pick."""
    if not frame_scores:
        return []
    limit = int(max_candidate_frames or max_candidates or 12)
    best = max(score for _, score, _ in frame_scores)
    floor = best * float(top_fraction)
    cands = [item for item in frame_scores if item[1] >= floor]
    cands.sort(key=lambda item: (-item[1], item[0]))
    cands = cands[: max(limit, 1)]
    rng = random.Random(seed)
    rng.shuffle(cands)
    return cands
