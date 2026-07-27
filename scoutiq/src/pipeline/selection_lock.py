"""Identity-aware selection lock — keep focused preview on clicked bodies.

ByteTrack / person_group_id can hand off to a nearby player during occlusion.
After Confirm we re-associate each selected person to detections by spatial
continuity plus a deep person-ReID reference, not raw group ID.

Important: never reward a detection merely for carrying the selected
person_group_id — after a handoff that ID sits on the *wrong* body.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pipeline.selected_player_reid import reid_vector, strip_reid_vector


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _same_body(box_a: list[float], box_b: list[float]) -> bool:
    """Whether two boxes describe one person rather than two touching ones.

    SAM 2 reports a tight silhouette box while the detector reports a looser
    person box, so plain IoU is not enough to recognise the duplicate. Leaving
    the duplicate in place makes isolate erase the selected player's limbs.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return False
    contained = inter / smaller
    return contained >= 0.75 and _box_iou(box_a, box_b) >= 0.40


def _center(box: list[float]) -> tuple[float, float]:
    return (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))


def _shift_box(box: list[float], dx: float, dy: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


@dataclass
class BodyLock:
    selection_gid: int
    label: str
    jersey_color_name: str | None = None
    last_box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    # Exact box the user clicked on the snapshot — never overwritten by a
    # nearby stranger before the exclusive seed step.
    click_box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    # The clicked body is the anchor. The rolling template updates only on
    # unambiguous matches, so it cannot drift to a defender/referee.
    reid_anchor: np.ndarray | None = None
    reid_template: np.ndarray | None = None
    reid_updates: int = 0


def _det_box(det: dict) -> list[float]:
    return [float(v) for v in (det.get("bbox_xyxy") or [0, 0, 0, 0])[:4]]


def _det_gid(det: dict) -> int:
    value = det.get("person_group_id")
    if value is None:
        value = det.get("track_id", -1)
    return int(value)


def _jersey_hint(det: dict, team_meta_by_group: dict[int, dict] | None) -> str | None:
    """Jersey color for a candidate *body*.

    Prefer appearance stamped on the detection. Do not fall back to the
    person_group_id's team meta — after a ByteTrack handoff that meta sits on
    the wrong body and makes close defense swaps worse.
    """
    del team_meta_by_group  # intentionally unused for body color
    name = det.get("jersey_color_name")
    return str(name).lower() if name else None


def _det_role(det: dict, team_meta_by_group: dict[int, dict] | None) -> str | None:
    role = det.get("role")
    if role:
        return str(role).lower()
    # Official role on group meta is usually stable (refs rarely steal player gids).
    meta = (team_meta_by_group or {}).get(_det_gid(det)) or {}
    role = meta.get("role")
    return str(role).lower() if role else None


def _is_non_player_det(det: dict, team_meta_by_group: dict[int, dict] | None) -> bool:
    role = _det_role(det, team_meta_by_group)
    jersey = (det.get("jersey_color_name") or "").lower() or None
    if role == "official":
        return True
    if jersey in {"white", "official", "gray", "grey"}:
        return True
    meta = (team_meta_by_group or {}).get(_det_gid(det)) or {}
    meta_j = str(meta.get("jersey_color_name") or "").lower()
    if meta.get("role") == "official" or meta_j in {"white", "official", "gray", "grey"}:
        return True
    return False


def _box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def _reid_similarity(lock: BodyLock, det: dict) -> float | None:
    """Similarity to both the immutable click anchor and safe rolling view."""
    vector = reid_vector(det)
    if vector is None:
        return None
    anchor = _cosine(lock.reid_anchor, vector)
    rolling = _cosine(lock.reid_template, vector)
    scores = [score for score in (anchor, rolling) if score is not None]
    if not scores:
        return None
    # Keep the click view dominant — the rolling template only helps after
    # lighting/camera changes and must never overwrite the selected identity.
    if anchor is not None and rolling is not None:
        return float(0.75 * anchor + 0.25 * rolling)
    return float(scores[0])


def _in_contact(lock: BodyLock, peer_locks: list[BodyLock] | None) -> bool:
    if not peer_locks:
        return False
    for peer in peer_locks:
        if peer.selection_gid == lock.selection_gid:
            continue
        if _box_iou(lock.last_box, peer.last_box) >= 0.04:
            return True
        lx, ly = _center(lock.last_box)
        px, py = _center(peer.last_box)
        if float(np.hypot(lx - px, ly - py)) <= 1.2 * max(
            float(lock.last_box[2] - lock.last_box[0]),
            float(peer.last_box[2] - peer.last_box[0]),
            1.0,
        ):
            return True
    return False


def _score_lock_to_det(
    lock: BodyLock,
    det: dict,
    *,
    frame_diag: float,
    team_meta_by_group: dict[int, dict] | None,
    peer_locks: list[BodyLock] | None = None,
) -> float:
    box = _det_box(det)
    pred = _shift_box(lock.last_box, lock.velocity[0], lock.velocity[1])
    iou = _box_iou(pred, box)
    # Also compare to last box (no motion) — helps when velocity is wrong.
    iou_last = _box_iou(lock.last_box, box)
    iou = max(iou, 0.85 * iou_last)

    cx, cy = _center(box)
    px, py = _center(pred)
    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
    # Basketball contact: keep the search radius tight so a close defender
    # cannot steal the lock via a slightly higher IoU on the wrong body.
    dist_score = 1.0 - min(dist / max(frame_diag * 0.12, 1.0), 1.0)
    spatial_score = 0.62 * iou + 0.38 * dist_score

    # Hard reject teleports — these look like "tracking goes crazy."
    if dist > frame_diag * 0.14:
        return -1.0

    # Player locks must never latch onto officials / grey shirts.
    lock_jersey = (lock.jersey_color_name or "").lower() or None
    if lock_jersey not in {None, "white", "official", "gray", "grey"}:
        if _is_non_player_det(det, team_meta_by_group):
            return -1.0

    # Fused mega-box covering this lock AND a peer → reject (common when
    # offense/defense collide and the detector merges them).
    if peer_locks and _box_area(box) > 1.55 * max(_box_area(lock.last_box), 1.0):
        covers_peer = False
        for peer in peer_locks:
            if peer.selection_gid == lock.selection_gid:
                continue
            if _box_iou(box, peer.last_box) >= 0.35 and _box_iou(box, lock.last_box) >= 0.35:
                covers_peer = True
                break
        if covers_peer:
            return -1.0

    # Tiny tie-break only when already spatially locked onto this body.
    if iou >= 0.45 and _det_gid(det) == lock.selection_gid:
        spatial_score += 0.02

    dj = _jersey_hint(det, team_meta_by_group)
    if lock_jersey and lock_jersey not in {"white", "official", "gray", "grey"}:
        if dj and dj == lock_jersey:
            # Strong keep signal during on-ball defense — red stays on red.
            spatial_score += 0.28
        elif dj and dj not in {"white", "official", "gray", "grey"} and dj != lock_jersey:
            # Near-hard veto: wrong team color even if boxes heavily overlap.
            spatial_score -= 0.55

    appearance_score = _reid_similarity(lock, det)
    if appearance_score is None:
        return spatial_score

    # When players close down, spatial overlap is exactly what becomes
    # ambiguous. Let the target's visual identity dominate the decision.
    reid_weight = 0.68 if _in_contact(lock, peer_locks) else 0.52
    return (1.0 - reid_weight) * spatial_score + reid_weight * appearance_score


def _identity_is_clear(
    lock: BodyLock,
    dets: list[dict],
    candidate_index: int,
    *,
    frame_diag: float,
    team_meta_by_group: dict[int, dict] | None,
    peer_locks: list[BodyLock],
) -> bool:
    """Reject an ambiguous ReID choice during physical contact.

    A short gap is safer than putting a target's label on a defender or a
    referee. Outside contact, normal motion continuity remains sufficient.
    """
    if lock.reid_anchor is None:
        return True
    candidate = _reid_similarity(lock, dets[candidate_index])
    if candidate is None:
        return not _in_contact(lock, peer_locks)
    if candidate < 0.46:
        return False
    if not _in_contact(lock, peer_locks):
        return True

    pred = _shift_box(lock.last_box, lock.velocity[0], lock.velocity[1])
    px, py = _center(pred)
    competing: list[float] = []
    for index, det in enumerate(dets):
        if index == candidate_index or _is_non_player_det(det, team_meta_by_group):
            continue
        cx, cy = _center(_det_box(det))
        if float(np.hypot(cx - px, cy - py)) > frame_diag * 0.14:
            continue
        score = _reid_similarity(lock, det)
        if score is not None:
            competing.append(score)
    # A 0.03 gap is intentionally conservative at broadcast resolution. If
    # ReID cannot separate two touching bodies, retain neither identity.
    return not competing or candidate - max(competing) >= 0.03


def _commit_match(
    lock: BodyLock,
    det: dict,
    *,
    peer_locks: list[BodyLock],
    frame_diag: float,
) -> dict:
    """Advance a lock after an accepted candidate and update safe appearance."""
    box = _det_box(det)
    old_c = _center(lock.last_box)
    new_c = _center(box)
    vx = 0.55 * (new_c[0] - old_c[0]) + 0.45 * lock.velocity[0]
    vy = 0.55 * (new_c[1] - old_c[1]) + 0.45 * lock.velocity[1]
    max_step = frame_diag * 0.06
    speed = (vx * vx + vy * vy) ** 0.5
    if speed > max_step:
        scale = max_step / max(speed, 1e-6)
        vx *= scale
        vy *= scale

    appearance = _reid_similarity(lock, det)
    stable_view = (
        appearance is not None
        and appearance >= 0.72
        and not _in_contact(lock, peer_locks)
        and _box_iou(lock.last_box, box) >= 0.40
    )
    vector = reid_vector(det)
    if stable_view and vector is not None:
        previous = lock.reid_template if lock.reid_template is not None else vector
        updated = 0.90 * previous + 0.10 * vector
        norm = float(np.linalg.norm(updated))
        lock.reid_template = updated / norm if norm > 1e-8 else previous
        lock.reid_updates += 1

    lock.velocity = (vx, vy)
    lock.last_box = box
    lock.missed = 0
    enriched = dict(det)
    enriched["person_group_id"] = lock.selection_gid
    enriched["selection_locked"] = True
    if appearance is not None:
        enriched["reid_similarity"] = round(float(appearance), 4)
    return enriched


def _seed_locks_exclusive_iou(
    locks: list[BodyLock],
    snapshot_dets: list[dict],
    *,
    min_iou: float = 0.20,
) -> dict[int, dict]:
    """Attach each lock to exactly one snapshot det by *click* box IoU only.

    No Hungarian / no person_group_id bonus — adjacent selected players must
    not steal each other's seed body. Score against click_box (what the user
    saw), not a previously remapped last_box.
    """
    if not locks:
        return {}
    pairs: list[tuple[float, int, int]] = []
    for i, lock in enumerate(locks):
        ref = lock.click_box if any(lock.click_box) else lock.last_box
        for j, det in enumerate(snapshot_dets):
            iou = _box_iou(ref, _det_box(det))
            if iou >= min_iou:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_locks: set[int] = set()
    used_dets: set[int] = set()
    chosen: dict[int, dict] = {}
    for iou, i, j in pairs:
        if i in used_locks or j in used_dets:
            continue
        used_locks.add(i)
        used_dets.add(j)
        lock = locks[i]
        det = snapshot_dets[j]
        lock.last_box = _det_box(det)
        lock.velocity = (0.0, 0.0)
        lock.missed = 0
        vector = reid_vector(det)
        if vector is not None:
            lock.reid_anchor = vector.copy()
            lock.reid_template = vector.copy()
            lock.reid_updates = 0
        enriched = dict(det)
        enriched["person_group_id"] = lock.selection_gid
        enriched["selection_locked"] = True
        chosen[lock.selection_gid] = enriched

    # Always paint the snapshot frame on the click even if no det overlapped
    # enough — never invent a distant body just to fill the label.
    for lock in locks:
        if lock.selection_gid in chosen:
            continue
        box = lock.click_box if any(lock.click_box) else lock.last_box
        lock.last_box = list(box)
        lock.velocity = (0.0, 0.0)
        lock.missed = 0
        chosen[lock.selection_gid] = {
            "track_id": lock.selection_gid,
            "person_group_id": lock.selection_gid,
            "bbox_xyxy": list(box),
            "confidence": 1.0,
            "mask_polygon": None,
            "selection_locked": True,
        }
    return chosen


def _assign_locks(
    locks: list[BodyLock],
    dets: list[dict],
    *,
    frame_diag: float,
    team_meta_by_group: dict[int, dict] | None,
    min_score: float,
    allow_reacquire: bool = True,
) -> dict[int, dict]:
    """Map selected IDs to a one-to-one, identity-safe body assignment."""
    if not locks or not dets:
        for lock in locks:
            lock.missed += 1
        return {}

    scores = np.zeros((len(locks), len(dets)), dtype=np.float32)
    for i, lock in enumerate(locks):
        for j, det in enumerate(dets):
            scores[i, j] = _score_lock_to_det(
                lock,
                det,
                frame_diag=frame_diag,
                team_meta_by_group=team_meta_by_group,
                peer_locks=locks,
            )

    chosen: dict[int, dict] = {}
    row_ind, col_ind = linear_sum_assignment(-scores)
    for i, j in zip(row_ind, col_ind):
        lock = locks[i]
        score = float(scores[i, j])
        gate = min_score + (0.08 if lock.missed >= 1 else 0.0)
        if lock.missed >= 4:
            gate += 0.06
        if score < gate:
            continue

        det = dets[j]
        appearance = _reid_similarity(lock, det)
        if lock.missed >= 1:
            if not allow_reacquire:
                continue
            # After an occlusion, never reacquire from a raw tracker ID or
            # color alone. This is the anti-referee / anti-defender rule.
            if lock.reid_anchor is not None and (
                appearance is None or appearance < 0.58
            ):
                continue
        if not _identity_is_clear(
            lock,
            dets,
            j,
            frame_diag=frame_diag,
            team_meta_by_group=team_meta_by_group,
            peer_locks=locks,
        ):
            continue

        chosen[lock.selection_gid] = _commit_match(
            lock,
            det,
            peer_locks=locks,
            frame_diag=frame_diag,
        )

    for lock in locks:
        if lock.selection_gid not in chosen:
            lock.missed += 1
    return chosen


def _init_locks_at_snapshot(
    *,
    selected_players: list[dict],
    selected_group_ids: set[int],
    snapshot_dets: list[dict],
    label_by_group_id: dict[int, str],
    team_meta_by_group: dict[int, dict],
) -> list[BodyLock]:
    locks: list[BodyLock] = []
    seeds: list[tuple[int, list[float], str | None, str]] = []
    for player in selected_players or []:
        gid = player.get("person_group_id")
        if gid is None:
            gid = player.get("track_id")
        if gid is None:
            continue
        gid = int(gid)
        if gid not in selected_group_ids:
            continue
        if player.get("x1") is None:
            continue
        box = [
            float(player["x1"]),
            float(player["y1"]),
            float(player["x2"]),
            float(player["y2"]),
        ]
        jersey = (
            player.get("jersey_color_name")
            or (team_meta_by_group.get(gid) or {}).get("jersey_color_name")
        )
        label = (
            player.get("player_name")
            or player.get("label")
            or label_by_group_id.get(gid)
            or f"Person #{gid}"
        )
        seeds.append((gid, box, str(jersey).lower() if jersey else None, label))

    if not seeds:
        for gid in sorted(selected_group_ids):
            match = next((d for d in snapshot_dets if _det_gid(d) == gid), None)
            if match is None:
                continue
            jersey = (team_meta_by_group.get(gid) or {}).get("jersey_color_name")
            seeds.append(
                (
                    gid,
                    _det_box(match),
                    str(jersey).lower() if jersey else None,
                    label_by_group_id.get(gid, f"Person #{gid}"),
                )
            )

    for gid, seed_box, jersey, label in seeds:
        # Keep click_box = what the user selected. Exclusive seed maps it to a
        # det later — do not rematch here (that was swapping adjacent picks).
        locks.append(
            BodyLock(
                selection_gid=gid,
                label=label,
                jersey_color_name=jersey,
                last_box=list(seed_box),
                click_box=list(seed_box),
            )
        )
    return locks


def lock_selected_bodies(
    frames: list[dict],
    *,
    snapshot_frame_number: int,
    selected_group_ids: set[int],
    selected_players: list[dict] | None,
    label_by_group_id: dict[int, str],
    team_meta_by_group: dict[int, dict] | None = None,
    frame_width: int = 1280,
    frame_height: int = 720,
    min_score: float = 0.32,
) -> dict[int, dict[int, dict]]:
    """Return {frame_number: {selection_gid: det}} locked to clicked bodies."""
    team_meta_by_group = team_meta_by_group or {}
    frame_diag = float(np.hypot(frame_width, frame_height))
    by_frame = {int(fr["frame_number"]): fr for fr in frames}
    if not by_frame:
        return {}

    snap = int(snapshot_frame_number)
    if snap not in by_frame:
        if snap - 1 in by_frame:
            snap = snap - 1
        else:
            snap = min(by_frame.keys(), key=lambda f: abs(f - snap))

    snapshot_dets = list(by_frame[snap].get("tracks") or [])
    locks = _init_locks_at_snapshot(
        selected_players=selected_players or [],
        selected_group_ids=selected_group_ids,
        snapshot_dets=snapshot_dets,
        label_by_group_id=label_by_group_id,
        team_meta_by_group=team_meta_by_group,
    )
    if not locks:
        return {}

    locked: dict[int, dict[int, dict]] = {}

    # Seed ONLY from the clicked boxes (exclusive IoU). Never Hungarian here —
    # two adjacent selections were swapping at init and "starting wrong."
    snap_chosen = _seed_locks_exclusive_iou(locks, snapshot_dets, min_iou=0.18)
    locked[snap] = snap_chosen

    ordered = sorted(by_frame.keys())
    snap_idx = ordered.index(snap)

    fwd_locks = [
        BodyLock(
            selection_gid=l.selection_gid,
            label=l.label,
            jersey_color_name=l.jersey_color_name,
            last_box=list(l.last_box),
            click_box=list(l.click_box),
            velocity=(0.0, 0.0),
            missed=0,
            reid_anchor=(
                l.reid_anchor.copy() if l.reid_anchor is not None else None
            ),
            reid_template=(
                l.reid_template.copy() if l.reid_template is not None else None
            ),
        )
        for l in locks
    ]
    for lock in fwd_locks:
        det = snap_chosen.get(lock.selection_gid)
        if det is not None:
            lock.last_box = _det_box(det)
    for frame_number in ordered[snap_idx + 1 :]:
        dets = list(by_frame[frame_number].get("tracks") or [])
        locked[frame_number] = _assign_locks(
            fwd_locks,
            dets,
            frame_diag=frame_diag,
            team_meta_by_group=team_meta_by_group,
            min_score=min_score,
            allow_reacquire=True,
        )

    bwd_locks = [
        BodyLock(
            selection_gid=l.selection_gid,
            label=l.label,
            jersey_color_name=l.jersey_color_name,
            last_box=list(l.last_box),
            click_box=list(l.click_box),
            velocity=(0.0, 0.0),
            missed=0,
            reid_anchor=(
                l.reid_anchor.copy() if l.reid_anchor is not None else None
            ),
            reid_template=(
                l.reid_template.copy() if l.reid_template is not None else None
            ),
        )
        for l in locks
    ]
    for lock in bwd_locks:
        det = snap_chosen.get(lock.selection_gid)
        if det is not None:
            lock.last_box = _det_box(det)
    # Backward is stricter and does not reacquire after a miss — that was
    # painting wrong people at t=0 when the user hit play.
    for frame_number in reversed(ordered[:snap_idx]):
        dets = list(by_frame[frame_number].get("tracks") or [])
        locked[frame_number] = _assign_locks(
            bwd_locks,
            dets,
            frame_diag=frame_diag,
            team_meta_by_group=team_meta_by_group,
            min_score=min_score + 0.08,
            allow_reacquire=False,
        )

    return locked


def build_locked_frame_records(
    frames: list[dict],
    locked: dict[int, dict[int, dict]],
    selected_group_ids: set[int],
    *,
    lock_last_boxes: dict[int, list[float]] | None = None,
) -> list[dict]:
    """Rewrite tracks so selected gids stick to spatially locked bodies.

    Demote a selected person_group_id on a body only when:
      - another detection is the locked body for that gid this frame, OR
      - the body is far from every lock's last known box (clear handoff victim).

    Do NOT demote merely because the lock missed this frame — that erased the
    true selected player in isolate ("deselected").
    """
    out: list[dict] = []
    last_boxes = dict(lock_last_boxes or {})
    for fr in frames:
        fid = int(fr["frame_number"])
        chosen = locked.get(fid) or {}
        claimed_boxes = [_det_box(d) for d in chosen.values()]
        for gid, det in chosen.items():
            last_boxes[int(gid)] = _det_box(det)

        rewritten: list[dict] = []
        for det in chosen.values():
            rewritten.append(strip_reid_vector(det))

        for det in fr.get("tracks") or []:
            box = _det_box(det)
            if any(
                _box_iou(box, cb) >= 0.85 or _same_body(box, cb)
                for cb in claimed_boxes
            ):
                continue
            enriched = strip_reid_vector(det)
            gid = _det_gid(enriched)
            if gid in selected_group_ids:
                # Locked body for this gid exists elsewhere this frame → demote.
                if gid in chosen:
                    enriched["person_group_id"] = -1
                else:
                    # Lock missed: only demote if this box is far from the
                    # selection's last known location (handoff victim).
                    ref = last_boxes.get(gid)
                    if ref is not None and _box_iou(box, ref) < 0.15:
                        enriched["person_group_id"] = -1
                    # else keep gid so isolate still preserves the player
            rewritten.append(enriched)

        out.append({**fr, "tracks": rewritten})
    return out


def write_locked_tracks_jsonl(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for fr in frames:
            handle.write(json.dumps(fr, separators=(",", ":")) + "\n")


def resolve_tracks_jsonl(artifact_dir: Path) -> Path:
    """Prefer spatially locked tracks when Confirm has written them."""
    locked = artifact_dir / "locked_tracks.jsonl.gz"
    if locked.is_file():
        return locked
    return artifact_dir / "tracks.jsonl.gz"
