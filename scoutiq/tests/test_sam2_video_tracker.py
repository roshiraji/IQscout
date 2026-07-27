"""SAM 2 video propagation guardrails (model-free logic only).

These cover the rules that decide whether a propagated mask is allowed to carry
a selected player's label — the behaviour that stopped Confirm from relabelling
a target onto the defender pressed against them.
"""

from __future__ import annotations

import numpy as np

from src.pipeline.sam2_video_tracker import (
    Sam2LockSettings,
    _Target,
    drop_duplicate_bodies,
    judge_mask,
    mask_to_geometry,
)
from src.pipeline.selection_lock import build_locked_frame_records
from src.track_players import _selection_seed_boxes


def _blob(x1: int, y1: int, x2: int, y2: int, *, size: int = 240) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _target(gid: int = 7, anchor=None) -> _Target:
    target = _Target(gid=gid, label=f"Person #{gid}", seed_box=[0, 0, 10, 20])
    if anchor is not None:
        target.anchor = np.asarray(anchor, dtype=np.float32)
        target.template = target.anchor.copy()
    return target


def test_mask_to_geometry_uses_largest_blob_only():
    mask = _blob(40, 40, 80, 140)
    # A stray fragment (ball, shadow, teammate limb) must not widen the box.
    mask[10:16, 200:206] = True
    geometry = mask_to_geometry(mask)
    assert geometry is not None
    x1, y1, x2, y2 = geometry["bbox"]
    assert (x1, y1, x2, y2) == (40.0, 40.0, 80.0, 140.0)
    assert geometry["polygon"] and len(geometry["polygon"]) >= 3


def test_mask_to_geometry_rejects_specks():
    assert mask_to_geometry(_blob(10, 10, 13, 14)) is None
    assert mask_to_geometry(np.zeros((64, 64), dtype=bool)) is None


def test_contact_with_weak_identity_is_left_unmarked():
    settings = Sam2LockSettings()
    target = _target()
    target.ref_area = 4000.0
    mine = {"bbox": [100.0, 100.0, 150.0, 220.0], "polygon": None, "area": 4000.0}
    defender = {"bbox": [105.0, 100.0, 155.0, 220.0], "polygon": None, "area": 4000.0}

    # Same evidence, no contact: the mask is accepted.
    assert judge_mask(target, mine, [], 0.40, settings) == "ok"
    # Pressed against another tracked body, a middling match is refused rather
    # than painted onto the wrong player.
    assert judge_mask(target, mine, [defender], 0.40, settings) == "contested"
    # A clear match survives contact.
    assert judge_mask(target, mine, [defender], 0.80, settings) == "ok"


def test_wrong_person_is_rejected_even_when_isolated():
    settings = Sam2LockSettings()
    target = _target()
    geometry = {"bbox": [10.0, 10.0, 60.0, 130.0], "polygon": None, "area": 4000.0}
    assert judge_mask(target, geometry, [], 0.10, settings) == "identity"


def test_fused_or_collapsed_mask_is_rejected_by_area():
    settings = Sam2LockSettings()
    target = _target()
    target.ref_area = 4000.0
    fused = {"bbox": [10.0, 10.0, 200.0, 200.0], "polygon": None, "area": 30000.0}
    sliver = {"bbox": [10.0, 10.0, 20.0, 40.0], "polygon": None, "area": 300.0}
    assert judge_mask(target, fused, [], 0.90, settings) == "shape"
    assert judge_mask(target, sliver, [], 0.90, settings) == "shape"


def test_missing_reid_falls_back_to_trusting_propagation():
    settings = Sam2LockSettings()
    geometry = {"bbox": [10.0, 10.0, 60.0, 130.0], "polygon": None, "area": 4000.0}
    assert judge_mask(_target(), geometry, [], None, settings) == "ok"


def test_two_selections_never_share_one_body():
    settings = Sam2LockSettings()
    box = [100.0, 100.0, 150.0, 220.0]
    accepted = {
        7: {5: {"bbox_xyxy": box, "reid_similarity": 0.81}},
        13: {5: {"bbox_xyxy": [102.0, 101.0, 152.0, 221.0], "reid_similarity": 0.44}},
    }
    dropped = drop_duplicate_bodies(accepted, settings)
    assert dropped == 1
    assert 5 in accepted[7]
    assert 5 not in accepted[13]


def test_distinct_bodies_are_both_kept():
    settings = Sam2LockSettings()
    accepted = {
        7: {5: {"bbox_xyxy": [100.0, 100.0, 150.0, 220.0], "reid_similarity": 0.81}},
        13: {5: {"bbox_xyxy": [400.0, 100.0, 450.0, 220.0], "reid_similarity": 0.79}},
    }
    assert drop_duplicate_bodies(accepted, settings) == 0
    assert 5 in accepted[7] and 5 in accepted[13]


def test_tight_sam2_mask_absorbs_the_loose_detector_box():
    # SAM 2 reports the silhouette, the detector reports a looser person box.
    # If the detector copy survives it becomes an "erase this person" target and
    # isolate cuts the selected player's limbs out of the frame.
    frames = [
        {
            "frame_number": 12,
            "tracks": [
                {"track_id": 7, "person_group_id": 7, "bbox_xyxy": [100, 100, 160, 260]},
                {"track_id": 9, "person_group_id": 9, "bbox_xyxy": [600, 100, 660, 260]},
            ],
        }
    ]
    locked = {
        12: {
            7: {
                "track_id": 7,
                "person_group_id": 7,
                "bbox_xyxy": [106, 108, 154, 252],
                "selection_locked": True,
            }
        }
    }
    records = build_locked_frame_records(frames, locked, {7})
    tracks = records[0]["tracks"]

    assert [det["person_group_id"] for det in tracks].count(7) == 1
    assert not any(det["person_group_id"] == -1 for det in tracks)
    assert any(det["person_group_id"] == 9 for det in tracks)


def test_touching_opponent_is_not_absorbed():
    frames = [
        {
            "frame_number": 12,
            "tracks": [
                {"track_id": 9, "person_group_id": 9, "bbox_xyxy": [140, 100, 200, 260]}
            ],
        }
    ]
    locked = {
        12: {
            7: {
                "track_id": 7,
                "person_group_id": 7,
                "bbox_xyxy": [100, 100, 160, 260],
                "selection_locked": True,
            }
        }
    }
    records = build_locked_frame_records(frames, locked, {7})
    assert any(det["person_group_id"] == 9 for det in records[0]["tracks"])


def test_seed_boxes_prefer_the_clicked_box():
    frame_records = [
        {
            "frame_number": 58,
            "tracks": [
                {"person_group_id": 7, "bbox_xyxy": [0, 0, 10, 20]},
                {"person_group_id": 13, "bbox_xyxy": [500, 300, 560, 460]},
            ],
        }
    ]
    selected_players = [
        {"person_group_id": 7, "x1": 100, "y1": 200, "x2": 160, "y2": 360}
    ]
    seeds = _selection_seed_boxes(selected_players, {7, 13}, frame_records, 58)

    # Clicked box wins for #7; #13 falls back to its snapshot detection.
    assert seeds[7] == [100.0, 200.0, 160.0, 360.0]
    assert seeds[13] == [500.0, 300.0, 560.0, 460.0]
