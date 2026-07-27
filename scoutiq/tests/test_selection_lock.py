"""Spatial selection lock resists ByteTrack ID handoffs."""

from __future__ import annotations

import numpy as np

from src.pipeline.selection_lock import lock_selected_bodies


def _det(tid: int, gid: int, box: list[int], conf: float = 0.9) -> dict:
    return {
        "track_id": tid,
        "person_group_id": gid,
        "bbox_xyxy": box,
        "confidence": conf,
        "mask_polygon": None,
    }


def test_lock_follows_body_not_swapped_group_id():
    # Snapshot: user picks red body on the right as Person #6.
    # Later frames: group id 6 jumps to a left blue body (handoff), while the
    # true red body keeps moving right under a different track/group id.
    frames = []
    # frame 0 (snapshot): red right = gid 6, blue left = gid 11
    frames.append(
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "tracks": [
                _det(11, 11, [200, 300, 260, 420]),
                _det(6, 6, [900, 350, 980, 520]),
            ],
        }
    )
    # frame 1: both move slightly, IDs still correct
    frames.append(
        {
            "frame_number": 1,
            "timestamp_seconds": 0.03,
            "tracks": [
                _det(11, 11, [210, 300, 270, 420]),
                _det(6, 6, [920, 350, 1000, 520]),
            ],
        }
    )
    # frame 2: HANDOFF — raw gid 6 is now on the left blue body; red body is gid 99
    frames.append(
        {
            "frame_number": 2,
            "timestamp_seconds": 0.06,
            "tracks": [
                _det(6, 6, [220, 300, 280, 420]),  # wrong body carrying gid 6
                _det(99, 99, [940, 350, 1020, 520]),  # true selected body
            ],
        }
    )

    selected_players = [
        {
            "person_group_id": 6,
            "label": "Person #6 · Red",
            "jersey_color_name": "red",
            "x1": 900,
            "y1": 350,
            "x2": 980,
            "y2": 520,
        },
        {
            "person_group_id": 11,
            "label": "Person #11 · Blue",
            "jersey_color_name": "blue",
            "x1": 200,
            "y1": 300,
            "x2": 260,
            "y2": 420,
        },
    ]

    locked = lock_selected_bodies(
        frames,
        snapshot_frame_number=0,
        selected_group_ids={6, 11},
        selected_players=selected_players,
        label_by_group_id={6: "Person #6 · Red", 11: "Person #11 · Blue"},
        team_meta_by_group={
            6: {"jersey_color_name": "red"},
            11: {"jersey_color_name": "blue"},
            99: {"jersey_color_name": "red"},
        },
        frame_width=1280,
        frame_height=720,
    )

    assert 2 in locked
    assert 6 in locked[2]
    # Must stay on the right-side body, not the hijacked left detection.
    box = locked[2][6]["bbox_xyxy"]
    cx = 0.5 * (box[0] + box[2])
    assert cx > 700, f"handoff: locked Person #6 drifted left to cx={cx}"
    assert locked[2][6]["person_group_id"] == 6


def test_lock_resists_close_occlusion_gid_swap():
    """Two nearby players swap ByteTrack/group IDs — lock must stay spatial."""
    frames = [
        {
            "frame_number": 0,
            "tracks": [
                _det(1, 1, [500, 300, 560, 420]),  # selected A
                _det(2, 2, [580, 300, 640, 420]),  # selected B (close)
            ],
        },
        {
            "frame_number": 1,
            "tracks": [
                _det(1, 1, [505, 300, 565, 420]),
                _det(2, 2, [585, 305, 645, 425]),
            ],
        },
        {
            # Swap: gid 1 now on B's trajectory, gid 2 on A's.
            "frame_number": 2,
            "tracks": [
                _det(2, 2, [510, 300, 570, 420]),
                _det(1, 1, [590, 305, 650, 425]),
            ],
        },
    ]
    selected_players = [
        {
            "person_group_id": 1,
            "label": "A",
            "jersey_color_name": "red",
            "x1": 500,
            "y1": 300,
            "x2": 560,
            "y2": 420,
        },
        {
            "person_group_id": 2,
            "label": "B",
            "jersey_color_name": "blue",
            "x1": 580,
            "y1": 300,
            "x2": 640,
            "y2": 420,
        },
    ]
    locked = lock_selected_bodies(
        frames,
        snapshot_frame_number=0,
        selected_group_ids={1, 2},
        selected_players=selected_players,
        label_by_group_id={1: "A", 2: "B"},
        team_meta_by_group={
            1: {"jersey_color_name": "red"},
            2: {"jersey_color_name": "blue"},
        },
        frame_width=1280,
        frame_height=720,
    )
    assert 1 in locked[2] and 2 in locked[2]
    a_cx = 0.5 * (locked[2][1]["bbox_xyxy"][0] + locked[2][1]["bbox_xyxy"][2])
    b_cx = 0.5 * (locked[2][2]["bbox_xyxy"][0] + locked[2][2]["bbox_xyxy"][2])
    assert a_cx < b_cx, f"occlusion swap: A={a_cx} should stay left of B={b_cx}"


def test_seed_uses_click_boxes_not_gid_swap_on_adjacent():
    """Adjacent red/blue picks must seed on the clicked bodies, not swap IDs."""
    frames = [
        {
            "frame_number": 10,
            "tracks": [
                _det(8, 8, [400, 300, 480, 450]),   # blue body (gid 8)
                _det(14, 14, [300, 280, 380, 440]),  # red body (gid 14)
                _det(99, 99, [900, 200, 980, 360]),  # distant stranger
            ],
        },
        {
            "frame_number": 0,
            # At t=0 those pixels are different people — backward must not
            # latch onto them just because jersey color matches.
            "tracks": [
                _det(8, 8, [900, 200, 980, 360]),
                _det(14, 14, [850, 220, 930, 380]),
                _det(50, 50, [310, 285, 390, 445]),
                _det(51, 51, [410, 305, 490, 455]),
            ],
        },
    ]
    selected_players = [
        {
            "person_group_id": 14,
            "label": "Red ball-handler",
            "jersey_color_name": "red",
            "x1": 300,
            "y1": 280,
            "x2": 380,
            "y2": 440,
        },
        {
            "person_group_id": 8,
            "label": "Blue defender",
            "jersey_color_name": "blue",
            "x1": 400,
            "y1": 300,
            "x2": 480,
            "y2": 450,
        },
    ]
    locked = lock_selected_bodies(
        frames,
        snapshot_frame_number=10,
        selected_group_ids={8, 14},
        selected_players=selected_players,
        label_by_group_id={8: "Blue defender", 14: "Red ball-handler"},
        team_meta_by_group={
            8: {"jersey_color_name": "blue"},
            14: {"jersey_color_name": "red"},
            50: {"jersey_color_name": "red"},
            51: {"jersey_color_name": "blue"},
        },
        frame_width=1280,
        frame_height=720,
    )
    # Snapshot seed must be on the clicked adjacent pair (left red, right blue).
    assert 10 in locked
    red_cx = 0.5 * (
        locked[10][14]["bbox_xyxy"][0] + locked[10][14]["bbox_xyxy"][2]
    )
    blue_cx = 0.5 * (
        locked[10][8]["bbox_xyxy"][0] + locked[10][8]["bbox_xyxy"][2]
    )
    assert red_cx < 400, f"seed red drifted right cx={red_cx}"
    assert blue_cx > 350, f"seed blue drifted left cx={blue_cx}"
    assert red_cx < blue_cx
    # Frame 0: either empty (stop-on-miss) or still near the snapshot side
    # of the court — not the distant gid carriers at x~900.
    if 14 in locked.get(0, {}):
        cx0 = 0.5 * (
            locked[0][14]["bbox_xyxy"][0] + locked[0][14]["bbox_xyxy"][2]
        )
        assert cx0 < 600, f"backward latched onto distant stranger cx={cx0}"


def test_close_defense_keeps_jersey_not_swap_or_ref():
    """On-ball defense: overlapping boxes + a nearby ref must not steal locks.

    Jersey colors are stamped on the *body* (appearance), not derived from the
    handed-off group id — matching what Confirm does after video enrich.
    """
    def det_with_color(tid, gid, box, color, role="player", conf=0.9):
        d = _det(tid, gid, box, conf)
        d["jersey_color_name"] = color
        d["role"] = role
        return d

    frames = [
        {
            "frame_number": 0,
            "tracks": [
                det_with_color(14, 14, [300, 280, 380, 440], "red"),
                det_with_color(30, 30, [360, 290, 440, 450], "blue"),
                det_with_color(208, 208, [700, 500, 780, 640], "official", "official"),
            ],
        },
        {
            "frame_number": 1,
            "tracks": [
                # IDs swapped; appearance still matches the body positions.
                det_with_color(30, 30, [305, 285, 385, 445], "red"),
                det_with_color(14, 14, [350, 295, 430, 455], "blue"),
                det_with_color(208, 208, [690, 490, 770, 630], "official", "official"),
                # Fused mega-box temptation
                det_with_color(99, 99, [300, 280, 440, 455], "red"),
            ],
        },
        {
            "frame_number": 2,
            "tracks": [
                det_with_color(14, 14, [310, 288, 390, 448], "red"),
                det_with_color(30, 30, [355, 298, 435, 458], "blue"),
                det_with_color(208, 208, [680, 480, 760, 620], "official", "official"),
            ],
        },
    ]
    selected_players = [
        {
            "person_group_id": 14,
            "label": "Offense",
            "jersey_color_name": "red",
            "x1": 300,
            "y1": 280,
            "x2": 380,
            "y2": 440,
        },
        {
            "person_group_id": 30,
            "label": "Defense",
            "jersey_color_name": "blue",
            "x1": 360,
            "y1": 290,
            "x2": 440,
            "y2": 450,
        },
    ]
    locked = lock_selected_bodies(
        frames,
        snapshot_frame_number=0,
        selected_group_ids={14, 30},
        selected_players=selected_players,
        label_by_group_id={14: "Offense", 30: "Defense"},
        team_meta_by_group={
            14: {"jersey_color_name": "red", "role": "player"},
            30: {"jersey_color_name": "blue", "role": "player"},
            208: {"jersey_color_name": "official", "role": "official"},
            99: {"jersey_color_name": "red", "role": "player"},
        },
        frame_width=1280,
        frame_height=720,
    )
    for fid in (1, 2):
        assert 14 in locked[fid] and 30 in locked[fid]
        red_box = locked[fid][14]["bbox_xyxy"]
        blue_box = locked[fid][30]["bbox_xyxy"]
        red_cx = 0.5 * (red_box[0] + red_box[2])
        blue_cx = 0.5 * (blue_box[0] + blue_box[2])
        assert red_cx < 500, f"frame {fid}: red lock jumped to cx={red_cx}"
        assert blue_cx < 550, f"frame {fid}: blue lock jumped to cx={blue_cx}"
        assert red_cx < blue_cx + 80
        assert locked[fid][14]["track_id"] != 208
        assert locked[fid][30]["track_id"] != 208
        assert locked[fid][14].get("jersey_color_name") == "red"
        assert locked[fid][30].get("jersey_color_name") == "blue"


def test_deep_reid_keeps_targets_when_ids_swap_in_contact():
    """OSNet vectors, not ByteTrack IDs, decide an on-ball contact frame."""
    def with_reid(
        tid: int,
        gid: int,
        box: list[int],
        vector: list[float],
    ) -> dict:
        det = _det(tid, gid, box)
        det["_reid_vector"] = np.asarray(vector, dtype=np.float32)
        return det

    frames = [
        {
            "frame_number": 0,
            "tracks": [
                with_reid(14, 14, [300, 280, 380, 440], [1.0, 0.0, 0.0]),
                with_reid(8, 8, [360, 290, 440, 450], [0.0, 1.0, 0.0]),
            ],
        },
        {
            "frame_number": 1,
            "tracks": [
                # Raw IDs have exchanged bodies, but ReID features have not.
                with_reid(8, 8, [305, 285, 385, 445], [1.0, 0.0, 0.0]),
                with_reid(14, 14, [355, 295, 435, 455], [0.0, 1.0, 0.0]),
            ],
        },
    ]
    selected_players = [
        {
            "person_group_id": 14,
            "label": "Offense",
            "jersey_color_name": "red",
            "x1": 300,
            "y1": 280,
            "x2": 380,
            "y2": 440,
        },
        {
            "person_group_id": 8,
            "label": "Defense",
            "jersey_color_name": "blue",
            "x1": 360,
            "y1": 290,
            "x2": 440,
            "y2": 450,
        },
    ]
    locked = lock_selected_bodies(
        frames,
        snapshot_frame_number=0,
        selected_group_ids={8, 14},
        selected_players=selected_players,
        label_by_group_id={14: "Offense", 8: "Defense"},
        team_meta_by_group={},
        frame_width=1280,
        frame_height=720,
    )
    assert locked[1][14]["track_id"] == 8
    assert locked[1][8]["track_id"] == 14
    assert locked[1][14]["reid_similarity"] == 1.0
    assert locked[1][8]["reid_similarity"] == 1.0


def test_miss_does_not_demote_nearby_selected_body():
    from src.pipeline.selection_lock import build_locked_frame_records

    frames = [
        {
            "frame_number": 0,
            "tracks": [_det(5, 5, [400, 300, 460, 420])],
        },
        {
            "frame_number": 1,
            "tracks": [_det(5, 5, [405, 300, 465, 420])],
        },
    ]
    # Lock only on frame 0; frame 1 "missed".
    locked = {
        0: {
            5: {
                **_det(5, 5, [400, 300, 460, 420]),
                "person_group_id": 5,
                "selection_locked": True,
            }
        },
        1: {},
    }
    rewritten = build_locked_frame_records(frames, locked, {5})
    # Nearby continuation of selected body must keep gid for isolate keep.
    f1 = rewritten[1]["tracks"]
    assert any(int(t.get("person_group_id", -1)) == 5 for t in f1)
