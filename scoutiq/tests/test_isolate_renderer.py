"""Tests for the isolate-renderer alignment/blend/feather improvements.

These target the pure, model-free pieces of ``isolate_renderer`` — the
homography sanity check, wider neighbor sampling, outward edge feather, and
the median-vs-mean multi-neighbor blend inside ``_reconstruct_frame``. Real
video/LaMa paths are covered by manual smoke tests (GPU-only).
"""

from __future__ import annotations

import gzip
import json

import cv2
import numpy as np
import pytest

from src.pipeline import isolate_renderer as ir


def _solid_frame(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_neighbor_offsets_narrow_matches_base_four():
    offsets = ir._neighbor_offsets(radius=12, half=6, wide=False)
    assert set(offsets) == {-12, -6, 6, 12}


def test_neighbor_offsets_wide_has_more_candidates_within_radius():
    narrow = ir._neighbor_offsets(radius=12, half=6, wide=False)
    wide = ir._neighbor_offsets(radius=12, half=6, wide=True)
    assert len(wide) > len(narrow)
    assert set(narrow).issubset(set(wide))
    assert all(0 < abs(o) <= 12 for o in wide)
    # No duplicates, deterministic ordering.
    assert len(wide) == len(set(wide))
    assert list(wide) == sorted(wide)


def test_transform_is_sane_accepts_identity():
    identity = np.eye(3, dtype=np.float64)
    assert ir._transform_is_sane(identity, width=640, height=360)


def test_transform_is_sane_rejects_wild_extrapolation():
    # A small translation is fine at any frame size...
    translate = np.array(
        [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    assert ir._transform_is_sane(translate, width=640, height=360)

    # ...but a homography whose perspective term blows up the far corners
    # should be rejected even though it might fit a local patch of matches.
    wild = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.004, 0.0, 1.0]], dtype=np.float64
    )
    assert not ir._transform_is_sane(wild, width=640, height=360)


def test_feather_seam_smooths_boundary_band_leaves_far_pixels_alone():
    height, width = 60, 60
    original = _solid_frame(height, width, (200, 200, 200))  # real untouched frame
    erase = np.zeros((height, width), dtype=np.uint8)
    erase[20:40, 20:40] = 255  # a square hole in the middle
    # A hard-composite output matches ``original`` everywhere except inside
    # the hole (that invariant is what real callers always pass in).
    filled = original.copy()
    filled[erase > 0] = (10, 10, 10)

    out = ir._feather_seam(filled, erase, feather_px=3)

    # Deep inside the hole (far from the edge): unaffected by the local blur.
    assert tuple(int(v) for v in out[30, 30]) == (10, 10, 10)
    # Far outside the hole (beyond the feather band): untouched original.
    assert tuple(int(v) for v in out[5, 5]) == (200, 200, 200)
    # Right at/near the true boundary: blurred toward a value strictly
    # between the two solid colors — this is what actually hides the seam.
    for probe in (out[19, 30], out[20, 30]):
        value = int(probe[0])
        assert 10 < value < 200


def test_feather_seam_never_touches_keep_region():
    height, width = 60, 60
    original = _solid_frame(height, width, (200, 200, 200))
    erase = np.zeros((height, width), dtype=np.uint8)
    erase[20:40, 20:40] = 255
    keep = np.zeros((height, width), dtype=np.uint8)
    keep[18:22, 18:22] = 255  # selected player overlapping right at the edge
    filled = original.copy()
    filled[erase > 0] = (10, 10, 10)
    filled[keep > 0] = (50, 60, 70)  # the selected player's real pixels

    out = ir._feather_seam(filled, erase, feather_px=3, keep_mask=keep)
    assert tuple(int(v) for v in out[19, 19]) == (50, 60, 70)


def test_feather_seam_noop_when_feather_px_is_zero():
    height, width = 20, 20
    filled = _solid_frame(height, width, (10, 10, 10))
    erase = np.zeros((height, width), dtype=np.uint8)
    erase[5:15, 5:15] = 255
    out = ir._feather_seam(filled, erase, feather_px=0)
    assert np.array_equal(out, filled)


def _make_target(height: int, width: int, erase_box: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = erase_box
    frame = _solid_frame(height, width, (0, 0, 0))  # the erased player's real pixels
    erase = np.zeros((height, width), dtype=np.uint8)
    erase[y0:y1, x0:x1] = 255
    occupancy = erase.copy()
    keep = np.zeros((height, width), dtype=np.uint8)
    return {
        "frame": frame,
        "occupancy": occupancy,
        "keep": keep,
        "erase": erase,
        "keypoints": [],
        "descriptors": None,
    }


def _make_neighbor(height: int, width: int, bgr: tuple[int, int, int]) -> dict:
    return {
        "frame": _solid_frame(height, width, bgr),
        "occupancy": np.zeros((height, width), dtype=np.uint8),
        "keep": np.zeros((height, width), dtype=np.uint8),
        "erase": np.zeros((height, width), dtype=np.uint8),
        "keypoints": [],
        "descriptors": None,
    }


def test_reconstruct_frame_median_blend_rejects_single_outlier_neighbor(monkeypatch):
    """Median across neighbors should ignore one bad sample; a mean would not."""
    height, width = 40, 40
    target = _make_target(height, width, erase_box=(10, 10, 30, 30))

    # Two neighbors agree at (100,100,100); one outlier (from e.g. a
    # misaligned crowd frame) sits far off at (250,250,250).
    neighbors = [
        _make_neighbor(height, width, (100, 100, 100)),
        _make_neighbor(height, width, (100, 100, 100)),
        _make_neighbor(height, width, (250, 250, 250)),
    ]

    monkeypatch.setattr(
        ir, "_neighbor_to_target_matrix", lambda *a, **k: np.eye(3, dtype=np.float64)
    )

    out, leftover = ir._reconstruct_frame(
        target,
        neighbors,
        width,
        height,
        use_spatial_fallback=False,
        min_temporal_samples=1,
        hard_composite=True,
    )

    pixel = out[20, 20]
    # Median of [100,100,250] is 100 — the outlier is outvoted, not averaged in.
    assert abs(int(pixel[0]) - 100) <= 1
    mean_value = (100 + 100 + 250) / 3
    assert abs(int(pixel[0]) - mean_value) > 20
    assert not leftover.any()


def test_reconstruct_frame_leftover_when_below_min_temporal_samples():
    height, width = 40, 40
    target = _make_target(height, width, erase_box=(10, 10, 30, 30))
    out, leftover = ir._reconstruct_frame(
        target,
        [],  # no neighbors align at all
        width,
        height,
        use_spatial_fallback=False,
        min_temporal_samples=1,
        hard_composite=True,
    )
    # Nothing to fill from temporal neighbors and no spatial fallback —
    # every erase pixel should be reported as leftover for LaMa to own.
    assert leftover[20, 20] == 255
    assert leftover.sum() == int((target["erase"] > 0).sum()) * 255


def test_reconstruct_frame_only_touches_erase_bbox_region(monkeypatch):
    """Unrelated pixels far from the hole must be untouched (bbox-cropped math)."""
    height, width = 100, 100
    target = _make_target(height, width, erase_box=(10, 10, 20, 20))
    neighbors = [_make_neighbor(height, width, (77, 77, 77))]
    monkeypatch.setattr(
        ir, "_neighbor_to_target_matrix", lambda *a, **k: np.eye(3, dtype=np.float64)
    )
    out, _leftover = ir._reconstruct_frame(
        target,
        neighbors,
        width,
        height,
        use_spatial_fallback=False,
        min_temporal_samples=1,
        hard_composite=True,
    )
    # Far corner, well outside the erase bbox, must equal the original frame.
    assert tuple(out[90, 90]) == (0, 0, 0)


def _write_synthetic_job(job_dir, *, width=220, height=120, n_frames=20, fps=10.0) -> None:
    """A tiny real video + tracks.jsonl.gz to exercise the full render path.

    Static, richly-textured background (so ORB actually finds keypoints) with
    a moving "non-selected" player to erase and a static "selected" player to
    keep — close to a real clip's temporal-fill scenario without needing a
    real tracking job.
    """
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    video_path = job_dir / "source.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()

    records = []
    keep_box = [160, 10, 190, 35]
    for i in range(n_frames):
        frame = base.copy()
        erase_box = [20 + 5 * i, 30, 45 + 5 * i, 65]
        cv2.rectangle(
            frame, (erase_box[0], erase_box[1]), (erase_box[2], erase_box[3]), (0, 0, 255), -1
        )
        cv2.rectangle(
            frame, (keep_box[0], keep_box[1]), (keep_box[2], keep_box[3]), (0, 255, 0), -1
        )
        writer.write(frame)
        records.append(
            {
                "tracks": [
                    {"person_group_id": 1, "bbox_xyxy": keep_box},
                    {"person_group_id": 2, "bbox_xyxy": erase_box},
                ]
            }
        )
    writer.release()

    with gzip.open(job_dir / "tracks.jsonl.gz", "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    (job_dir / "summary.json").write_text(
        json.dumps({"video_path": str(video_path), "source_frame_count": n_frames}),
        encoding="utf-8",
    )


def test_render_temporal_isolate_end_to_end_on_synthetic_clip(tmp_path):
    """Full render path (real video I/O + ORB + homography) end-to-end.

    CPU-only (temporal mode, no LaMa/GPU needed) so this runs anywhere, and
    catches real runtime bugs the pure unit tests above can't (broadcasting,
    dtype, codec, gzip I/O, etc.).
    """
    _write_synthetic_job(tmp_path)

    output = ir.render_temporal_isolate(
        tmp_path,
        selected_group_ids={1},
        generation="smoke",
        isolate_cfg={"write_quality_metrics": True, "warp_workers": 2},
        quiet=True,
    )

    assert output.is_file()
    assert output.stat().st_size > 0

    quality_path = tmp_path / "isolate_quality.json"
    assert quality_path.is_file()
    payload = json.loads(quality_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "temporal"
    assert payload["generation"] == "smoke"
    assert payload["erase_pixels_total"] > 0
    # The non-selected player moves every frame, so at least some neighbors
    # should align cleanly and resolve part of the hole without the NS/Telea
    # spatial fallback.
    assert payload["hole_resolved_without_fallback_fraction"] is not None


def test_hybrid_full_erase_skips_temporal_and_uses_fast_path(tmp_path, monkeypatch):
    """Default hybrid + lama_full_erase must not burn CPU on a discarded temporal seed."""
    _write_synthetic_job(tmp_path, n_frames=4)
    called = {"pipelined": 0, "reconstruct": 0}

    def fake_pipelined(*args, **kwargs):
        called["pipelined"] += 1
        out = tmp_path / "focused_isolated.mp4"
        out.write_bytes(b"fake")
        return out

    def boom_reconstruct(*args, **kwargs):
        called["reconstruct"] += 1
        raise AssertionError("temporal reconstruct should be skipped")

    monkeypatch.setattr(ir, "_render_lama_pipelined", fake_pipelined)
    monkeypatch.setattr(ir, "_reconstruct_frame", boom_reconstruct)

    result = ir.render_hybrid_isolate(
        tmp_path,
        selected_group_ids={1},
        generation="fast",
        isolate_cfg={
            "lama_full_erase": True,
            "skip_temporal_when_full_erase": True,
        },
        quiet=True,
    )
    assert called["pipelined"] == 1
    assert called["reconstruct"] == 0
    assert result.name == "focused_isolated.mp4"


def test_hybrid_keeps_temporal_when_full_erase_disabled(tmp_path, monkeypatch):
    _write_synthetic_job(tmp_path, n_frames=4)
    called = {"pipelined": 0}

    def fake_pipelined(*args, **kwargs):
        called["pipelined"] += 1
        return tmp_path / "focused_isolated.mp4"

    monkeypatch.setattr(ir, "_render_lama_pipelined", fake_pipelined)

    # Without GPU LaMa this would fail further down — stub the leftover path
    # by forcing skip-off and catching the LaMa load.
    def fake_lama(cfg):
        class _Stub:
            def inpaint_bgr(self, image, mask):
                return image

        return _Stub()

    monkeypatch.setattr(ir, "_make_lama", fake_lama)

    result = ir.render_hybrid_isolate(
        tmp_path,
        selected_group_ids={1},
        generation="temporal",
        isolate_cfg={
            "lama_full_erase": False,
            "skip_temporal_when_full_erase": True,
            "write_quality_metrics": True,
            "warp_workers": 2,
            "wide_neighbor_search": False,
        },
        quiet=True,
    )
    assert called["pipelined"] == 0
    assert result.is_file()


def test_auto_ai_working_side_speed_smaller_than_quality():
    from src.pipeline.hardware import _auto_ai_working_side

    hw = {"vram_gb": 8.0, "cpu_count": 24, "cuda": True}
    speed = _auto_ai_working_side(hw, pull="hard", prefer="speed")
    quality = _auto_ai_working_side(hw, pull="hard", prefer="quality")
    assert speed is not None and quality is not None
    assert speed < quality
    assert speed == 640  # 8GB hard/speed default


@pytest.mark.parametrize("wide", [True, False])
def test_quality_accumulator_reports_resolved_fraction(wide):
    quality = ir._QualityAccumulator()
    erase = np.zeros((10, 10), dtype=np.uint8)
    erase[0:10, 0:5] = 255  # 50 erase pixels
    leftover = np.zeros((10, 10), dtype=np.uint8)
    leftover[0:5, 0:5] = 255  # 25 left unresolved
    quality.record_frame(erase, leftover)
    payload = quality.as_dict(mode="temporal", cfg={**ir.DEFAULT_ISOLATE_CFG, "wide_neighbor_search": wide})
    assert payload["erase_pixels_total"] == 50
    assert payload["leftover_pixels_total"] == 25
    assert payload["hole_resolved_without_fallback_fraction"] == pytest.approx(0.5)
    assert payload["settings"]["wide_neighbor_search"] is wide
