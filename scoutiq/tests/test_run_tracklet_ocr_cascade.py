"""Tests for cascading OCR preprocessing and call budgets."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.ocr.run_tracklet_ocr import _process_keyframe_reads


def test_process_keyframe_stops_after_first_confident_read(monkeypatch, tmp_path):
    crop_path = tmp_path / "frame_000001.jpg"
    img = np.full((200, 100, 3), 128, dtype=np.uint8)
    import cv2

    cv2.imwrite(str(crop_path), img)

    ocr_engine = MagicMock()
    calls = {"count": 0}

    def fake_read(_engine, _image, *, use_angle_cls=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return "23", 0.95
        return None, 0.0

    monkeypatch.setattr("src.ocr.run_tracklet_ocr._ocr_read", fake_read)
    monkeypatch.setattr(
        "src.ocr.run_tracklet_ocr._crop_is_ocr_viable",
        lambda *_args, **_kwargs: (True, "usable"),
    )

    reads, ocr_calls = _process_keyframe_reads(
        {"frame_idx": 1, "crop_path": str(crop_path), "opportunity_score": 0.8},
        ocr_engine=ocr_engine,
        viable_kwargs={
            "min_height": 10,
            "min_width": 10,
            "min_sharpness": 1.0,
            "min_contrast": 1.0,
        },
        min_read_confidence=0.55,
        save_debug_crops=False,
        use_angle_cls=False,
    )

    assert ocr_calls == 1
    assert len(reads) == 1
    assert reads[0]["value"] == "23"


def test_process_keyframe_runs_up_to_three_variants_when_all_fail(monkeypatch, tmp_path):
    crop_path = tmp_path / "frame_000002.jpg"
    img = np.full((200, 100, 3), 128, dtype=np.uint8)
    import cv2

    cv2.imwrite(str(crop_path), img)

    ocr_engine = MagicMock()
    monkeypatch.setattr(
        "src.ocr.run_tracklet_ocr._ocr_read",
        lambda *_args, **_kwargs: (None, 0.0),
    )
    monkeypatch.setattr(
        "src.ocr.run_tracklet_ocr._crop_is_ocr_viable",
        lambda *_args, **_kwargs: (True, "usable"),
    )

    reads, ocr_calls = _process_keyframe_reads(
        {"frame_idx": 2, "crop_path": str(crop_path), "opportunity_score": 0.8},
        ocr_engine=ocr_engine,
        viable_kwargs={
            "min_height": 10,
            "min_width": 10,
            "min_sharpness": 1.0,
            "min_contrast": 1.0,
        },
        min_read_confidence=0.55,
        save_debug_crops=False,
        use_angle_cls=False,
    )

    assert ocr_calls == 3
    assert len(reads) == 3
