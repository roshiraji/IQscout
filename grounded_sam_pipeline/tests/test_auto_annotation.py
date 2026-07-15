"""Tests for auto_annotation logic (no GPU models required)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from auto_annotation.config import load_config, validate_config, PROJECT_ROOT
from auto_annotation.filtering import clip_box, filter_detections, iou, is_valid_box, nms_per_class
from auto_annotation.prompts import build_class_prompts, match_phrase_to_class
from auto_annotation.reporting import load_progress, save_progress
from auto_annotation.utils import mask_to_polygons, normalize_polygon


def test_prompt_enhancement_and_overrides():
    pm = build_class_prompts(
        ["person", "ball", "referee"],
        template="all {class_name}s",
        overrides={"ball": "the basketball", "person": "all people"},
    )
    assert pm.prompts[0] == "all people"
    assert pm.prompts[1] == "the basketball"
    assert pm.class_to_id["referee"] == 2
    assert match_phrase_to_class("the basketball", pm) == "ball"


def test_clip_and_invalid_box():
    box = clip_box([-10, -5, 200, 300], 100, 100)
    assert is_valid_box(box, 100, 100)
    assert not is_valid_box([10, 10, 10, 20], 100, 100)


def test_nms_removes_overlap():
    boxes = [[0, 0, 50, 50], [5, 5, 55, 55], [100, 100, 150, 150]]
    scores = [0.9, 0.8, 0.7]
    class_ids = [0, 0, 1]
    keep = nms_per_class(boxes, scores, class_ids, iou_threshold=0.3)
    assert 0 in keep
    assert 2 in keep
    assert len(keep) == 2


def test_filter_detections_min_area():
    boxes = [[0, 0, 5, 5], [10, 10, 60, 60]]
    scores = [0.9, 0.8]
    class_ids = [0, 0]
    names = ["person", "person"]
    fb, _, _, _ = filter_detections(
        boxes, scores, class_ids, names, 200, 200,
        min_area_px=100, min_area_frac=0.0, max_area_frac=1.0,
        box_threshold=0.3, nms_iou=0.5,
    )
    assert len(fb) == 1


def test_mask_to_polygon_and_yolo_normalize():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    polys = mask_to_polygons(mask.astype(np.uint8) * 255, epsilon=1.0, min_area=10)
    assert polys
    norm = normalize_polygon(polys[0], 100, 100)
    assert all(0.0 <= v <= 1.0 for v in norm)
    assert len(norm) >= 6


def test_config_validation(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "auto_annotation.yaml"
    if not cfg_path.is_file():
        pytest.skip("config not present")
    cfg = load_config(cfg_path)
    # Don't require input dir for this test
    cfg.images_directory = tmp_path / "missing_input"
    errors = validate_config(cfg, require_models=False, require_input=False)
    assert isinstance(errors, list)


def test_resume_progress_roundtrip(tmp_path):
    p = tmp_path / "progress.json"
    save_progress(p, {"a.jpg", "b.jpg"})
    loaded = load_progress(p)
    assert loaded == {"a.jpg", "b.jpg"}


def test_empty_detection_handling():
    fb, fs, fc, fn = filter_detections([], [], [], [], 640, 480,
        min_area_px=10, min_area_frac=0.0, max_area_frac=1.0,
        box_threshold=0.3, nms_iou=0.5)
    assert fb == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
