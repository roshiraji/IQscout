"""Unit tests for jersey OCR early-exit helpers."""

from __future__ import annotations

from src.ocr.run_tracklet_ocr import (
    _is_confident_read,
    _max_confident_agreement,
    _should_stop_ocr_early,
)


def test_is_confident_read_requires_value_confidence_and_viability():
    assert _is_confident_read(
        {"value": "23", "conf": 0.8, "ocr_viable": True},
        min_read_confidence=0.55,
    )
    assert not _is_confident_read(
        {"value": "23", "conf": 0.4, "ocr_viable": True},
        min_read_confidence=0.55,
    )
    assert not _is_confident_read(
        {"value": None, "conf": 0.9, "ocr_viable": True},
        min_read_confidence=0.55,
    )
    assert not _is_confident_read(
        {"value": "23", "conf": 0.9, "ocr_viable": False},
        min_read_confidence=0.55,
    )


def test_max_confident_agreement_counts_identical_reads():
    reads = [
        {"value": "12", "conf": 0.9, "ocr_viable": True},
        {"value": "12", "conf": 0.85, "ocr_viable": True},
        {"value": "7", "conf": 0.95, "ocr_viable": True},
    ]
    assert _max_confident_agreement(reads, min_read_confidence=0.55) == 2


def test_should_stop_after_sufficient_agreement():
    reads = [
        {"value": "4", "conf": 0.9, "ocr_viable": True},
        {"value": "4", "conf": 0.88, "ocr_viable": True},
    ]
    stop, reason = _should_stop_ocr_early(
        reads,
        processed_frames=1,
        consecutive_frame_failures=0,
        total_ocr_calls=2,
        min_agreeing_reads=2,
        min_read_confidence=0.55,
        max_consecutive_failures=3,
        max_processed_frames=5,
        max_ocr_calls=8,
    )
    assert stop is True
    assert reason == "sufficient_agreeing_reads"


def test_should_stop_after_consecutive_failures():
    stop, reason = _should_stop_ocr_early(
        [],
        processed_frames=3,
        consecutive_frame_failures=3,
        total_ocr_calls=3,
        min_agreeing_reads=2,
        min_read_confidence=0.55,
        max_consecutive_failures=3,
        max_processed_frames=5,
        max_ocr_calls=8,
    )
    assert stop is True
    assert reason == "consecutive_frame_failures"


def test_should_stop_after_max_processed_frames():
    stop, reason = _should_stop_ocr_early(
        [],
        processed_frames=5,
        consecutive_frame_failures=1,
        total_ocr_calls=5,
        min_agreeing_reads=2,
        min_read_confidence=0.55,
        max_consecutive_failures=3,
        max_processed_frames=5,
        max_ocr_calls=8,
    )
    assert stop is True
    assert reason == "max_processed_frames"


def test_should_stop_after_max_ocr_calls():
    stop, reason = _should_stop_ocr_early(
        [],
        processed_frames=4,
        consecutive_frame_failures=1,
        total_ocr_calls=8,
        min_agreeing_reads=2,
        min_read_confidence=0.55,
        max_consecutive_failures=3,
        max_processed_frames=5,
        max_ocr_calls=8,
    )
    assert stop is True
    assert reason == "max_ocr_calls"


def test_should_continue_when_evidence_is_insufficient():
    reads = [{"value": "9", "conf": 0.9, "ocr_viable": True}]
    stop, reason = _should_stop_ocr_early(
        reads,
        processed_frames=1,
        consecutive_frame_failures=1,
        total_ocr_calls=1,
        min_agreeing_reads=2,
        min_read_confidence=0.55,
        max_consecutive_failures=3,
        max_processed_frames=5,
        max_ocr_calls=8,
    )
    assert stop is False
    assert reason is None
