"""Play-outcome tag storage — make/miss/foul_drawn/turnover/no_shot."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.play_outcome import (
    PLAY_OUTCOME_TAGS,
    load_play_outcome,
    write_play_outcome,
)


def test_load_play_outcome_missing_file_returns_none(tmp_path: Path):
    assert load_play_outcome(tmp_path) is None


def test_write_then_load_play_outcome_roundtrips(tmp_path: Path):
    written = write_play_outcome(tmp_path, "make", note="and-1")
    assert written["tag"] == "make"
    assert written["note"] == "and-1"

    loaded = load_play_outcome(tmp_path)
    assert loaded["tag"] == "make"
    assert loaded["note"] == "and-1"
    assert loaded["set_by"] == "user"


def test_write_play_outcome_rejects_unknown_tag(tmp_path: Path):
    with pytest.raises(ValueError):
        write_play_outcome(tmp_path, "and_one")


def test_write_play_outcome_none_clears_existing_tag(tmp_path: Path):
    write_play_outcome(tmp_path, "turnover")
    assert load_play_outcome(tmp_path) is not None

    write_play_outcome(tmp_path, None)
    assert load_play_outcome(tmp_path) is None


def test_load_play_outcome_ignores_corrupt_file(tmp_path: Path):
    path = tmp_path / "play_outcome.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_play_outcome(tmp_path) is None


def test_all_tags_are_writable_and_loadable(tmp_path: Path):
    for tag in PLAY_OUTCOME_TAGS:
        sub = tmp_path / tag
        sub.mkdir()
        write_play_outcome(sub, tag)
        loaded = load_play_outcome(sub)
        assert loaded["tag"] == tag
