"""Offense/Defense grade blending with an optional tagged play outcome."""

from __future__ import annotations

import math

import pytest

from src.pipeline.coach_insights import (
    OUTCOME_DEFENSE_ADJUST,
    OUTCOME_OFFENSE_POINTS,
    OUTCOME_WEIGHT,
    PROCESS_WEIGHT,
    compute_pairwise_metrics,
)

FRAME_WIDTH = 1000
FRAME_HEIGHT = 500
DIAG = math.hypot(FRAME_WIDTH, FRAME_HEIGHT)


def _constant_gap_series(n_frames: int, gap_px: float):
    """Two players holding a fixed on-screen gap for n_frames at 30fps."""
    series_a = []
    series_b = []
    for i in range(n_frames):
        t = i / 30.0
        series_a.append(
            {
                "frame_number": i,
                "timestamp_seconds": t,
                "stance_x": 100.0,
                "stance_y": 100.0,
            }
        )
        series_b.append(
            {
                "frame_number": i,
                "timestamp_seconds": t,
                "stance_x": 100.0 + gap_px,
                "stance_y": 100.0,
            }
        )
    return series_a, series_b


def _pairwise(tag=None, note=None):
    series_a, series_b = _constant_gap_series(20, gap_px=50.0)
    play_outcome = {"tag": tag, "note": note} if tag else None
    return compute_pairwise_metrics(
        series_a,
        series_b,
        gid_a=1,
        gid_b=2,
        meta_a={"team_id": 0, "player_label": "Defender"},
        meta_b={"team_id": 1, "player_label": "Attacker"},
        source_frame_count=20,
        frame_diagonal=DIAG,
        possession=None,
        play_outcome=play_outcome,
    )


def test_process_only_grades_when_no_outcome_tagged():
    pair = _pairwise(tag=None)
    scorecard = pair["matchup_scorecard"]
    assert scorecard["grading_mode"] == "process_only"
    assert scorecard["play_outcome_tag"] is None
    # Grades fall back exactly to the legacy process-only proxies.
    assert scorecard["offense_grade"] == scorecard["create_space_score"]
    assert scorecard["defense_grade"] == scorecard["stay_close_score"]


def test_make_tag_boosts_offense_and_trims_defense():
    baseline = _pairwise(tag=None)["matchup_scorecard"]
    tagged = _pairwise(tag="make")["matchup_scorecard"]

    assert tagged["grading_mode"] == "outcome_plus_process"
    assert tagged["play_outcome_tag"] == "make"

    expected_offense = round(
        OUTCOME_WEIGHT * (OUTCOME_OFFENSE_POINTS["make"] / 100.0)
        + PROCESS_WEIGHT * baseline["create_space_score"],
        4,
    )
    expected_defense = round(
        min(1.0, baseline["stay_close_score"] + OUTCOME_DEFENSE_ADJUST["make"] / 100.0),
        4,
    )
    assert tagged["offense_grade"] == pytest.approx(expected_offense, abs=1e-4)
    assert tagged["defense_grade"] == pytest.approx(expected_defense, abs=1e-4)
    # A make should out-score the untagged process-only offense estimate —
    # the whole point is that scoring counts even against tight coverage.
    assert tagged["offense_grade"] > baseline["offense_grade"]
    # Defense grade should still be respectable — a make doesn't erase a
    # good contest, it only trims a little off the process score.
    assert tagged["defense_grade"] >= baseline["defense_grade"] - 0.10


def test_turnover_tag_tanks_offense_and_credits_defense():
    baseline = _pairwise(tag=None)["matchup_scorecard"]
    make = _pairwise(tag="make")["matchup_scorecard"]
    turnover = _pairwise(tag="turnover")["matchup_scorecard"]

    # A turnover should grade far worse for the offense than a make on the
    # exact same underlying separation data — the result, not just the
    # process, drives the offense grade once it's tagged.
    assert turnover["offense_grade"] < make["offense_grade"]
    assert turnover["defense_grade"] > baseline["defense_grade"]
    assert turnover["defense_grade"] == pytest.approx(
        min(1.0, baseline["stay_close_score"] + OUTCOME_DEFENSE_ADJUST["turnover"] / 100.0),
        abs=1e-4,
    )


def test_headline_reflects_result_and_defense_quality():
    make = _pairwise(tag="make")["matchup_scorecard"]
    assert "scored" in make["headline"]

    turnover = _pairwise(tag="turnover")["matchup_scorecard"]
    assert "turned it over" in turnover["headline"]

    untagged = _pairwise(tag=None)["matchup_scorecard"]
    assert "no play result tagged" in untagged["headline"].lower() or (
        "estimate" in untagged["headline"].lower()
    )


def test_same_team_pair_still_reports_off_ball_score_unaffected_by_tag():
    series_a, series_b = _constant_gap_series(20, gap_px=50.0)
    pair = compute_pairwise_metrics(
        series_a,
        series_b,
        gid_a=1,
        gid_b=2,
        meta_a={"team_id": 0, "player_label": "Player A"},
        meta_b={"team_id": 0, "player_label": "Player B"},
        source_frame_count=20,
        frame_diagonal=DIAG,
        possession=None,
        play_outcome={"tag": "make"},
    )
    scorecard = pair["matchup_scorecard"]
    assert scorecard["mode"] == "offense_spacing"
    assert scorecard["off_ball_score"] is not None
