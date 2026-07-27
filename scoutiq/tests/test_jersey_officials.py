"""Official vs player jersey / court-prior classification."""

from __future__ import annotations

from src.pipeline.jersey_team_colors import _court_official_prior, _is_official


def _sig(**kwargs) -> dict[str, float]:
    base = {
        "red_frac": 0.0,
        "dark_frac": 0.0,
        "blue_frac": 0.0,
        "gray_frac": 0.0,
        "white_frac": 0.0,
        "stripe": 0.0,
        "axis": 0.0,
        "mean_v": 0.5,
        "mean_s": 0.4,
        "keep_frac": 1.0,
    }
    base.update(kwargs)
    return base


def test_strong_red_player_never_official_even_with_stripe_noise():
    sig = _sig(red_frac=0.55, stripe=0.6, gray_frac=0.2, white_frac=0.1, axis=0.7)
    assert _is_official(sig, court_prior=0.4) is False


def test_zebra_sideline_ref_is_official():
    sig = _sig(
        gray_frac=0.55,
        white_frac=0.2,
        stripe=0.7,
        red_frac=0.02,
        blue_frac=0.02,
        dark_frac=0.2,
        mean_s=0.12,
        axis=-0.1,
    )
    assert _is_official(sig, court_prior=0.4) is True


def test_navy_player_on_court_not_official():
    sig = _sig(
        blue_frac=0.7,
        dark_frac=0.25,
        red_frac=0.02,
        mean_s=0.45,
        axis=-0.9,
        stripe=0.3,
    )
    assert _is_official(sig, court_prior=-0.2) is False


def test_black_sideline_ref_without_stripes():
    sig = _sig(
        dark_frac=0.55,
        gray_frac=0.2,
        red_frac=0.02,
        blue_frac=0.05,
        mean_s=0.18,
        stripe=0.2,
        axis=-0.2,
    )
    assert _is_official(sig, court_prior=0.45) is True


def test_weak_stripe_on_court_not_enough_for_official():
    sig = _sig(
        stripe=0.48,
        gray_frac=0.2,
        white_frac=0.1,
        red_frac=0.05,
        blue_frac=0.08,
        mean_s=0.25,
    )
    assert _is_official(sig, court_prior=-0.3) is False


def test_court_prior_sideline_vs_upcourt():
    # feet near bottom → official bias
    assert _court_official_prior([600, 500, 700, 700], 1280, 720) > 0.2
    # mid-court wood → player bias
    assert _court_official_prior([600, 200, 700, 400], 1280, 720) < 0.0
