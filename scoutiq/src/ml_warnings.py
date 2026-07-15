"""Suppress known benign ML-library warnings for cleaner pipeline logs."""

from __future__ import annotations

import logging
import os
import warnings

_CONFIGURED = False


def configure_ml_runtime(*, quiet: bool = True) -> None:
    """Apply one-time runtime tweaks for torch/transformers/GroundingDINO noise."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not quiet:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*Failed to load custom C\+\+ ops.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.meshgrid.*indexing argument.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*`device` argument is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*use_reentrant parameter should be passed explicitly.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*None of the inputs have requires_grad=True.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*invalid escape sequence.*",
        category=SyntaxWarning,
    )

    for name in ("transformers", "transformers.modeling_utils", "torch.utils.checkpoint"):
        logging.getLogger(name).setLevel(logging.ERROR)
