"""Shared helpers for filesystem, logging, and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .settings import PipelinePaths

LOGGER_NAME = "sports_autolabel"


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_map(paths: PipelinePaths) -> dict[int, str]:
    if not paths.classes_yaml_path.is_file():
        return {0: "player", 1: "referee", 2: "ball", 3: "hoop"}
    payload = yaml.safe_load(paths.classes_yaml_path.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    return {int(key): str(value) for key, value in names.items()}


def write_default_classes_yaml(paths: PipelinePaths) -> None:
    if paths.classes_yaml_path.is_file():
        return
    payload = {
        "names": {
            0: "player",
            1: "referee",
            2: "ball",
            3: "hoop",
        }
    }
    paths.classes_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    write_json(
        paths.classes_summary_path,
        {
            "enabled_for_training": ["player", "referee", "ball", "hoop"],
            "ignored_by_default": ["crowd", "bench_players"],
        },
    )


def normalize_windows_path(path_value: str | Path) -> Path:
    return Path(str(path_value).replace("\\", "/")).expanduser().resolve()


def default_court_roi(width: int, height: int) -> list[list[int]]:
    margin_x = max(int(width * 0.08), 1)
    margin_y = max(int(height * 0.10), 1)
    return [
        [margin_x, margin_y],
        [width - margin_x, margin_y],
        [width - margin_x, height - margin_y],
        [margin_x, height - margin_y],
    ]
