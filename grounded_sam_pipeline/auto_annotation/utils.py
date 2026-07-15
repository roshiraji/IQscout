"""Shared utilities."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def setup_logging(level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def list_images(directory: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_image_to_output(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def mask_to_polygons(
    mask: np.ndarray,
    *,
    epsilon: float = 2.0,
    min_area: int = 10,
) -> list[list[list[float]]]:
    """Convert boolean mask to list of polygon point lists [[x,y], ...]."""
    if mask.dtype != np.uint8:
        binary = (mask > 0).astype(np.uint8) * 255
    else:
        binary = mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        poly = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
        polygons.append(poly)
    return polygons


def polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    pts = np.array(poly, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

def normalize_polygon(poly: list[list[float]], width: int, height: int) -> list[float]:
    coords: list[float] = []
    for x, y in poly:
        coords.extend([max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))])
    return coords
