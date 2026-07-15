"""Side-by-side visuals for segmentation lab review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.segmentation_lab.sam_eval import MaskResult
from src.segmentation_lab.yolo_prompts import BoxPrompt

_PALETTE = [
    (0, 255, 128),
    (255, 128, 0),
    (0, 180, 255),
    (255, 64, 192),
    (255, 255, 0),
    (128, 128, 255),
    (255, 96, 96),
    (96, 255, 96),
]


def _color(index: int) -> tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def draw_yolo_boxes(frame_bgr: np.ndarray, boxes: list[BoxPrompt]) -> np.ndarray:
    canvas = frame_bgr.copy()
    for index, box in enumerate(boxes):
        color = _color(index)
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = f"{box.label} {box.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (box.x1, max(18, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_sam_masks(
    frame_bgr: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[BoxPrompt],
    results: list[MaskResult],
) -> np.ndarray:
    canvas = frame_bgr.copy().astype(np.float32)
    for index, (mask, box, result) in enumerate(zip(masks, boxes, results)):
        color = np.array(_color(index), dtype=np.float32)
        overlay = np.zeros_like(canvas)
        overlay[mask] = color
        canvas = np.where(
            mask[..., None],
            canvas * 0.45 + overlay * 0.55,
            canvas,
        )
        cv2.rectangle(canvas.astype(np.uint8), (box.x1, box.y1), (box.x2, box.y2), _color(index), 1)
        label = f"{result.label} SAM {result.sam_score:.2f}"
        cv2.putText(
            canvas.astype(np.uint8),
            label,
            (box.x1, min(canvas.shape[0] - 6, box.y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _color(index),
            1,
            cv2.LINE_AA,
        )
    return canvas.astype(np.uint8)


def build_comparison_panel(
    frame_bgr: np.ndarray,
    boxes: list[BoxPrompt],
    masks: list[np.ndarray],
    results: list[MaskResult],
) -> np.ndarray:
    original = frame_bgr.copy()
    yolo_view = draw_yolo_boxes(frame_bgr, boxes)
    sam_view = draw_sam_masks(frame_bgr, masks, boxes, results)

    header_h = 36
    gap = 8
    panel_w = original.shape[1]
    panel_h = original.shape[0] + header_h
    row_h = panel_h
    out_w = panel_w * 3 + gap * 2
    out_h = row_h
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    panels = [
        ("Original", original),
        ("YOLO boxes (prompts)", yolo_view),
        ("SAM masks", sam_view),
    ]
    x = 0
    for title, image in panels:
        canvas[header_h : header_h + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(
            canvas,
            title,
            (x + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        x += panel_w + gap
    return canvas


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))
