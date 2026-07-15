"""YOLO11-seg primary detector with color-based jersey classification.

Detects persons via YOLO11-seg, then classifies player jersey colors
into player_red vs player_black_blue using HSV analysis of the torso region.
Ball detection uses a dedicated HSV orange-ball detector (YOLO's sports ball
class is unreliable for small fast-moving basketballs in broadcast footage).
Referee detection uses a dark+white uniform heuristic.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class indices for YOLO11.
_COCO_PERSON = 0
_COCO_BALL = 32

# Minimum sizes (px) to reject tiny false positives.
_MIN_PERSON_HEIGHT = 40
_MIN_PERSON_WIDTH = 15

# Confidence thresholds.
_PERSON_CONF = 0.30
_BALL_CONF = 0.15

_MODEL_NAME = "yolo11n-seg.pt"
_model_cache: dict = {}

# Orange-ball HSV detector parameters.
_BALL_MIN_AREA = 60        # px²
_BALL_MAX_AREA = 3500      # px² — basketball at broadcast distance ≈ 15-50px diam
_BALL_MIN_CIRCULARITY = 0.45
_BALL_ORANGE_CONF = 0.55   # confidence assigned to color-detected balls


def _get_model(weights: str = _MODEL_NAME):
    if weights not in _model_cache:
        from ultralytics import YOLO
        logger.info("Loading YOLO11-seg model: %s", weights)
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


# ---------------------------------------------------------------------------
# Jersey color classification
# ---------------------------------------------------------------------------

def _classify_jersey_color(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """Analyze torso region to distinguish red / black-blue / referee uniforms."""
    h_total = y2 - y1
    w_total = x2 - x1
    if h_total < 10 or w_total < 8:
        return "player_black_blue"

    # Sample the torso: middle 25-65% vertically, center 20-80% horizontally.
    ty1 = y1 + int(h_total * 0.25)
    ty2 = y1 + int(h_total * 0.65)
    tx1 = x1 + int(w_total * 0.20)
    tx2 = x2 - int(w_total * 0.20)
    tx1, tx2 = max(tx1, 0), min(tx2, image_bgr.shape[1])
    ty1, ty2 = max(ty1, 0), min(ty2, image_bgr.shape[0])
    if ty2 <= ty1 or tx2 <= tx1:
        return "player_black_blue"

    torso = image_bgr[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    red_lo1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red_lo2 = cv2.inRange(hsv, (168, 80, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_lo1, red_lo2)

    blue_mask = cv2.inRange(hsv, (95, 50, 50), (135, 255, 255))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    total_px = torso.shape[0] * torso.shape[1]
    if total_px == 0:
        return "player_black_blue"

    red_ratio = red_mask.sum() / (255 * total_px)
    blue_ratio = blue_mask.sum() / (255 * total_px)
    dark_ratio = dark_mask.sum() / (255 * total_px)
    white_ratio = white_mask.sum() / (255 * total_px)

    # Referee heuristic: mostly dark with notable white.
    if dark_ratio > 0.35 and white_ratio > 0.12:
        return "referee"

    if red_ratio > 0.12 and red_ratio > blue_ratio * 1.2:
        return "player_red"

    return "player_black_blue"


# ---------------------------------------------------------------------------
# HSV orange-ball detector
# ---------------------------------------------------------------------------

def _detect_orange_balls(
    image_bgr: np.ndarray,
    frame_id: int,
    timestamp: float,
    image_name: str,
) -> list[dict]:
    """Detect orange basketball using HSV color + contour circularity."""
    h, w = image_bgr.shape[:2]

    # Blur to reduce court-line noise.
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Orange: H 6-22, high saturation and value.
    orange_mask = cv2.inRange(hsv, (6, 130, 130), (22, 255, 255))

    # Morphological cleanup.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _BALL_MIN_AREA or area > _BALL_MAX_AREA:
            continue

        # Circularity = 4π·area / perimeter².
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        if circularity < _BALL_MIN_CIRCULARITY:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Aspect ratio should be roughly square.
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 2.0:
            continue

        # Pad the bounding box slightly.
        pad = 4
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        # Confidence proxy: circularity × saturation score.
        roi_hsv = hsv[by: by + bh, bx: bx + bw]
        avg_sat = float(roi_hsv[:, :, 1].mean()) / 255.0
        conf = round(min(circularity * avg_sat + 0.3, 1.0), 3)
        if conf < 0.3:
            continue

        detections.append({
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "image": image_name,
            "width": w,
            "height": h,
            "class_id": 0,
            "class_name": "ball",
            "prompt": "orange basketball on the court",
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "mask_polygon": None,
            "source": "color-detector",
        })

    return detections


# ---------------------------------------------------------------------------
# Main detector entry point
# ---------------------------------------------------------------------------

def detect_frame(
    image_bgr: np.ndarray,
    frame_id: int,
    timestamp: float,
    image_name: str,
    weights: str = _MODEL_NAME,
) -> list[dict]:
    """Run YOLO11-seg (persons) + HSV ball detector on one BGR frame."""
    model = _get_model(weights)
    h, w = image_bgr.shape[:2]

    results = model(
        image_bgr,
        conf=_BALL_CONF,
        iou=0.45,
        classes=[_COCO_PERSON, _COCO_BALL],
        verbose=False,
    )

    detections: list[dict] = []
    if results:
        res = results[0]
        boxes = res.boxes
        masks_data = res.masks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            box_w = x2 - x1
            box_h = y2 - y1

            mask_poly: list[list[float]] | None = None
            if masks_data is not None and i < len(masks_data.xy):
                pts = masks_data.xy[i]
                if pts is not None and len(pts) > 2:
                    mask_poly = [[float(p[0]), float(p[1])] for p in pts]

            if cls_id == _COCO_BALL:
                if 6 <= box_w <= 80 and 6 <= box_h <= 80:
                    detections.append({
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 4),
                        "image": image_name,
                        "width": w,
                        "height": h,
                        "class_id": 0,
                        "class_name": "ball",
                        "prompt": "orange basketball on the court",
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "mask_polygon": mask_poly,
                        "source": "yolo11-seg",
                    })
            elif cls_id == _COCO_PERSON:
                if box_h < _MIN_PERSON_HEIGHT or box_w < _MIN_PERSON_WIDTH:
                    continue
                if conf < _PERSON_CONF:
                    continue
                class_name = _classify_jersey_color(image_bgr, x1, y1, x2, y2)
                class_map = {"ball": 0, "player_red": 1, "player_black_blue": 2, "referee": 3}
                detections.append({
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "image": image_name,
                    "width": w,
                    "height": h,
                    "class_id": class_map[class_name],
                    "class_name": class_name,
                    "prompt": f"basketball {class_name} on the court",
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "mask_polygon": mask_poly,
                    "source": "yolo11-seg",
                })

    # Color-based orange ball detector (more reliable for small fast balls).
    color_balls = _detect_orange_balls(image_bgr, frame_id, timestamp, image_name)
    detections.extend(color_balls)

    return detections
