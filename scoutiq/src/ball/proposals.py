"""HSV orange + geometry QA proposals for basketball labels."""

from __future__ import annotations

import cv2
import numpy as np

# Broadcast basketball at court distance is typically ~12–50px diameter.
_MIN_AREA = 40
_MAX_AREA = 4500
_MIN_CIRCULARITY = 0.40
_MIN_CONF = 0.28


def propose_hsv_balls(
    image_bgr: np.ndarray,
    *,
    spatial_filter: dict | None = None,
) -> list[dict]:
    """Return candidate ball boxes using orange HSV + circularity gates."""
    from src.ball.spatial import spatial_gates_from_cfg

    h, w = image_bgr.shape[:2]
    gates = spatial_gates_from_cfg({"spatial_filter": spatial_filter or {}})
    y_min = float(gates["y_min"]) if gates["enabled"] else 0.06
    y_max = float(gates["y_max"]) if gates["enabled"] else 0.96
    x_min = float(gates["x_min"]) if gates["enabled"] else 0.0
    x_max = float(gates["x_max"]) if gates["enabled"] else 1.0

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 110, 110), (25, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < _MIN_AREA or area > _MAX_AREA:
            continue
        peri = float(cv2.arcLength(cnt, True))
        circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
        if circ < _MIN_CIRCULARITY:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 2.2:
            continue
        # Reject huge boxes that are likely jerseys / logos.
        if max(bw, bh) > min(w, h) * 0.12:
            continue
        # Prefer on-court band (exclude upper crowd / scorebug).
        cx = (bx + bw / 2.0) / max(w, 1)
        cy = (by + bh / 2.0) / max(h, 1)
        if cy < y_min or cy > y_max or cx < x_min or cx > x_max:
            continue
        pad = 3
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)
        roi = hsv[by : by + max(bh, 1), bx : bx + max(bw, 1)]
        avg_sat = float(roi[:, :, 1].mean()) / 255.0 if roi.size else 0.0
        conf = float(min(circ * avg_sat + 0.25, 0.95))
        if conf < _MIN_CONF:
            continue
        out.append(
            {
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": round(conf, 4),
                "circularity": round(float(circ), 4),
                "area": round(area, 1),
                "source": "hsv",
            }
        )
    # Keep top few per frame (usually 0–2 true balls).
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:3]


def propose_yolo_coco_balls(
    image_bgr: np.ndarray,
    *,
    model=None,
    conf: float = 0.15,
) -> list[dict]:
    """Optional COCO sports-ball class proposals (class 32). Soft-fails."""
    if model is None:
        return []
    try:
        results = model(image_bgr, conf=conf, classes=[32], verbose=False)
    except Exception:
        return []
    out: list[dict] = []
    for r in results or []:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            c = float(box.conf[0]) if box.conf is not None else 0.0
            out.append(
                {
                    "bbox_xyxy": [float(v) for v in xyxy[:4]],
                    "confidence": round(c, 4),
                    "circularity": None,
                    "area": None,
                    "source": "yolo_coco_ball",
                }
            )
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:3]


def merge_proposals(*groups: list[dict], iou_thr: float = 0.35) -> list[dict]:
    """Greedy NMS merge across proposal sources."""
    flat = [d for g in groups for d in g]
    flat.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for det in flat:
        if any(_iou(det["bbox_xyxy"], k["bbox_xyxy"]) >= iou_thr for k in kept):
            continue
        kept.append(det)
    return kept[:2]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 1.0) * max(ay2 - ay1, 1.0)
    area_b = max(bx2 - bx1, 1.0) * max(by2 - by1, 1.0)
    return inter / (area_a + area_b - inter)
