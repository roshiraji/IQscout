"""Court ROI utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .settings import PipelinePaths
from .utils import default_court_roi, write_json


def set_roi_from_image(paths: PipelinePaths, image_path: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    points: list[tuple[int, int]] = []
    canvas = image.copy()

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    window_name = "Set Court ROI - left click add points, enter to save, r reset"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
        while True:
            canvas[:] = image
            for idx, point in enumerate(points):
                cv2.circle(canvas, point, 4, (0, 255, 255), -1)
                if idx > 0:
                    cv2.line(canvas, points[idx - 1], point, (0, 255, 255), 2)
            if len(points) > 2:
                cv2.line(canvas, points[-1], points[0], (0, 255, 255), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # Enter
                if len(points) >= 3:
                    break
            elif key == ord("r"):
                points = []
            elif key == 27:  # ESC
                break
        cv2.destroyAllWindows()
    except Exception:
        # Headless fallback.
        points = [tuple(p) for p in default_court_roi(width, height)]

    if len(points) < 3:
        points = [tuple(p) for p in default_court_roi(width, height)]

    payload = {
        "image": str(image_path.resolve()),
        "width": width,
        "height": height,
        "polygon": [[int(x), int(y)] for x, y in points],
    }
    write_json(paths.court_roi_path, payload)
    return payload


def load_roi_polygon(paths: PipelinePaths, width: int, height: int) -> np.ndarray:
    from .utils import read_json

    payload = read_json(paths.court_roi_path, default={})
    polygon = payload.get("polygon")
    if not polygon:
        polygon = default_court_roi(width, height)
    return np.array(polygon, dtype=np.int32)
