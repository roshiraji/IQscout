"""Auto-label generation from mock or LocateAnything-like sources."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .contracts import DetectionRecord, FrameDetections
from .settings import PipelinePaths
from .utils import load_class_map, read_json, write_json


def load_court_roi(paths: PipelinePaths, width: int, height: int) -> np.ndarray:
    payload = read_json(paths.court_roi_path, default={})
    points = payload.get("polygon")
    if not points:
        return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    return np.array(points, dtype=np.int32)


def _point_in_roi(x: float, y: float, roi: np.ndarray) -> bool:
    return cv2.pointPolygonTest(roi.astype(np.float32), (float(x), float(y)), False) >= 0


def _mk_detection(
    *,
    class_id: int,
    class_name: str,
    prompt: str,
    confidence: float,
    bbox: list[int],
    source: str,
) -> DetectionRecord:
    return DetectionRecord(
        class_id=class_id,
        class_name=class_name,
        prompt=prompt,
        confidence=confidence,
        bbox_xyxy=bbox,
        source=source,
        review_status="unreviewed",
    )


def _sample_bbox(
    rng: random.Random,
    width: int,
    height: int,
    roi: np.ndarray,
    *,
    min_w: int,
    max_w: int,
    min_h: int,
    max_h: int,
) -> list[int]:
    for _ in range(200):
        box_w = rng.randint(min_w, max_w)
        box_h = rng.randint(min_h, max_h)
        x1 = rng.randint(0, max(width - box_w - 1, 1))
        y1 = rng.randint(0, max(height - box_h - 1, 1))
        x2 = x1 + box_w
        y2 = y1 + box_h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if _point_in_roi(cx, cy, roi):
            return [x1, y1, x2, y2]
    return [0, 0, max(min_w, 2), max(min_h, 2)]


def _mock_frame_detections(
    *,
    image_path: Path,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    roi: np.ndarray,
) -> FrameDetections:
    rng = random.Random(frame_id * 131 + width + height)
    players = rng.randint(8, 12)
    referees = max(1, min(3, players // 4))
    include_ball = rng.random() > 0.12

    rows: list[DetectionRecord] = []
    for _ in range(players):
        bbox = _sample_bbox(
            rng,
            width,
            height,
            roi,
            min_w=max(width // 42, 22),
            max_w=max(width // 18, 70),
            min_h=max(height // 9, 70),
            max_h=max(height // 4, 260),
        )
        rows.append(
            _mk_detection(
                class_id=0,
                class_name="player",
                prompt="basketball player on the court",
                confidence=rng.uniform(0.62, 0.96),
                bbox=bbox,
                source="mock",
            )
        )

    for _ in range(referees):
        bbox = _sample_bbox(
            rng,
            width,
            height,
            roi,
            min_w=max(width // 52, 18),
            max_w=max(width // 22, 58),
            min_h=max(height // 10, 64),
            max_h=max(height // 3, 220),
        )
        rows.append(
            _mk_detection(
                class_id=1,
                class_name="referee",
                prompt="basketball referee near play",
                confidence=rng.uniform(0.58, 0.91),
                bbox=bbox,
                source="mock",
            )
        )

    if include_ball:
        bbox = _sample_bbox(
            rng,
            width,
            height,
            roi,
            min_w=max(width // 180, 8),
            max_w=max(width // 90, 16),
            min_h=max(height // 180, 8),
            max_h=max(height // 90, 16),
        )
        rows.append(
            _mk_detection(
                class_id=2,
                class_name="ball",
                prompt="basketball in game action",
                confidence=rng.uniform(0.40, 0.86),
                bbox=bbox,
                source="mock",
            )
        )

    hoop_bbox = [int(width * 0.74), int(height * 0.08), int(width * 0.82), int(height * 0.23)]
    rows.append(
        _mk_detection(
            class_id=3,
            class_name="hoop",
            prompt="basketball hoop and backboard",
            confidence=0.88,
            bbox=hoop_bbox,
            source="mock",
        )
    )

    return FrameDetections(
        image=image_path.name,
        frame_id=frame_id,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=rows,
    )


def _filter_to_roi(frame_data: FrameDetections, roi: np.ndarray) -> FrameDetections:
    keep: list[DetectionRecord] = []
    for det in frame_data.detections:
        if det.class_name in {"player", "referee", "ball"}:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if not _point_in_roi(cx, cy, roi):
                continue
        keep.append(det)
    frame_data.detections = keep
    return frame_data


def _run_locateanything(executable: Path, image_path: Path) -> list[dict]:
    # Expect LocateAnything output as JSON lines or a JSON file path.
    proc = subprocess.run(
        [str(executable), "--image", str(image_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"LocateAnything exited with code {proc.returncode}")
    import json

    text = proc.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        return payload.get("detections", [])
    if isinstance(payload, list):
        return payload
    return []


def _locateanything_to_contract(
    *,
    image_path: Path,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    raw_rows: list[dict],
) -> FrameDetections:
    class_map = load_class_map_stub()
    rows: list[DetectionRecord] = []
    for row in raw_rows:
        class_name = str(row.get("class_name", "player"))
        class_id = int(row.get("class_id", class_map.get(class_name, 0)))
        bbox = [int(v) for v in row.get("bbox_xyxy", [0, 0, 1, 1])]
        rows.append(
            _mk_detection(
                class_id=class_id,
                class_name=class_name,
                prompt=str(row.get("prompt", f"{class_name} on basketball court")),
                confidence=float(row.get("confidence", 0.5)),
                bbox=bbox,
                source="locateanything",
            )
        )
    return FrameDetections(
        image=image_path.name,
        frame_id=frame_id,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=rows,
    )


def load_class_map_stub() -> dict[str, int]:
    return {"player": 0, "referee": 1, "ball": 2, "hoop": 3}


def autolabel_frames(
    *,
    paths: PipelinePaths,
    source: str,
    locateanything_executable: Path | None = None,
) -> dict:
    frames = sorted(paths.frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(
            f"No frames found in {paths.frames_dir}. Run extract first or use demo-mock."
        )

    classes = load_class_map(paths)
    class_name_by_id = {name: class_id for class_id, name in classes.items()}
    locate_ready = (
        source == "locateanything"
        and locateanything_executable is not None
        and locateanything_executable.is_file()
    )
    effective_source = "locateanything" if locate_ready else "mock"

    frame_count = 0
    all_detections = 0
    for image_path in tqdm(frames, desc=f"Auto-label ({effective_source})"):
        frame_id = int(image_path.stem.split("_")[-1])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        roi = load_court_roi(paths, width=width, height=height)
        timestamp = frame_id / 30.0

        if locate_ready:
            try:
                raw_rows = _run_locateanything(locateanything_executable, image_path)
                frame_data = _locateanything_to_contract(
                    image_path=image_path,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    raw_rows=raw_rows,
                )
            except Exception:
                frame_data = _mock_frame_detections(
                    image_path=image_path,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    roi=roi,
                )
        else:
            frame_data = _mock_frame_detections(
                image_path=image_path,
                frame_id=frame_id,
                timestamp=timestamp,
                width=width,
                height=height,
                roi=roi,
            )

        frame_data = _filter_to_roi(frame_data, roi)
        # Ensure class IDs align to classes.yaml when class names exist.
        for det in frame_data.detections:
            if det.class_name in class_name_by_id:
                det.class_id = class_name_by_id[det.class_name]

        write_json(paths.raw_predictions_dir / f"{image_path.stem}.json", frame_data.to_dict())
        frame_count += 1
        all_detections += len(frame_data.detections)

    summary = {
        "source_requested": source,
        "source_used": effective_source,
        "locateanything_available": bool(locate_ready),
        "frames_labeled": frame_count,
        "detections_total": all_detections,
        "output_dir": str(paths.raw_predictions_dir),
    }
    write_json(paths.raw_predictions_dir / "autolabel_summary.json", summary)
    return summary
