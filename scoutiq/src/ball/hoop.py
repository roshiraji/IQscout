"""Dedicated hoop / rim detector (separate weights) for attacking-direction.

Factory mirrors the ball pipeline:
  - autolabel via Grounding DINO prompts ("basketball hoop", "rim", "backboard")
  - build a scene-segment YOLO-detect dataset
  - train best_hoop_det.pt (reuses src.ball.train)
Live: detect_hoops() returns rim boxes; HoopField aggregates a stable rim x
position per scene segment so we can compute which basket is being attacked.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.ball.autolabel import _load_scene_cuts, _segment_of, _yolo_det_line
from src.showcase.ingest import read_manifest

logger = logging.getLogger(__name__)

HOOP_PROMPTS = ["basketball hoop", "basketball rim", "backboard"]

_model_cache: dict[str, Any] = {}


def _get_yolo(weights: str | Path):
    key = str(weights)
    if key not in _model_cache:
        from ultralytics import YOLO

        logger.info("Loading hoop detector: %s", key)
        _model_cache[key] = YOLO(key)
    return _model_cache[key]


def detect_hoops(
    image_bgr,
    *,
    weights: str | Path | None = None,
    conf: float = 0.30,
    device: str | int = "cpu",
    max_hoops: int = 2,
) -> list[dict]:
    """Return rim/backboard boxes. Empty list if no weights / no detection."""
    if weights is None or not Path(weights).is_file():
        return []
    try:
        model = _get_yolo(weights)
        results = model.predict(source=image_bgr, conf=conf, device=device, verbose=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hoop detect failed: %s", exc)
        return []
    out: list[dict] = []
    for r in results or []:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()[:4]]
            c = float(box.conf[0]) if box.conf is not None else 0.0
            out.append(
                {
                    "bbox_xyxy": xyxy,
                    "confidence": round(c, 4),
                    "class_id": 0,
                    "class_name": "hoop",
                    "source": "hoop_yolo",
                }
            )
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:max_hoops]


def autolabel_hoop_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    frames_root: Path,
    max_frames: int | None = None,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    skip_existing: bool = True,
    stride: int = 3,
) -> dict[str, Any]:
    """Grounding DINO hoop boxes → YOLO detect labels (class 0: hoop)."""
    from src.fusion.grounding_dino_runner import (
        GroundingDinoSettings,
        detect_with_grounding_dino,
        grounding_dino_available,
    )

    gd = GroundingDinoSettings(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        class_prompts=HOOP_PROMPTS,
        class_names=["hoop", "hoop", "hoop"],
    )
    if not grounding_dino_available(gd):
        raise RuntimeError(
            "Grounding DINO not available — cannot autolabel hoops. "
            "Install GDINO weights or label a small hoop set manually."
        )

    rows = [
        r
        for r in read_manifest(manifest_path)
        if str(r.get("split") or "train") in {"train", "val"}
        and int(r.get("label_candidate") or 0) == 1
    ]
    rows = rows[::max(stride, 1)]
    if max_frames is not None:
        rows = rows[: int(max_frames)]

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    qa_dir = output_root / "qa_previews"
    for d in (images_dir, labels_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats = {"frames_considered": len(rows), "frames_with_hoop": 0, "instances": 0}
    for row in tqdm(rows, desc="Hoop autolabel", unit="frame"):
        image_path = Path(row["path"])
        if not image_path.is_file():
            continue
        clip_id = row["clip_id"]
        frame_id = int(row["frame_id"])
        stem = f"{clip_id}_{frame_id:06d}"
        label_path = labels_dir / f"{stem}.txt"
        if skip_existing and label_path.is_file() and (images_dir / f"{stem}.jpg").is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        dets = detect_with_grounding_dino(
            image,
            frame_id=frame_id,
            timestamp=float(row.get("timestamp") or 0.0),
            image_name=stem + ".jpg",
            settings=gd,
        )
        # Hoops are upper-court, small-ish; drop implausible boxes.
        kept = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            bw, bh = x2 - x1, y2 - y1
            cy = (y1 + y2) / 2.0 / max(h, 1)
            if bw <= 0 or bh <= 0:
                continue
            if bw > w * 0.35 or bh > h * 0.35:
                continue
            if cy > 0.75:
                continue
            kept.append(d)

        if not kept and skip_existing:
            continue

        shutil.copy2(image_path, images_dir / f"{stem}.jpg")
        lines = []
        for d in kept:
            line = _yolo_det_line(d["bbox_xyxy"], w, h)
            if line:
                lines.append(line)
                stats["instances"] += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if kept:
            stats["frames_with_hoop"] += 1
            canvas = image.copy()
            for d in kept:
                x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.imwrite(str(qa_dir / f"{stem}.jpg"), canvas)

    (output_root / "hoop_autolabel_summary.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("Hoop autolabel complete → %s | %s", output_root, stats)
    return stats


def build_hoop_yolo_dataset(
    *,
    autolabel_root: Path,
    dataset_root: Path,
    frames_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Scene-segment train/val split for hoop detect (class 0: hoop)."""
    images_src = autolabel_root / "images"
    labels_src = autolabel_root / "labels"
    if not images_src.is_dir():
        raise FileNotFoundError(images_src)

    by_clip: dict[str, list[str]] = {}
    for img in sorted(images_src.glob("*.jpg")):
        parts = img.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if not (labels_src / f"{img.stem}.txt").is_file():
            continue
        by_clip.setdefault(parts[0], []).append(img.stem)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for cid, stems in by_clip.items():
        cuts = _load_scene_cuts(frames_root, cid)
        groups: dict[int, list[str]] = {}
        for stem in stems:
            fid = int(stem.rsplit("_", 1)[1])
            groups.setdefault(_segment_of(fid, cuts), []).append(stem)
        seg_ids = sorted(groups)
        rng.shuffle(seg_ids)
        target_val = int(round(len(stems) * val_ratio))
        val_count = 0
        for seg in seg_ids:
            split = "val" if val_count < target_val else "train"
            if split == "val":
                val_count += len(groups[seg])
            for stem in groups[seg]:
                assignment[stem] = split

    for split in ("train", "val"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

    train_n = val_n = 0
    for stem, split in assignment.items():
        shutil.copy2(images_src / f"{stem}.jpg", dataset_root / split / "images" / f"{stem}.jpg")
        shutil.copy2(labels_src / f"{stem}.txt", dataset_root / split / "labels" / f"{stem}.txt")
        if split == "train":
            train_n += 1
        else:
            val_n += 1

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: hoop\n".format(
            str(dataset_root.resolve()).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    (dataset_root / "dataset_meta.json").write_text(
        json.dumps(
            {"train_images": train_n, "val_images": val_n, "clips": sorted(by_clip)},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Hoop YOLO dataset ready: %s (train=%d val=%d)", data_yaml, train_n, val_n)
    return data_yaml
