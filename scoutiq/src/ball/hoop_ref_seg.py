"""Hoop + referee YOLO-seg factory (GDINO boxes → SAM2 masks → train).

Separate from player seg and ball/hoop detect weights. Classes:
  0: hoop
  1: referee

Target: high-confidence live detection (conf 0.80–0.90) after training.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from src.ball.autolabel import _load_scene_cuts, _segment_of
from src.showcase.autolabel import (
    _box_to_fallback_polygon,
    _label_qa_check,
    _polygon_to_yolo_seg_line,
)
from src.showcase.ingest import read_manifest

logger = logging.getLogger(__name__)

HOOP_PROMPTS = ["basketball hoop", "basketball rim", "backboard"]
REF_PROMPTS = ["basketball referee", "referee official", "striped referee"]
CLASS_PROMPTS = HOOP_PROMPTS + REF_PROMPTS
CLASS_NAMES = ["hoop", "hoop", "hoop", "referee", "referee", "referee"]
CLASS_TO_ID = {"hoop": 0, "referee": 1}

# Distinct QA colors (BGR)
_QA_COLORS = {
    "hoop": (0, 200, 255),
    "referee": (255, 140, 40),
}


def _keep_hoop(det: dict, width: int, height: int) -> bool:
    x1, y1, x2, y2 = det["bbox_xyxy"]
    bw, bh = x2 - x1, y2 - y1
    cy = (y1 + y2) / 2.0 / max(height, 1)
    if bw <= 0 or bh <= 0:
        return False
    if bw > width * 0.35 or bh > height * 0.35:
        return False
    if cy > 0.75:
        return False
    return True


def _keep_referee(det: dict, width: int, height: int) -> bool:
    x1, y1, x2, y2 = det["bbox_xyxy"]
    bw, bh = x2 - x1, y2 - y1
    area = (bw * bh) / max(width * height, 1)
    aspect = bh / max(bw, 1)
    cy = (y1 + y2) / 2.0 / max(height, 1)
    if bw <= 0 or bh <= 0:
        return False
    # Person-sized upright boxes; drop tiny noise / huge crowd blobs.
    if area < 0.002 or area > 0.25:
        return False
    if aspect < 1.1 or aspect > 5.0:
        return False
    if cy < 0.15:
        return False
    return True


def _draw_qa_preview(image: np.ndarray, dets: list[dict], out_path: Path) -> None:
    canvas = image.copy()
    for det in dets:
        name = str(det.get("class_name") or "")
        color = _QA_COLORS.get(name, (180, 180, 180))
        box = det.get("bbox_xyxy") or []
        if len(box) >= 4:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        poly = det.get("mask_polygon")
        if poly and len(poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        conf = float(det.get("confidence") or 0.0)
        if len(box) >= 4:
            label = f"{name[:3]} {conf:.2f}"
            cv2.putText(
                canvas,
                label,
                (int(box[0]), max(int(box[1]) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def autolabel_hoop_ref_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    frames_root: Path,
    max_frames: int | None = None,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    min_confidence: float = 0.28,
    skip_existing: bool = True,
    stride: int = 3,
    sam2_weights: str = "sam2.1_b.pt",
    write_qa_previews: bool = True,
) -> dict[str, Any]:
    """Grounding DINO hoop+ref boxes → SAM2 masks → YOLO-seg labels."""
    from src.fusion.grounding_dino_runner import (
        GroundingDinoSettings,
        detect_with_grounding_dino,
        grounding_dino_available,
    )
    from src.sam2_segmenter import refine_detections_with_sam2, sam2_available

    gd = GroundingDinoSettings(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        class_prompts=CLASS_PROMPTS,
        class_names=CLASS_NAMES,
    )
    if not grounding_dino_available(gd):
        raise RuntimeError(
            "Grounding DINO not available — cannot autolabel hoop/refs. "
            "Install GDINO weights or label a small set manually."
        )

    rows = [
        r
        for r in read_manifest(manifest_path)
        if str(r.get("split") or "train") in {"train", "val"}
        and int(r.get("label_candidate") or 0) == 1
    ]
    rows = rows[:: max(stride, 1)]
    if max_frames is not None:
        rows = rows[: int(max_frames)]

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    qa_dir = output_root / "qa_previews"
    raw_dir = output_root / "raw_json"
    for d in (images_dir, labels_dir, qa_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    sam_ok = sam2_available()
    stats: dict[str, Any] = {
        "frames_considered": len(rows),
        "frames_labeled": 0,
        "frames_skipped_existing": 0,
        "frames_with_hoop": 0,
        "frames_with_referee": 0,
        "instances_hoop": 0,
        "instances_referee": 0,
        "empty_frames": 0,
        "sam2": sam_ok,
        "stride": stride,
    }

    for row in tqdm(rows, desc="Hoop+ref seg autolabel", unit="frame"):
        image_path = Path(row["path"])
        if not image_path.is_file():
            continue
        clip_id = row["clip_id"]
        frame_id = int(row["frame_id"])
        stem = f"{clip_id}_{frame_id:06d}"
        label_path = labels_dir / f"{stem}.txt"
        if skip_existing and label_path.is_file() and (images_dir / f"{stem}.jpg").is_file():
            stats["frames_skipped_existing"] += 1
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

        kept: list[dict] = []
        for d in dets:
            name = str(d.get("class_name") or "")
            conf = float(d.get("confidence") or 0.0)
            if conf < min_confidence:
                continue
            if name == "hoop" and _keep_hoop(d, w, h):
                kept.append(d)
            elif name == "referee" and _keep_referee(d, w, h):
                kept.append(d)

        if kept and sam_ok:
            kept = refine_detections_with_sam2(image, kept, sam2_weights)

        shutil.copy2(image_path, images_dir / f"{stem}.jpg")
        lines: list[str] = []
        has_hoop = has_ref = False
        for det in kept:
            name = str(det.get("class_name") or "")
            class_id = CLASS_TO_ID.get(name)
            if class_id is None:
                continue
            poly = det.get("mask_polygon")
            if not poly or len(poly) < 3:
                poly = _box_to_fallback_polygon(det["bbox_xyxy"])
                det["mask_polygon"] = poly
                det["mask_source"] = det.get("mask_source") or "box_fallback"
            line = _polygon_to_yolo_seg_line(class_id, poly, w, h)
            if not line:
                continue
            lines.append(line)
            if name == "hoop":
                stats["instances_hoop"] += 1
                has_hoop = True
            else:
                stats["instances_referee"] += 1
                has_ref = True

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        raw_dir.joinpath(f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": stem + ".jpg",
                    "detections": kept,
                    "clip_id": clip_id,
                    "frame_id": frame_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if write_qa_previews and kept:
            _draw_qa_preview(image, kept, qa_dir / f"{stem}.jpg")

        stats["frames_labeled"] += 1
        if has_hoop:
            stats["frames_with_hoop"] += 1
        if has_ref:
            stats["frames_with_referee"] += 1
        if not lines:
            stats["empty_frames"] += 1

    (output_root / "hoop_ref_autolabel_summary.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("Hoop+ref seg autolabel complete → %s | %s", output_root, stats)
    return stats


def build_hoop_ref_yolo_seg_dataset(
    *,
    autolabel_root: Path,
    dataset_root: Path,
    frames_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    require_positive: bool = True,
) -> Path:
    """Scene-segment train/val split for hoop+ref YOLO-seg."""
    images_src = autolabel_root / "images"
    labels_src = autolabel_root / "labels"
    if not images_src.is_dir():
        raise FileNotFoundError(images_src)

    by_clip: dict[str, list[str]] = {}
    for img in sorted(images_src.glob("*.jpg")):
        parts = img.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        label = labels_src / f"{img.stem}.txt"
        if not label.is_file():
            continue
        text = label.read_text(encoding="utf-8").strip()
        if require_positive and not text:
            continue
        by_clip.setdefault(parts[0], []).append(img.stem)

    # Label QA
    qa_report: dict[str, list[str]] = {}
    clean_lines: dict[str, list[str]] = {}
    for cid, stems in list(by_clip.items()):
        kept: list[str] = []
        for stem in stems:
            valid, reasons = _label_qa_check(labels_src / f"{stem}.txt")
            if reasons:
                qa_report[stem] = reasons
            if valid:
                clean_lines[stem] = valid
                kept.append(stem)
        by_clip[cid] = kept
    by_clip = {c: s for c, s in by_clip.items() if s}

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
        (dataset_root / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(clean_lines[stem]) + "\n", encoding="utf-8"
        )
        if split == "train":
            train_n += 1
        else:
            val_n += 1

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        (
            "path: {}\n"
            "train: train/images\n"
            "val: val/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: hoop\n"
            "  1: referee\n"
        ).format(str(dataset_root.resolve()).replace("\\", "/")),
        encoding="utf-8",
    )
    meta = {
        "train_images": train_n,
        "val_images": val_n,
        "clips": sorted(by_clip),
        "classes": CLASS_TO_ID,
        "labels_with_qa_rejections": len(qa_report),
        "data_yaml": str(data_yaml),
    }
    (dataset_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dataset_root / "label_qa_report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    logger.info("Hoop+ref YOLO-seg dataset ready: %s (%s)", data_yaml, meta)
    return data_yaml
