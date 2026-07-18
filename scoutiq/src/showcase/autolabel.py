"""Auto-label factory: Grounding DINO boxes → SAM2 masks → YOLO-seg labels.

Player-only. Applies exclusion filters before writing labels.
Optional dual-GPU: set gdino_device / sam_device in settings.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.fusion.grounding_dino_runner import (
    GroundingDinoSettings,
    detect_with_grounding_dino,
    grounding_dino_available,
)
from src.sam2_segmenter import refine_detections_with_sam2, sam2_available
from src.showcase.exclusions import (
    ExclusionSet,
    filter_detections,
    load_exclusion_set,
    resolve_exclusion_path,
)
from src.showcase.ingest import read_manifest

logger = logging.getLogger(__name__)

PLAYER_PROMPTS = ["basketball player"]
PLAYER_NAMES = ["player"]


@dataclass
class AutolabelSettings:
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    sam2_weights: str = "sam2.1_b.pt"
    min_confidence: float = 0.25
    write_qa_previews: bool = True
    max_frames: int | None = None
    gdino_device: str | None = None
    sam_device: str | None = None
    skip_existing: bool = True


def _polygon_to_yolo_seg_line(
    class_id: int,
    polygon: list[list[float]],
    width: int,
    height: int,
) -> str | None:
    if not polygon or len(polygon) < 3 or width <= 0 or height <= 0:
        return None
    coords: list[str] = [str(class_id)]
    for x, y in polygon:
        coords.append(f"{float(x) / width:.6f}")
        coords.append(f"{float(y) / height:.6f}")
    return " ".join(coords)


def _box_to_fallback_polygon(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _draw_qa_preview(image: np.ndarray, dets: list[dict], out_path: Path) -> None:
    canvas = image.copy()
    for det in dets:
        box = det.get("bbox_xyxy") or []
        if len(box) >= 4:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 180, 40), 2)
        poly = det.get("mask_polygon")
        if poly and len(poly) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (40, 180, 40))
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        conf = det.get("confidence", 0.0)
        if len(box) >= 4:
            cv2.putText(
                canvas, f"p {conf:.2f}", (int(box[0]), max(int(box[1]) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def autolabel_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    exclusions_dir: Path,
    gdino_settings: GroundingDinoSettings | None = None,
    settings: AutolabelSettings | None = None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict:
    """Label manifest rows marked label_candidate=1 (never holdout)."""
    cfg = settings or AutolabelSettings()
    gd = gdino_settings or GroundingDinoSettings(
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
        class_prompts=PLAYER_PROMPTS,
        class_names=PLAYER_NAMES,
    )
    # Force player-only prompts.
    gd.class_prompts = PLAYER_PROMPTS
    gd.class_names = PLAYER_NAMES

    if cfg.gdino_device:
        from src.fusion.grounding_dino_runner import set_grounding_dino_device

        set_grounding_dino_device(cfg.gdino_device)
        logger.info("Grounding DINO device → %s", cfg.gdino_device)
    if cfg.sam_device:
        from src.sam2_segmenter import set_sam_device

        set_sam_device(cfg.sam_device)
        logger.info("SAM2 device → %s", cfg.sam_device)

    if not grounding_dino_available(gd):
        raise FileNotFoundError(
            f"Grounding DINO weights missing:\n  {gd.config_path}\n  {gd.checkpoint_path}"
        )

    rows = read_manifest(manifest_path)
    candidates = [
        r for r in rows
        if r.get("split") in splits
        and int(r.get("label_candidate") or 0) == 1
    ]
    if cfg.max_frames is not None:
        candidates = candidates[: cfg.max_frames]

    labels_dir = output_root / "labels"
    images_dir = output_root / "images"
    qa_dir = output_root / "qa_previews"
    raw_dir = output_root / "raw_json"
    for d in (labels_dir, images_dir, qa_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    sam_ok = sam2_available()
    stats = {
        "frames_considered": len(candidates),
        "frames_labeled": 0,
        "frames_skipped_existing": 0,
        "instances": 0,
        "dropped_by_exclusion": 0,
        "sam2": sam_ok,
        "empty_frames": 0,
    }

    exclusion_cache: dict[str, ExclusionSet] = {}

    for row in tqdm(candidates, desc="Auto-label (GDINO→SAM2)", unit="frame"):
        image_path = Path(row["path"])
        if not image_path.is_file():
            continue
        clip_id = row["clip_id"]
        frame_id = int(row["frame_id"])
        stem = f"{clip_id}_{frame_id:06d}"
        label_path = labels_dir / f"{stem}.txt"
        if cfg.skip_existing and label_path.is_file():
            stats["frames_skipped_existing"] += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if clip_id not in exclusion_cache:
            excl_path = resolve_exclusion_path(exclusions_dir, clip_id)
            exclusion_cache[clip_id] = load_exclusion_set(
                excl_path, width=w, height=h, clip_id=clip_id,
            )
        exclusions = exclusion_cache[clip_id]

        ts = float(row.get("timestamp") or 0.0)
        dets = detect_with_grounding_dino(
            image,
            frame_id=frame_id,
            timestamp=ts,
            image_name=image_path.name,
            settings=gd,
        )
        # Player only (already prompted), then exclusions.
        before = len(dets)
        dets = filter_detections(dets, exclusions, class_names=("player",))
        stats["dropped_by_exclusion"] += max(before - len(dets), 0)
        dets = [d for d in dets if float(d.get("confidence") or 0) >= cfg.min_confidence]

        if dets and sam_ok:
            dets = refine_detections_with_sam2(image, dets, cfg.sam2_weights)

        # Copy/link image into dataset staging.
        dest_img = images_dir / f"{stem}.jpg"
        if not dest_img.is_file():
            shutil.copy2(image_path, dest_img)

        lines: list[str] = []
        for det in dets:
            poly = det.get("mask_polygon")
            if not poly or len(poly) < 3:
                poly = _box_to_fallback_polygon(det["bbox_xyxy"])
                det["mask_polygon"] = poly
                det["mask_source"] = det.get("mask_source") or "box_fallback"
            line = _polygon_to_yolo_seg_line(0, poly, w, h)
            if line:
                lines.append(line)
                stats["instances"] += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        raw_dir.joinpath(f"{stem}.json").write_text(
            json.dumps({"image": stem + ".jpg", "detections": dets, "clip_id": clip_id, "frame_id": frame_id}, indent=2),
            encoding="utf-8",
        )
        if cfg.write_qa_previews and dets:
            _draw_qa_preview(image, dets, qa_dir / f"{stem}.jpg")

        stats["frames_labeled"] += 1
        if not lines:
            stats["empty_frames"] += 1

    summary_path = output_root / "autolabel_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Autolabel complete → %s | %s", output_root, stats)
    return stats


def _label_qa_check(
    label_path: Path,
    *,
    min_points: int = 3,
    min_norm_area: float = 0.00005,
    coord_tolerance: float = 0.02,
) -> tuple[list[str], list[str]]:
    """Validate YOLO-seg label lines. Returns (valid_lines, rejection_reasons)."""
    valid: list[str] = []
    reasons: list[str] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError:
        return [], ["unreadable_file"]
    for ln, line in enumerate(text.splitlines()):
        parts = line.split()
        if not parts:
            continue
        coords = parts[1:]
        if len(coords) < min_points * 2 or len(coords) % 2 != 0:
            reasons.append(f"line{ln}:too_few_points")
            continue
        try:
            vals = [float(v) for v in coords]
        except ValueError:
            reasons.append(f"line{ln}:non_numeric")
            continue
        xs = vals[0::2]
        ys = vals[1::2]
        if (min(xs) < -coord_tolerance or max(xs) > 1 + coord_tolerance
                or min(ys) < -coord_tolerance or max(ys) > 1 + coord_tolerance):
            reasons.append(f"line{ln}:out_of_bounds")
            continue
        # Shoelace polygon area in normalized units.
        area = 0.0
        n = len(xs)
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j] - xs[j] * ys[i]
        area = abs(area) / 2.0
        if area < min_norm_area:
            reasons.append(f"line{ln}:tiny_area")
            continue
        if area > 0.60:
            reasons.append(f"line{ln}:implausibly_large")
            continue
        valid.append(line)
    return valid, reasons


def _load_scene_cuts(frames_root: Path, clip_id: str) -> list[int]:
    """Find scene cut indices recorded by ingest for a clip."""
    for meta_path in frames_root.glob(f"*/{clip_id}/ingest_meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return sorted(int(c) for c in meta.get("scene_cuts") or [])
        except (OSError, ValueError):
            continue
    return []


def _segment_of(frame_id: int, cuts: list[int]) -> int:
    """Scene segment index for a frame (segments bounded by scene cuts)."""
    import bisect

    return bisect.bisect_right(cuts, frame_id)


def build_yolo_seg_dataset(
    *,
    autolabel_root: Path,
    dataset_root: Path,
    holdout_clip_ids: set[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    scene_split_frames_root: Path | None = None,
) -> Path:
    """Assemble YOLO-seg train/val from autolabel images+labels.

    Never includes holdout clip IDs. When ``scene_split_frames_root`` is given,
    train/val are split by scene segment (frames between camera cuts stay in
    the same split) so every clip contributes to both splits without
    adjacent-frame leakage. Otherwise falls back to clip-level split.
    Labels failing QA (invalid/tiny/out-of-bounds polygons) are cleaned or
    dropped, with a QA report written next to the dataset.
    """
    import random

    holdout_clip_ids = holdout_clip_ids or set()
    images_src = autolabel_root / "images"
    labels_src = autolabel_root / "labels"
    if not images_src.is_dir():
        raise FileNotFoundError(f"Missing autolabel images: {images_src}")

    # Group by clip_id prefix (stem = clipid_frameid).
    by_clip: dict[str, list[str]] = {}
    for img in sorted(images_src.glob("*.jpg")):
        # clip_id may contain underscores; frame id is last _XXXXXX
        parts = img.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            clip_id = img.stem
        else:
            clip_id = parts[0]
        if clip_id in holdout_clip_ids:
            continue
        label = labels_src / f"{img.stem}.txt"
        if not label.is_file():
            continue
        by_clip.setdefault(clip_id, []).append(img.stem)

    clip_ids = sorted(by_clip.keys())
    rng = random.Random(seed)

    # Label QA pass: keep stems whose labels have >=1 valid polygon.
    qa_report: dict[str, list[str]] = {}
    clean_lines: dict[str, list[str]] = {}
    for cid in clip_ids:
        kept: list[str] = []
        for stem in by_clip[cid]:
            valid, reasons = _label_qa_check(labels_src / f"{stem}.txt")
            if reasons:
                qa_report[stem] = reasons
            if valid:
                clean_lines[stem] = valid
                kept.append(stem)
        by_clip[cid] = kept
    clip_ids = [c for c in clip_ids if by_clip[c]]

    # Decide split assignment per stem.
    assignment: dict[str, str] = {}
    split_mode = "clip"
    if scene_split_frames_root is not None and clip_ids:
        split_mode = "scene_segment"
        for cid in clip_ids:
            cuts = _load_scene_cuts(scene_split_frames_root, cid)
            groups: dict[int, list[str]] = {}
            for stem in by_clip[cid]:
                frame_id = int(stem.rsplit("_", 1)[1])
                groups.setdefault(_segment_of(frame_id, cuts), []).append(stem)
            seg_ids = sorted(groups)
            rng.shuffle(seg_ids)
            total = len(by_clip[cid])
            target_val = int(round(total * val_ratio))
            val_count = 0
            for seg in seg_ids:
                split = "val" if val_count < target_val else "train"
                if split == "val":
                    val_count += len(groups[seg])
                for stem in groups[seg]:
                    assignment[stem] = split
            # Guarantee both splits are non-empty per clip when possible.
            splits_used = {assignment[s] for s in by_clip[cid]}
            if len(splits_used) == 1 and len(seg_ids) >= 2:
                flip_to = "val" if "train" in splits_used else "train"
                for stem in groups[seg_ids[0]]:
                    assignment[stem] = flip_to
    elif len(clip_ids) >= 2:
        shuffled = clip_ids[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        val_clips = set(shuffled[:n_val])
        for cid in clip_ids:
            for stem in by_clip[cid]:
                assignment[stem] = "val" if cid in val_clips else "train"
    else:
        stems = by_clip[clip_ids[0]] if clip_ids else []
        shuffled = stems[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 5 else max(1, len(shuffled) // 5)
        n_val = min(n_val, max(len(shuffled) - 1, 0))
        val_set = set(shuffled[:n_val])
        for stem in stems:
            assignment[stem] = "val" if stem in val_set else "train"

    for split in ("train", "val"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

    train_n = val_n = 0
    for cid in clip_ids:
        for stem in by_clip[cid]:
            split = assignment[stem]
            shutil.copy2(images_src / f"{stem}.jpg", dataset_root / split / "images" / f"{stem}.jpg")
            # Write QA-cleaned labels rather than copying raw files.
            (dataset_root / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(clean_lines[stem]) + "\n", encoding="utf-8",
            )
            if split == "train":
                train_n += 1
            else:
                val_n += 1

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: player\n".format(
            str(dataset_root.resolve()).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    meta = {
        "train_images": train_n,
        "val_images": val_n,
        "clips": clip_ids,
        "split_mode": split_mode,
        "holdout_excluded": sorted(holdout_clip_ids),
        "labels_with_qa_rejections": len(qa_report),
        "data_yaml": str(data_yaml),
    }
    (dataset_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dataset_root / "label_qa_report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    logger.info("YOLO-seg dataset ready: %s (%s)", data_yaml, meta)
    return data_yaml
