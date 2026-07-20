"""Offline ball label factory: HSV (+ optional COCO) proposals → YOLO-det labels."""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.ball.proposals import merge_proposals, propose_hsv_balls, propose_yolo_coco_balls
from src.showcase.ingest import read_manifest

logger = logging.getLogger(__name__)


def _yolo_det_line(bbox: list[float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    if bw <= 0 or bh <= 0:
        return None
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _load_scene_cuts(frames_root: Path, clip_id: str) -> list[int]:
    for meta_path in frames_root.glob(f"*/{clip_id}/ingest_meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return sorted(int(c) for c in meta.get("scene_cuts") or [])
        except (OSError, ValueError, TypeError):
            continue
    return []


def _segment_of(frame_id: int, cuts: list[int]) -> int:
    import bisect

    return bisect.bisect_right(cuts, frame_id)


def _sam_refine_box(image, box: list[float]):
    """Tighten a ball box using SAM mask → bbox. Soft-fails to original box."""
    try:
        from src.segmentation import predict_mask_from_box
        import numpy as np

        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        # Pad slightly so SAM sees context around a tiny ball.
        pad = max((x2 - x1), (y2 - y1)) * 0.4 + 4
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        mask = predict_mask_from_box(image, [px1, py1, px2, py2])
        ys, xs = np.where(mask)
        if xs.size == 0:
            return box, False
        nx1, ny1, nx2, ny2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        # Reject implausible refinements (SAM grabbed a player/limb).
        area_ratio = ((nx2 - nx1) * (ny2 - ny1)) / max((x2 - x1) * (y2 - y1), 1.0)
        if area_ratio > 4.0 or (nx2 - nx1) > (x2 - x1) * 3 or (ny2 - ny1) > (y2 - y1) * 3:
            return box, False
        return [nx1, ny1, nx2, ny2], True
    except Exception:
        return box, False


def autolabel_ball_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    frames_root: Path | None = None,
    max_frames: int | None = None,
    min_confidence: float = 0.32,
    use_coco_ball: bool = True,
    skip_existing: bool = True,
    write_qa_previews: bool = True,
    stride_keep_empty: int = 40,
    use_sam_refine: bool = False,
    mine_hard_negatives: bool = True,
    scorebug_top_frac: float = 0.10,
    scorebug_bottom_frac: float = 0.08,
) -> dict[str, Any]:
    """Label manifest candidates with ball proposals; keep some empty negatives."""
    rows = [
        r
        for r in read_manifest(manifest_path)
        if str(r.get("split") or "train") in {"train", "val"}
        and int(r.get("label_candidate") or 0) == 1
    ]
    if max_frames is not None and len(rows) > int(max_frames):
        # Stratify across clips so one game does not dominate the budget.
        by_clip: dict[str, list] = {}
        for r in rows:
            by_clip.setdefault(str(r.get("clip_id") or "unknown"), []).append(r)
        budget = int(max_frames)
        clips = sorted(by_clip)
        per = max(budget // max(len(clips), 1), 1)
        sampled: list = []
        for i, cid in enumerate(clips):
            take = per if i < len(clips) - 1 else max(budget - len(sampled), 0)
            sampled.extend(by_clip[cid][:take])
        rows = sampled[:budget]
    elif max_frames is not None:
        rows = rows[: int(max_frames)]

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    qa_dir = output_root / "qa_previews"
    raw_dir = output_root / "raw_json"
    for d in (images_dir, labels_dir, qa_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    coco_model = None
    if use_coco_ball:
        try:
            from ultralytics import YOLO

            coco_model = YOLO("yolo11n.pt")
            logger.info("COCO ball proposals enabled (yolo11n class 32)")
        except Exception as exc:
            logger.warning("COCO ball proposals unavailable: %s", exc)

    stats = {
        "frames_considered": len(rows),
        "frames_with_ball": 0,
        "frames_empty_negatives": 0,
        "frames_hard_negatives": 0,
        "frames_skipped_existing": 0,
        "instances": 0,
        "sam_refined": 0,
        "sources": {"hsv": 0, "yolo_coco_ball": 0},
        "use_sam_refine": bool(use_sam_refine),
    }
    review_rows: list[dict] = []

    for idx, row in enumerate(tqdm(rows, desc="Ball autolabel", unit="frame")):
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
        hsv = propose_hsv_balls(image)
        coco = propose_yolo_coco_balls(image, model=coco_model) if coco_model else []
        raw_all = merge_proposals(hsv, coco)

        # Scorebug / graphic bands: proposals here are logos/clocks, not balls.
        def _in_scorebug(det: dict) -> bool:
            cy = (det["bbox_xyxy"][1] + det["bbox_xyxy"][3]) / 2.0 / max(h, 1)
            return cy < scorebug_top_frac or cy > (1.0 - scorebug_bottom_frac)

        merged = [
            d
            for d in raw_all
            if float(d.get("confidence") or 0) >= min_confidence and not _in_scorebug(d)
        ]
        # Rejected orange blobs (jersey/logo/scorebug) → mark frame hard negative.
        rejected = [d for d in raw_all if d not in merged]
        # One basketball per frame for weak labels (keep strongest).
        merged = merged[:1]
        for d in merged:
            stats["sources"][d.get("source") or "hsv"] = (
                stats["sources"].get(d.get("source") or "hsv", 0) + 1
            )

        if merged and use_sam_refine:
            refined_box, ok = _sam_refine_box(image, merged[0]["bbox_xyxy"])
            if ok:
                merged[0] = {**merged[0], "bbox_xyxy": refined_box, "sam_refined": True}
                stats["sam_refined"] += 1

        is_hard_negative = bool(not merged and rejected and mine_hard_negatives)
        # Keep frame if: has ball, is a hard negative, or on the empty stride.
        if not merged and not is_hard_negative and (idx % max(stride_keep_empty, 1) != 0):
            continue

        dest = images_dir / f"{stem}.jpg"
        if not dest.is_file():
            shutil.copy2(image_path, dest)

        lines: list[str] = []
        for det in merged:
            line = _yolo_det_line(det["bbox_xyxy"], w, h)
            if line:
                lines.append(line)
                stats["instances"] += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        raw_dir.joinpath(f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": stem + ".jpg",
                    "clip_id": clip_id,
                    "frame_id": frame_id,
                    "detections": merged,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if write_qa_previews and merged:
            canvas = image.copy()
            for det in merged:
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"][:4]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    canvas,
                    f"ball {det['confidence']:.2f}",
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(qa_dir / f"{stem}.jpg"), canvas)

        # Review queue: low-confidence or multi-proposal frames.
        if merged and (
            any(float(d.get("confidence") or 0) < 0.45 for d in merged) or len(merged) > 1
        ):
            review_rows.append(
                {
                    "stem": stem,
                    "clip_id": clip_id,
                    "frame_id": frame_id,
                    "n_dets": len(merged),
                    "max_conf": max(float(d.get("confidence") or 0) for d in merged),
                    "sources": "|".join(sorted({d.get("source") or "?" for d in merged})),
                }
            )

        if merged:
            stats["frames_with_ball"] += 1
        elif is_hard_negative:
            stats["frames_hard_negatives"] += 1
        else:
            stats["frames_empty_negatives"] += 1

    if review_rows:
        review_path = output_root / "review_queue.csv"
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stem", "clip_id", "frame_id", "n_dets", "max_conf", "sources"],
            )
            writer.writeheader()
            writer.writerows(review_rows)
        stats["review_queue_rows"] = len(review_rows)

    summary_path = output_root / "ball_autolabel_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Ball autolabel complete → %s | %s", output_root, stats)
    return stats


def build_ball_yolo_dataset(
    *,
    autolabel_root: Path,
    dataset_root: Path,
    frames_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Assemble YOLO-detect train/val with scene-segment split."""
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
        used = {assignment[s] for s in stems}
        if len(used) == 1 and len(seg_ids) >= 2:
            flip = "val" if "train" in used else "train"
            for stem in groups[seg_ids[0]]:
                assignment[stem] = flip

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
        "path: {}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: ball\n".format(
            str(dataset_root.resolve()).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    meta = {
        "train_images": train_n,
        "val_images": val_n,
        "clips": sorted(by_clip),
        "split_mode": "scene_segment",
        "data_yaml": str(data_yaml),
    }
    (dataset_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Ball YOLO dataset ready: %s (%s)", data_yaml, meta)
    return data_yaml
