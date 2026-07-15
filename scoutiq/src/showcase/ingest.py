"""Multi-video ingest with million-frame-ready layout and CSV manifest.

Layout:
  data/footage/{train,val,holdout}/<clip>.mp4   (or absolute paths in config)
  data/frames/<game_id>/<clip_id>/frame_XXXXXX.jpg
  data/frames/manifest.csv
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.frame_extractor import extract_all_frames, get_video_info

logger = logging.getLogger(__name__)

MANIFEST_FIELDS = [
    "game_id",
    "clip_id",
    "split",
    "frame_id",
    "path",
    "timestamp",
    "width",
    "height",
    "fps",
    "video_path",
    "label_candidate",
]


@dataclass
class ClipSpec:
    video_path: Path
    split: str  # train | val | holdout
    game_id: str | None = None
    clip_id: str | None = None


def slugify(name: str, *, max_len: int = 64) -> str:
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    if not slug:
        slug = "clip"
    return slug[:max_len]


def clip_checksum(path: Path, *, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(nbytes))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def resolve_clip_ids(video_path: Path, game_id: str | None = None, clip_id: str | None = None) -> tuple[str, str]:
    cid = clip_id or slugify(video_path.name)
    gid = game_id or cid
    return gid, cid


def select_label_candidates(
    frame_count: int,
    *,
    stride: int = 8,
    max_candidates: int | None = 8000,
    scene_cut_indices: list[int] | None = None,
) -> set[int]:
    """Smart subset: every Nth frame + scene cuts, capped for QA budget."""
    selected: set[int] = set(range(0, frame_count, max(stride, 1)))
    if scene_cut_indices:
        selected.update(i for i in scene_cut_indices if 0 <= i < frame_count)
    ordered = sorted(selected)
    if max_candidates is not None and len(ordered) > max_candidates:
        # Keep evenly spaced subset of candidates.
        step = len(ordered) / max_candidates
        ordered = [ordered[int(i * step)] for i in range(max_candidates)]
    return set(ordered)


def detect_scene_cuts(
    video_path: Path,
    *,
    sample_stride: int = 2,
    hist_threshold: float = 0.55,
    max_frames: int | None = None,
) -> list[int]:
    """Lightweight histogram-diff scene cut detector (frame indices)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    cuts: list[int] = []
    prev_hist = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        if idx % sample_stride == 0:
            small = cv2.resize(frame, (160, 90))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                if score < hist_threshold:
                    cuts.append(idx)
            prev_hist = hist
        idx += 1
    cap.release()
    return cuts


def ingest_clip(
    clip: ClipSpec,
    *,
    frames_root: Path,
    stride: int = 8,
    max_label_candidates: int | None = 8000,
    detect_cuts: bool = True,
    copy_footage_to: Path | None = None,
) -> list[dict]:
    """Extract all frames for one clip and return manifest rows."""
    if not clip.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {clip.video_path}")
    if clip.split not in {"train", "val", "holdout"}:
        raise ValueError(f"split must be train|val|holdout, got {clip.split}")

    game_id, clip_id = resolve_clip_ids(clip.video_path, clip.game_id, clip.clip_id)
    info = get_video_info(clip.video_path)

    if copy_footage_to is not None:
        dest_dir = copy_footage_to / clip.split
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{clip_id}{clip.video_path.suffix.lower()}"
        if not dest.is_file():
            shutil.copy2(clip.video_path, dest)
            logger.info("Copied footage → %s", dest)

    out_dir = frames_root / game_id / clip_id
    frame_paths = extract_all_frames(clip.video_path, out_dir)

    cuts = detect_scene_cuts(clip.video_path) if detect_cuts else []
    candidates = select_label_candidates(
        len(frame_paths),
        stride=stride,
        max_candidates=max_label_candidates if clip.split != "holdout" else 0,
        scene_cut_indices=cuts if clip.split != "holdout" else None,
    )
    # Holdout: never mark as label candidates.
    if clip.split == "holdout":
        candidates = set()

    rows: list[dict] = []
    for fp in frame_paths:
        try:
            frame_id = int(fp.stem.split("_")[-1])
        except ValueError:
            continue
        rows.append({
            "game_id": game_id,
            "clip_id": clip_id,
            "split": clip.split,
            "frame_id": frame_id,
            "path": str(fp.resolve()),
            "timestamp": round(frame_id / info.fps, 4) if info.fps else 0.0,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "video_path": str(clip.video_path.resolve()),
            "label_candidate": int(frame_id in candidates),
        })

    meta = {
        "game_id": game_id,
        "clip_id": clip_id,
        "split": clip.split,
        "checksum": clip_checksum(clip.video_path),
        "frames": len(rows),
        "label_candidates": sum(r["label_candidate"] for r in rows),
        "scene_cuts": cuts[:200],
        "video": str(clip.video_path.resolve()),
    }
    (out_dir / "ingest_meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8",
    )
    logger.info(
        "Ingested %s/%s split=%s frames=%d label_candidates=%d",
        game_id, clip_id, clip.split, len(rows), meta["label_candidates"],
    )
    return rows


def write_manifest(rows: list[dict], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    logger.info("Wrote manifest %s (%d rows)", manifest_path, len(rows))
    return manifest_path


def read_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ingest_from_config(cfg: dict, scoutiq_root: Path) -> Path:
    """Ingest all clips listed in config; return manifest path."""
    data_root = _resolve(cfg.get("data_root", "data"), scoutiq_root)
    frames_root = data_root / "frames"
    footage_root = data_root / "footage"
    manifest_path = frames_root / "manifest.csv"

    stride = int(cfg.get("label_stride", 8))
    max_cands = cfg.get("max_label_candidates", 8000)
    detect_cuts = bool(cfg.get("detect_scene_cuts", True))
    copy_footage = bool(cfg.get("copy_footage", False))

    clips_cfg = cfg.get("clips") or []
    if not clips_cfg:
        raise ValueError("Config must include clips: [{path, split}, ...]")

    all_rows: list[dict] = []
    holdout_ids: list[str] = []

    for item in clips_cfg:
        video = _resolve(item["path"], scoutiq_root)
        split = str(item.get("split", "train")).lower()
        clip = ClipSpec(
            video_path=video,
            split=split,
            game_id=item.get("game_id"),
            clip_id=item.get("clip_id"),
        )
        rows = ingest_clip(
            clip,
            frames_root=frames_root,
            stride=stride,
            max_label_candidates=int(max_cands) if max_cands else None,
            detect_cuts=detect_cuts,
            copy_footage_to=footage_root if copy_footage else None,
        )
        all_rows.extend(rows)
        if split == "holdout":
            _, cid = resolve_clip_ids(video, clip.game_id, clip.clip_id)
            holdout_ids.append(cid)

    write_manifest(all_rows, manifest_path)

    # Hard gate: record holdout IDs so train never merges them.
    gate = {
        "holdout_clip_ids": sorted(set(holdout_ids)),
        "manifest": str(manifest_path.resolve()),
        "total_frames": len(all_rows),
        "label_candidates": sum(int(r["label_candidate"]) for r in all_rows),
    }
    gate_path = frames_root / "holdout_gate.json"
    gate_path.write_text(__import__("json").dumps(gate, indent=2), encoding="utf-8")
    logger.info("Holdout gate: %s", gate_path)
    return manifest_path


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()
