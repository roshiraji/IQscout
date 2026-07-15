"""Ingest NBA annotation data from multiple folder formats into unified YOLO layout."""

from __future__ import annotations

import csv
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CLASSES: dict[int, str] = {
    0: "player",
    1: "referee",
    2: "hoop",
    3: "ball",
}


@dataclass
class IngestStats:
    images: int = 0
    labels: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


@dataclass
class Sample:
    stem: str
    image_path: Path
    label_lines: list[str]
    source: str


def _name_to_id(name: str, classes: dict[int, str]) -> int | None:
    lookup = {v.lower(): k for k, v in classes.items()}
    return lookup.get(name.strip().lower())


def _pixel_bbox_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy, w / img_w, h / img_h


def _read_yolo_label(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def _find_image(stem: str, search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = directory / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def ingest_yolo_split(
    yolo_root: Path,
    classes: dict[int, str],
    pool: dict[str, Sample],
) -> int:
    """Copy images+labels from an existing YOLO train/val/test tree."""
    added = 0
    for split in ("train", "val", "test"):
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        if not img_dir.is_dir():
            continue
        for image_path in sorted(img_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = lbl_dir / f"{image_path.stem}.txt"
            lines = _read_yolo_label(label_path)
            if not lines:
                continue
            pool[image_path.stem] = Sample(
                stem=image_path.stem,
                image_path=image_path,
                label_lines=lines,
                source=f"yolo_dataset/{split}",
            )
            added += 1
    return added


def ingest_raw_labels(
    labels_dir: Path,
    frames_dirs: list[Path],
    classes: dict[int, str],
    pool: dict[str, Sample],
) -> int:
    """Pair flat YOLO .txt labels with frame images."""
    added = 0
    if not labels_dir.is_dir():
        return 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = _read_yolo_label(label_path)
        if not lines:
            continue
        image_path = _find_image(label_path.stem, frames_dirs)
        if image_path is None:
            continue
        pool[label_path.stem] = Sample(
            stem=label_path.stem,
            image_path=image_path,
            label_lines=lines,
            source="raw_labels",
        )
        added += 1
    return added


def ingest_annotations_csv(
    csv_path: Path,
    frames_dirs: list[Path],
    classes: dict[int, str],
    pool: dict[str, Sample],
) -> int:
    """Convert MakeSense / Roboflow-style CSV bboxes to YOLO lines."""
    if not csv_path.is_file():
        return 0

    rows_by_image: dict[str, list[str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_name = row.get("image_name", "").strip()
            label_name = row.get("label_name", "").strip()
            if not image_name or not label_name:
                continue
            class_id = _name_to_id(label_name, classes)
            if class_id is None:
                continue
            try:
                img_w = int(float(row["image_width"]))
                img_h = int(float(row["image_height"]))
                bx = float(row["bbox_x"])
                by = float(row["bbox_y"])
                bw = float(row["bbox_width"])
                bh = float(row["bbox_height"])
            except (KeyError, ValueError):
                continue
            cx, cy, nw, nh = _pixel_bbox_to_yolo(bx, by, bw, bh, img_w, img_h)
            line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            rows_by_image.setdefault(Path(image_name).stem, []).append(line)

    added = 0
    for stem, lines in rows_by_image.items():
        image_path = _find_image(stem, frames_dirs)
        if image_path is None:
            continue
        pool[stem] = Sample(stem=stem, image_path=image_path, label_lines=lines, source="annotations_csv")
        added += 1
    return added


def ingest_legacy_yolo_zip(
    zip_path: Path,
    frames_dirs: list[Path],
    classes: dict[int, str],
    pool: dict[str, Sample],
) -> int:
    """Read legacy Roboflow YOLO zip exports (flat .txt files)."""
    if not zip_path.is_file():
        return 0
    added = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            stem = Path(name).stem
            lines = [ln.strip() for ln in zf.read(name).decode("utf-8").splitlines() if ln.strip()]
            if not lines:
                continue
            image_path = _find_image(stem, frames_dirs)
            if image_path is None:
                continue
            pool[stem] = Sample(stem=stem, image_path=image_path, label_lines=lines, source="legacy_yolo_zip")
            added += 1
    return added


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train_n = max(int(total * train_ratio), 1)
    val_n = max(int(total * val_ratio), 1) if total > 2 else 0
    test_n = max(total - train_n - val_n, 0)
    while train_n + val_n + test_n > total:
        if train_n > 1:
            train_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            test_n -= 1
    return train_n, val_n, test_n


def build_merged_dataset(
    *,
    output_dir: Path,
    classes: dict[int, str],
    yolo_dataset: Path | None = None,
    raw_labels: Path | None = None,
    frames_dirs: list[Path] | None = None,
    annotations_csv: Path | None = None,
    legacy_yolo_zip: Path | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Path, IngestStats]:
    """
    Merge all available sources into output_dir with train/val/test splits.

    Later sources overwrite earlier ones for the same image stem (priority:
    yolo_dataset < raw_labels < csv < legacy zip).
    """
    frames_dirs = frames_dirs or []
    pool: dict[str, Sample] = {}
    stats = IngestStats()

    order = [
        ("yolo_dataset", lambda: ingest_yolo_split(yolo_dataset, classes, pool) if yolo_dataset else 0),
        ("raw_labels", lambda: ingest_raw_labels(raw_labels, frames_dirs, classes, pool) if raw_labels else 0),
        ("annotations_csv", lambda: ingest_annotations_csv(annotations_csv, frames_dirs, classes, pool) if annotations_csv else 0),
        ("legacy_yolo_zip", lambda: ingest_legacy_yolo_zip(legacy_yolo_zip, frames_dirs, classes, pool) if legacy_yolo_zip else 0),
    ]
    for source_name, fn in order:
        count = fn()
        if count:
            stats.sources[source_name] = count

    samples = list(pool.values())
    if not samples:
        raise FileNotFoundError("No labeled samples found from configured NBA data sources.")

    random.Random(seed).shuffle(samples)
    train_n, val_n, test_n = _split_counts(len(samples), train_ratio, val_ratio)
    splits = {
        "train": samples[:train_n],
        "val": samples[train_n : train_n + val_n],
        "test": samples[train_n + val_n : train_n + val_n + test_n],
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for sample in items:
            dest_img = output_dir / split / "images" / f"{sample.stem}{sample.image_path.suffix.lower()}"
            dest_lbl = output_dir / split / "labels" / f"{sample.stem}.txt"
            shutil.copy2(sample.image_path, dest_img)
            dest_lbl.write_text("\n".join(sample.label_lines) + "\n", encoding="utf-8")
            stats.images += 1
            stats.labels += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    manifest = {
        "total_samples": len(samples),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "sources": stats.sources,
        "classes": classes,
    }
    (output_dir / "ingest_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8",
    )

    return output_dir / "data.yaml", stats
