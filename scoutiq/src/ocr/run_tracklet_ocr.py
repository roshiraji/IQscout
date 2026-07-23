"""Run jersey OCR on best-frame tracklet crops and aggregate per tracklet."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OCR_GATES = {
    "min_agreeing_reads": 2,
    "min_consistency": 0.60,
    "min_read_confidence": 0.55,
    "min_crop_height": 70,
    "min_crop_width": 35,
    "min_sharpness": 25.0,
    "min_contrast": 18.0,
    "max_consecutive_frame_failures": 3,
    "max_processed_frames": 5,
    "max_ocr_calls_per_person": 8,
    "use_angle_cls": False,
    "save_debug_crops": False,
}

_OCR_ENGINE: Any | None = None
_OCR_BACKEND: str | None = None
_OCR_ENGINE_KEY: tuple[bool, str] | None = None


def _extract_digits(text: str) -> str | None:
    digits = re.findall(r"\d", text or "")
    if not digits:
        return None
    if len(digits) >= 2:
        return "".join(digits[:2])
    return digits[0]


def get_ocr_engine(*, use_angle_cls: bool = False, lang: str = "en") -> tuple[Any | None, str]:
    """Initialize PaddleOCR once and reuse it for the process lifetime."""
    global _OCR_ENGINE, _OCR_BACKEND, _OCR_ENGINE_KEY
    key = (use_angle_cls, lang)
    if _OCR_ENGINE is not None and _OCR_ENGINE_KEY == key:
        return _OCR_ENGINE, _OCR_BACKEND or "paddleocr"

    try:
        from paddleocr import PaddleOCR

        init_start = time.perf_counter()
        _OCR_ENGINE = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        _OCR_BACKEND = "paddleocr"
        _OCR_ENGINE_KEY = key
        logger.info(
            "PaddleOCR initialized in %.2fs (use_angle_cls=%s)",
            time.perf_counter() - init_start,
            use_angle_cls,
        )
        return _OCR_ENGINE, _OCR_BACKEND
    except Exception as exc:
        logger.warning("PaddleOCR unavailable, using fallback OCR: %s", exc)
        _OCR_ENGINE = None
        _OCR_BACKEND = "fallback"
        _OCR_ENGINE_KEY = key
        return None, "fallback"


def _parse_ocr_result(result: Any) -> tuple[str | None, float]:
    texts: list[tuple[str, float]] = []

    def _append(text: str, conf: float) -> None:
        texts.append((text, conf))

    for line in result or []:
        if isinstance(line, (list, tuple)):
            for item in line:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                ):
                    _append(str(item[1][0]), float(item[1][1]))
                elif (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], str)
                ):
                    _append(str(item[1]), float(item[0]))
        elif isinstance(line, dict):
            text = line.get("rec_text") or line.get("text")
            conf = line.get("rec_score") or line.get("score") or 0.0
            if text:
                _append(str(text), float(conf))

    if not texts:
        return None, 0.0
    best_text, conf = max(texts, key=lambda x: x[1])
    return _extract_digits(best_text), conf


def _ocr_read(
    ocr_engine,
    image: np.ndarray | Path | str,
    *,
    use_angle_cls: bool = False,
) -> tuple[str | None, float]:
    if ocr_engine is None:
        return None, 0.0
    try:
        if isinstance(image, (str, Path)):
            result = ocr_engine.predict(str(image))
        elif hasattr(ocr_engine, "ocr"):
            result = ocr_engine.ocr(image, cls=use_angle_cls)
        else:
            result = ocr_engine.predict(image)
        return _parse_ocr_result(result)
    except Exception:
        return None, 0.0


def _crop_is_ocr_viable(
    torso,
    *,
    min_height: int = 70,
    min_width: int = 35,
    min_sharpness: float = 25.0,
    min_contrast: float = 18.0,
) -> tuple[bool, str]:
    if torso is None or torso.size == 0:
        return False, "empty_crop"

    height, width = torso.shape[:2]

    if height < min_height or width < min_width:
        return False, "too_small"

    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < min_sharpness:
        return False, "too_blurry"

    contrast = float(gray.std())

    if contrast < min_contrast:
        return False, "low_contrast"

    return True, "usable"


def _preprocess_torso_for_ocr(torso):
    """Upscale and boost contrast so small jersey digits are easier to read."""
    if torso is None or torso.size == 0:
        return torso
    enlarged = cv2.resize(
        torso,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _extract_torso_regions(img):
    """Return front and back torso strips for OCR."""
    h, w = img.shape[:2]
    regions = []
    for x1_frac, x2_frac in ((0.20, 0.80), (0.15, 0.85)):
        x1, x2 = int(w * x1_frac), int(w * x2_frac)
        y1, y2 = int(h * 0.25), int(h * 0.80)
        torso = img[y1:y2, x1:x2]
        if torso.size:
            regions.append(torso)
    return regions


def _maybe_save_debug_crop(image: np.ndarray, path: Path, *, enabled: bool) -> str:
    if enabled:
        cv2.imwrite(str(path), image)
        return str(path.resolve())
    return f"memory://{path.name}"


def _make_read_record(
    *,
    frame_idx: int,
    opportunity: float,
    value: str | None,
    conf: float,
    crop_ref: str,
    is_viable: bool,
    viability_reason: str,
    preprocessed: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "frame_idx": frame_idx,
        "value": value,
        "conf": round(conf, 4),
        "opportunity_score": opportunity,
        "torso_crop": crop_ref,
        "ocr_viable": is_viable,
        "viability_reason": viability_reason,
    }
    if preprocessed:
        record["preprocessed"] = True
    return record


def _vote_weight(read: dict) -> float:
    """Weight OCR votes by confidence and crop opportunity score."""
    conf = max(float(read.get("conf") or 0.0), 0.05)
    opportunity = float(read.get("opportunity_score") or 0.0)
    opportunity_factor = 0.55 + 0.45 * max(0.0, min(opportunity, 1.0))
    return conf * opportunity_factor


def _is_confident_read(read: dict, *, min_read_confidence: float) -> bool:
    """True when a read counts as usable evidence for aggregation."""
    return (
        bool(read.get("value"))
        and float(read.get("conf") or 0.0) >= min_read_confidence
        and read.get("ocr_viable", True)
    )


def _max_confident_agreement(reads: list[dict], *, min_read_confidence: float) -> int:
    """Highest count of identical confident jersey reads collected so far."""
    counts: Counter[str] = Counter()
    for read in reads:
        if _is_confident_read(read, min_read_confidence=min_read_confidence):
            counts[str(read["value"])] += 1
    return max(counts.values()) if counts else 0


def _should_stop_ocr_early(
    reads: list[dict],
    *,
    processed_frames: int,
    consecutive_frame_failures: int,
    total_ocr_calls: int,
    min_agreeing_reads: int,
    min_read_confidence: float,
    max_consecutive_failures: int,
    max_processed_frames: int,
    max_ocr_calls: int,
) -> tuple[bool, str | None]:
    """Return (stop, reason) once enough evidence exists or failure is clear."""
    if _max_confident_agreement(
        reads, min_read_confidence=min_read_confidence
    ) >= min_agreeing_reads:
        return True, "sufficient_agreeing_reads"
    if consecutive_frame_failures >= max_consecutive_failures:
        return True, "consecutive_frame_failures"
    if processed_frames >= max_processed_frames:
        return True, "max_processed_frames"
    if total_ocr_calls >= max_ocr_calls:
        return True, "max_ocr_calls"
    return False, None


def _run_single_ocr(
    *,
    ocr_engine,
    image: np.ndarray,
    crop_path: Path,
    suffix: str,
    save_debug_crops: bool,
    use_angle_cls: bool,
    viable_kwargs: dict,
    frame_idx: int,
    opportunity: float,
    preprocessed: bool = False,
) -> tuple[dict, int]:
    """Run one OCR attempt and return (read_record, ocr_calls_used)."""
    debug_path = crop_path.with_name(f"{crop_path.stem}_{suffix}.jpg")
    crop_ref = _maybe_save_debug_crop(
        image,
        debug_path,
        enabled=save_debug_crops,
    )
    is_viable, viability_reason = _crop_is_ocr_viable(image, **viable_kwargs)
    if is_viable:
        value, conf = _ocr_read(
            ocr_engine,
            image if not save_debug_crops else debug_path,
            use_angle_cls=use_angle_cls,
        )
        ocr_calls = 1
    else:
        value, conf = None, 0.0
        ocr_calls = 0
    return (
        _make_read_record(
            frame_idx=frame_idx,
            opportunity=opportunity,
            value=value,
            conf=conf,
            crop_ref=crop_ref,
            is_viable=is_viable,
            viability_reason=viability_reason,
            preprocessed=preprocessed,
        ),
        ocr_calls,
    )


def _process_keyframe_reads(
    keyframe: dict,
    *,
    ocr_engine,
    viable_kwargs: dict,
    min_read_confidence: float,
    save_debug_crops: bool,
    use_angle_cls: bool,
) -> tuple[list[dict], int]:
    """Run OCR on one candidate keyframe with cascading preprocessing."""
    crop_path = Path(keyframe["crop_path"])
    img = cv2.imread(str(crop_path))
    if img is None:
        return [], 0

    opportunity = float(keyframe.get("opportunity_score") or 0.0)
    frame_idx = int(keyframe["frame_idx"])
    reads: list[dict] = []
    ocr_calls = 0

    h, w = img.shape[:2]
    x1, x2 = int(w * 0.2), int(w * 0.8)
    y1, y2 = int(h * 0.25), int(h * 0.8)
    torso = img[y1:y2, x1:x2]

    # 1) Original torso crop.
    read, used = _run_single_ocr(
        ocr_engine=ocr_engine,
        image=torso,
        crop_path=crop_path,
        suffix="torso",
        save_debug_crops=save_debug_crops,
        use_angle_cls=use_angle_cls,
        viable_kwargs=viable_kwargs,
        frame_idx=frame_idx,
        opportunity=opportunity,
    )
    reads.append(read)
    ocr_calls += used
    if _is_confident_read(read, min_read_confidence=min_read_confidence):
        return reads, ocr_calls

    regions = _extract_torso_regions(img)

    # 2) First preprocessing variant only if the original crop failed.
    if regions:
        prepped = _preprocess_torso_for_ocr(regions[0])
        read, used = _run_single_ocr(
            ocr_engine=ocr_engine,
            image=prepped,
            crop_path=crop_path,
            suffix="torso_prepped_0",
            save_debug_crops=save_debug_crops,
            use_angle_cls=use_angle_cls,
            viable_kwargs=viable_kwargs,
            frame_idx=frame_idx,
            opportunity=opportunity,
            preprocessed=True,
        )
        reads.append(read)
        ocr_calls += used
        if _is_confident_read(read, min_read_confidence=min_read_confidence):
            return reads, ocr_calls

    # 3) Second preprocessing variant only if the first two failed.
    if len(regions) > 1:
        prepped = _preprocess_torso_for_ocr(regions[1])
        read, used = _run_single_ocr(
            ocr_engine=ocr_engine,
            image=prepped,
            crop_path=crop_path,
            suffix="torso_prepped_1",
            save_debug_crops=save_debug_crops,
            use_angle_cls=use_angle_cls,
            viable_kwargs=viable_kwargs,
            frame_idx=frame_idx,
            opportunity=opportunity,
            preprocessed=True,
        )
        reads.append(read)
        ocr_calls += used

    return reads, ocr_calls


def _aggregate_jersey(
    reads: list[dict],
    *,
    min_agreeing_reads: int,
    min_consistency: float,
    min_read_confidence: float,
) -> tuple[str | None, float, str, str]:
    valid_reads = [
        r
        for r in reads
        if r.get("value")
        and float(r.get("conf") or 0.0) >= min_read_confidence
        and r.get("ocr_viable", True)
    ]
    if not valid_reads:
        return None, 0.0, "unreadable", "no_viable_jersey_crops"

    weighted = Counter()
    for read in valid_reads:
        weighted[read["value"]] += _vote_weight(read)
    jersey, score = weighted.most_common(1)[0]
    total = sum(weighted.values())
    consistency = float(score / max(0.001, total))
    agreeing_reads = [read for read in valid_reads if read["value"] == jersey]

    if len(agreeing_reads) < min_agreeing_reads or consistency < min_consistency:
        distinct = [value for value, mass in weighted.items() if mass > 0]
        if len(distinct) >= 2 and consistency < min_consistency:
            return None, consistency, "unreadable", "conflicting_jersey_predictions"
        return None, consistency, "unreadable", "insufficient_repeated_agreement"

    return jersey, consistency, "readable", "repeated_ocr_agreement"


def run_tracklet_ocr(
    manifest_path: Path,
    output_path: Path,
    *,
    jersey_ocr: dict | None = None,
    profile: bool = True,
) -> dict:
    gates = {**DEFAULT_OCR_GATES, **(jersey_ocr or {})}
    min_agreeing = int(gates.get("min_agreeing_reads", 2))
    min_consistency = float(gates.get("min_consistency", 0.60))
    min_read_conf = float(gates.get("min_read_confidence", 0.55))
    max_consecutive_failures = int(
        gates.get("max_consecutive_frame_failures", 3)
    )
    max_processed_frames = int(gates.get("max_processed_frames", 5))
    max_ocr_calls = int(gates.get("max_ocr_calls_per_person", 8))
    save_debug_crops = bool(gates.get("save_debug_crops", False))
    use_angle_cls = bool(gates.get("use_angle_cls", False))
    viable_kwargs = {
        "min_height": int(gates.get("min_crop_height", 70)),
        "min_width": int(gates.get("min_crop_width", 35)),
        "min_sharpness": float(gates.get("min_sharpness", 25.0)),
        "min_contrast": float(gates.get("min_contrast", 18.0)),
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    init_start = time.perf_counter()
    ocr_engine, backend = get_ocr_engine(use_angle_cls=use_angle_cls)
    ocr_init_seconds = time.perf_counter() - init_start

    out_rows = []
    per_track_profiles: list[dict[str, Any]] = []
    total_ocr_seconds = 0.0

    for track in manifest:
        track_id = int(track["track_id"])
        track_start = time.perf_counter()
        reads: list[dict] = []
        consecutive_frame_failures = 0
        processed_frames = 0
        total_ocr_calls = 0
        early_exit_reason: str | None = None
        keyframes = track.get("keyframes") or []

        for keyframe in keyframes:
            keyframe_reads, keyframe_calls = _process_keyframe_reads(
                keyframe,
                ocr_engine=ocr_engine,
                viable_kwargs=viable_kwargs,
                min_read_confidence=min_read_conf,
                save_debug_crops=save_debug_crops,
                use_angle_cls=use_angle_cls,
            )
            if not keyframe_reads:
                continue

            reads.extend(keyframe_reads)
            total_ocr_calls += keyframe_calls
            processed_frames += 1

            frame_had_confident_read = any(
                _is_confident_read(read, min_read_confidence=min_read_conf)
                for read in keyframe_reads
            )
            if frame_had_confident_read:
                consecutive_frame_failures = 0
            else:
                consecutive_frame_failures += 1

            should_stop, stop_reason = _should_stop_ocr_early(
                reads,
                processed_frames=processed_frames,
                consecutive_frame_failures=consecutive_frame_failures,
                total_ocr_calls=total_ocr_calls,
                min_agreeing_reads=min_agreeing,
                min_read_confidence=min_read_conf,
                max_consecutive_failures=max_consecutive_failures,
                max_processed_frames=max_processed_frames,
                max_ocr_calls=max_ocr_calls,
            )
            if should_stop:
                early_exit_reason = stop_reason
                break

        track_seconds = time.perf_counter() - track_start
        total_ocr_seconds += track_seconds

        jersey, consistency, status, reason = _aggregate_jersey(
            reads,
            min_agreeing_reads=min_agreeing,
            min_consistency=min_consistency,
            min_read_confidence=min_read_conf,
        )

        out_rows.append(
            {
                "track_id": track_id,
                "jersey_prediction": jersey,
                "ocr_consistency": round(consistency, 4),
                "identity_evidence_status": status,
                "identity_evidence_reason": reason,
                "candidate_keyframes": len(keyframes),
                "processed_keyframes": processed_frames,
                "ocr_calls": total_ocr_calls,
                "early_exit_reason": early_exit_reason,
                "viable_crop_count": sum(
                    1 for read in reads if read.get("ocr_viable", True)
                ),
                "ocr_reads": reads,
            }
        )
        per_track_profiles.append(
            {
                "track_id": track_id,
                "ocr_seconds": round(track_seconds, 2),
                "ocr_calls": total_ocr_calls,
                "processed_keyframes": processed_frames,
                "early_exit_reason": early_exit_reason,
            }
        )

        if profile:
            logger.info(
                "Player %s\n---------\nOCR: %.2fs (%d calls, %d frames)\nEarly exit: %s",
                track_id,
                track_seconds,
                total_ocr_calls,
                processed_frames,
                early_exit_reason or "none",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"backend": backend, "tracklets": out_rows}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "backend": backend,
        "tracklets": len(out_rows),
        "output": str(output_path.resolve()),
        "ocr_init_seconds": round(ocr_init_seconds, 2),
        "ocr_total_seconds": round(total_ocr_seconds, 2),
        "per_track_profiles": per_track_profiles,
        "limits": {
            "max_processed_frames": max_processed_frames,
            "max_ocr_calls_per_person": max_ocr_calls,
            "max_consecutive_frame_failures": max_consecutive_failures,
        },
    }
    if profile:
        logger.info(
            "OCR summary: init=%.2fs total=%.2fs tracks=%d max_calls=%d",
            ocr_init_seconds,
            total_ocr_seconds,
            len(out_rows),
            max_ocr_calls,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on best-frame tracklet torso crops."
    )
    parser.add_argument(
        "--tracklets", required=True, help="Path to tracklet_keyframes.json"
    )
    parser.add_argument("--output", required=True, help="Path to output OCR JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run_tracklet_ocr(Path(args.tracklets), Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
