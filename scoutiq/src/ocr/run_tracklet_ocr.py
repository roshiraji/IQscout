"""Run jersey OCR on best-frame tracklet crops and aggregate per tracklet."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2

DEFAULT_OCR_GATES = {
    "min_agreeing_reads": 2,
    "min_consistency": 0.60,
    "min_read_confidence": 0.50,
    "min_crop_height": 70,
    "min_crop_width": 35,
    "min_sharpness": 25.0,
    "min_contrast": 18.0,
}


def _extract_digits(text: str) -> str | None:
    digits = re.findall(r"\d", text or "")
    if not digits:
        return None
    if len(digits) >= 2:
        return "".join(digits[:2])
    return digits[0]


def _init_ocr():
    try:
        from paddleocr import PaddleOCR

        return PaddleOCR(use_angle_cls=True, lang="en"), "paddleocr"
    except Exception:
        return None, "fallback"


def _ocr_read(ocr_engine, image_path: Path) -> tuple[str | None, float]:
    if ocr_engine is None:
        return None, 0.0
    try:
        result = ocr_engine.predict(str(image_path))
        texts: list[tuple[str, float]] = []
        for line in result or []:
            for item in line or []:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                ):
                    text = str(item[1][0])
                    conf = float(item[1][1])
                    texts.append((text, conf))
        if not texts:
            return None, 0.0
        best_text, conf = max(texts, key=lambda x: x[1])
        return _extract_digits(best_text), conf
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


def _vote_weight(read: dict) -> float:
    """Weight OCR votes by confidence and crop opportunity score."""
    conf = max(float(read.get("conf") or 0.0), 0.05)
    opportunity = float(read.get("opportunity_score") or 0.0)
    opportunity_factor = 0.55 + 0.45 * max(0.0, min(opportunity, 1.0))
    return conf * opportunity_factor


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
        # Distinguish weak agreement from strong conflicting predictions.
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
) -> dict:
    gates = {**DEFAULT_OCR_GATES, **(jersey_ocr or {})}
    min_agreeing = int(gates.get("min_agreeing_reads", 2))
    min_consistency = float(gates.get("min_consistency", 0.60))
    min_read_conf = float(gates.get("min_read_confidence", 0.50))
    viable_kwargs = {
        "min_height": int(gates.get("min_crop_height", 70)),
        "min_width": int(gates.get("min_crop_width", 35)),
        "min_sharpness": float(gates.get("min_sharpness", 25.0)),
        "min_contrast": float(gates.get("min_contrast", 18.0)),
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ocr_engine, backend = _init_ocr()
    out_rows = []
    for track in manifest:
        track_id = int(track["track_id"])
        reads = []
        for keyframe in track.get("keyframes", []):
            crop_path = Path(keyframe["crop_path"])
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            opportunity = float(keyframe.get("opportunity_score") or 0.0)
            h, w = img.shape[:2]
            x1, x2 = int(w * 0.2), int(w * 0.8)
            y1, y2 = int(h * 0.25), int(h * 0.8)
            torso = img[y1:y2, x1:x2]
            torso_path = crop_path.with_name(crop_path.stem + "_torso.jpg")
            cv2.imwrite(str(torso_path), torso)
            is_viable, viability_reason = _crop_is_ocr_viable(
                torso, **viable_kwargs
            )
            if is_viable:
                value, conf = _ocr_read(ocr_engine, torso_path)
            else:
                value, conf = None, 0.0
            reads.append(
                {
                    "frame_idx": keyframe["frame_idx"],
                    "value": value,
                    "conf": round(conf, 4),
                    "opportunity_score": opportunity,
                    "torso_crop": str(torso_path.resolve()),
                    "ocr_viable": is_viable,
                    "viability_reason": viability_reason,
                }
            )
            for region_idx, region in enumerate(_extract_torso_regions(img)):
                prepped = _preprocess_torso_for_ocr(region)
                prepped_path = crop_path.with_name(
                    f"{crop_path.stem}_torso_prepped_{region_idx}.jpg"
                )
                cv2.imwrite(str(prepped_path), prepped)
                is_viable, viability_reason = _crop_is_ocr_viable(
                    prepped, **viable_kwargs
                )
                if is_viable:
                    prepped_value, prepped_conf = _ocr_read(
                        ocr_engine,
                        prepped_path,
                    )
                else:
                    prepped_value, prepped_conf = None, 0.0
                reads.append(
                    {
                        "frame_idx": keyframe["frame_idx"],
                        "value": prepped_value,
                        "conf": round(prepped_conf, 4),
                        "opportunity_score": opportunity,
                        "torso_crop": str(prepped_path.resolve()),
                        "preprocessed": True,
                        "ocr_viable": is_viable,
                        "viability_reason": viability_reason,
                    }
                )

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
                "candidate_keyframes": len(track.get("keyframes") or []),
                "viable_crop_count": sum(
                    1 for read in reads if read.get("ocr_viable", True)
                ),
                "ocr_reads": reads,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"backend": backend, "tracklets": out_rows}, indent=2),
        encoding="utf-8",
    )
    return {
        "backend": backend,
        "tracklets": len(out_rows),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on best-frame tracklet torso crops."
    )
    parser.add_argument(
        "--tracklets", required=True, help="Path to tracklet_keyframes.json"
    )
    parser.add_argument("--output", required=True, help="Path to output OCR JSON")
    args = parser.parse_args()
    summary = run_tracklet_ocr(Path(args.tracklets), Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
