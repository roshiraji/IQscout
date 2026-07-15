"""Run jersey OCR on tracklet keyframes and aggregate per tracklet."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2


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
        # fallback: no OCR dependency installed
        return None, 0.0
    try:
        result = ocr_engine.predict(str(image_path))
        # PaddleOCR v3 nested structure
        texts: list[tuple[str, float]] = []
        for line in result or []:
            for item in line or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (list, tuple)):
                    text = str(item[1][0])
                    conf = float(item[1][1])
                    texts.append((text, conf))
        if not texts:
            return None, 0.0
        best_text, conf = max(texts, key=lambda x: x[1])
        return _extract_digits(best_text), conf
    except Exception:
        return None, 0.0


def run_tracklet_ocr(manifest_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ocr_engine, backend = _init_ocr()
    out_rows = []
    for track in manifest:
        track_id = int(track["track_id"])
        reads = []
        for keyframe in track.get("keyframes", []):
            crop_path = Path(keyframe["crop_path"])
            # read torso region only: center vertical strip for jersey
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            x1, x2 = int(w * 0.2), int(w * 0.8)
            y1, y2 = int(h * 0.25), int(h * 0.8)
            torso = img[y1:y2, x1:x2]
            torso_path = crop_path.with_name(crop_path.stem + "_torso.jpg")
            cv2.imwrite(str(torso_path), torso)
            value, conf = _ocr_read(ocr_engine, torso_path)
            reads.append(
                {
                    "frame_idx": keyframe["frame_idx"],
                    "value": value,
                    "conf": round(conf, 4),
                    "torso_crop": str(torso_path.resolve()),
                }
            )

        valid_reads = [r for r in reads if r["value"]]
        if valid_reads:
            weighted = Counter()
            for r in valid_reads:
                weighted[r["value"]] += max(r["conf"], 0.05)
            jersey, score = weighted.most_common(1)[0]
            consistency = float(score / max(0.001, sum(weighted.values())))
        else:
            jersey, consistency = None, 0.0
        out_rows.append(
            {
                "track_id": track_id,
                "jersey_prediction": jersey,
                "ocr_consistency": round(consistency, 4),
                "ocr_reads": reads,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"backend": backend, "tracklets": out_rows}, indent=2), encoding="utf-8")
    return {"backend": backend, "tracklets": len(out_rows), "output": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on tracklet torso keyframes.")
    parser.add_argument("--tracklets", required=True, help="Path to tracklet_keyframes.json")
    parser.add_argument("--output", required=True, help="Path to output OCR JSON")
    args = parser.parse_args()
    summary = run_tracklet_ocr(Path(args.tracklets), Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
