"""Extract body embeddings for tracklets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _simple_reid(image) -> list[float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s], axis=0)
    norm = np.linalg.norm(vec) or 1.0
    return (vec / norm).astype(float).tolist()


def embed_tracklets(manifest_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for track in manifest:
        embs = []
        for keyframe in track.get("keyframes", []):
            crop = cv2.imread(str(keyframe["crop_path"]))
            if crop is None or crop.size == 0:
                continue
            embs.append({"frame_idx": keyframe["frame_idx"], "embedding": _simple_reid(crop)})
        rows.append({"track_id": int(track["track_id"]), "reid_embeddings": embs})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"backend": "colorhist_fallback", "tracklets": rows}, indent=2), encoding="utf-8")
    return {"tracklets": len(rows), "output": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fallback ReID embeddings for tracklets.")
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(embed_tracklets(Path(args.tracklets), Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
