"""Extract face embeddings from keyframe crops (best-effort)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def _init_face_app(force_insightface: bool):
    if not force_insightface:
        return None, "fallback"
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0)
        return app, "insightface"
    except Exception:
        return None, "fallback"


def _fallback_embedding(image) -> list[float]:
    # small color-hist fallback so the pipeline stays runnable
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    norm = np.linalg.norm(hist) or 1.0
    return (hist / norm).astype(float).tolist()


def embed_tracklets(manifest_path: Path, output_path: Path, *, use_insightface: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    force = use_insightface or os.getenv("SCOUTIQ_ENABLE_INSIGHTFACE") == "1"
    face_app, backend = _init_face_app(force)
    rows = []
    for track in manifest:
        embeddings = []
        for keyframe in track.get("keyframes", []):
            crop = cv2.imread(str(keyframe["crop_path"]))
            if crop is None:
                continue
            h, w = crop.shape[:2]
            head = crop[0 : max(1, int(h * 0.35)), int(w * 0.2) : int(w * 0.8)]
            if head.size == 0 or min(head.shape[:2]) < 24:
                continue
            vec = None
            if face_app is not None:
                faces = face_app.get(head)
                if faces:
                    vec = faces[0].embedding.astype(float).tolist()
            if vec is None:
                vec = _fallback_embedding(head)
            embeddings.append({"frame_idx": keyframe["frame_idx"], "embedding": vec})
        rows.append({"track_id": int(track["track_id"]), "face_embeddings": embeddings})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"backend": backend, "tracklets": rows}, indent=2), encoding="utf-8")
    return {"backend": backend, "tracklets": len(rows), "output": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face embeddings for tracklets.")
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-insightface", action="store_true", help="Enable InsightFace backend")
    args = parser.parse_args()
    print(
        json.dumps(
            embed_tracklets(
                Path(args.tracklets),
                Path(args.output),
                use_insightface=args.use_insightface,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
