"""LocateAnything runner wrapper.

If the LocateAnything executable is present (set LOCATEANYTHING_EXE env var or
pass executable= explicitly), it is used for text-prompted grounding.

Otherwise this module logs exactly what is needed and returns an empty list
so the pipeline continues using YOLO11-seg as the primary detector.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ----- class prompts ---------------------------------------------------------
CLASS_PROMPTS: dict[str, str] = {
    "ball": "orange basketball on the court",
    "player_red": "basketball player wearing a red uniform on the court",
    "player_black_blue": "basketball player wearing a black or blue uniform on the court",
    "referee": "basketball referee on the court",
}

CLASS_IDS: dict[str, int] = {
    "ball": 0,
    "player_red": 1,
    "player_black_blue": 2,
    "referee": 3,
}


def _find_executable() -> Path | None:
    env_path = os.environ.get("LOCATEANYTHING_EXE", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        logger.warning(
            "LOCATEANYTHING_EXE is set but file not found: %s", p
        )
    # Try common relative paths.
    candidates = [
        Path("Z:/AI/models/LocateAnything-Q8/locate-anything-q8_0.gguf"),
        Path("Z:/AI/models/LocateAnything/locate-anything.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def locate_anything_available() -> bool:
    return _find_executable() is not None


def _run_locateanything(executable: Path, image_path: Path, class_name: str, prompt: str) -> list[dict]:
    cmd = [
        str(executable),
        "--image", str(image_path),
        "--prompt", prompt,
        "--class", class_name,
        "--output-json",
    ]
    logger.debug("LocateAnything cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            logger.warning("LocateAnything exited %d: %s", proc.returncode, proc.stderr[:200])
            return []
        raw = json.loads(proc.stdout.strip())
        if isinstance(raw, list):
            return raw
        return raw.get("detections", [])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
        logger.warning("LocateAnything failed for %s: %s", image_path.name, exc)
        return []


def run_locateanything_on_frame(
    image_path: Path,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    executable: Path | None = None,
) -> list[dict]:
    exe = executable or _find_executable()
    if exe is None:
        logger.debug(
            "LocateAnything not available. Set LOCATEANYTHING_EXE to:\n"
            "  Z:/AI/models/LocateAnything-Q8/locate-anything-q8_0.gguf\n"
            "Pipeline will use YOLO11-seg as primary detector instead."
        )
        return []

    all_detections: list[dict] = []
    for class_name, prompt in CLASS_PROMPTS.items():
        raw_rows = _run_locateanything(exe, image_path, class_name, prompt)
        for row in raw_rows:
            all_detections.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "class_id": CLASS_IDS.get(class_name, -1),
                    "class_name": class_name,
                    "prompt": prompt,
                    "confidence": float(row.get("confidence", row.get("score", 0.5))),
                    "bbox_xyxy": [int(v) for v in row.get("bbox_xyxy", row.get("box", [0, 0, 1, 1]))],
                    "source": "locateanything",
                }
            )
    return all_detections
