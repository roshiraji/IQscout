"""Deep visual identity features for Confirm-selected basketball players.

ByteTrack is deliberately not used here: it provides candidate detections but
can exchange IDs whenever players overlap.  OSNet supplies a person
re-identification vector for each detection, anchored to the body the user
clicked.  Vectors are kept in memory only; they are never written into the
tracking JSON artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VECTOR_KEY = "_reid_vector"
_MODEL_NAME = "osnet_x0_25"
_MODEL_FILENAME = "osnet_x0_25_market1501.pth"


def reid_vector(det: dict) -> np.ndarray | None:
    """Return a normalized transient embedding attached to a detection."""
    value = det.get(_VECTOR_KEY)
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def strip_reid_vector(det: dict) -> dict:
    """Copy a detection without the non-JSON, in-memory ReID embedding."""
    clean = dict(det)
    clean.pop(_VECTOR_KEY, None)
    return clean


class SelectedPlayerReId:
    """Batch OSNet embeddings for candidate bodies during Confirm.

    The model is trained for person re-identification (Market1501), not merely
    ImageNet classification. It is intentionally used only after the user
    selects targets, so Analyze remains lightweight.
    """

    def __init__(
        self,
        *,
        weights_path: Path | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.backend = "unavailable"
        self.error: str | None = None
        self._extractor = None
        project_root = Path(__file__).resolve().parents[2]
        self.weights_path = weights_path or (
            project_root / "models" / "reid" / _MODEL_FILENAME
        )
        self.device = device
        self._load()

    @property
    def available(self) -> bool:
        return self._extractor is not None

    def _load(self) -> None:
        if not self.weights_path.is_file():
            self.error = f"Missing ReID weights: {self.weights_path}"
            logger.warning("%s", self.error)
            return
        try:
            import torch
            from torchreid.reid.utils import FeatureExtractor  # type: ignore

            target = self.device
            if not target:
                target = "cuda:0" if torch.cuda.is_available() else "cpu"
            if str(target).startswith("cuda") and not torch.cuda.is_available():
                target = "cpu"
            self._extractor = FeatureExtractor(
                model_name=_MODEL_NAME,
                model_path=str(self.weights_path),
                device=str(target),
            )
            self.backend = f"{_MODEL_NAME}_market1501@{target}"
            logger.info("Confirm ReID enabled: %s", self.backend)
        except Exception as exc:  # noqa: BLE001 - Confirm retains a safe fallback
            self.error = f"{type(exc).__name__}: {exc}"
            self._extractor = None
            logger.warning("Deep ReID unavailable; retaining safe spatial fallback: %s", exc)

    @staticmethod
    def _crop_box(frame_bgr: np.ndarray, box) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in (box or [0, 0, 0, 0])[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 18 or y2 - y1 < 36:
            return None
        # OSNet was trained on full human crops. Keep the complete bbox rather
        # than just the jersey; shoes, hair and silhouette help separate a
        # defender from an official in similar dark colors.
        return frame_bgr[y1:y2, x1:x2]

    def _crop_person(self, frame_bgr: np.ndarray, det: dict) -> np.ndarray | None:
        return self._crop_box(frame_bgr, det.get("bbox_xyxy"))

    def embed_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: list,
    ) -> list[np.ndarray | None]:
        """Normalized embeddings for xyxy boxes; None where a crop is unusable."""
        vectors: list[np.ndarray | None] = [None] * len(boxes)
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            return vectors

        crops: list[np.ndarray] = []
        indexes: list[int] = []
        for index, box in enumerate(boxes):
            crop = self._crop_box(frame_bgr, box)
            if crop is None:
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            indexes.append(index)

        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size]
            try:
                features = self._extractor(batch)
                if hasattr(features, "detach"):
                    features = features.detach().cpu().numpy()
                features = np.asarray(features, dtype=np.float32).reshape(len(batch), -1)
            except Exception as exc:  # noqa: BLE001 - leave failed frame unembedded
                logger.debug("ReID extraction failed for one Confirm batch: %s", exc)
                continue
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-8)
            for offset, vector in enumerate(features):
                vectors[indexes[start + offset]] = vector
        return vectors

    def embed_detections(self, frame_bgr: np.ndarray, detections: list[dict]) -> int:
        """Attach transient normalized ReID vectors to viable detections."""
        vectors = self.embed_boxes(
            frame_bgr, [det.get("bbox_xyxy") for det in detections]
        )
        encoded = 0
        for det, vector in zip(detections, vectors):
            if vector is None:
                continue
            det[_VECTOR_KEY] = vector
            encoded += 1
        return encoded
