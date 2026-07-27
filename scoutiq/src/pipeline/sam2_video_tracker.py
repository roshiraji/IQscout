"""SAM 2.1 video propagation for the bodies a coach clicked on the snapshot.

Detector + ByteTrack + a spatial lock re-decides "who is this?" on every frame,
so two players in contact can trade identities. SAM 2's video predictor instead
carries one memory per selected body forward and backward through the clip, so
close defense becomes a segmentation problem rather than a re-association one.

OSNet re-identification is used only as a referee: it verifies each propagated
mask still looks like the clicked player and finds a re-entry box after a loss.
A target that cannot be verified is left unmarked for those frames — it is
never relabelled onto a teammate or an official.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "models" / "sam2" / "sam2.1_hiera_base_plus.pt"
_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# JPEG keeps the whole clip addressable for the reverse pass at roughly a tenth
# of the RAM SAM 2's own float-tensor frame loader needs.
_FRAME_BUFFER_BUDGET_BYTES = 1_600_000_000
_JPEG_QUALITY = 92

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_predictor_cache: dict[tuple[str, str, str], object] = {}


class Sam2Unavailable(RuntimeError):
    """Raised when SAM 2 video propagation cannot run in this environment."""


@dataclass
class Sam2LockSettings:
    """Thresholds for accepting a propagated mask as the selected player."""

    # Below this the mask is simply not the clicked person.
    min_similarity: float = 0.34
    # Applied only while another tracked body overlaps: contact is exactly when
    # a wrong mask is cheap to produce and expensive to display.
    contested_similarity: float = 0.48
    contested_iou: float = 0.35
    # Re-entry after a loss has to be clearly the same person.
    recover_similarity: float = 0.62
    min_area_ratio: float = 0.18
    max_area_ratio: float = 4.5
    min_mask_pixels: int = 80
    max_consecutive_misses: int = 18
    max_recoveries: int = 4
    recovery_scan_frames: int = 240
    template_update_similarity: float = 0.72
    # Two selections resolved onto one body: keep the better match only.
    duplicate_iou: float = 0.60


@dataclass
class _Target:
    gid: int
    label: str
    seed_box: list[float]
    anchor: np.ndarray | None = None
    template: np.ndarray | None = None
    ref_area: float = 0.0
    recoveries: int = 0
    stats: dict = field(default_factory=dict)

    def note(self, verdict: str) -> None:
        self.stats[verdict] = self.stats.get(verdict, 0) + 1


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class ClipFrameBuffer:
    """Random-access clip storage shared by SAM 2 and the ReID referee.

    ``__getitem__`` returns the normalized square tensor SAM 2's image encoder
    expects; ``bgr`` returns the stored-resolution frame used for OSNet crops.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        image_size: int,
        expected_frames: int | None = None,
        quality: int = _JPEG_QUALITY,
        budget_bytes: int = _FRAME_BUFFER_BUDGET_BYTES,
    ) -> None:
        import torch

        self._torch = torch
        self.image_size = int(image_size)
        self.scale = 1.0
        self._payloads: list[bytes] = []
        self._bgr_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._tensor_cache: OrderedDict[int, object] = OrderedDict()
        self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32)[:, None, None]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Sam2Unavailable(f"Could not open source video: {video_path}")
        try:
            ok, frame = cap.read()
            if not ok:
                raise Sam2Unavailable(f"Source video has no frames: {video_path}")
            self.height, self.width = frame.shape[:2]

            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            probe = cv2.imencode(".jpg", frame, params)[1]
            planned = max(int(expected_frames or 0), 1)
            predicted = int(probe.nbytes) * planned
            if predicted > budget_bytes:
                self.scale = max(
                    0.5, min(1.0, float(np.sqrt(budget_bytes / max(predicted, 1))))
                )

            stored_bytes = 0
            while True:
                payload = self._encode(frame, params)
                self._payloads.append(payload)
                stored_bytes += len(payload)
                if stored_bytes > budget_bytes:
                    raise Sam2Unavailable(
                        "Clip is too long for in-memory SAM 2 propagation "
                        f"({len(self._payloads)} frames buffered)."
                    )
                ok, frame = cap.read()
                if not ok:
                    break
        finally:
            cap.release()

        self.stored_width = int(round(self.width * self.scale))
        self.stored_height = int(round(self.height * self.scale))

    def _encode(self, frame: np.ndarray, params: list[int]) -> bytes:
        if self.scale < 0.999:
            frame = cv2.resize(
                frame,
                (
                    max(int(round(frame.shape[1] * self.scale)), 2),
                    max(int(round(frame.shape[0] * self.scale)), 2),
                ),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise Sam2Unavailable("Failed to buffer a frame for SAM 2 propagation.")
        return buffer.tobytes()

    def __len__(self) -> int:
        return len(self._payloads)

    def bgr(self, index: int) -> np.ndarray:
        cached = self._bgr_cache.get(index)
        if cached is not None:
            self._bgr_cache.move_to_end(index)
            return cached
        image = cv2.imdecode(
            np.frombuffer(self._payloads[index], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self._bgr_cache[index] = image
        while len(self._bgr_cache) > 6:
            self._bgr_cache.popitem(last=False)
        return image

    def scale_box(self, box) -> list[float]:
        """Map a full-resolution xyxy box onto the stored frame scale."""
        return [float(v) * self.scale for v in list(box)[:4]]

    def __getitem__(self, index: int):
        cached = self._tensor_cache.get(index)
        if cached is not None:
            self._tensor_cache.move_to_end(index)
            return cached
        resized = cv2.resize(
            self.bgr(index),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb))
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        self._tensor_cache[index] = tensor
        while len(self._tensor_cache) > 3:
            self._tensor_cache.popitem(last=False)
        return tensor


def checkpoint_path() -> Path:
    override = os.environ.get("SCOUTIQ_SAM2_CHECKPOINT")
    return Path(override) if override else _DEFAULT_CHECKPOINT


def config_name() -> str:
    return os.environ.get("SCOUTIQ_SAM2_CONFIG") or _DEFAULT_CONFIG


def is_enabled() -> bool:
    flag = str(os.environ.get("SCOUTIQ_SAM2_VIDEO", "1")).strip().lower()
    return flag not in {"0", "false", "off", "no"}


def load_predictor(device: str | None = None):
    """Build (and memoize) the SAM 2.1 video predictor."""
    if not is_enabled():
        raise Sam2Unavailable("SAM 2 video propagation disabled by SCOUTIQ_SAM2_VIDEO.")

    ckpt = checkpoint_path()
    if not ckpt.is_file():
        raise Sam2Unavailable(f"Missing SAM 2 video checkpoint: {ckpt}")

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:  # noqa: BLE001 - Confirm keeps its spatial fallback
        raise Sam2Unavailable(f"SAM 2 package unavailable: {exc}") from exc

    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(target).startswith("cuda") and not torch.cuda.is_available():
        target = "cpu"

    key = (str(ckpt), config_name(), str(target))
    cached = _predictor_cache.get(key)
    if cached is not None:
        return cached

    try:
        predictor = build_sam2_video_predictor(
            config_name(),
            str(ckpt),
            device=str(target),
            hydra_overrides_extra=[
                # Two selected players may never own the same pixels; this is
                # the mask-level form of "do not swap onto the defender".
                "++model.non_overlap_masks=true",
                "++model.non_overlap_masks_for_mem_enc=true",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise Sam2Unavailable(f"Could not build SAM 2 video predictor: {exc}") from exc

    _predictor_cache[key] = predictor
    return predictor


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in list(box_a)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in list(box_b)[:4]]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def _similarity(target: _Target, vector: np.ndarray | None) -> float | None:
    anchor = _cosine(target.anchor, vector)
    rolling = _cosine(target.template, vector)
    if anchor is not None and rolling is not None:
        # The click stays dominant; the rolling view only absorbs lighting and
        # camera changes, so it can never drift onto a defender.
        return float(0.75 * anchor + 0.25 * rolling)
    return anchor if anchor is not None else rolling


def mask_to_geometry(mask: np.ndarray, *, min_pixels: int = 80) -> dict | None:
    """Largest connected blob of a propagated mask as bbox + polygon."""
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    if int(mask_u8.sum()) < min_pixels:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < float(min_pixels):
        return None
    x, y, w, h = cv2.boundingRect(largest)
    if w < 6 or h < 12:
        return None
    approx = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
    return {
        "bbox": [float(x), float(y), float(x + w), float(y + h)],
        "polygon": polygon if len(polygon) >= 3 else None,
        "area": area,
    }


def judge_mask(
    target: _Target,
    geometry: dict,
    peers: list[dict],
    similarity: float | None,
    settings: Sam2LockSettings,
) -> str:
    """``"ok"`` or the reason this mask must not carry the player's label."""
    if target.ref_area > 0.0:
        ratio = geometry["area"] / target.ref_area
        if ratio < settings.min_area_ratio or ratio > settings.max_area_ratio:
            return "shape"
    if similarity is None:
        return "ok"
    if similarity < settings.min_similarity:
        return "identity"
    contested = any(
        _box_iou(geometry["bbox"], peer["bbox"]) >= settings.contested_iou
        for peer in peers
    )
    if contested and similarity < settings.contested_similarity:
        return "contested"
    return "ok"


def _make_detection(
    target: _Target,
    geometry: dict,
    similarity: float | None,
    *,
    recovered: bool,
) -> dict:
    x1, y1, x2, y2 = geometry["bbox"]
    det = {
        "track_id": target.gid,
        "person_group_id": target.gid,
        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "confidence": 1.0,
        "mask_polygon": geometry["polygon"],
        "mask_source": "sam2_video",
        "selection_locked": True,
        "selection_source": "sam2_recovered" if recovered else "sam2_video",
    }
    if similarity is not None:
        det["reid_similarity"] = round(float(similarity), 4)
    return det


class _PassRunner:
    """One SAM 2 propagation run over a contiguous slice of the clip."""

    def __init__(
        self,
        predictor,
        buffer: ClipFrameBuffer,
        reid,
        settings: Sam2LockSettings,
    ) -> None:
        self.predictor = predictor
        self.buffer = buffer
        self.reid = reid
        self.settings = settings

    @property
    def reid_ready(self) -> bool:
        return self.reid is not None and bool(getattr(self.reid, "available", False))

    def _init_state(self) -> dict:
        import torch

        return {
            "images": self.buffer,
            "num_frames": len(self.buffer),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_height": self.buffer.height,
            "video_width": self.buffer.width,
            "device": self.predictor.device,
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }

    def run(
        self,
        targets: list[_Target],
        prompts: dict[int, list[float]],
        start_index: int,
        *,
        reverse: bool,
        accepted: dict[int, dict[int, dict]],
        recovered: bool = False,
        on_frame=None,
    ) -> dict[int, int | None]:
        """Propagate ``targets`` from ``start_index`` and fill ``accepted``.

        Returns the frame index where each target's losing streak began, or
        ``None`` when the target survived to the end of the pass.
        """
        import torch

        by_gid = {target.gid: target for target in targets}
        state = self._init_state()
        for target in targets:
            self.predictor.add_new_points_or_box(
                state,
                frame_idx=start_index,
                obj_id=target.gid,
                box=np.asarray(prompts[target.gid], dtype=np.float32),
            )

        step = -1 if reverse else 1
        misses = {target.gid: 0 for target in targets}
        lost_at: dict[int, int | None] = {target.gid: None for target in targets}
        active = {target.gid for target in targets}

        on_cuda = str(self.predictor.device).startswith("cuda")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else _NullContext()
        )

        try:
            with autocast:
                stream = self.predictor.propagate_in_video(
                    state, start_frame_idx=start_index, reverse=reverse
                )
                for frame_index, obj_ids, mask_logits in stream:
                    masks = (mask_logits[:, 0] > 0.0).cpu().numpy()
                    geometries: dict[int, dict] = {}
                    for slot, gid in enumerate(obj_ids):
                        if int(gid) not in active:
                            continue
                        geometry = mask_to_geometry(
                            masks[slot], min_pixels=self.settings.min_mask_pixels
                        )
                        if geometry is not None:
                            geometries[int(gid)] = geometry

                    scored = self._score(frame_index, by_gid, geometries)

                    for gid in sorted(active):
                        target = by_gid[gid]
                        geometry = geometries.get(gid)
                        similarity, vector = scored.get(gid, (None, None))
                        if geometry is None:
                            verdict = "occluded"
                        else:
                            peers = [
                                other
                                for other_gid, other in geometries.items()
                                if other_gid != gid
                            ]
                            verdict = judge_mask(
                                target, geometry, peers, similarity, self.settings
                            )

                        if verdict == "ok" and geometry is not None:
                            misses[gid] = 0
                            target.note("accepted")
                            accepted.setdefault(gid, {})[frame_index] = _make_detection(
                                target, geometry, similarity, recovered=recovered
                            )
                            self._absorb(target, geometry, similarity, vector)
                            continue

                        misses[gid] += 1
                        target.note("rejected")
                        target.note(verdict)
                        if misses[gid] >= self.settings.max_consecutive_misses:
                            active.discard(gid)
                            lost_at[gid] = frame_index - step * (misses[gid] - 1)

                    if on_frame is not None:
                        on_frame(frame_index)
                    if not active:
                        break
        finally:
            state.clear()
            if on_cuda:
                torch.cuda.empty_cache()

        return lost_at

    def _score(
        self,
        frame_index: int,
        by_gid: dict[int, _Target],
        geometries: dict[int, dict],
    ) -> dict[int, tuple[float | None, np.ndarray | None]]:
        if not geometries or not self.reid_ready:
            return {}
        gids = sorted(geometries)
        boxes = [self.buffer.scale_box(geometries[gid]["bbox"]) for gid in gids]
        vectors = self.reid.embed_boxes(self.buffer.bgr(frame_index), boxes)
        return {
            gid: (_similarity(by_gid[gid], vector), vector)
            for gid, vector in zip(gids, vectors)
        }

    def _absorb(
        self,
        target: _Target,
        geometry: dict,
        similarity: float | None,
        vector: np.ndarray | None,
    ) -> None:
        area = float(geometry["area"])
        target.ref_area = (
            area if target.ref_area <= 0.0 else 0.9 * target.ref_area + 0.1 * area
        )
        if (
            vector is None
            or similarity is None
            or similarity < self.settings.template_update_similarity
        ):
            return
        previous = target.template if target.template is not None else vector
        blended = 0.9 * previous + 0.1 * vector
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            target.template = (blended / norm).astype(np.float32)


def _detections_for_index(frame_records: list[dict], index: int) -> list[dict]:
    if index < 0 or index >= len(frame_records):
        return []
    return list(frame_records[index].get("tracks") or [])


def _find_reentry(
    target: _Target,
    accepted: dict[int, dict[int, dict]],
    *,
    frame_records: list[dict],
    buffer: ClipFrameBuffer,
    reid,
    start_index: int,
    reverse: bool,
    settings: Sam2LockSettings,
    bounds: tuple[int, int],
) -> tuple[int, list[float]] | None:
    """First detection after a loss that ReID clearly attributes to this player."""
    low, high = bounds
    step = -1 if reverse else 1
    for offset in range(1, settings.recovery_scan_frames + 1):
        index = start_index + step * offset
        if index < low or index > high:
            return None
        candidates = _detections_for_index(frame_records, index)
        if not candidates:
            continue

        claimed = [
            det["bbox_xyxy"]
            for gid, per_index in accepted.items()
            if gid != target.gid and index in per_index
            for det in [per_index[index]]
        ]
        boxes: list[list[float]] = []
        keep: list[list[float]] = []
        for det in candidates:
            box = [float(v) for v in (det.get("bbox_xyxy") or [0, 0, 0, 0])[:4]]
            if any(_box_iou(box, other) >= 0.5 for other in claimed):
                continue
            keep.append(box)
            boxes.append(buffer.scale_box(box))
        if not keep:
            continue

        vectors = reid.embed_boxes(buffer.bgr(index), boxes)
        best_score = -1.0
        best_box: list[float] | None = None
        for box, vector in zip(keep, vectors):
            score = _cosine(target.anchor, vector)
            if score is not None and score > best_score:
                best_score = score
                best_box = box
        if best_box is not None and best_score >= settings.recover_similarity:
            return index, best_box
    return None


def _recover_lost_targets(
    runner: _PassRunner,
    targets: list[_Target],
    accepted: dict[int, dict[int, dict]],
    *,
    frame_records: list[dict],
    lost_by_direction: list[tuple[bool, dict[int, int | None]]],
    buffer: ClipFrameBuffer,
    settings: Sam2LockSettings,
    bounds: tuple[int, int],
) -> int:
    """Re-seed SAM 2 wherever a target was abandoned and ReID finds it again."""
    if not runner.reid_ready:
        return 0

    total = 0
    by_gid = {target.gid: target for target in targets}
    for reverse, lost in lost_by_direction:
        for gid, lost_index in lost.items():
            target = by_gid.get(gid)
            if target is None or target.anchor is None:
                continue
            cursor = lost_index
            while cursor is not None and target.recoveries < settings.max_recoveries:
                found = _find_reentry(
                    target,
                    accepted,
                    frame_records=frame_records,
                    buffer=buffer,
                    reid=runner.reid,
                    start_index=cursor,
                    reverse=reverse,
                    settings=settings,
                    bounds=bounds,
                )
                if found is None:
                    break
                index, box = found
                target.recoveries += 1
                total += 1
                logger.info(
                    "SAM 2 re-seeded %s at frame index %d (%s pass)",
                    target.label,
                    index,
                    "reverse" if reverse else "forward",
                )
                cursor = runner.run(
                    [target],
                    {gid: box},
                    index,
                    reverse=reverse,
                    accepted=accepted,
                    recovered=True,
                ).get(gid)
    return total


def drop_duplicate_bodies(
    accepted: dict[int, dict[int, dict]], settings: Sam2LockSettings
) -> int:
    """Two selections resolved onto one body: keep only the better ReID match.

    Independent re-seeded runs do not share SAM 2's non-overlap constraint, so
    this is the last guard against showing one player under two names.
    """
    dropped = 0
    gids = sorted(accepted)
    for i, gid_a in enumerate(gids):
        for gid_b in gids[i + 1 :]:
            shared = set(accepted[gid_a]) & set(accepted[gid_b])
            for index in shared:
                det_a = accepted[gid_a][index]
                det_b = accepted[gid_b][index]
                if (
                    _box_iou(det_a["bbox_xyxy"], det_b["bbox_xyxy"])
                    < settings.duplicate_iou
                ):
                    continue
                score_a = float(det_a.get("reid_similarity") or 0.0)
                score_b = float(det_b.get("reid_similarity") or 0.0)
                loser = gid_b if score_a >= score_b else gid_a
                accepted[loser].pop(index, None)
                dropped += 1
    return dropped


def propagate_selected_players(
    *,
    video_path: str | Path,
    frame_records: list[dict],
    snapshot_frame_number: int,
    seeds: dict[int, list[float]],
    labels: dict[int, str] | None = None,
    reid=None,
    settings: Sam2LockSettings | None = None,
    progress_cb=None,
) -> tuple[dict[int, dict[int, dict]], dict]:
    """Track each clicked body through the clip with SAM 2 video propagation.

    Returns ``({frame_number: {person_group_id: detection}}, stats)`` — the same
    shape the spatial selection lock produces, so Confirm can swap between them
    without touching the renderer.
    """
    settings = settings or Sam2LockSettings()
    labels = labels or {}
    if not seeds:
        raise Sam2Unavailable("No clicked boxes to seed SAM 2 propagation.")

    started = time.perf_counter()
    predictor = load_predictor()

    ordered = [int(record["frame_number"]) for record in frame_records]
    if not ordered:
        raise Sam2Unavailable("Canonical tracks contain no frames.")

    def _emit(progress: float, detail: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(progress, detail)
        except Exception:  # noqa: BLE001 - progress must never break tracking
            pass

    _emit(0.02, "Loading clip for SAM 2 propagation…")
    buffer = ClipFrameBuffer(
        video_path,
        image_size=int(predictor.image_size),
        expected_frames=len(ordered),
    )
    usable = min(len(buffer), len(ordered))
    if usable <= 0:
        raise Sam2Unavailable("No overlapping frames between video and tracks.")
    ordered = ordered[:usable]
    frame_records = frame_records[:usable]

    snapshot_index = min(
        range(usable), key=lambda i: abs(ordered[i] - int(snapshot_frame_number))
    )

    targets = [
        _Target(
            gid=int(gid),
            label=labels.get(int(gid), f"Person #{int(gid)}"),
            seed_box=[float(v) for v in list(box)[:4]],
        )
        for gid, box in sorted(seeds.items())
    ]

    if reid is not None and getattr(reid, "available", False):
        anchors = reid.embed_boxes(
            buffer.bgr(snapshot_index),
            [buffer.scale_box(target.seed_box) for target in targets],
        )
        for target, vector in zip(targets, anchors):
            if vector is not None:
                target.anchor = np.asarray(vector, dtype=np.float32)
                target.template = target.anchor.copy()

    runner = _PassRunner(predictor, buffer, reid, settings)
    accepted: dict[int, dict[int, dict]] = {}
    prompts = {target.gid: target.seed_box for target in targets}

    progress_state = {"count": 0}

    def _tick(_frame_index: int) -> None:
        progress_state["count"] += 1
        if progress_state["count"] % 15:
            return
        fraction = min(progress_state["count"] / max(usable, 1), 1.0)
        _emit(0.05 + 0.80 * fraction, "Propagating your selected players…")

    _emit(0.05, "Propagating forward from the snapshot…")
    lost_forward = runner.run(
        targets, prompts, snapshot_index, reverse=False, accepted=accepted, on_frame=_tick
    )
    _emit(0.55, "Propagating back to the start of the clip…")
    lost_backward = runner.run(
        targets, prompts, snapshot_index, reverse=True, accepted=accepted, on_frame=_tick
    )

    _emit(0.88, "Recovering targets that were lost in traffic…")
    recoveries = _recover_lost_targets(
        runner,
        targets,
        accepted,
        frame_records=frame_records,
        lost_by_direction=[(False, lost_forward), (True, lost_backward)],
        buffer=buffer,
        settings=settings,
        bounds=(0, usable - 1),
    )

    _emit(0.95, "Resolving overlapping identities…")
    duplicates = drop_duplicate_bodies(accepted, settings)

    locked: dict[int, dict[int, dict]] = {}
    for gid, per_index in accepted.items():
        for index, det in per_index.items():
            locked.setdefault(ordered[index], {})[gid] = det

    stats = {
        "backend": "sam2.1_video",
        "checkpoint": checkpoint_path().name,
        "device": str(predictor.device),
        "frames": usable,
        "snapshot_index": snapshot_index,
        "snapshot_frame_number": ordered[snapshot_index],
        "recoveries": recoveries,
        "duplicate_masks_dropped": duplicates,
        "frames_with_all_targets": sum(
            1 for chosen in locked.values() if len(chosen) == len(targets)
        ),
        "reid_backend": getattr(reid, "backend", None),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "per_player": {
            str(target.gid): {
                "label": target.label,
                "frames_locked": len(accepted.get(target.gid) or {}),
                "rejected_frames": target.stats.get("rejected", 0),
                "identity_rejects": target.stats.get("identity", 0),
                "contested_rejects": target.stats.get("contested", 0),
                "occluded_frames": target.stats.get("occluded", 0),
                "recoveries": target.recoveries,
                "has_reid_anchor": target.anchor is not None,
            }
            for target in targets
        },
    }
    _emit(1.0, "SAM 2 propagation complete.")
    return locked, stats
