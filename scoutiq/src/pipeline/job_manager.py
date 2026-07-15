"""Persistent single-GPU queue for full-clip player tracking jobs."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.player_track_pipeline import JOBS_DIR, run_player_seg_tracking

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrackingJobManager:
    """Serialize GPU inference while exposing durable progress on disk."""

    def __init__(self, jobs_dir: Path = JOBS_DIR) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, Path, dict, int | None]] = queue.Queue()
        self._isolate_queue: queue.Queue[tuple[str, set[int], str]] = queue.Queue()
        self._lock = threading.Lock()
        self._recover_interrupted_jobs()
        self._worker = threading.Thread(
            target=self._run, name="scoutiq-tracking-worker", daemon=True
        )
        self._worker.start()
        self._isolate_worker = threading.Thread(
            target=self._run_isolate,
            name="scoutiq-isolate-worker",
            daemon=True,
        )
        self._isolate_worker.start()

    def _recover_interrupted_jobs(self) -> None:
        for state_path in self.jobs_dir.glob("*/job.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in {"queued", "tracking", "identity"}:
                state.update(
                    {
                        "status": "failed",
                        "stage": "interrupted",
                        "error": "API restarted before the job completed. Re-upload to retry.",
                        "updated_at": _now(),
                    }
                )
                self._write_state(state["job_id"], state)
        for state_path in self.jobs_dir.glob("*/isolate_status.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in {"queued", "processing"}:
                state.update(
                    {
                        "status": "failed",
                        "stage": "interrupted",
                        "error": "API restarted before isolate processing completed. Re-Confirm to retry.",
                        "updated_at": _now(),
                    }
                )
                self._write_isolate_state(state["job_id"], state)

    def submit(
        self,
        video_path: str | Path,
        *,
        game_context: dict | None = None,
        max_frames: int | None = None,
    ) -> dict:
        job_id = uuid.uuid4().hex
        video = Path(video_path).resolve()
        state = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "frames_processed": 0,
            "total_frames": 0,
            "video_path": str(video),
            "game_context": game_context or {},
            "result": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._write_state(job_id, state)
        self._queue.put((job_id, video, game_context or {}, max_frames))
        return state

    def get(self, job_id: str) -> dict | None:
        path = self.jobs_dir / Path(job_id).name / "job.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def artifact_dir(self, job_id: str) -> Path:
        return self.jobs_dir / Path(job_id).name

    def submit_isolate(
        self,
        job_id: str,
        selected_group_ids: set[int],
        generation: str,
    ) -> dict:
        """Queue a CPU-heavy temporal isolate render without blocking Confirm."""
        canonical = self.get(job_id)
        if canonical is None:
            raise KeyError(job_id)
        total = int((canonical.get("result") or {}).get("source_frame_count") or 0)
        state = {
            "job_id": job_id,
            "generation": generation,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "frames_processed": 0,
            "total_frames": total,
            "eta_seconds": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        # Never let an old isolate artifact masquerade as the new selection.
        (self.artifact_dir(job_id) / "focused_isolated.mp4").unlink(missing_ok=True)
        self._write_isolate_state(job_id, state)
        self._isolate_queue.put((job_id, set(selected_group_ids), generation))
        return state

    def get_isolate(self, job_id: str) -> dict | None:
        path = self.artifact_dir(job_id) / "isolate_status.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_state(self, job_id: str, state: dict) -> None:
        with self._lock:
            job_dir = self.artifact_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            path = job_dir / "job.json"
            tmp = job_dir / "job.tmp.json"
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _write_isolate_state(self, job_id: str, state: dict) -> None:
        with self._lock:
            job_dir = self.artifact_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            path = job_dir / "isolate_status.json"
            tmp = job_dir / "isolate_status.tmp.json"
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _update_isolate(
        self,
        job_id: str,
        generation: str,
        **changes,
    ) -> dict | None:
        state = self.get_isolate(job_id)
        if state is None or state.get("generation") != generation:
            return None
        state.update(changes)
        state["updated_at"] = _now()
        self._write_isolate_state(job_id, state)
        return state

    def _mark_isolate_ready_in_summary(
        self,
        job_id: str,
        generation: str,
        output: Path,
    ) -> None:
        path = self.artifact_dir(job_id) / "selection_summary.json"
        if not path.is_file():
            return
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if summary.get("isolation_generation") != generation:
            return
        summary.update(
            {
                "isolated_tracking_video": str(output.resolve()),
                "isolation_available": True,
                "isolation_status": "ready",
            }
        )
        with self._lock:
            tmp = self.artifact_dir(job_id) / "selection_summary.tmp.json"
            tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tmp.replace(path)

    def _update(self, job_id: str, **changes) -> dict:
        state = self.get(job_id)
        if state is None:
            raise KeyError(job_id)
        state.update(changes)
        state["updated_at"] = _now()
        self._write_state(job_id, state)
        return state

    def _progress(self, job_id: str, current: int, total: int, stage: str) -> None:
        self._update(
            job_id,
            status="tracking",
            stage=stage,
            frames_processed=current,
            total_frames=total,
            progress=round(current / max(total, 1), 4),
        )

    def _run(self) -> None:
        while True:
            job_id, video, game_context, max_frames = self._queue.get()
            try:
                self._update(job_id, status="tracking", stage="loading_model")
                result = run_player_seg_tracking(
                    video,
                    output_dir=self.artifact_dir(job_id),
                    job_id=job_id,
                    game_context=game_context,
                    max_frames=max_frames,
                    quiet=True,
                    progress_callback=lambda current, total, stage: self._progress(
                        job_id, current, total, stage
                    ),
                )
                self._update(
                    job_id,
                    status="ready",
                    stage=(
                        "ready"
                        if result.get("complete_clip")
                        else "ready_partial_smoke_test"
                    ),
                    progress=1.0,
                    frames_processed=result["frames_tracked"],
                    total_frames=result["source_frame_count"],
                    partial=not bool(result.get("complete_clip")),
                    result=result,
                )
            except Exception as exc:  # noqa: BLE001 — persisted for API clients
                logger.exception("Tracking job %s failed", job_id)
                self._update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=str(exc),
                )
            finally:
                self._queue.task_done()

    def _run_isolate(self) -> None:
        from src.pipeline.isolate_renderer import (
            IsolateCancelled,
            render_temporal_isolate,
        )

        while True:
            job_id, selected_group_ids, generation = self._isolate_queue.get()
            started = time.perf_counter()
            last_update = 0.0
            try:
                if self._update_isolate(
                    job_id,
                    generation,
                    status="processing",
                    stage="preparing_temporal_background",
                ) is None:
                    continue

                def cancelled() -> bool:
                    state = self.get_isolate(job_id)
                    return (
                        state is None
                        or state.get("generation") != generation
                    )

                def progress(current: int, total: int, stage: str) -> None:
                    nonlocal last_update
                    now = time.perf_counter()
                    if current < total and now - last_update < 0.5:
                        return
                    elapsed = max(now - started, 0.001)
                    fraction = current / max(total, 1)
                    eta = (
                        round(elapsed * (1.0 - fraction) / fraction)
                        if fraction > 0
                        else None
                    )
                    self._update_isolate(
                        job_id,
                        generation,
                        status="processing",
                        stage=stage,
                        frames_processed=current,
                        total_frames=total,
                        progress=round(min(fraction, 1.0), 4),
                        eta_seconds=eta,
                    )
                    last_update = now

                output = render_temporal_isolate(
                    self.artifact_dir(job_id),
                    selected_group_ids,
                    generation,
                    progress_callback=progress,
                    cancelled=cancelled,
                    quiet=True,
                )
                if cancelled():
                    output.unlink(missing_ok=True)
                    raise IsolateCancelled(
                        "A newer selection replaced this isolate job."
                    )
                self._mark_isolate_ready_in_summary(
                    job_id, generation, output
                )
                self._update_isolate(
                    job_id,
                    generation,
                    status="ready",
                    stage="ready",
                    progress=1.0,
                    frames_processed=(
                        self.get_isolate(job_id) or {}
                    ).get("total_frames", 0),
                    eta_seconds=0,
                    artifact=str(output.resolve()),
                    method="aligned_temporal_background_plus_spatial_fallback",
                )
            except IsolateCancelled:
                # A newer Confirm owns the durable status and output path.
                pass
            except Exception as exc:  # noqa: BLE001 — persisted for API clients
                logger.exception("Isolate job %s (%s) failed", job_id, generation)
                self._update_isolate(
                    job_id,
                    generation,
                    status="failed",
                    stage="failed",
                    error=str(exc),
                    eta_seconds=None,
                )
            finally:
                self._isolate_queue.task_done()


_manager: TrackingJobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> TrackingJobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TrackingJobManager()
    return _manager
