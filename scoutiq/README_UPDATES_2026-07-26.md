# ScoutIQ VisionPlay — Updates (2026-07-26)

Date: July 26, 2026

Confirm no longer re-decides "who is this player?" on every frame. The selected
bodies are now carried through the clip by **SAM 2.1 video propagation**, with
OSNet re-identification acting only as a referee. This replaces the association
tuning that kept swapping a target onto the defender guarding them.

---

## Why the old path failed

Detector → ByteTrack → spatial/appearance lock re-answers the identity question
on each frame. During on-ball defense the two candidate boxes overlap heavily,
their jersey crops bleed into each other, and the highest-scoring candidate is
frequently the wrong body. No threshold fixes that, and no retraining cycle
fixes it either — the detector was already finding both players.

## What runs now

```text
snapshot click boxes
  → SAM 2.1 video predictor (one mask memory per selected body)
      forward  snapshot → end of clip
      backward snapshot → start of clip
  → OSNet ReID verification per frame
      unverified frame  → target left unmarked (never relabelled)
      long loss         → re-seed SAM 2 from a confident ReID re-detection
  → mask → bbox + polygon → the same locked_tracks / CSV / preview outputs
```

Two guarantees worth keeping in mind when tuning:

- SAM 2 runs with `non_overlap_masks` on, so two selected players can never own
  the same pixels.
- A target that cannot be verified is dropped for those frames. An empty frame
  is a correct answer; a teammate wearing the target's label is not.

## Measured on job `e7e2d2be…` (259 frames, 1920×1080, two players in contact)

| Path | Frames holding both selections | Confirm identity runtime |
|------|-------------------------------|--------------------------|
| `osnet_reid+spatial` (previous) | 181 / 259 (69.9%) | ~12 s |
| `sam2_video+osnet_reid` (new)   | 259 / 259 (100%)  | ~65 s |

`selection_summary.json` now reports `selection_lock`, plus a `selection_sam2`
block with per-player locked/rejected/recovered counts, and
`selection_sam2_error` when the pipeline fell back.

---

## Setup

The SAM 2 package comes from `requirements.txt`. Weights are gitignored, so
download them once:

```bat
scripts\download_sam2_video_checkpoint.bat base_plus
```

That writes `models\sam2\sam2.1_hiera_base_plus.pt` (~308 MB).

Verify propagation on a finished job without starting the API:

```bat
python scripts\smoke_sam2_video.py <job_id> <gid> <gid>
python scripts\smoke_confirm_render.py <job_id> <gid> <gid>
```

## Environment toggles

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCOUTIQ_SAM2_VIDEO` | `1` | Set `0` to force the previous spatial lock |
| `SCOUTIQ_SAM2_CHECKPOINT` | `models/sam2/sam2.1_hiera_base_plus.pt` | Swap model size |
| `SCOUTIQ_SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | Must match the checkpoint |

For a faster (slightly less accurate) run, download `small` and point both
variables at `sam2.1_hiera_small.pt` / `configs/sam2.1/sam2.1_hiera_s.yaml`.

## Fallback behaviour

Confirm falls back to `osnet_reid+spatial` — and still produces a clip — when
the checkpoint is missing, the `sam2` package fails to import, a selected
player has no clicked box, the clip is too long to buffer in RAM, or
propagation raises for any other reason. The reason is written to
`selection_sam2_error`.

## Thresholds

`Sam2LockSettings` in [`src/pipeline/sam2_video_tracker.py`](src/pipeline/sam2_video_tracker.py):

| Field | Default | Effect |
|-------|---------|--------|
| `min_similarity` | 0.34 | Below this the mask is not the clicked person |
| `contested_similarity` | 0.48 | Stricter bar while another target overlaps |
| `recover_similarity` | 0.62 | Bar for re-seeding after a loss |
| `min/max_area_ratio` | 0.18 / 4.5 | Rejects slivers and fused two-player blobs |
| `max_consecutive_misses` | 18 | Frames of failure before the target is abandoned |
| `max_recoveries` | 4 | Re-seed attempts per target per direction |

## Known limits

- Runtime scales with clip length at roughly 4–5 frames/s on an RTX 3070.
- The whole clip is buffered as JPEG in RAM (~1.6 GB cap); longer clips fall
  back rather than swap.
- `tests/test_analytics_api.py` still fails to collect on this machine because
  `httpx` is not installed. Unrelated to this change.
