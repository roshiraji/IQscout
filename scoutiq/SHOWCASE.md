# ScoutIQ VisionPlay — Player Tracking + Segmentation Showcase

Professional demo path: **custom `yolo11n-seg` weights** + exclusions + tracker.
Grounding DINO and SAM2 are used **only** in the offline label factory.

## Quick start (from `scoutiq/`)

```powershell
# 0) Folders + default exclusion template
..\.venv\Scripts\python.exe -m src.showcase init

# 1) Extract all frames + manifest (label candidates = stride/scene cuts)
..\.venv\Scripts\python.exe -m src.showcase ingest --config configs/showcase_ingest.yaml

# 2) Auto-label players (GDINO boxes → SAM2 masks → YOLO-seg)
#    Smoke test: add --max-frames 50
..\.venv\Scripts\python.exe -m src.showcase autolabel --config configs/showcase_ingest.yaml

# 3) Build train/val dataset (holdout clip IDs excluded)
..\.venv\Scripts\python.exe -m src.showcase build-dataset --config configs/showcase_ingest.yaml

# 4) Train lightweight seg model (uses 2 GPUs when available)
..\.venv\Scripts\python.exe -m src.showcase train --config configs/showcase_train.yaml

# 5) Showcase on unseen / holdout clip (no GDINO)
..\.venv\Scripts\python.exe -m src.showcase run --config configs/showcase_player_seg.yaml
```

## Live user path (React dashboard → FastAPI)

The trained `best_player_seg.pt` also powers the online product path — no
Grounding DINO / SAM2 at inference. Config: `configs/player_tracking_mvp.yaml`
(`weights`, `conf`, `iou`, `device`, `exclusions_dir`, `mask_overlay`, `max_frames`).

Flow when a user uploads a clip:

1. `POST /api/analyze` → intake validation + player snapshot (existing).
2. **Auto-Track All Players** button → `POST /api/track/auto` with `{video_path}`.
3. Backend `src/pipeline/player_track_pipeline.py`:
   streams frames → `detect_players_seg` (player-only YOLO-seg) → exclusions
   (soft-fail to defaults) → ByteTrack (`update_tracker`) → semi-transparent
   masks + bbox + track-ID overlay → browser-safe H.264.
4. Outputs land in the existing dirs and are served over existing endpoints:
   - `outputs/tracking_videos/<clip>_seg_tracked.mp4` → `GET /api/files/tracking/<name>`
   - `outputs/tracking/<clip>_seg_tracks.csv` + `_seg_tracking_summary.json`
   - `outputs/player_snapshots/<clip>_seg_snapshot.jpg`

Smoke test without the UI (from `scoutiq/`):

```powershell
..\.venv\Scripts\python.exe -m src.pipeline.player_track_pipeline `
  --video "path\to\clip.mp4" --max-frames 60
```

The run logs `weights=…/best_player_seg.pt` so you can confirm the deployed
model is in use. Optional `--focus Player_3 Player_7` highlights a subset;
`AutoTrackRequest.focus_labels` exposes the same over the API.

## Layout

See `data/README_SHOWCASE.md` after `init`.

| Path | Role |
|------|------|
| `data/frames/<game>/<clip>/` | Frame store (scales toward 1M frames) |
| `data/frames/manifest.csv` | Per-frame index + `label_candidate` |
| `data/frames/holdout_gate.json` | Holdout clip IDs — never train |
| `data/exclusions/*.json` | Court ROI + shot clock / scorebug / crowd |
| `outputs/auto_labels/` | Factory labels + QA previews |
| `outputs/training_loop/best_player_seg.pt` | Deployed weights |
| `outputs/showcase/*.mp4` | Final overlay (mask + bbox + ID) |

## Exclusions

Edit `data/exclusions/default.json` or add `data/exclusions/<clip_id>.json`:

- **court_roi** / **margins** — keep on-floor players
- **exclude_rects** — shot clock, scorebug
- **exclude_polygons** — crowd / benches

Applied at autolabel export **and** showcase inference.

## Unseen clip gate

1. Add a second video under `clips:` with `split: holdout` in `showcase_ingest.yaml`.
2. Point `configs/showcase_player_seg.yaml` → that holdout path.
3. Confirm its `clip_id` appears in `holdout_gate.json` and **not** in `data/datasets/nba_player_seg`.
4. Run showcase; ship only if coverage / boxes / masks / exclusions look professional.

Highlight hard cuts can reset track IDs — expected limitation.

## Dual GPU

- **Autolabel:** sequential GDINO then SAM2 today (both on default CUDA). Split across cards later via `CUDA_VISIBLE_DEVICES` per process if needed.
- **Train:** `device: null` auto-selects `[0, 1]` when two GPUs are visible (`configs/showcase_train.yaml`).
