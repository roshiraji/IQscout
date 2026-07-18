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

1. `POST /api/analyze` validates the upload and returns a background `job_id`.
2. `GET /api/jobs/<job_id>` reports frame progress while one isolated
   `ObjectTracker` (ByteTrack) processes the full clip from frame 0.
3. `src/pipeline/player_track_pipeline.py` writes canonical artifacts under
   `outputs/jobs/<job_id>/`, then `src/pipeline/tracklet_stitcher.py` merges
   short raw tracklets into stable `person_group_id`s using temporal gap,
   spatial proximity, jersey-color histograms, and optional OSNet appearance.
4. Artifacts include `tracks.csv` / `tracks.jsonl.gz` (with `person_group_id`),
   `person_groups.json`, all-player H.264, `quality.json`, and a snapshot whose
   selectable units are **Person #** groups (not ephemeral ByteTrack IDs).
5. The user selects person groups. `POST /api/tracking/selection` filters member
   tracklets for those groups and renders `focused_players.mp4` from frame 0 →
   EOF; it does **not** run YOLO or ByteTrack again (`canonical_reuse=true`).
6. `identity_attachments.json` provides safe `pending/unresolved/resolved`
   fields keyed by person group. **Resolve jersey identities** searches each
   person's full track for high-quality jersey OCR opportunities, OCR's only
   the top temporally diverse candidates (`identity.jersey_ocr` in config),
   then votes with consistency gates + roster match; Manual jersey remains the
   fallback. Tracking never invents player names.

Smoke test without the UI (from `scoutiq/`):

```powershell
..\.venv\Scripts\python.exe -m src.pipeline.player_track_pipeline `
  --video "path\to\clip.mp4" --max-frames 60
```

The run logs `weights=…/best_player_seg.pt` and `raw=N groups=M` so you can
confirm stitching reduced fragmented IDs.

### Continuity + retention (selling-point metric)

- **Problem ByteTrack alone solves poorly:** one player becomes many short
  tracklets after occlusions/cuts, so selecting `Track #N` only overlays while
  that ephemeral ID exists.
- **Stitching:** raw `track_id`s are kept for debug; `person_group_id` is the
  product identity. Config under `tracker:` / `stitch:` in
  `configs/player_tracking_mvp.yaml`.
- **`selected_player_retention`** = frames with ≥1 selected person group /
  `source_frame_count`. Shown in the dashboard TrackingResults panel and in
  `selection_summary.json` / focused `quality` block after Confirm.
- Also reported: `unique_raw_tracks` vs `unique_person_groups`, and
  `average_id_switches_per_person` pre vs post stitch.

### Focused preview + isolate (after Confirm)

Confirm reuses canonical `tracks.jsonl.gz` (no second YOLO/ByteTrack pass).
The normal focused clip returns first; a CPU background worker then builds the
higher-quality isolate while the dashboard polls real progress + ETA:

| Artifact | Behavior |
|----------|----------|
| `focused_players.mp4` | Full scene + selected overlays (boxes, tinted masks, labels) |
| `focused_isolated.mp4` | Full scene; **non-selected** masks are reconstructed from feature-aligned clean pixels in nearby frames, with Navier-Stokes fallback only for uncovered pixels; court/hoop/crowd and selected players remain |

Dashboard toggle **Full scene | Isolate** switches the preview `src` (same
cache-bust `?v=&g=` as Confirm). Isolate stays disabled with percentage and ETA
until `isolate_status.json` reports ready. A new Confirm supersedes the older
background generation. Coach insights / CSV are unchanged.

### Coach insights (after Confirm, ≥2 players)

Confirm requires **at least two** person groups. After focused render,
`src/pipeline/coach_insights.py` computes clip proxies from
`selected_tracks.csv` using **mask-bottom stance** when segmentation exists
(else bbox bottom / midpoint fallback). Distances stay frame-diagonal (no
court feet):

| Layer | Metrics |
|-------|---------|
| Single-player | coverage trust, path length, mean/p90 speed, high-effort fraction |
| Pair | both-visible %, separation bands, closeout bursts, chase role (opp), sticky matchup (opp), screen-shape soft label (same team), cut-timing overlap |
| Scorecard | **Stay Close** (defense) vs **Create Space** (offense) /100 from time attached, closeouts, get-away bursts — or same-team pick/cut/spacing |
| Time series | `pair_distance_series` in JSON + `pair_distances.csv` (`gap_norm`, `gap_band`, `closing_speed_norm`) |

Artifacts: `outputs/jobs/<job_id>/coach_insights.json`,
`pair_distances.csv` (also on the tracking API as `coach_insights` +
`coach_narrative`; file kind `pair_distances`). The dashboard **Matchup
Outcomes** panel shows Stay Close / Create Space scores (not camera
coverage), metric tiles, breakdown bars, and a separation sparkline.

**Non-goals this stage:** ball tracking, shot make/miss, named identity,
calibrated feet/s, true defensive rating, help/rotation without a ball model.
Create Space means “got away,” not points scored. Speeds/distances are
normalized to the frame diagonal.

### Full-clip quality gate

`quality.json` reports detector confidence separately from tracking continuity:
frame coverage, short-track ratio, per-track gaps, camera cuts, and stitch
compression. Thresholds live under `quality:` in
`configs/player_tracking_mvp.yaml`. A detection gate failure calls for held-out
mask/recall review; a continuity failure calls for stitch/tracker tuning — not
automatic retraining of `best_player_seg.pt`.

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
