# ScoutIQ VisionPlay — July 14, 2026 Session

This document records the ScoutIQ dashboard, tracking, coaching, and isolate
preview work completed during this development session.

Editable diagrams:

- [`docs/ScoutIQ_Architecture.drawio`](docs/ScoutIQ_Architecture.drawio) — full
  repository architecture: live dashboard, showcase training, fusion, identity,
  models, hardware, and data.
- [`docs/ScoutIQ_VisionPlay_Architecture.drawio`](docs/ScoutIQ_VisionPlay_Architecture.drawio)
  — detailed live dashboard, job, artifact, and asynchronous isolate flow.

Open either with [draw.io / diagrams.net](https://app.diagrams.net/).

## Product workflow

1. Upload a basketball MP4 in the React dashboard.
2. FastAPI validates the clip and queues one canonical full-clip analysis.
3. The trained `best_player_seg.pt` model detects and segments court players.
4. ByteTrack creates raw tracklets.
5. The stitcher joins compatible tracklets into stable `person_group_id` values.
6. ScoutIQ chooses a dense snapshot and displays selectable players with team
   colors.
7. Select at least two players and press **Confirm**.
8. ScoutIQ reuses canonical detections—there is no second YOLO or ByteTrack
   pass—to produce the focused preview, tracking CSV, and Coach Insights.
9. A separate CPU worker builds the higher-quality **Isolate** preview in the
   background. The button displays real progress and an estimated time until
   the preview is ready.

The live online path is:

```text
React upload
  → FastAPI
  → best_player_seg.pt
  → ByteTrack
  → tracklet stitching
  → stable person groups
  → snapshot selection
  → focused preview + CSV + Coach Insights
  → asynchronous temporal isolate preview
```

GroundingDINO and SAM/SAM2 experiments remain offline tools. They are not in
the dashboard's online analysis path.

## Features added in this session

### 1. Stable person groups across the full clip

Raw ByteTrack IDs can fragment when a player is briefly occluded, leaves the
frame, or crosses a camera cut. The post-pass stitcher in
`src/pipeline/tracklet_stitcher.py` compares time gaps, position, jersey color,
and optional appearance features. Compatible raw tracklets are assigned one
stable `person_group_id`.

The dashboard selects and follows these stable groups rather than treating
each raw track ID as a different person. Quality output reports raw track count,
stitched group count, and identity-switch estimates before and after stitching.

### 2. Correct canonical selection mapping

Snapshot selections already contain explicit `person_group_id` values. The
Confirm path now gives those IDs priority and only maps raw `track_id` values
as a legacy fallback. This prevents a selected snapshot person from being
incorrectly remapped to a different stitched group.

### 3. Jersey-color team grouping

`src/pipeline/jersey_team_colors.py` samples jersey regions and clusters visible
players into team-color buckets. The snapshot and focused overlays use those
colors consistently. Officials and uncertain neutral detections use a separate
neutral treatment.

These are visual team groups, not named roster identity.

### 4. Snapshot display modes

The snapshot selector now includes:

- **Boxes + labels** — selectable person boxes, labels, and team colors.
- **Segmentation only** — a cleaner mask-focused snapshot.

Both modes use the same player selection state.

### 5. Focused preview refresh and re-selection

Confirm writes `focused_players.mp4` from canonical tracks. Re-Confirm now:

- clears the previous preview immediately;
- regenerates artifacts for the new selected group set;
- applies `?v=&g=` cache-busting;
- remounts the video element with `key={url}`;
- serves rewritten artifacts with `Cache-Control: no-store`; and
- provides **New selection** to return to the snapshot picker.

This prevents the browser from replaying a stale focused clip.

### 6. Browser-safe video encoding

`src/video_encoding.py` resolves the bundled `imageio-ffmpeg` executable and
re-encodes OpenCV `mp4v` output to H.264/yuv420p with `faststart`. Encoding
soft-fails to the original MP4 so a missing or failed ffmpeg process does not
destroy a completed tracking job.

### 7. Coach Insights MVP

Confirm requires at least two selected people. From the selected tracking CSV,
`src/pipeline/coach_insights.py` calculates explainable clip-level proxies
using **mask-bottom stance** (segmentation foot proxy) when available, else
bbox:

- selected-player coverage and trust;
- normalized path length;
- mean and p90 movement speed;
- high-effort movement fraction;
- pair visibility and separation bands (plus bbox gap for comparison);
- closeout bursts and chase-role indicators;
- sticky-matchup proxy for opposing teams;
- screen-shape proxy for same-team pairs;
- cut-timing overlap; and
- per-frame `pair_distances.csv` / `pair_distance_series` for performance work.

The React `CoachInsights` component presents glance cards and a plain-language
narrative. This MVP does not claim ball possession, shot outcome, calibrated
feet, or named player identity.

### 8. Scene-preserving Isolate preview

The first isolate prototype placed selected player pixels on black. It was
changed to preserve the court, hoop, crowd, score graphics, and selected
players while removing only non-selected tracked players.

The current high-quality renderer is
`src/pipeline/isolate_renderer.py`:

1. Reuses `tracks.jsonl.gz` masks and source video frames.
2. Builds removal masks for non-selected players.
3. Protects selected-player pixels during overlaps.
4. Finds static-scene ORB features outside player masks.
5. Aligns nearby frames to the current frame with robust affine estimation.
6. Reconstructs hidden court/background pixels from clean temporal samples.
7. Uses Navier-Stokes spatial inpainting only where no clean neighboring sample
   exists.
8. Feathers the reconstructed boundary and writes browser-ready H.264.

This removes the large translucent smears produced by frame-local inpainting
in most normal camera motion. A long occlusion can still require spatial
fallback when no neighboring frame contains the hidden background.

### 9. Asynchronous isolate processing

Confirm no longer waits for the expensive temporal isolate render.

- The normal focused preview, tracking CSV, and Coach Insights return first.
- `TrackingJobManager` owns a separate daemon CPU queue for isolate work.
- `isolate_status.json` stores generation, stage, frame progress, percentage,
  ETA, error, and final artifact.
- React polls `GET /api/jobs/{job_id}/isolate-status`.
- **Isolate** remains disabled and shows percentage/ETA until ready.
- A new Confirm generation supersedes the previous one.
- The final MP4 is atomically replaced and served with no-store headers.

This background job does not rerun the detector or tracker.

### 10. Best-frame jersey OCR (identity Resolve)

**Resolve jersey identities** does not OCR every frame and does not rely on a
single snapshot. For each non-official person group it:

1. Scores the full track with cheap crop metrics (size, sharpness, contrast,
   aspect, brightness).
2. Keeps top-N temporally spaced candidates (`identity.jersey_ocr` in
   `configs/player_tracking_mvp.yaml`, default 48).
3. Runs PaddleOCR only on those torso crops.
4. Aggregates weighted votes with ≥2 agreeing reads and consistency ≥ 0.60.
5. Matches unique jersey + team hint against the roster.

Manual jersey entry remains the fallback when OCR fails. Identity stays optional
for Confirm / Coach Insights.

## Current architecture

### Frontend

Primary files:

- `FrontEnd/src/pages/Dashboard.jsx` — workflow state, Analyze polling, Confirm,
  isolate polling, and cache-busted artifact URLs.
- `FrontEnd/src/components/VideoUpload.jsx` — video selection and game context.
- `FrontEnd/src/components/ProcessingStatus.jsx` — canonical job stage/progress.
- `FrontEnd/src/components/PlayerSnapshotSelector.jsx` — stable-group selection
  and snapshot view toggle.
- `FrontEnd/src/components/TrackingResults.jsx` — focused video, Full scene /
  Isolate control, isolate ETA, and continuity metrics.
- `FrontEnd/src/components/CoachInsights.jsx` — derived coaching metrics and
  narrative.

### FastAPI

Primary endpoints in `app/api_server.py`:

- `POST /api/analyze` — validate upload and queue canonical analysis.
- `GET /api/jobs/{job_id}` — canonical job state and result.
- `POST /api/tracking/selection` — save selected groups, render focused output,
  compute Coach Insights, and queue isolate processing.
- `GET /api/jobs/{job_id}/isolate-status` — isolate stage, progress, ETA, and
  readiness.
- `GET /api/jobs/{job_id}/files/{kind}` — snapshots, videos, CSV, quality,
  person groups, selection summary, and Coach Insights artifacts.

### Canonical analysis worker

`src/pipeline/job_manager.py` serializes GPU analysis jobs. The canonical
pipeline in `src/pipeline/player_track_pipeline.py` performs:

```text
validated source MP4
  → player segmentation
  → ByteTrack association
  → tracklet stitching
  → stable person groups
  → jersey team colors
  → dense snapshot
  → continuity quality report
  → canonical artifacts
```

### Focused Confirm path

`src/track_players.py` filters canonical tracks by selected
`person_group_id`. It decodes the source once for the normal focused overlay
and writes:

- `focused_players.mp4`;
- `selected_tracks.csv`;
- `selection_summary.json`; and
- `coach_insights.json`.

### Isolate worker

The independent CPU worker reads the same source video and canonical masks,
performs temporal background reconstruction, and writes:

- `isolate_status.json`; and
- `focused_isolated.mp4`.

## Job artifacts

Each canonical run writes under `outputs/jobs/<job_id>/`.

Important artifacts:

- `job.json` — canonical queue status and result.
- `tracks.jsonl.gz` — per-frame canonical detections, masks, track IDs, and
  person groups.
- `tracks.csv` — canonical position export.
- `person_groups.json` — raw-track to stable-group mapping and metadata.
- `quality.json` — detector and continuity quality.
- `snapshot.jpg` — boxes and labels.
- `snapshot_masks.jpg` — segmentation-only snapshot.
- `focused_players.mp4` — selected overlays on the full scene.
- `selected_tracks.csv` — selected group positions.
- `selection_summary.json` — selected render metadata and flags.
- `coach_insights.json` — coaching proxies and narrative.
- `isolate_status.json` — asynchronous isolate progress.
- `focused_isolated.mp4` — scene-preserving selected-player view.

Runtime artifacts and videos are intentionally local and should normally be
excluded from Git. Source, configuration, documentation, and lightweight test
fixtures belong in version control.

## Configuration

The live path is configured by `configs/player_tracking_mvp.yaml`, including:

- trained segmentation weights;
- detector confidence and IoU;
- CUDA/CPU device;
- ByteTrack thresholds and lost-track buffer;
- stitch timing, spatial, jersey-color, and appearance weights;
- team-color sampling; and
- detector/continuity quality gates.

## Running locally

From `scoutiq/` on Windows:

```powershell
.\run_scoutiq.bat
```

Services:

- React/Vite: `http://localhost:5173`
- FastAPI: `http://localhost:8000`
- API health: `http://localhost:8000/api/health`

After backend changes, restart the stack before testing. To exercise the new
workflow:

1. Analyze a short WNBA clip.
2. Select at least two snapshot players.
3. Confirm and verify the normal focused preview.
4. Watch the disabled Isolate button report progress and ETA.
5. Switch to Isolate when ready.
6. Change the selected groups and Confirm again to verify regeneration and
   cache-busting.

## Validation performed

During this session:

- changed Python modules compiled successfully;
- the FastAPI module imported with the isolate worker enabled;
- a synthetic temporal reconstruction smoke test passed;
- changed React files passed targeted ESLint checks; and
- the Vite production build completed successfully.

The repository-wide ESLint command still reports an existing React hook rule
issue in `PlayerSnapshotSelector.jsx`; it is unrelated to the asynchronous
isolate implementation.

## Known limitations

- Identity is a stable anonymous person group until Resolve/Manual jersey
  attaches a roster name; OCR searches best frames, not every frame.
- Team grouping is jersey-color based.
- Coach Insights use mask-bottom stance distances when segmentation exists;
  values remain frame-diagonal proxies, not calibrated court coordinates.
- Isolate quality depends on mask quality and whether nearby frames expose the
  hidden court.
- The isolate queue is process-local. Restarting FastAPI interrupts the active
  isolate job; Re-Confirm queues it again.
- Ball tracking, OCR-based possession, shot outcome, and tactical event
  classification remain future work.
