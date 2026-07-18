# ScoutIQ VisionPlay — Updates (2026-07-17)

Branch: `cursor/7-17-26`  
Date: July 17, 2026

This README summarizes what shipped on today’s branch: stronger jersey OCR, mask-based distance metrics, a multi-game segmentation training loop (v2 weights), and a coach-facing **Matchup Outcomes** scorecard.

---

## Highlights

1. **Best-frame jersey OCR** — search each player’s full track for readable jersey moments, OCR only the best candidates, vote with consistency gates; never guess identity.
2. **Mask-bottom stance distances** — pairwise gaps use segmentation foot proxies when available, exported as JSON + CSV.
3. **Segmentation training v2** — trained on all three unique Fever highlight games; `best_player_seg_v2.pt` beats prior production metrics on held-out scenes.
4. **Matchup Outcomes UI** — Stay Close / Create Space (or Off-Ball Action) scores out of 100 for selected pairs — performance outcomes, not camera coverage.

---

## 1. Jersey identity (best-frame OCR)

**Flow:** Track → opportunity score full track → torso crops → PaddleOCR → consistency gates → roster lookup → manual fallback if unreadable.

**Config** (`configs/player_tracking_mvp.yaml` → `identity.jersey_ocr`):

- `max_candidates_per_person: 48`
- `min_spacing_frames: 3`
- `min_agreeing_reads: 2`
- `min_consistency: 0.60`

**Files:** `src/crops/build_tracklet_crops.py`, `src/ocr/run_tracklet_ocr.py`, `src/pipeline/resolve_job_identity.py`

---

## 2. Stance-based pair distances & coach metrics

After Confirm with ≥2 players:

| Artifact | Purpose |
|----------|---------|
| `selected_tracks.csv` | Includes `stance_x`, `stance_y`, `position_source` |
| `coach_insights.json` | Scorecard + pair summaries (v3) |
| `pair_distances.csv` | Per-frame `gap_norm`, `gap_band`, `closing_speed_norm` |

**Matchup Outcomes (opposite teams / unknown):**

- **Stay Close /100** — time attached, tight stretches, closeouts
- **Create Space /100** — time separated, get-away bursts

**Same team / same jersey color:**

- **Off-Ball Action /100** — screen shape, cut timing, spacing

Honest limit: Create Space means “got away,” not points scored (no ball/shot model yet).

**Files:** `src/pipeline/geometry_points.py`, `src/pipeline/coach_insights.py`, `src/track_players.py`, `FrontEnd/src/components/CoachInsights.jsx`

---

## 3. Segmentation training v2

**Source videos** (folder `input videos for training/` — 4 files on disk, **3 unique** games; Aces file was duplicated):

- Fever vs Sparks (Jul 8, 2026)
- Fever vs Mercury (Jul 9, 2026)
- Fever vs Aces (Jul 12, 2026)

**Pipeline:** ingest → GDINO+SAM2 autolabel → scene-segment train/val split + label QA → YOLO11-seg train → compare vs production.

| Metric | Production v1 | **v2** |
|--------|---------------|--------|
| Box mAP50 | 0.955 | **0.980** |
| Mask mAP50 | 0.943 | **0.972** |
| Box recall | 0.891 | **0.938** |
| Mask mAP | 0.736 | **0.797** |

**Weights (live path now points here):**

```yaml
# configs/player_tracking_mvp.yaml
weights: outputs/training_loop_v2/best_player_seg_v2.pt
max_frames: null   # full uploaded clip (not 100-frame smoke)
```

**Also:** `outputs/training_loop_v2/TRAINING_REPORT_v2.md`, `v2_comparison_report.json`  
**Configs:** `configs/showcase_ingest_v2.yaml`, `configs/showcase_train_v2.yaml`  
**Orchestrator:** `scripts/run_training_v2.py`

---

## 4. Live product path (unchanged contract)

```text
Upload → Analyze (YOLO-seg v2 + ByteTrack + stitch)
  → Select ≥2 person groups
  → Confirm → focused MP4 + CSV + Matchup Outcomes
  → Optional Resolve jersey identities
  → Async Isolate preview
```

Confirm still **reuses** `tracks.jsonl.gz` (no second detect/track). Identity remains optional.

---

## 5. Diagrams

- `docs/ScoutIQ_DataFlow.drawio` — pipeline + job artifacts
- `docs/ScoutIQ_ModelFlow.drawio` — model family data flow

Open with [diagrams.net](https://app.diagrams.net/).

---

## How to run

From `scoutiq/`:

```powershell
.\run_scoutiq.bat
```

- React: `http://localhost:5173`
- API: `http://localhost:8000`

Restart the backend after pulling so it loads v2 weights and `max_frames: null`.

**Quick check**

1. Upload a full clip (should process entire length).
2. Select ≥2 players → Confirm.
3. See **Matchup Outcomes** with numeric /100 scores.
4. Optional: Resolve jersey identities.

---

## Non-goals (still)

- Ball tracking / shot make-miss
- Calibrated court feet
- True defensive rating / box-score substitutes
- Forcing track IDs across hard scene cuts (future: global identity layer)

---

## Key paths

| Item | Path |
|------|------|
| Live config | `scoutiq/configs/player_tracking_mvp.yaml` |
| v2 weights | `scoutiq/outputs/training_loop_v2/best_player_seg_v2.pt` |
| Training report | `scoutiq/outputs/training_loop_v2/TRAINING_REPORT_v2.md` |
| Scorecard UI | `scoutiq/FrontEnd/src/components/CoachInsights.jsx` |
| Showcase notes | `scoutiq/SHOWCASE.md` |
