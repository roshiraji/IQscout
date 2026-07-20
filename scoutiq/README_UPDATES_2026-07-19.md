# ScoutIQ VisionPlay — Updates (2026-07-19)

Branch: `7/19/2026`  
Date: July 19, 2026

This README summarizes what shipped on today’s branch: a **desktop-style one-click app**, a **story-driven loading experience**, **Anne’s React design wired to FastAPI**, **faster GPU Clean Isolate**, durable job writes on Windows, and clearer ingest/confirm progress UX.

---

## Highlights

1. **Single launcher** — `run_scoutiq.bat` starts the whole stack; closing the ScoutIQ window (or Quit) stops API + dashboard.
2. **Dynamic loading experience** — splash + in-app boot screen walk through how ScoutIQ works (ingest → track → select → isolate → possession → coach).
3. **Anne React integration** — `FrontEnd/src/API/scoutiqAPI.js` (from `f-Anne_ReactComponent`) rewired to FastAPI via Vite proxy; WNBA branding kept.
4. **Faster Clean Isolate on local GPU** — crop-to-erase + 768 working side + VRAM probe cache + LaMa warmup on RTX 3070.
5. **Windows-safe job writes** — atomic JSON with retry/fallback fixes `WinError 5` on mapped `Z:` drives.
6. **Live progress UX** — IngestProgress + ConfirmProgress stage checklists (upload → track → ready; confirm → isolate).

---

## 1. Desktop app launcher (one `.bat`)

**Entry point:** repo-root [`run_scoutiq.bat`](../run_scoutiq.bat)

| Behavior | Detail |
|----------|--------|
| Start | Double-click → minimized session → native splash → Edge/Chrome `--app` window |
| Quit | Close window **X** or in-app **Quit** → stops FastAPI (8000) + Vite (5173) + app browser profile |
| Single instance | Launcher lock file under `scoutiq/.scoutiq_runtime/` |
| Internals | `scoutiq/scripts/launch_app.ps1`, `stop_services.ps1`, `POST /api/shutdown` |

Removed separate `scoutiq/run_scoutiq.bat`, `stop_scoutiq.bat`, and console/silent helpers so there is **one** user-facing launcher.

---

## 2. Dynamic loading / “how ScoutIQ works”

### Native splash (WinForms)
Rotating story beats while services boot:

1. Ingest the clip  
2. See every player  
3. Keep identities stable  
4. Pick the matchup  
5. Clear the noise  
6. Read possession  
7. Coach the film  

ASCII-only status lines (no mojibake ellipsis on Windows).

### In-app `BootScreen`
Full-window experience after the app window opens:

- Pipeline step pills that light up in order  
- Rotating “How it works” cards  
- Live phase: warming → connecting → ready  

**Files:** `FrontEnd/src/components/BootScreen.jsx`, `FrontEnd/src/App.jsx`, `FrontEnd/src/index.css`, `scripts/launch_app.ps1`

---

## 3. Anne React branch integration

Pulled design/API patterns from `f-Anne_ReactComponent` ([IQscout](https://github.com/K9quantum1/IQscout.git)) and resolved conflicts against the live FastAPI pipeline:

| Anne brought | Kept / adapted |
|--------------|----------------|
| Dark hero + card layout | Already in `index.css` / Dashboard |
| `src/API/scoutiqAPI.js` | Rewired from Flask `:5000` → `/api` (Vite → `:8000`) |
| Simple checkbox selection | Kept click-to-select `PlayerSnapshotSelector` |
| Placeholder dashboard flow | Kept full ingest → confirm → isolate → coach flow |
| WNBA upload copy | Restored WNBA branding |

**Files:** `FrontEnd/src/API/scoutiqAPI.js`, `Dashboard.jsx`, `VideoUpload.jsx`, `PlayerSnapshotSelector.jsx`, `CoachInsights.jsx`

---

## 4. Clean Isolate — faster on user GPU

Hybrid isolate (temporal fill + LaMa) now uses local compute more efficiently:

| Optimization | Effect |
|--------------|--------|
| Crop to erase-mask bbox | Skip empty full-frame tiles |
| `ai_working_side: 768` | Downscale for LaMa, paste back |
| VRAM probe cache | `.scoutiq_runtime/lama_tile.json` |
| GPU warmup | Isolate worker preloads LaMa at API start |
| Lighter ORB | Faster temporal align on CPU |

**Config** (`configs/player_tracking_mvp.yaml` → `isolate`):

```yaml
mode: hybrid
ai_working_side: 768
crop_pad: 48
orb_features: 700
lama_full_erase: true
hard_composite: true
```

**Files:** `src/pipeline/lama_inpaint.py`, `src/pipeline/isolate_renderer.py`, `src/pipeline/job_manager.py`

---

## 5. Windows job durability (`WinError 5`)

Mapped-drive / AV locks broke `job.tmp.json → job.json` renames during ingest.

**Fix:** `src/pipeline/atomic_json.py` — retry + in-place overwrite fallback.

Wired into:

- `job.json`, `isolate_status.json`, `selection_summary.json`  
- `confirm_status.json` (API)

---

## 6. Progress UX (ingest + confirm)

### Ingest (`IngestProgress.jsx`)
Upload → Validate → Queue → Load model → Track → Stitch → Ball → Ready  

XHR upload progress, then poll `/api/jobs/{id}`.

### Confirm (`ConfirmProgress.jsx`)
Saving selection → focused preview stages → isolate (hybrid/AI) in background  

Poll `/api/jobs/{id}/confirm-status` and `isolate-status`.

### Preview modes (`TrackingResults.jsx`)
Full scene | Segmentation only (`focused_masks`) | Clean Isolate (hybrid)

---

## 7. Ball → offense / defense (continued)

Still part of the live stack on this branch:

- `src/ball/` — detect, track, fusion, possession, hoop factory, live hooks  
- Possession windows + offense-team override API  
- Matchup Outcomes roles when possession is confident  
- Eval harness: `scripts/eval_offense_defense.py`

---

## How to run (2026-07-19)

1. Double-click **`run_scoutiq.bat`** at the repo root.  
2. Wait for splash → ScoutIQ app window + boot story.  
3. Analyze clip → select players → Confirm → watch isolate progress.  
4. Close the window or click **Quit** to shut everything down.

Advanced / debug: services still live at `http://127.0.0.1:8000` (API) and `http://127.0.0.1:5173` (UI) while the app session is open.

---

## Key new / updated paths

```
run_scoutiq.bat
scoutiq/README_UPDATES_2026-07-19.md
scoutiq/scripts/launch_app.ps1
scoutiq/src/pipeline/atomic_json.py
scoutiq/src/pipeline/lama_inpaint.py
scoutiq/src/pipeline/isolate_renderer.py
scoutiq/src/ball/
scoutiq/FrontEnd/src/API/scoutiqAPI.js
scoutiq/FrontEnd/src/components/BootScreen.jsx
scoutiq/FrontEnd/src/components/IngestProgress.jsx
scoutiq/FrontEnd/src/components/ConfirmProgress.jsx
scoutiq/configs/player_tracking_mvp.yaml   # isolate: fast GPU path
scoutiq/app/api_server.py                  # /api/shutdown, confirm-status, …
```

---

## Notes / follow-ups

- Ball detector quality still needs better labels for production mAP targets.  
- Hoop detector factory exists; weights not fully trained.  
- ProPainter / SAM refine on erase masks discussed, not shipped on this branch.  
- Prefer `ai_working_side: null` (or `"full"`) only when you want max isolate quality over speed.
