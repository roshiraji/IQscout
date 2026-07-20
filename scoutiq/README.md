# ScoutIQ / VisionPlay AI MVP

Beginner-friendly Python project for processing WNBA MP4 video clips with OpenCV, YOLO, and tracking tools. The React dashboard and FastAPI backend drive the main workflow; Streamlit remains available as an optional legacy UI.

## Latest updates (2026-07-19)

See [`README_UPDATES_2026-07-19.md`](README_UPDATES_2026-07-19.md) for today’s branch: one-click desktop launcher, story-driven loading screen, Anne React + FastAPI integration, faster GPU Clean Isolate, Windows-safe job writes, and ingest/confirm progress UX.

**Run the app:** double-click repo-root `run_scoutiq.bat` (close the ScoutIQ window to quit everything).

## Latest dashboard architecture

See [`SESSION_README.md`](SESSION_README.md) for the July 14, 2026 tracking,
Coach Insights, and asynchronous isolate-preview build. The editable draw.io
diagram is at
[`docs/ScoutIQ_Architecture.drawio`](docs/ScoutIQ_Architecture.drawio), with a
detailed live-flow diagram at
[`docs/ScoutIQ_VisionPlay_Architecture.drawio`](docs/ScoutIQ_VisionPlay_Architecture.drawio).

## New modular identity pipeline (6/23/2026)

The repo now includes a runnable, report-aligned v1 pipeline under `src/`:

- `python -m src.ingest.extract_frames --video ... --output ...`
- `python -m src.detect.train --config configs/wnba_players.yaml`
- `python -m src.track.run --video ... --weights yolo26s.pt --output outputs/identity/<video>/tracking`
- `python -m src.crops.build_tracklet_crops --video ... --tracks ... --output ...`
- `python -m src.ocr.run_tracklet_ocr --tracklets ... --output ...`
- `python -m src.face.embed_tracklets --tracklets ... --output ...`
- `python -m src.reid.embed_tracklets --tracklets ... --output ...`
- `python -m src.db.ingest_rosters --season 2026 --roster-dir outputs/identity/rosters`
- `python -m src.resolver.resolve_game --video-id ... --tracklets ... --ocr ... --face ... --reid ... --rosters ... --output ...`
- `python -m src.clips.cut_resolved_clips --video ... --tracks ... --resolved ... --output ...`
- Full orchestration: `python -m src.resolver.run_end_to_end --video data/input_videos/<clip>.mp4 --season 2026 --weights yolo26s.pt`
- API: `uvicorn src.api.app:app --reload`

Outputs are written under `outputs/identity/`.

## Segment Anything (SAM)

Meta's [Segment Anything](https://github.com/facebookresearch/segment-anything) package lives at the repo root (`segment_anything/`). ScoutIQ wraps it in `src/segmentation/`.

**One-time setup** (from `scoutiq/` with `.venv` active):

```bat
pip install -e ..[all]
scripts\download_sam_checkpoint.bat vit_b
python scripts\verify_sam.py
```

Checkpoints are stored in `models/sam/` (not committed). Default model is **vit_b** (~357 MB, CUDA when available).

**Use in Python:**

```python
import cv2
from src.segmentation import load_sam_predictor, predict_mask_from_box

image = cv2.imread("path/to/frame.jpg")
mask = predict_mask_from_box(image, [x1, y1, x2, y2])  # boolean HxW array
```

Larger models: `scripts\download_sam_checkpoint.bat vit_l` or `vit_h`, then pass `model_type="vit_l"` to the helpers.

## Segmentation lab (SAM sandbox)

Isolated evaluation on your **fine-tune clip** before wiring SAM into the main pipeline.
YOLO boxes are used as SAM prompts; results are saved as side-by-side images + HTML report.

**Config:** `configs/segmentation_lab.yaml` (default clip: `Caitlin Clark clip.mp4`)

```bat
scripts\run_segmentation_lab.bat
scripts\run_segmentation_lab.bat --video "data/input_videos/new tune.mp4"
scripts\run_segmentation_lab.bat --use-snapshot
python -m src.segmentation_lab.run --max-frames 5 --frame-stride 15
```

**Outputs:** `outputs/segmentation_lab/<clip_name>/`
- `index.html` — browser gallery (Original | YOLO boxes | SAM masks)
- `report.json` — per-frame SAM scores, fill ratios, mask paths
- `frames/` — comparison JPGs
- `masks/` — per-player binary mask PNGs

## Build log — 6/15/2026

Summary of work completed on this date:

### Environment and launcher
- Set up `scoutiq/.venv` on a new machine with CUDA PyTorch (`torch 2.6.0+cu124`) for RTX 3070
- Replaced the old Streamlit-only launcher with a unified **`run_scoutiq.bat`** / **`stop_scoutiq.bat`** stack
- Added `scripts/runtime_config.json`, `check_environment.ps1`, `start_services.ps1`, and `stop_services.ps1`
- Default startup is now **React (port 5173) + FastAPI (port 8000)**; Streamlit (8501) is opt-in via config
- Removed legacy launchers: `run_gui.bat`, `run_streamlit.bat`, and related one-off scripts

### FastAPI backend (`app/api_server.py`)
- `POST /api/analyze` — video intake validation, player snapshot, optional labeling
- `POST /api/tracking/selection` — saves user player picks and runs ByteTrack tracking
- `GET /api/files/snapshot/`, `/heatmap/`, `/annotated/`, `/tracking/` — serve outputs to the React UI
- `GET /api/health` — reports available pipeline features

### React dashboard (`FrontEnd/`)
- Upload → Analyze → click `Player_#` boxes → Confirm tracking flow on `Dashboard.jsx`
- New components: `PlayerSnapshotSelector`, `TrackingResults`, updated `VideoUpload` and `ProcessingStatus`
- Vite proxies `/api` to `http://localhost:8000`
- Fixed blank-page crash (`isSaving` prop missing in `VideoUpload.jsx`)

### Vision pipeline
- **`src/player_snapshot_selector.py`** — samples frames, picks best snapshot, exports JPG + JSON
- **`src/track_players.py`** — YOLO + ByteTrack; locks onto user-selected players from snapshot frame
- **`src/heatmap.py`** — heatmap generation wired into analyze flow
- **`src/video_encoding.py`** — re-encodes OpenCV `mp4v` output to **H.264** for in-browser `<video>` playback
- Bug fix: restored path variables in `player_labeling.py` (`annotated_path` NameError)
- Bug fix: disabled `augment=True` for YOLO26 in `yolo_config.py` (YOLO11 only)

### Outputs added this build
| Path | Purpose |
|------|---------|
| `outputs/player_snapshots/` | Best-frame JPG + player box JSON |
| `outputs/tracking_selections/` | User selection JSON per clip |
| `outputs/tracking/` | Per-player position CSV + summary JSON |
| `outputs/tracking_videos/` | Tracked-players preview MP4 (H.264) |

### Verified on test clip
`data/input_videos/Caitlin Clark clip.mp4` — intake, snapshot selection, and tracking preview (3/4 players anchored; `Player_6` failed IoU match at snapshot frame).

## Project structure

```
scoutiq/
├── data/input_videos/     # Place your MP4 clips here
├── outputs/
│   ├── intake_reports/    # JSON validation reports per upload
│   ├── annotated_videos/
│   ├── detections/
│   └── frames/
├── src/                   # Video processing and validation scripts
├── app/                   # Streamlit apps
├── requirements.txt
└── README.md
```

## Setup

### Windows PowerShell

Run these commands from the **`scoutiq/`** folder (not the parent `scoutIQ VisionPlay` folder):

```powershell
cd scoutiq
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Troubleshooting `ModuleNotFoundError: No module named 'cv2'`**

Packages are installed in `scoutiq/.venv`, not a `.venv` at the workspace root. Activate the correct environment:

```powershell
cd Z:\scoutIQ VisionPlay\scoutiq
.\.venv\Scripts\Activate.ps1
python src/process_video.py data/input_videos/sample_wnba_clip.mp4
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** `pip install torch` uses CPU wheels from PyPI by default. For GPU/CUDA support, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/) later.

## Test video

Copy any WNBA (or other) MP4 clip to:

```
data/input_videos/sample_wnba_clip.mp4
```

## Quick test

With the virtual environment activated, from the `scoutiq/` folder:

```text
python src/process_video.py data/input_videos/sample_wnba_clip.mp4
```

You should see printed metadata: FPS, width, height, and frame count.

## Video Intake Filter (upload gate)

Before any YOLO or tracking runs, upload and validate basketball MP4 clips through the Streamlit intake UI. The easiest way to start it:

```text
.\run_scoutiq.bat
```

Or with the virtual environment activated, from the `scoutiq/` folder:

```text
.\.venv\Scripts\activate
streamlit run app/video_upload_filter.py
```

**Important:** Run Streamlit from `scoutiq/.venv`, not the workspace-root `.venv`. The root environment has CPU-only PyTorch and will show `YOLO will run on: CPU`.

The app will:

- Accept drag-and-drop **MP4** uploads only
- Save the file to `data/input_videos/`
- Validate duration (5–15 seconds), resolution, FPS, file size, and readability
- Show **Approved** (green), **Rejected** (red), or **Warnings** (yellow)
- Write a JSON report to `outputs/intake_reports/<filename>_intake_report.json`

You can also validate from the command line:

```text
python src/video_intake_filter.py data/input_videos/sample_wnba_clip.mp4
```

### Desktop GUI (Tkinter)

For a simple file-picker window (no terminal path typing), from the `scoutiq/` folder:

```powershell
.\run_gui.ps1
```

This uses `scoutiq/.venv-gui` (Python 3.12), which includes working Tkinter on Windows. The main `.venv` may not support GUI windows on all Python installs.

Click **Select Video File**, choose an MP4 in File Explorer, and view approval status, metadata, and the saved JSON report.

**If no window appears:** make sure you are not using a Python install without Tkinter/Tcl. Use `.\run_gui.ps1` instead of `python app/video_intake_gui.py` directly.

**Future pipeline flow:** upload → validate → if approved → `player_labeling.py` → tracking → dashboard

## Player labeling (YOLO26)

After a clip passes intake validation, label detected players with **Ultralytics YOLO26**. From the `scoutiq/` folder:

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4"
```

Default model: `yolo26n.pt` (nano). Choose a larger model for more accuracy:

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --model yolo26s.pt
```

Both **YOLO26** (default) and **YOLO11** families are selectable in the Streamlit sidebar and via `--model`. Each family has five sizes (n/s/m/l/x):

| Family | Example | Best for |
|--------|---------|----------|
| YOLO26 | `yolo26n.pt` | Edge-first default; fastest on CPU/GPU |
| YOLO26 | `yolo26x.pt` | Maximum YOLO26 quality |
| YOLO11 | `yolo11n.pt` | Prior-gen baseline / comparison runs |
| YOLO11 | `yolo11m.pt` | YOLO11 mid-size accuracy |

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --model yolo11s.pt
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --model yolo26m.pt
```

Weights auto-download on first use. Model metadata lives in `src/yolo_config.py` and applies to the Streamlit pipeline too.

### Quality mode (default)

**Quality mode is ON by default** — it favors detection quality and tracking-readiness over raw speed:

| Setting | Quality mode | Speed mode (`--fast`) |
|---------|--------------|----------------------|
| `imgsz` | 1280 (or 1920) | 640 |
| Primary confidence | 0.25 | 0.35 |
| Augmentation | Yes | No |
| IoU | 0.45 | 0.50 |

Inference runs at a **review floor** of `conf=0.15` so distant or partially occluded players are captured. Detections are then split:

- **Primary** (`conf >= 0.25` in quality mode): drawn on video, written to main CSV and ByteTrack export
- **Review** (`0.15 <= conf < 0.25`): logged to `low_confidence_candidates.csv` only (not drawn)

Quality mode is slower but preferred for ByteTrack, jersey-number OCR, and player identification prep.

CLI examples:

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4"
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --imgsz 1920
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --fast
```

On an **RTX 3070 Ti**, start at `imgsz=1280`. Try `--imgsz 1920` if VRAM allows; if CUDA runs out of memory, retry at 1280.

The Streamlit intake UI includes a **Quality mode** checkbox and **imgsz** selector in the sidebar (same settings as CLI).

### Outputs

| File | Purpose |
|------|---------|
| `outputs/annotated_videos/<name>_labeled_players.mp4` | Video with Player_1, Player_2, ... boxes (primary detections only) |
| `outputs/detections/<name>_player_labels.csv` | Per-frame box coordinates, confidence, and `class_id` |
| `outputs/detections/<name>_bytetrack_detections.csv` | ByteTrack-ready feed: `frame,class_id,confidence,x1,y1,x2,y2` |
| `outputs/detections/<name>_frame_statistics.csv` | Per-frame player count and confidence stats |
| `outputs/detections/<name>_low_confidence_candidates.csv` | Review-tier boxes (0.15–0.25 conf in quality mode) |

At the end of each run, timing metrics are printed: total runtime, average inference ms/frame, and pipeline FPS.

Labels are per-frame only (no cross-frame tracking yet). Phase 2 will add ByteTrack.

### GPU acceleration (recommended)

Your machine has an NVIDIA GPU, but the default `pip install torch` build is CPU-only. To enable GPU inference, reinstall PyTorch with CUDA from the `scoutiq/` folder:

```powershell
.\.venv\Scripts\activate
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU support:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then run labeling (auto-uses GPU when available):

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4"
```

Force CPU if needed:

```text
python src/player_labeling.py "data/input_videos/Caitlin Clark clip.mp4" --device cpu
```

## Local Development Launcher

Start every local ScoutIQ service from one place instead of opening multiple terminals.

### Start everything

Double-click:

```
run_scoutiq.bat
```

Or from Command Prompt / PowerShell inside the `scoutiq/` folder:

```cmd
.\run_scoutiq.bat
```

The launcher will:

- Verify `.venv\Scripts\python.exe` exists
- Check required folders and Python packages (`cv2`, `fastapi`, `uvicorn`, `ultralytics`)
- Install React dependencies on first run (`FrontEnd/node_modules`) if Node.js is available
- Start the **FastAPI backend** on port **8000**
- Start the **React dashboard** (Vite) on port **5173**
- Open **http://localhost:5173** in your browser
- Keep one command window open until you press **Ctrl+C** or close the window

Optional: set `"streamlit": { "enabled": true }` in `scripts/runtime_config.json` to also run the legacy Streamlit UI on port **8501**.

### Stop everything

Double-click:

```
stop_scoutiq.bat
```

Or run:

```cmd
.\stop_scoutiq.bat
```

This terminates React/FastAPI/Streamlit processes started by the launcher, frees ports **5173**, **5174**, **8501**, and **8000**, and removes `.scoutiq_runtime/pids.json`.

### Services started

| Service | Port | URL |
|---------|------|-----|
| React dashboard (default) | 5173 | http://localhost:5173 |
| FastAPI backend | 8000 | http://localhost:8000/api/health |
| Streamlit (optional, legacy) | 8501 | http://localhost:8501 |

Runtime process IDs are stored in `.scoutiq_runtime/pids.json`. Service logs are written to `.scoutiq_runtime/react.log`, `.scoutiq_runtime/fastapi.log`, and `.scoutiq_runtime/streamlit.log` when enabled.

### Troubleshooting

- **Missing dependencies:** activate `.venv` and run `pip install -r requirements.txt`
- **React won't start:** install [Node.js](https://nodejs.org/), then `cd FrontEnd && npm install`
- **Ports stuck / orphaned servers:** run `stop_scoutiq.bat`
- **Wrong Cursor interpreter:** select `.venv\Scripts\python.exe` in the IDE Python interpreter picker
- **Legacy Streamlit UI:** set `streamlit.enabled` to `true` in `scripts/runtime_config.json`

Launcher settings live in `scripts/runtime_config.json`.

## Staged auto-label pipeline (`src/main.py`)

This repo now includes a staged basketball auto-label CLI designed to keep progress moving even when advanced dependencies are missing.

### Working vertical slice (ready now)

`python src/main.py demo-mock` runs end-to-end with no LocateAnything or SAM2 required:

1. Creates sample frames (or uses existing frames)
2. Generates mock detections (strict JSON contract)
3. Exports YOLO detection labels
4. Draws debug frames
5. Builds `outputs/review_queue.json`
6. Builds dataset version `data/datasets/basketball_yolo_seg_v001`
7. Renders annotated demo videos

### CLI commands

```bash
python src/main.py init
python src/main.py set-roi --image data/frames/frame_000001.jpg
python src/main.py extract --video data/raw_videos/play.mp4 --fps 2
python src/main.py autolabel --source locateanything
python src/main.py autolabel --source mock
python src/main.py debug
python src/main.py export-yolo-det
python src/main.py export-yolo-seg
python src/main.py export-coco
python src/main.py review-select --percent 20
python src/main.py review
python src/main.py build-dataset --version v001
python src/main.py train --version v001 --epochs 50
python src/main.py infer --weights outputs/training_runs/run_001/best.pt --video data/raw_videos/play.mp4
python src/main.py track
python src/main.py render
python src/main.py full --video data/raw_videos/play.mp4 --fps 2 --review_percent 20
python src/main.py demo-mock
```

### Data contracts

Raw detections are stored per-frame in:

```text
data/autolabel/raw_predictions/frame_XXXXXX.json
```

Corrected labels are stored separately (never overwrite raw predictions):

```text
data/autolabel/corrected_predictions/corrected_labels.json
```

### Dataset + run versioning

- Dataset versions: `data/datasets/basketball_yolo_seg_v001`, `v002`, ...
- Training runs: `outputs/training_runs/run_001`, `run_002`, ...

Each dataset version includes:
- `images/`, `labels/`, `data.yaml`
- `dataset_manifest.json`
- `label_source_summary.json`
- `reviewed_frames.json`
- `rejected_labels.json`

### Extended troubleshooting (auto-label CLI)

- **ffmpeg not found:** install ffmpeg or rely on OpenCV-only render outputs
- **CUDA not available:** training automatically falls back to CPU
- **CUDA out of memory:** training retries with smaller settings (`imgsz=640`, reduced batch)
- **LocateAnything executable not found:** `autolabel --source locateanything` falls back to mock detections
- **SAM2 not installed:** `export-yolo-seg` uses box-polygon fallback
- **Ultralytics not installed:** install with `pip install ultralytics`
- **labels are empty:** verify `data/frames/*.jpg` exists, then rerun `autolabel` + export
- **too many crowd detections:** set court ROI with `set-roi` and rerun autolabel
- **ball not detected:** expected in some motion/blur frames; use review queue + corrections
- **training starts but mAP is bad:** review labels, increase reviewed sample percent, rebuild dataset version
- **video render has wrong FPS:** rerun `render --fps <target>`
- **Windows path problems:** use quoted absolute paths or `pathlib`-style normalized paths

## What's next

- Jersey number recognition and persistent player identity across frames
- Spacing, speed, and defensive-pressure metrics from tracking CSV
- Heatmap and full labeling on the React dashboard after tracking confirm
- `src/utils.py` — shared helpers (placeholder)
- `app/dashboard.py` — legacy results dashboard (placeholder)

Intake validation, YOLO labeling, snapshot selection, and ByteTrack tracking are wired through the React + FastAPI stack.


## WNBA Player Database

### Overview

This component builds a local SQLite database of WNBA player and roster information using the SportsDataverse WNBA API.

### Purpose

The player database is intended to support:

* Player search and lookup
* Jersey number matching
* Team roster information
* Future FastAPI endpoints
* Future RAG (Retrieval-Augmented Generation) functionality

### Data Source

SportsDataverse:

```python
from sportsdataverse.wnba import (
    espn_wnba_teams_site,
    espn_wnba_team_roster,
)
```

### Current Status

Implemented:

* Pull all WNBA teams
* Pull roster data for each team
* Store player information in SQLite

Database output:

```text
outputs/wnba_players.db
```

Implementation file:

```text
src/wnba_player_database.py
```

### Next Steps

* Add FastAPI endpoints
* Connect player search to React dashboard
* Add player profile pages
* Build RAG layer for player information retrieval
