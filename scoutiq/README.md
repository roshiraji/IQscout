# ScoutIQ / VisionPlay AI MVP

Beginner-friendly Python project for processing WNBA MP4 video clips with OpenCV, YOLO, and tracking tools. This MVP stage sets up dependencies and a simple video metadata script. YOLO detection and tracking are not wired up yet.

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

Before any YOLO or tracking runs, upload and validate basketball MP4 clips through the Streamlit intake UI. With the virtual environment activated, from the `scoutiq/` folder:

```text
.\run_streamlit.ps1
```

Or:

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

```text
.\run_gui.bat
```

Or in PowerShell:

```powershell
.\run_gui.ps1
```

This uses `scoutiq/.venv-gui` (Python 3.12), which includes working Tkinter on Windows. The main `.venv` may not support GUI windows on all Python installs.

Click **Select Video File**, choose an MP4 in File Explorer, and view approval status, metadata, and the saved JSON report.

**If no window appears:** make sure you are not using a Python install without Tkinter/Tcl. Run `.\run_gui.bat` instead of `python app/video_intake_gui.py` directly.

**Future pipeline flow:** upload → validate → if approved → `player_labeling.py` → tracking → dashboard

## Player labeling (YOLO)

After a clip passes intake validation, label detected players with YOLOv8. From the `scoutiq/` folder:

```text
python src/player_labeling.py data/input_videos/caitlin_clark_clip.mp4
```

Outputs:

- `outputs/annotated_videos/<name>_labeled_players.mp4` — video with Player_1, Player_2, ... boxes
- `outputs/detections/<name>_player_labels.csv` — per-frame box coordinates and confidence

Labels are per-frame only (no cross-frame tracking yet).

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
python src/player_labeling.py data/input_videos/caitlin_clark_clip.mp4
```

Force CPU if needed:

```text
python src/player_labeling.py data/input_videos/caitlin_clark_clip.mp4 --device cpu
```

## What's next

- `src/video_intake_filter.py` — pre-processing validation gate (ready)
- `app/video_upload_filter.py` — Streamlit upload UI (ready)
- `src/track_players.py` — player tracking (placeholder)
- `src/utils.py` — shared helpers (placeholder)
- `app/dashboard.py` — results dashboard (placeholder)

YOLO detection and tracking are not wired up yet. The intake filter runs before those steps.
