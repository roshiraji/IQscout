# ScoutIQ / VisionPlay AI MVP

Beginner-friendly Python project for processing WNBA MP4 video clips with OpenCV, YOLO, and tracking tools. This MVP stage sets up dependencies and a simple video metadata script. YOLO detection and tracking are not wired up yet.

## Project structure

```
scoutiq/
├── data/input_videos/     # Place your MP4 clips here
├── outputs/               # Annotated videos, detections, frames (future)
├── src/                   # Video processing and tracking scripts
├── app/                   # Streamlit dashboard (coming soon)
├── requirements.txt
└── README.md
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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

## What's next

- `src/track_players.py` — player tracking (placeholder)
- `src/utils.py` — shared helpers (placeholder)
- `app/dashboard.py` — Streamlit dashboard (placeholder)

Dependencies for YOLO, tracking, Streamlit, and FastAPI are installed but not used in scripts yet.
