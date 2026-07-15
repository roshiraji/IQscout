# Grounded SAM Auto-Annotation Pipeline

Self-contained folder for automated instance-segmentation dataset generation
(Grounding DINO → SAM → COCO / YOLO-seg export).

Reference notebook: `notes/NOTES_DONOTUSE_automated_dataset_annotation_and_evaluation_with_grounding_dino_and_sam.ipynb`

## Local testing (start here)

Everything runs on your machine — no Colab or cloud required. Weights are in `../weights/`, venv in `../.venv/`.

**One-click smoke test** (validate + mock + one real image):

```powershell
cd grounded_sam_pipeline
.\test_local.bat
```

**Manual steps:**

```powershell
cd grounded_sam_pipeline
.\run.bat --validate-only
.\run.bat --mock --limit 2 --config config\auto_annotation_dog_demo.yaml
.\run.bat --config config\auto_annotation_dog_demo.yaml --single-image data\input_images\dog-3.jpeg --overwrite --save-previews
```

Check outputs in `data/auto_annotated_dog_demo/previews/`.

For basketball/WNBA classes, use the default `config/auto_annotation.yaml` once dog demo looks good.

## WNBA pipeline bridge

Annotate players from ScoutIQ video frames (separate connector script):

```powershell
cd scoutiq
..\.venv\Scripts\python.exe -m src.ingest.wnba_grounded_sam ^
  --frames-dir "data/annotation_frames/new tune" ^
  --clip-name new_tune --limit 5 --overwrite
```

Or extract + annotate in one step:

```powershell
..\.venv\Scripts\python.exe -m src.ingest.wnba_grounded_sam ^
  --video "data/input_videos/new tune.mp4" ^
  --clip-name new_tune --fps 2 --limit 5 --overwrite
```

From `grounded_sam_pipeline/` you can also use `run_wnba.bat` (forwards to the bridge).

Outputs land in `scoutiq/data/annotations/wnba_grounded_sam/<clip>/` with YOLO-seg labels using classes `player`, `referee`, `ball` (matches `configs/wnba_players.yaml`).

## Folder layout

```text
grounded_sam_pipeline/
├── auto_annotation/          # Python package + CLI
├── config/auto_annotation.yaml
├── data/input_images/        # put images here
├── data/auto_annotated/      # outputs (created on run)
├── tests/
├── scripts/setup.bat
├── requirements.txt
└── run.bat
```

Shared repo assets (not duplicated here):

- `../GroundingDINO/` — Grounding DINO install
- `../weights/` — model checkpoints
- `../segment_anything/` — SAM package (via `pip install -e ..` from repo root)
- `../.venv/` — Python virtual environment

## Setup (Windows)

From **ScoutIQ repo root**:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r grounded_sam_pipeline\requirements.txt
pip install -e GroundingDINO
pip install -e .
```

Or run `grounded_sam_pipeline\scripts\setup.bat`

## Commands

All commands run from **`grounded_sam_pipeline/`**:

```powershell
cd grounded_sam_pipeline
..\.venv\Scripts\python.exe -m auto_annotation.run --validate-only
```

Or use `run.bat`:

```powershell
.\run.bat --validate-only
.\run.bat --single-image data\input_images\dog-3.jpeg --save-previews
.\run.bat --limit 10 --save-previews
.\run.bat --config config\auto_annotation.yaml --resume
.\run.bat --mock --limit 3
```

## Tests

```powershell
cd grounded_sam_pipeline
..\.venv\Scripts\python.exe -m pytest tests\test_auto_annotation.py -v
```

See `auto_annotation/README.md` for full protocol, troubleshooting, and output format details.
