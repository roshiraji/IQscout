# Auto-Annotation Pipeline (Grounding DINO + SAM)

Automated instance-segmentation dataset generation for ScoutIQ.

## Protocol

```
text classes → enhanced prompts → Grounding DINO detections
  → bounding-box filtering (clip, area, NMS)
  → SAM box prompts → highest-scoring mask per object
  → preview visualization → COCO + YOLO-seg export
```

This reproduces the workflow from the reference notebook without Colab shell commands, Roboflow uploads, or notebook-only syntax.

## Installation (Windows)

Uses the shared ScoutIQ repo `.venv`. From repo root:

```powershell
pip install -r grounded_sam_pipeline\requirements.txt
pip install -e GroundingDINO
pip install -e .
```

Run all commands from **`grounded_sam_pipeline/`** (or use `run.bat`).

Place model weights in **`../weights/`** (repo root, or use `--download-weights`):

| File | Size | Source |
|------|------|--------|
| `groundingdino_swint_ogc.pth` | ~660 MB | Grounding DINO releases |
| `sam_vit_h_4b8939.pth` | ~2.5 GB | Meta SAM |

Copy images into `data/input_images/` (jpg/png).

## Commands

**Validate environment only:**

```powershell
python -m auto_annotation.run --validate-only
```

**One-image test:**

```powershell
python -m auto_annotation.run --single-image data\input_images\dog-3.jpeg --save-previews
```

**Small batch (10 images):**

```powershell
python -m auto_annotation.run --limit 10 --save-previews
```

**Full dataset:**

```powershell
python -m auto_annotation.run --config config\auto_annotation.yaml --resume
```

**Mock mode (no models, tests pipeline structure):**

```powershell
python -m auto_annotation.run --mock --limit 3
```

**Custom classes via CLI:**

```powershell
python -m auto_annotation.run --classes "person,ball,referee" --box-threshold 0.35
```

## Configuration

Edit `config/auto_annotation.yaml`:

- `input.images_directory` — source images
- `detection.classes` — class names (used as Grounding DINO prompts)
- `detection.prompt_overrides` — per-class prompt customization
- `detection.box_threshold` / `text_threshold` — detection sensitivity
- `models.*` — checkpoint paths
- `export.formats` — `coco`, `yolo_segmentation`, `voc`

## Output structure

```
data/auto_annotated/
├── images/           # copied source images
├── masks/            # individual mask PNGs
├── previews/         # annotated preview JPGs
├── annotations/
│   ├── coco/instances.json
│   ├── yolo_seg/classes.txt + labels/
│   └── voc/          # bbox-only XML (+ README)
└── reports/
    ├── run_summary.json
    ├── image_results.csv
    ├── class_counts.csv
    ├── review_queue.csv
    ├── failures.log
    └── progress.json
```

## Supervision API note

The original notebook used `supervision==0.6.0` (`sv.Dataset`, tuple unpacking on detections). This pipeline uses **supervision 0.29+** with `Detections` attribute access, separate `LabelAnnotator`, and manual COCO/YOLO export. Functional outcome is identical.

## Manual review

Grounding DINO + SAM produce **initial labels**. Always inspect `reports/review_queue.csv` and preview images before training a downstream model.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA unavailable | Install CUDA PyTorch wheel; or `--device cpu` |
| `get_head_mask` error | Use `transformers>=4.36,<5.0` |
| `_C` / ms_deform_attn error | Windows patch in `GroundingDINO/.../ms_deform_attn.py` (PyTorch fallback) |
| Missing checkpoints | Download manually or `--download-weights` |
| No detections | Lower `box_threshold`; improve prompts in `prompt_overrides` |
| CUDA OOM | Pipeline retries on CPU if `retry_on_cpu_after_oom: true` |
| Empty SAM masks | Check box validity; lower detection thresholds |

## Tests

```powershell
python -m pytest tests\test_auto_annotation.py -v
```
