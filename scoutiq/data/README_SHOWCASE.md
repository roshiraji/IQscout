# Showcase data layout

```
data/
  footage/{train,val,holdout}/   # optional copies of source mp4s
  frames/<game_id>/<clip_id>/    # frame_XXXXXX.jpg (scales toward 1M frames)
  frames/manifest.csv
  frames/holdout_gate.json       # holdout clip IDs — never train on these
  exclusions/<clip_id>.json      # court ROI + shot clock / scorebug / crowd
  datasets/nba_player_seg/       # YOLO-seg train/val

outputs/
  auto_labels/                   # factory labels (GDINO+SAM2)
  training_loop/best_player_seg.pt
  showcase/                      # final unseen overlay mp4
```

## Commands (from scoutiq/)

```powershell
..\.venv\Scripts\python.exe -m src.showcase init
..\.venv\Scripts\python.exe -m src.showcase ingest --config configs/showcase_ingest.yaml
..\.venv\Scripts\python.exe -m src.showcase autolabel --config configs/showcase_ingest.yaml
..\.venv\Scripts\python.exe -m src.showcase build-dataset --config configs/showcase_ingest.yaml
..\.venv\Scripts\python.exe -m src.showcase train --config configs/showcase_train.yaml
..\.venv\Scripts\python.exe -m src.showcase run --config configs/showcase_player_seg.yaml
```

## Unseen clip gate

1. Put the holdout video in `clips:` with `split: holdout` in `showcase_ingest.yaml` (or only pass it to `run`).
2. `holdout_gate.json` records those clip IDs; `build-dataset` refuses them.
3. Never point training `data.yaml` at holdout frames.
4. Showcase `run` uses only `best_player_seg.pt` — no Grounding DINO.

Edit `data/exclusions/default.json` (or per-clip JSON) for shot clock / scorebug rectangles.
