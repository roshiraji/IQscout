# ScoutIQ Segmentation Training v2 — Final Report

**Completed:** 2026-07-17 ~02:30 local  
**Elapsed:** 264.5 minutes (~4.4 hours)  
**Hardware:** NVIDIA GeForce RTX 3070 (8 GB), single GPU  

## Source videos

Three unique WNBA highlight packages (fourth file on disk was an MD5-identical Aces duplicate and was ingested once):

| Clip | Frames | Label candidates |
|------|--------|------------------|
| Fever vs Sparks (Jul 8, 2026) | 18,146 | 2,315 |
| Fever vs Mercury (Jul 9, 2026) | 20,747 | 2,632 |
| Fever vs Aces (Jul 12, 2026) | 18,132 | 2,278 |
| **Total** | **57,025** | **7,225** |

## Dataset (`data/datasets/nba_player_seg_v2/`)

- Train: **5,783** images  
- Val: **1,439** images  
- Split mode: **scene_segment** (frames between camera cuts stay in the same split; every game contributes to train and val)  
- Label QA rejections: **5** (invalid/tiny/out-of-bounds polygons cleaned or dropped)  
- Label factory: Grounding DINO boxes → SAM2 masks → YOLO-seg polygons, with court/scorebug exclusions  

## Training

- Base model: `yolo11n-seg.pt`  
- Epochs: 80 (targets met in round 1 — no second round needed)  
- Image size: 960 · batch: 8 · device: 0  
- Targets were `box mAP50 ≥ 0.50` and `mask mAP50 ≥ 0.40`  

## Holdout-scene comparison (same v2 val split)

| Metric | Production v1 | **v2** | Δ |
|--------|---------------|--------|---|
| Box mAP50 | 0.955 | **0.980** | +0.026 |
| Box mAP | 0.820 | **0.891** | +0.071 |
| Box precision | 0.933 | **0.950** | +0.016 |
| Box recall | 0.891 | **0.938** | +0.047 |
| Mask mAP50 | 0.943 | **0.972** | +0.029 |
| Mask mAP | 0.736 | **0.797** | +0.061 |

v2 wins on every reported metric. Production weights were **not** overwritten.

## Saved weights (use these)

| File | Path |
|------|------|
| Best v2 | `outputs/training_loop_v2/best_player_seg_v2.pt` |
| Last checkpoint | `outputs/training_loop_v2/last_player_seg_v2.pt` |
| Comparison JSON | `outputs/training_loop_v2/v2_comparison_report.json` |
| Current production (unchanged) | `outputs/training_loop/best_player_seg.pt` |

## How to switch the live pipeline to v2

In `configs/player_tracking_mvp.yaml`, change:

```yaml
weights: outputs/training_loop_v2/best_player_seg_v2.pt
```

Leave Confirm / Coach Insights / Isolate / jersey OCR as-is — they consume detections from whatever weights path is configured.

## Still future work (not in this run)

- Scene-cut tracker reset + cross-scene `global_player_id` reacquisition  
- Appearance Re-ID gallery  
- Multi-signal identity fusion with roster/headshot retrieval  
- Controlled tracking HOTA/IDF1 evaluation on cut-heavy holdouts  

Those build on these weights; segmentation quality for the three-game mix is now the stronger baseline.
