"""Write JSON summary and HTML gallery for segmentation lab runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_summary(
    *,
    video_path: Path,
    sam_model: str,
    yolo_model: str,
    frame_reports: list[dict],
) -> dict:
    all_masks = [mask for frame in frame_reports for mask in frame.get("masks", [])]
    sam_scores = [float(m["sam_score"]) for m in all_masks]
    fill_ratios = [float(m["fill_ratio"]) for m in all_masks]
    inside_ratios = [float(m["mask_inside_box_ratio"]) for m in all_masks]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path.resolve()),
        "sam_model": sam_model,
        "yolo_model": yolo_model,
        "frames_evaluated": len(frame_reports),
        "total_masks": len(all_masks),
        "aggregate": {
            "avg_sam_score": _avg(sam_scores),
            "avg_fill_ratio": _avg(fill_ratios),
            "avg_mask_inside_box_ratio": _avg(inside_ratios),
            "min_sam_score": round(min(sam_scores), 4) if sam_scores else 0.0,
            "max_sam_score": round(max(sam_scores), 4) if sam_scores else 0.0,
        },
        "frames": frame_reports,
    }


def write_json_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_html_gallery(path: Path, summary: dict, run_dir: Path) -> None:
    frames = summary.get("frames", [])
    agg = summary.get("aggregate", {})
    cards = []
    for frame in frames:
        comparison = frame.get("comparison_image", "")
        rel = Path(comparison).name if comparison else ""
        masks = frame.get("masks", [])
        mask_lines = "<br>".join(
            f"{m['label']}: SAM {m['sam_score']:.2f}, fill {m['fill_ratio']:.2f}, "
            f"inside box {m['mask_inside_box_ratio']:.2f}"
            for m in masks
        )
        cards.append(
            f"""
            <section class="card">
              <h2>Frame {frame['frame_number']} @ {frame['timestamp_seconds']:.2f}s</h2>
              <p>{frame['player_count']} players</p>
              <img src="{rel}" alt="frame {frame['frame_number']} comparison" />
              <div class="metrics">{mask_lines or "No masks"}</div>
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Segmentation Lab — {run_dir.name}</title>
  <style>
    body {{ margin: 0; background: #0a0a0a; color: #f2f2f2; font-family: Arial, sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #b5b5b5; margin-bottom: 24px; line-height: 1.5; }}
    .card {{ background: #111; border: 1px solid #222; border-radius: 16px; padding: 20px; margin-bottom: 20px; }}
    .card img {{ width: 100%; border-radius: 10px; margin-top: 12px; }}
    .metrics {{ margin-top: 12px; color: #cfcfcf; font-size: 14px; line-height: 1.6; }}
    .agg {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 28px; }}
    .stat {{ background: #1a1a1a; border-radius: 12px; padding: 14px; }}
    .stat strong {{ display: block; color: #9a9a9a; font-size: 13px; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <main>
    <h1>Segmentation Lab</h1>
    <div class="meta">
      Video: {summary.get("video", "")}<br />
      YOLO: {summary.get("yolo_model", "")} · SAM: {summary.get("sam_model", "")}<br />
      Frames: {summary.get("frames_evaluated", 0)} · Masks: {summary.get("total_masks", 0)}
    </div>
    <div class="agg">
      <div class="stat"><strong>Avg SAM score</strong>{agg.get("avg_sam_score", 0)}</div>
      <div class="stat"><strong>Avg fill ratio</strong>{agg.get("avg_fill_ratio", 0)}</div>
      <div class="stat"><strong>Avg inside box</strong>{agg.get("avg_mask_inside_box_ratio", 0)}</div>
      <div class="stat"><strong>SAM score range</strong>{agg.get("min_sam_score", 0)} – {agg.get("max_sam_score", 0)}</div>
    </div>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
