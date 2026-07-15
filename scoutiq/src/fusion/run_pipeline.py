"""
Fusion video pipeline: trained NBA detector + Grounding DINO + SAM2 + confidence fusion.

Uses bounding boxes from multiple sources, refines with SAM2 segmentation, and
boosts confidence when bbox + mask + multi-source agreement align.

Usage (from scoutiq/):
  python -m src.fusion.run_pipeline --config configs/nba_fusion_pipeline.yaml
"""

from __future__ import annotations

from src.ml_warnings import configure_ml_runtime

configure_ml_runtime()

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from src.court_roi import apply_nms, filter_detections_by_roi, load_or_default_roi
from src.frame_extractor import extract_all_frames, get_video_info
from src.fusion.detection_fusion import (
    FusionWeights,
    apply_mask_confidence_boost,
    fuse_bbox_detections,
)
from src.fusion.grounding_dino_runner import (
    GroundingDinoSettings,
    detect_with_grounding_dino,
    grounding_dino_available,
)
from src.fusion.trained_detector import detect_with_trained_model
from src.locateanything_runner import locate_anything_available, run_locateanything_on_frame
from src.object_tracker import update_tracker
from src.render_tracking_overlay import render_video
from src.sam2_segmenter import refine_detections_with_sam2, sam2_available

logger = logging.getLogger(__name__)
SCOUTIQ_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_repo(path_str: str) -> Path:
    """Resolve paths relative to repo root (parent of scoutiq/)."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (SCOUTIQ_ROOT.parent / p).resolve()


def _build_gdino_settings(cfg: dict) -> GroundingDinoSettings:
    gd = cfg.get("grounding_dino", {})
    return GroundingDinoSettings(
        config_path=_resolve_repo(gd.get("config_path", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")),
        checkpoint_path=_resolve_repo(gd.get("checkpoint_path", "weights/groundingdino_swint_ogc.pth")),
        box_threshold=float(gd.get("box_threshold", 0.30)),
        text_threshold=float(gd.get("text_threshold", 0.25)),
        class_prompts=gd.get("class_prompts"),
        class_names=gd.get("class_names"),
    )


def run_fusion_pipeline(
    *,
    video: Path,
    output_video: Path,
    trained_weights: Path,
    frames_dir: Path,
    roi_path: Path,
    trained_conf: float = 0.25,
    use_grounding_dino: bool = True,
    grounding_dino_settings: GroundingDinoSettings | None = None,
    use_locateanything: bool = False,
    use_sam2: bool = True,
    sam2_weights: str = "sam2.1_b.pt",
    nms_iou: float = 0.45,
    fusion_match_iou: float = 0.45,
    min_fused_conf: float = 0.22,
    fusion_weights: FusionWeights | None = None,
    max_frames: int | None = None,
) -> Path:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not trained_weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {trained_weights}")

    info = get_video_info(video)
    frame_paths = extract_all_frames(video, frames_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    roi = load_or_default_roi(roi_path, width=info.width, height=info.height)
    gd_settings = grounding_dino_settings or GroundingDinoSettings()
    gd_ok = use_grounding_dino and grounding_dino_available(gd_settings)
    la_ok = use_locateanything and locate_anything_available()
    sam_ok = use_sam2 and sam2_available()

    logger.info(
        "Fusion pipeline | video=%s | trained=%s | GDINO=%s | LA=%s | SAM2=%s | frames=%d",
        video.name, trained_weights.name, gd_ok, la_ok, sam_ok, len(frame_paths),
    )
    if use_grounding_dino and not gd_ok:
        logger.warning(
            "Grounding DINO not available — check weights at %s",
            gd_settings.checkpoint_path,
        )

    all_tracked: dict[int, list[dict]] = {}
    fusion_stats = {"frames": 0, "detections": 0, "boosted": 0, "avg_conf": 0.0}

    for frame_path in tqdm(frame_paths, desc="Fusion detect+track", unit="frame"):
        try:
            fid = int(frame_path.stem.split("_")[-1])
        except ValueError:
            fid = 0
        ts = fid / info.fps

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            all_tracked[fid] = []
            continue

        sources: list[dict] = []

        # 1) Trained NBA player detector (primary)
        sources.extend(
            detect_with_trained_model(
                image_bgr,
                frame_id=fid,
                timestamp=ts,
                image_name=frame_path.name,
                weights=trained_weights,
                conf=trained_conf,
            )
        )

        # 2) Grounding DINO text grounding (LocateAnything substitute)
        if gd_ok:
            sources.extend(
                detect_with_grounding_dino(
                    image_bgr,
                    frame_id=fid,
                    timestamp=ts,
                    image_name=frame_path.name,
                    settings=gd_settings,
                )
            )

        # 3) LocateAnything (optional if installed)
        if la_ok:
            sources.extend(
                run_locateanything_on_frame(
                    image_path=frame_path,
                    frame_id=fid,
                    timestamp=ts,
                    width=info.width,
                    height=info.height,
                )
            )

        # 4) Fuse overlapping boxes across sources
        fused = fuse_bbox_detections(sources, match_iou=fusion_match_iou)
        for det in fused:
            if det.get("class_name") in ("player_red", "player_black_blue"):
                det["class_name"] = "player"
                det["canonical_class"] = "player"
        fused = filter_detections_by_roi(
            fused, roi,
            strict_classes=("player", "referee", "hoop"),
            lenient_classes=("ball",),
        )
        fused = apply_nms(fused, iou_threshold=nms_iou)

        # 5) SAM2 segmentation on fused boxes
        if sam_ok and fused:
            fused = refine_detections_with_sam2(image_bgr, fused, sam2_weights)

        # 6) Boost confidence using mask + multi-source agreement
        fused = apply_mask_confidence_boost(
            fused,
            weights=fusion_weights,
            min_fused_conf=min_fused_conf,
        )

        # 7) Track with fused high-confidence detections
        tracked = update_tracker(fused)
        all_tracked[fid] = tracked

        fusion_stats["frames"] += 1
        fusion_stats["detections"] += len(fused)
        fusion_stats["boosted"] += len(tracked)
        if fused:
            fusion_stats["avg_conf"] += sum(d.get("fused_confidence", d.get("confidence", 0)) for d in fused)

    if fusion_stats["frames"]:
        fusion_stats["avg_conf"] = round(
            fusion_stats["avg_conf"] / max(fusion_stats["detections"], 1), 4,
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    render_video(
        frame_paths=frame_paths,
        detections_by_frame=all_tracked,
        output_path=output_video,
        fps=info.fps,
        width=info.width,
        height=info.height,
    )

    report = {
        "video": str(video),
        "output_video": str(output_video),
        "trained_weights": str(trained_weights),
        "grounding_dino": gd_ok,
        "locateanything": la_ok,
        "sam2": sam_ok,
        "fusion_stats": fusion_stats,
    }
    report_path = output_video.parent / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Fusion complete → %s", output_video)
    logger.info("Avg fused confidence: %s", fusion_stats["avg_conf"])
    return output_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NBA fusion pipeline (detector + Grounding DINO + SAM2)")
    p.add_argument("--config", default="configs/nba_fusion_pipeline.yaml")
    p.add_argument("--video")
    p.add_argument("--max-frames", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    cfg = _load_config(_resolve(args.config, SCOUTIQ_ROOT))

    fusion_cfg = cfg.get("fusion", {})
    fw = FusionWeights(
        bbox=float(fusion_cfg.get("bbox_weight", 0.55)),
        mask_iou=float(fusion_cfg.get("mask_weight", 0.25)),
        multi_source=float(fusion_cfg.get("multi_source_weight", 0.20)),
    )

    video = _resolve(args.video or cfg["video"], SCOUTIQ_ROOT)
    gd_settings = _build_gdino_settings(cfg)
    run_fusion_pipeline(
        video=video,
        output_video=_resolve(cfg["output_video"], SCOUTIQ_ROOT),
        trained_weights=_resolve(cfg["trained_weights"], SCOUTIQ_ROOT),
        frames_dir=_resolve(cfg.get("frames_dir", f"data/working_frames/fusion/{video.stem}"), SCOUTIQ_ROOT),
        roi_path=_resolve(cfg.get("roi_path", "config/court_roi.json"), SCOUTIQ_ROOT),
        trained_conf=float(cfg.get("trained_conf", 0.25)),
        use_grounding_dino=bool(cfg.get("use_grounding_dino", True)),
        grounding_dino_settings=gd_settings,
        use_locateanything=bool(cfg.get("use_locateanything", False)),
        use_sam2=bool(cfg.get("use_sam2", True)),
        sam2_weights=str(cfg.get("sam2_weights", "sam2.1_b.pt")),
        nms_iou=float(cfg.get("nms_iou", 0.45)),
        fusion_match_iou=float(fusion_cfg.get("match_iou", 0.45)),
        min_fused_conf=float(fusion_cfg.get("min_confidence", 0.22)),
        fusion_weights=fw,
        max_frames=args.max_frames or cfg.get("max_frames"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
