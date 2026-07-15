"""Separate detector quality from full-clip tracking continuity."""

from __future__ import annotations


DEFAULT_THRESHOLDS = {
    "min_frame_coverage": 0.90,
    "min_mean_detection_confidence": 0.50,
    "max_short_track_ratio": 0.35,
    "max_track_gap_frames": 30,
}


def evaluate_quality(
    quality: dict,
    thresholds: dict | None = None,
) -> dict:
    """Return pass/fail gates and the correct next engineering action.

    A low detector gate may justify data review/retraining. A passing detector
    with poor continuity calls for tracker tuning or tracklet stitching instead.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    track_stats = quality.get("track_stats", {})
    total_tracks = max(len(track_stats), 1)
    short_ratio_raw = len(quality.get("short_track_ids", [])) / total_tracks
    # Prefer person-group continuity after stitch (product identity unit).
    unique_groups = int(quality.get("unique_person_groups") or 0)
    unique_raw = int(quality.get("unique_raw_tracks") or total_tracks)
    if unique_groups > 0:
        short_groups = 0
        groups_seen: dict[int, dict] = {}
        for item in track_stats.values():
            gid = item.get("person_group_id")
            if gid is None:
                continue
            gid = int(gid)
            bucket = groups_seen.setdefault(
                gid,
                {"detections": 0, "max_gap": 0},
            )
            bucket["detections"] += int(item.get("detections") or 0)
            bucket["max_gap"] = max(
                bucket["max_gap"], int(item.get("max_gap") or 0)
            )
        if groups_seen:
            # Matches stitch min_group_frames default: product unit maturity.
            short_thr = 8
            short_groups = sum(
                1 for g in groups_seen.values() if g["detections"] < short_thr
            )
            short_ratio = short_groups / max(len(groups_seen), 1)
            max_gap = max((g["max_gap"] for g in groups_seen.values()), default=0)
        else:
            short_ratio = short_ratio_raw
            max_gap = max(
                (int(item.get("max_gap", 0)) for item in track_stats.values()),
                default=0,
            )
    else:
        short_ratio = short_ratio_raw
        max_gap = max(
            (int(item.get("max_gap", 0)) for item in track_stats.values()),
            default=0,
        )

    mean_conf = quality.get("detection_confidence", {}).get("mean") or 0.0
    coverage = float(quality.get("frame_coverage") or 0.0)
    stitch_info = quality.get("stitch") or {}
    stitch_enabled = bool(stitch_info.get("enabled", unique_groups > 0))

    detection_checks = {
        "frame_coverage": coverage >= float(limits["min_frame_coverage"]),
        "mean_detection_confidence": mean_conf
        >= float(limits["min_mean_detection_confidence"]),
    }
    # After stitch, intra-group gaps are expected (occlusion join); continuity is
    # productized via person groups + selected_player_retention, not raw max_gap.
    gap_ok = (
        True
        if stitch_enabled and unique_groups > 0
        else max_gap <= int(limits["max_track_gap_frames"])
    )
    tracking_checks = {
        "short_track_ratio": short_ratio <= float(limits["max_short_track_ratio"]),
        "max_track_gap": gap_ok,
    }
    detection_pass = all(detection_checks.values())
    tracking_pass = all(tracking_checks.values())

    if not detection_pass:
        limiting_factor = "detection_or_segmentation"
        recommendation = (
            "Review false negatives/masks on held-out clips and retrain only if "
            "detector recall or mask quality is the confirmed bottleneck."
        )
    elif not tracking_pass:
        limiting_factor = "tracking_continuity"
        if stitch_enabled:
            recommendation = (
                "Keep the current segmentation weights; tighten stitch gaps/"
                "appearance weights or ByteTrack lost-buffer — do not retrain."
            )
        else:
            recommendation = (
                "Keep the current segmentation weights; tune ByteTrack lost-buffer/"
                "matching thresholds or add ReID-based tracklet stitching."
            )
    else:
        limiting_factor = "none"
        recommendation = "Quality gates passed; do not retrain for this clip."

    return {
        "passed": detection_pass and tracking_pass,
        "detection_pass": detection_pass,
        "tracking_pass": tracking_pass,
        "limiting_factor": limiting_factor,
        "recommendation": recommendation,
        "thresholds": limits,
        "observed": {
            "frame_coverage": round(coverage, 4),
            "mean_detection_confidence": round(float(mean_conf), 4),
            "short_track_ratio": round(short_ratio, 4),
            "short_track_ratio_raw": round(short_ratio_raw, 4),
            "unique_raw_tracks": unique_raw,
            "unique_person_groups": unique_groups,
            "max_track_gap_frames": max_gap,
        },
        "checks": {
            "detection": detection_checks,
            "tracking": tracking_checks,
        },
        "held_out_requirements": [
            "normal half-court play",
            "player occlusion",
            "camera pan/zoom",
            "highlight hard cuts",
        ],
        "identity_metrics": [
            "top_1_accuracy",
            "unresolved_rate",
            "false_name_rate",
            "selected_player_retention",
        ],
    }
