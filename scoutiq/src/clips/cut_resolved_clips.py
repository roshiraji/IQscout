"""Cut tracklet clips for resolved/review identities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def cut_resolved_clips(
    *,
    video: Path,
    tracks_json: Path,
    resolved_json: Path,
    output_dir: Path,
    min_status: str = "review",
) -> dict:
    status_rank = {"unresolved": 0, "review": 1, "resolved": 2}
    min_rank = status_rank.get(min_status, 1)

    tracks = json.loads(tracks_json.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_json.read_text(encoding="utf-8")).get("resolved_tracklets", [])
    resolved_by_tid = {}
    for item in resolved:
        tid = int(item["tracklet_id"].split("_")[-1])
        if status_rank.get(item["status"], 0) >= min_rank:
            resolved_by_tid[tid] = item

    frames_by_track: dict[int, list[int]] = defaultdict(list)
    for row in tracks:
        tid = int(row["track_id"])
        if tid in resolved_by_tid:
            frames_by_track[tid].append(int(row["frame_idx"]))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    spans = {}
    for tid, frames in frames_by_track.items():
        start_f, end_f = min(frames), max(frames)
        spans[tid] = (start_f, end_f)
        rid = resolved_by_tid[tid]
        clip_path = output_dir / f"{video.stem}_track_{tid:06d}_{rid['status']}.mp4"
        writers[tid] = {
            "writer": cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)),
            "clip_path": str(clip_path.resolve()),
        }

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, (start_f, end_f) in spans.items():
            if start_f <= frame_idx <= end_f:
                writers[tid]["writer"].write(frame)
        frame_idx += 1
    cap.release()

    records = []
    for tid, data in writers.items():
        data["writer"].release()
        rid = resolved_by_tid[tid]
        start_f, end_f = spans[tid]
        records.append(
            {
                "clip_id": f"{video.stem}_{tid:06d}",
                "video_id": video.stem,
                "tracklet_id": rid["tracklet_id"],
                "resolved_player_id": rid.get("resolved_player_id"),
                "status": rid.get("status"),
                "confidence": rid.get("resolver_confidence"),
                "clip_path": data["clip_path"],
                "start_frame": start_f,
                "end_frame": end_f,
            }
        )

    out_manifest = output_dir / "clips_manifest.json"
    out_manifest.write_text(json.dumps({"clips": records}, indent=2), encoding="utf-8")
    return {"clips": len(records), "manifest": str(out_manifest.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut clip segments from resolved tracklets.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--resolved", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-status", default="review", choices=["unresolved", "review", "resolved"])
    args = parser.parse_args()
    print(
        json.dumps(
            cut_resolved_clips(
                video=Path(args.video),
                tracks_json=Path(args.tracks),
                resolved_json=Path(args.resolved),
                output_dir=Path(args.output),
                min_status=args.min_status,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
