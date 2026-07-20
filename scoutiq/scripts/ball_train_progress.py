"""Print live ball-training progress from Ultralytics results.csv + log.

Usage (from scoutiq/):
  ..\\.venv\\Scripts\\python.exe scripts\\ball_train_progress.py
  ..\\.venv\\Scripts\\python.exe scripts\\ball_train_progress.py --watch 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

SCOUTIQ = Path(__file__).resolve().parents[1]
RUNS = SCOUTIQ / "outputs" / "training_loop_ball"
LOG = RUNS / "full_train.log"

_EPOCH_RE = re.compile(
    r"(?:^|\s)(\d+)/(\d+)\s+\S+\s+[\d.]+\S*\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:\s*(\d+)%|"
    r"(?:^|\s)(\d+)/(\d+)\s+.*?(\d+)/(\d+)\s",
)
_SIMPLE_EPOCH = re.compile(r"\b(\d+)/(\d+)\b.*?(\d+)/(\d+)\b")


def _read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_log_epoch(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    # Read last ~64KB for speed
    data = log_path.read_bytes()
    text = data[-65536:].decode("utf-8", errors="ignore")
    matches = list(_SIMPLE_EPOCH.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    a, b, c, d = (int(m.group(i)) for i in range(1, 5))
    # Prefer epoch/total when first pair looks like epochs (<= 120)
    if b <= 120 and d > b:
        return {"epoch": a, "epochs": b, "batch": c, "batches": d}
    if d <= 120:
        return {"epoch": c, "epochs": d, "batch": a, "batches": b}
    return {"epoch": a, "epochs": b, "batch": c, "batches": d}


def collect() -> dict:
    rounds = []
    for round_dir in sorted(RUNS.glob("round_*")):
        if not round_dir.is_dir():
            continue
        rows = _read_results(round_dir / "results.csv")
        last = rows[-1] if rows else None
        entry = {
            "name": round_dir.name,
            "epochs_logged": len(rows),
            "last_epoch": int(float(last["epoch"])) if last and last.get("epoch") else None,
            "mAP50": float(last["metrics/mAP50(B)"]) if last and last.get("metrics/mAP50(B)") else None,
            "recall": float(last["metrics/recall(B)"]) if last and last.get("metrics/recall(B)") else None,
            "precision": float(last["metrics/precision(B)"]) if last and last.get("metrics/precision(B)") else None,
            "history": [
                {
                    "epoch": int(float(r["epoch"])),
                    "mAP50": float(r.get("metrics/mAP50(B)") or 0),
                    "recall": float(r.get("metrics/recall(B)") or 0),
                }
                for r in rows[-8:]
            ],
        }
        rounds.append(entry)

    live = _latest_log_epoch(LOG)
    best = RUNS / "best_ball_det.pt"
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(RUNS),
        "live_from_log": live,
        "rounds": rounds,
        "best_ball_det": {
            "exists": best.is_file(),
            "mtime": datetime.fromtimestamp(best.stat().st_mtime).isoformat()
            if best.is_file()
            else None,
            "bytes": best.stat().st_size if best.is_file() else None,
        },
        "log": str(LOG) if LOG.is_file() else None,
    }
    progress_path = RUNS / "progress.json"
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def print_progress(payload: dict) -> None:
    print(f"Ball training progress  ({payload['updated_at']})")
    print(f"  runs: {payload['runs_root']}")
    live = payload.get("live_from_log") or {}
    if live:
        print(
            f"  LIVE: epoch {live.get('epoch')}/{live.get('epochs')}  "
            f"batch {live.get('batch')}/{live.get('batches')}"
        )
    else:
        print("  LIVE: (no epoch line found in log yet)")
    for r in payload.get("rounds") or []:
        print(
            f"  {r['name']}: {r['epochs_logged']} epochs logged | "
            f"last mAP50={r['mAP50']} recall={r['recall']} precision={r['precision']}"
        )
        for h in r.get("history") or []:
            print(f"      ep {h['epoch']:>3}: mAP50={h['mAP50']:.4f}  recall={h['recall']:.4f}")
    best = payload.get("best_ball_det") or {}
    if best.get("exists"):
        print(f"  best_ball_det.pt  mtime={best.get('mtime')}  size={best.get('bytes')}")
    print(f"  wrote {RUNS / 'progress.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=float, default=0, help="Refresh every N seconds")
    args = parser.parse_args()
    while True:
        payload = collect()
        print_progress(payload)
        if not args.watch or args.watch <= 0:
            break
        print("---")
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
