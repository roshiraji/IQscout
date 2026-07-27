"""Atomic-ish JSON writes tolerant of Windows file locks (mapped drives, AV)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_json_atomic(
    path: Path,
    payload: dict[str, Any] | list[Any],
    *,
    indent: int | None = 2,
    retries: int = 8,
    retry_delay_s: float = 0.03,
) -> None:
    """Write JSON to ``path``, retrying when replace fails with WinError 5.

    On Windows / network shares, ``Path.replace`` often fails if another process
    (frontend poll, AV) briefly locks ``job.json``. Prefer retries that leave
    the prior complete JSON visible; only use an in-place overwrite as a final
    fallback.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    last_err: OSError | None = None

    for attempt in range(retries):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(retry_delay_s * (attempt + 1))

    # Do not take this path until atomic replacement had every chance to work.
    # Callers reading job.json can retry through this tiny overwrite window.
    try:
        path.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)
        return
    except OSError as exc:
        last_err = exc
        tmp.unlink(missing_ok=True)

    if last_err is not None:
        raise last_err
    raise OSError(f"Failed to write {path}")
