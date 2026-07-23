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
    indent: int = 2,
    retries: int = 8,
    retry_delay_s: float = 0.03,
) -> None:
    """Write JSON to ``path``, retrying when replace fails with WinError 5.

    On Windows / network shares, ``Path.replace`` often fails if another process
    (frontend poll, AV) briefly locks ``job.json``. We retry, then fall back to
    an in-place overwrite of the destination file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    last_err: OSError | None = None

    for attempt in range(retries):
        try:
            tmp.write_text(text, encoding="utf-8")
            try:
                os.replace(str(tmp), str(path))
            except OSError:
                # Destination locked — overwrite in place (readers may see one stale read).
                path.write_text(text, encoding="utf-8")
                tmp.unlink(missing_ok=True)
            return
        except OSError as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(retry_delay_s * (attempt + 1))

    if last_err is not None:
        raise last_err
    raise OSError(f"Failed to write {path}")
