"""Make the project importable in tests the same way the API imports it.

``src/`` modules import each other by bare name (``player_snapshot_selector``),
so both the project root and ``src/`` have to be on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
