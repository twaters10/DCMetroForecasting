"""Make the collector's modules importable the way Lambda imports them.

The deployment zip puts every collector module at its root, so `handler.py`
imports `config`, `writers` and `static_gtfs` as top-level names. Tests have to
reproduce that layout or the imports fail — the same reason `local_run.py`
manipulates `sys.path`. Importing them as `infra.lambda_collector.config` instead
would test a module graph that does not exist in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

COLLECTOR_DIR = Path(__file__).resolve().parents[1] / "infra" / "lambda_collector"

if str(COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_DIR))
