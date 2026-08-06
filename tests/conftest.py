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

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_DIR = ROOT / "infra" / "lambda_collector"

# The ETL is a real package and is imported as `src.etl.*`, so only the repo root is
# needed for it. Deliberately NOT putting `src/etl` on the path: it uses relative
# imports, and a flat `config` there would shadow the collector's `config` — two
# different modules with the same name, resolved by path order.
for directory in (COLLECTOR_DIR, ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
