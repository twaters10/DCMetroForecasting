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

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_DIR = ROOT / "infra" / "lambda_collector"

# The ETL is a real package and is imported as `src.etl.*`, so only the repo root is
# needed for it. Deliberately NOT putting `src/etl` on the path: it uses relative
# imports, and a flat `config` there would shadow the collector's `config` — two
# different modules with the same name, resolved by path order.
for directory in (COLLECTOR_DIR, ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture(scope="session")
def spark():
    """One SparkSession for the whole run.

    Lives here rather than in `test_etl.py` because `test_arrivals_parity.py` needs it
    too — the parity test compares the Spark derivation against the pandas one, so it
    cannot avoid starting a session.

    Session-scoped, not module-scoped: a JVM start costs seconds, and two modules each
    starting their own doubles that for no benefit.
    """
    from src.etl.spark import build_session

    session = build_session(app_name="etl-tests", shuffle_partitions=1, cores="1")
    yield session
    session.stop()
