"""Categorical encoding, with the mapping persisted alongside the model.

`features.config` commits to native LightGBM categorical handling rather than target
encoding, for a stated reason: 785 segments is small enough that native splits suffice,
and target encoding would have to be fit inside a CV fold or a strictly-prior window to
be leakage-safe — real risk for modest lift.

Native handling needs integer codes, and the codes have to mean the same thing at
serving time as they did at training time. `pd.Categorical.codes` does **not** give that
for free: codes are assigned from whatever categories happen to be present, so a frame
missing one route silently renumbers everything after it. The mapping is therefore fit
once, saved, and reloaded — never re-derived from the data being scored.

**Unseen categories map to -1, which LightGBM reads as missing.** That is the honest
encoding: a segment the model never saw is not category zero, it is absent information.
The measured rate is not hypothetical — 89 segments appear in validation but never in
training on the current 12 service days.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..features.config import FeatureConfig

logger = logging.getLogger("models.encode")

# LightGBM treats any negative value in a categorical feature as missing.
UNSEEN_CODE = -1


@dataclass(frozen=True, slots=True)
class CategoricalEncoder:
    """A fixed category -> code mapping for each categorical column."""

    mapping: dict[str, dict[str, int]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        config: FeatureConfig | None = None,
        extra_columns: Sequence[str] = (),
    ) -> CategoricalEncoder:
        """Learn codes from the training frame only. Never call this on validation.

        `extra_columns` exists because `FeatureConfig.categorical_columns` lists the
        categoricals the feature layer *intends*, which is not the same as every column
        that is non-numeric. `fare_period` is a string and is absent from that list;
        unencoded it reaches LightGBM as a raw string. Callers pass what they actually
        found in the frame, so a new string feature cannot break a fit silently.
        """
        settings = config or FeatureConfig()
        wanted = list(dict.fromkeys([*settings.categorical_columns, *extra_columns]))
        mapping: dict[str, dict[str, int]] = {}
        for column in wanted:
            if column not in frame.columns:
                logger.warning("categorical column %s not present — skipped", column)
                continue
            # Sorted so the mapping is deterministic across runs and machines; an
            # encoder that renumbers itself between runs is unreproducible.
            values = sorted(frame[column].dropna().astype(str).unique())
            mapping[column] = {value: code for code, value in enumerate(values)}
        return cls(mapping=mapping)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace categorical columns with their codes. Returns a copy."""
        out = frame.copy()
        for column, codes in self.mapping.items():
            if column not in out.columns:
                continue
            out[column] = (
                out[column].astype(str).map(codes).fillna(UNSEEN_CODE).astype("int32")
            )
        return out

    def unseen_rate(self, frame: pd.DataFrame) -> dict[str, float]:
        """Fraction of rows per column whose category was never seen in training."""
        rates: dict[str, float] = {}
        for column, codes in self.mapping.items():
            if column in frame.columns:
                known = frame[column].astype(str).isin(codes)
                rates[column] = float(100 * (~known).mean())
        return rates

    @property
    def columns(self) -> list[str]:
        return list(self.mapping)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.mapping, indent=1, sort_keys=True))
        logger.info(
            "wrote encoder mapping for %d column(s) to %s", len(self.mapping), path
        )

    @classmethod
    def load(cls, path: Path) -> CategoricalEncoder:
        return cls(mapping=json.loads(Path(path).read_text()))
