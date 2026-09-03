"""Tests for the modelling layer, against synthetic fixtures only.

No fitting happens here. The parts worth testing are the ones that are wrong silently:
an encoder that renumbers itself between runs, and a journey window that sums across a
hole. Both produce entirely plausible numbers when broken.

The contiguity tests exist because this bug was hit for real. `features.io` applies the
quality filter, which drops ~2.5% of segments and punches holes into the middle of
otherwise intact trips — 767 broken pairs on 12 service days, the largest spanning
6,420s. Summing across one invents a journey that never ran.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.features.config import FeatureConfig
from src.models.encode import UNSEEN_CODE, CategoricalEncoder
from src.models.journey import contiguous_blocks, journey_windows
from src.models.train import (
    KEY_COLUMNS,
    MEASUREMENT_COLUMNS,
    build_matrix,
    feature_columns,
)

START = datetime(2026, 8, 11, 12, 0, tzinfo=UTC).replace(tzinfo=None)


def _segments(durations, *, trip="T1", run=0, first_seq=1, gap_before=None):
    """Build a contiguous run of segments whose arrivals chain exactly.

    `gap_before` inserts dead time before the run's first departure, simulating the hole
    a filtered-out segment leaves behind.
    """
    rows = []
    clock = START + (timedelta(seconds=gap_before) if gap_before else timedelta())
    for offset, duration in enumerate(durations):
        sequence = first_seq + offset
        rows.append(
            {
                "service_date": "2026-08-11",
                "trip_id": trip,
                "trip_run": run,
                "from_stop_sequence": sequence,
                "stop_sequence": sequence + 1,
                "actual_departure_ts": clock,
                "actual_duration_sec": duration,
            }
        )
        clock = clock + timedelta(seconds=duration)
    return rows


# --------------------------------------------------------------------------- blocks


def test_clean_trip_is_one_block():
    frame = pd.DataFrame(_segments([120, 180, 60, 240]))
    assert contiguous_blocks(frame).nunique() == 1


def test_hole_in_the_middle_splits_the_block():
    """A dropped segment leaves a time gap; the run must break there."""
    head = _segments([120, 180])
    # Resumes at sequence 5 after 300s of unexplained time — a filtered-out segment.
    tail = _segments([90, 150], first_seq=5, gap_before=300 + 120 + 180)
    frame = pd.DataFrame(head + tail)

    blocks = contiguous_blocks(frame)
    assert blocks.nunique() == 2
    assert blocks.iloc[0] == blocks.iloc[1]
    assert blocks.iloc[2] == blocks.iloc[3]
    assert blocks.iloc[1] != blocks.iloc[2]


def test_separate_trips_never_share_a_block():
    frame = pd.DataFrame(
        _segments([120, 120], trip="T1") + _segments([120, 120], trip="T2")
    )
    assert contiguous_blocks(frame).nunique() == 2


def test_repeat_run_of_one_trip_id_is_a_separate_block():
    """trip_id is reused within a service day; ~19% of real rows are a repeat run."""
    frame = pd.DataFrame(_segments([120, 120], run=0) + _segments([120, 120], run=1))
    assert contiguous_blocks(frame).nunique() == 2


def test_stop_span_greater_than_one_is_not_a_break():
    """A stop passed between polls widens a segment; the timeline stays continuous."""
    rows = _segments([120, 300, 120])
    rows[1]["stop_sequence"] = rows[1]["from_stop_sequence"] + 2  # spans two stops
    rows[2]["from_stop_sequence"] = rows[1]["stop_sequence"]
    frame = pd.DataFrame(rows)
    assert contiguous_blocks(frame).nunique() == 1


# -------------------------------------------------------------------------- windows


def test_perfect_prediction_scores_zero_error():
    frame = pd.DataFrame(_segments([120, 180, 60, 240]))
    frame["prediction"] = frame["actual_duration_sec"]

    windows = journey_windows(frame, "prediction", lengths=(1, 2, 4))
    assert (windows["mae_sec"] == 0).all()


def test_journey_actuals_are_the_sum_of_their_segments():
    frame = pd.DataFrame(_segments([120, 180, 60, 240]))
    frame["prediction"] = 0.0

    windows = journey_windows(frame, "prediction", lengths=(1, 2, 4)).set_index(
        "segments"
    )
    # With a zero prediction the residual IS the actual, so bias reads back the sums.
    assert windows.loc[4, "journeys"] == 1
    assert windows.loc[4, "median_duration_sec"] == pytest.approx(600)
    assert windows.loc[2, "journeys"] == 3  # (1,2) (2,3) (3,4)


def test_windows_never_sum_across_a_block_boundary():
    """The bug this module exists to prevent."""
    head = _segments([120, 120])
    tail = _segments([120, 120], first_seq=5, gap_before=9999)
    frame = pd.DataFrame(head + tail)
    frame["prediction"] = 0.0

    # Four segments, but never four contiguous ones — no journey of length 4 exists.
    windows = journey_windows(frame, "prediction", lengths=(2, 3, 4)).set_index(
        "segments"
    )
    assert windows.loc[2, "journeys"] == 2  # one per block, not three
    assert 3 not in windows.index or windows.loc[3, "journeys"] == 0
    assert 4 not in windows.index


def test_bias_sign_means_the_model_ran_short():
    """residual = actual - predicted, matching features.baselines._score."""
    frame = pd.DataFrame(_segments([120, 120]))
    frame["prediction"] = 100.0

    windows = journey_windows(frame, "prediction", lengths=(1,)).set_index("segments")
    assert windows.loc[1, "bias_sec"] == pytest.approx(20.0)


# -------------------------------------------------------------------------- encoder


def _categorical_frame():
    return pd.DataFrame(
        {
            "route_id": ["RED", "BLUE", "RED"],
            "direction_id": [0, 1, 0],
            "segment_id": ["A>B", "B>C", "A>B"],
            "from_station": ["A01", "B02", "A01"],
            "to_station": ["A02", "B03", "A02"],
        }
    )


def test_encoder_round_trips_known_categories():
    frame = _categorical_frame()
    encoder = CategoricalEncoder.fit(frame)
    encoded = encoder.transform(frame)

    assert (encoded["route_id"] >= 0).all()
    # Same input value must always get the same code.
    assert encoded["route_id"].iloc[0] == encoded["route_id"].iloc[2]
    assert encoded["route_id"].iloc[0] != encoded["route_id"].iloc[1]


def test_unseen_category_becomes_missing_not_a_crash():
    """89 segments appear in validation but never training on the real 12 days."""
    encoder = CategoricalEncoder.fit(_categorical_frame())
    unseen = _categorical_frame()
    unseen.loc[0, "segment_id"] = "NEVER>SEEN"

    encoded = encoder.transform(unseen)
    assert encoded.loc[0, "segment_id"] == UNSEEN_CODE
    assert UNSEEN_CODE < 0, "LightGBM only reads negatives as missing"


def test_unseen_rate_is_reported():
    encoder = CategoricalEncoder.fit(_categorical_frame())
    frame = _categorical_frame()
    frame.loc[0, "segment_id"] = "NEVER>SEEN"
    assert encoder.unseen_rate(frame)["segment_id"] == pytest.approx(100 / 3)


def test_codes_do_not_depend_on_which_rows_are_present():
    """The trap pd.Categorical.codes falls into: a missing category renumbers others."""
    full = _categorical_frame()
    encoder = CategoricalEncoder.fit(full)

    subset = full[full["route_id"] == "BLUE"]
    assert (
        encoder.transform(subset)["route_id"].iloc[0]
        == encoder.transform(full)["route_id"].iloc[1]
    )


def test_encoder_survives_save_and_load(tmp_path):
    encoder = CategoricalEncoder.fit(_categorical_frame())
    path = tmp_path / "encoder.json"
    encoder.save(path)

    reloaded = CategoricalEncoder.load(path)
    assert reloaded.mapping == encoder.mapping
    pd.testing.assert_frame_equal(
        reloaded.transform(_categorical_frame()),
        encoder.transform(_categorical_frame()),
    )


# ------------------------------------------------------------------- feature matrix


def _feature_frame():
    frame = pd.DataFrame(_segments([120, 180]))
    frame["arrival_bracket_sec"] = 60
    frame["arrival_source"] = "vehicle_position"
    frame["delay_sec"] = 0
    frame["actual_arrival_ts"] = frame["actual_departure_ts"]
    frame["observed_at_utc"] = frame["actual_departure_ts"]
    frame["from_stop_id"] = "PF_A01_C"
    frame["to_stop_id"] = "PF_A02_C"
    frame["local_hour"] = 8
    frame["scheduled_duration_sec"] = 120.0
    frame["route_id"] = "RED"
    frame["segment_id"] = "A>B"
    return frame


def test_measurement_columns_are_not_features():
    """They describe how the LABEL was measured — only knowable after the fact."""
    columns = feature_columns(_feature_frame())
    for column in MEASUREMENT_COLUMNS:
        assert column not in columns


def test_keys_and_label_side_columns_are_not_features():
    columns = feature_columns(_feature_frame())
    for column in (
        *KEY_COLUMNS,
        "actual_duration_sec",
        "delay_sec",
        "actual_arrival_ts",
    ):
        assert column not in columns


def test_real_features_survive_selection():
    columns = feature_columns(_feature_frame())
    for column in ("local_hour", "scheduled_duration_sec", "route_id", "segment_id"):
        assert column in columns


def test_matrix_preserves_the_persisted_column_order():
    frame = _feature_frame()
    encoder = CategoricalEncoder.fit(frame, FeatureConfig())
    columns = feature_columns(frame)

    matrix = build_matrix(frame, encoder, columns)
    assert list(matrix.columns) == columns


def test_matrix_refuses_a_frame_missing_a_feature():
    """Serving with a column silently absent would score against a shifted matrix."""
    frame = _feature_frame()
    encoder = CategoricalEncoder.fit(frame, FeatureConfig())
    columns = feature_columns(frame)

    with pytest.raises(ValueError, match="absent from the frame"):
        build_matrix(frame.drop(columns=["local_hour"]), encoder, columns)


def test_undeclared_string_feature_is_still_encoded():
    """`fare_period` is a string and is NOT in FeatureConfig.categorical_columns.

    Unencoded it reaches LightGBM as a raw string, which raises. Caught by the smoke
    check on real data rather than by any fixture, so it is pinned here.
    """
    from src.models.train import categorical_columns

    frame = _feature_frame()
    frame["fare_period"] = "weekday_standard"
    columns = feature_columns(frame)

    categoricals = categorical_columns(frame, columns)
    assert "fare_period" in categoricals

    encoder = CategoricalEncoder.fit(frame, extra_columns=categoricals)
    matrix = build_matrix(frame, encoder, columns)
    assert matrix["fare_period"].dtype.kind in "i"


def test_no_feature_reaches_the_model_as_a_string():
    """The general form of the bug above — dtype-driven, so new columns are covered."""
    from src.models.train import categorical_columns

    frame = _feature_frame()
    frame["fare_period"] = "weekday_standard"
    frame["some_future_string_feature"] = "x"
    columns = feature_columns(frame)

    encoder = CategoricalEncoder.fit(
        frame, extra_columns=categorical_columns(frame, columns)
    )
    matrix = build_matrix(frame, encoder, columns)
    assert all(matrix[c].dtype.kind in "ifb" for c in matrix.columns)


# ---------------------------------------------------------------- calibration


def test_calibration_recovers_a_known_injected_bias():
    """The whole point: find the constant the model is off by."""
    import numpy as np

    from src.models.calibrate import BiasCalibration

    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(120, 40, 5000))
    predicted = actual - 6.04  # model runs short, as the real one did

    calibration = BiasCalibration.fit(actual, predicted)
    assert calibration.offset_sec == pytest.approx(6.04, abs=0.01)
    assert np.mean(actual - calibration.apply(predicted)) == pytest.approx(0, abs=0.01)


def test_calibration_of_unbiased_predictions_is_a_no_op():

    from src.models.calibrate import BiasCalibration

    actual = pd.Series([100.0, 120.0, 140.0])
    predicted = pd.Series([100.0, 120.0, 140.0])
    assert BiasCalibration.fit(actual, predicted).offset_sec == pytest.approx(0.0)


def test_calibration_is_additive_and_order_preserving():
    """A constant shift must not reorder predictions — only translate them."""
    import numpy as np

    from src.models.calibrate import BiasCalibration

    predicted = pd.Series([60.0, 300.0, 120.0])
    shifted = BiasCalibration(offset_sec=7.5).apply(predicted)
    assert np.allclose(shifted - predicted, 7.5)
    assert list(np.argsort(shifted)) == list(np.argsort(predicted))


def test_calibration_survives_save_and_load(tmp_path):
    from src.models.calibrate import BiasCalibration

    path = tmp_path / "calibration.json"
    BiasCalibration(offset_sec=6.04).save(path)
    assert BiasCalibration.load(path).offset_sec == pytest.approx(6.04)


def test_bias_accumulates_linearly_over_a_journey():
    """Why a per-segment offset is the right level: it fixes every length at once."""
    from src.models.calibrate import BiasCalibration
    from src.models.journey import journey_windows

    frame = pd.DataFrame(_segments([120, 120, 120, 120]))
    frame["prediction"] = frame["actual_duration_sec"] - 6.0  # 6s short per segment

    raw = journey_windows(frame, "prediction", lengths=(1, 4)).set_index("segments")
    assert raw.loc[1, "bias_sec"] == pytest.approx(6.0)
    assert raw.loc[4, "bias_sec"] == pytest.approx(24.0)  # 4x, linear — the problem

    frame["prediction_calibrated"] = BiasCalibration(offset_sec=6.0).apply(
        frame["prediction"]
    )
    fixed = journey_windows(frame, "prediction_calibrated", lengths=(1, 4)).set_index(
        "segments"
    )
    assert fixed.loc[1, "bias_sec"] == pytest.approx(0.0)
    assert fixed.loc[4, "bias_sec"] == pytest.approx(0.0)  # fixed at every length


# ----------------------------------------------------------------- artifacts


def test_runs_are_immutable_across_retrains(tmp_path):
    """Retraining must never overwrite a previous run — that is the whole point."""
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir

    when = datetime(2026, 8, 20, 19, 0, 0, tzinfo=UTC)
    first = new_run_dir(tmp_path, when)
    (first / "model.txt").write_text("first")

    # Same clock second: must still land somewhere new, not merge into the first.
    second = new_run_dir(tmp_path, when)
    assert second != first
    assert (first / "model.txt").read_text() == "first"


def test_latest_points_at_the_newest_run(tmp_path):
    from datetime import UTC, datetime

    from src.models.artifacts import mark_latest, new_run_dir, resolve_run

    old = new_run_dir(tmp_path, datetime(2026, 8, 20, 10, 0, tzinfo=UTC))
    new = new_run_dir(tmp_path, datetime(2026, 8, 20, 11, 0, tzinfo=UTC))
    mark_latest(tmp_path, new)

    assert resolve_run(tmp_path) == new.resolve()
    assert resolve_run(tmp_path, old.name) == old
    # A plain-JSON pointer too, for anything that will not follow a symlink.
    assert json.loads((tmp_path / "latest.json").read_text())["run"] == new.name


def test_resolve_run_fails_loudly_when_there_is_nothing(tmp_path):
    from src.models.artifacts import resolve_run

    with pytest.raises(FileNotFoundError, match="train a model first"):
        resolve_run(tmp_path)


def test_manifest_records_provenance_and_checksums(tmp_path):
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir, write_manifest

    run = new_run_dir(tmp_path, datetime(2026, 8, 20, 12, 0, tzinfo=UTC))
    (run / "model.txt").write_text("tree ensemble")

    write_manifest(
        run,
        model_name="segment_duration",
        target="actual_duration_sec",
        trustworthy=False,
        training_data={"service_dates": ["2026-08-07"]},
        params={"objective": "regression_l1"},
        best_iteration=709,
        feature_columns=["local_hour", "route_id"],
        categorical_columns=["route_id"],
        headline_metrics={"mae": 22.6},
    )
    manifest = json.loads((run / "manifest.json").read_text())

    # A registry must never present a provisional score as validated.
    assert manifest["trustworthy"] is False
    assert manifest["feature_schema"]["n_features"] == 2
    assert "model.txt" in manifest["artifacts"]
    assert len(manifest["artifacts"]["model.txt"]) == 64  # sha256 hex
    assert "lightgbm" in manifest["environment"]


def test_package_flattens_for_sagemaker(tmp_path):
    """SageMaker extracts into /opt/ml/model, so members sit at the archive root."""
    import tarfile
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir, package

    run = new_run_dir(tmp_path, datetime(2026, 8, 20, 13, 0, tzinfo=UTC))
    for name in ("model.txt", "encoder.json", "feature_columns.json", "manifest.json"):
        (run / name).write_text("{}")
    (run / "plots" / "learning_curve.png").write_bytes(b"not really a png")
    (run / "validation_predictions.parquet").write_bytes(b"parquet")

    archive = package(run)
    with tarfile.open(archive) as tar:
        names = tar.getnames()

    assert "model.txt" in names
    assert not any("/" in n for n in names), "nested paths break the serving handler"
    # Evaluation evidence must not bloat every container pull.
    assert not any(n.endswith(".png") or n.endswith(".parquet") for n in names)


def test_package_puts_inference_code_under_code_dir(tmp_path):
    """Framework containers look for the handler in `code/`, not at the root.

    The flat layout is right for model files and wrong for inference code — SageMaker
    runs `pip install -r code/requirements.txt` at cold start, which is what puts
    LightGBM in the container.
    """
    import tarfile
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir, package

    run = new_run_dir(tmp_path / "m", datetime(2026, 8, 20, 14, 0, tzinfo=UTC))
    for name in ("model.txt", "encoder.json", "feature_columns.json", "manifest.json"):
        (run / name).write_text("{}")

    code = tmp_path / "code"
    code.mkdir()
    (code / "inference.py").write_text("def model_fn(d): ...")
    (code / "requirements.txt").write_text("lightgbm==4.7.0")

    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "station_index.json").write_text("{}")

    archive = package(
        run,
        serving_dir=serving,
        code_files={p.name: p for p in code.iterdir()},
    )
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())

    assert "model.txt" in names
    assert "code/inference.py" in names
    assert "code/requirements.txt" in names
    # Serving inputs travel WITH the model — the handler loads them from the model dir,
    # so an endpoint without them starts and then fails on its first request.
    assert "station_index.json" in names


def test_package_default_stays_flat(tmp_path):
    """Without code_files the archive is still flat — the old contract is unchanged."""
    import tarfile
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir, package

    run = new_run_dir(tmp_path / "m", datetime(2026, 8, 20, 15, 0, tzinfo=UTC))
    (run / "model.txt").write_text("x")
    (run / "plots" / "learning_curve.png").write_bytes(b"png")

    with tarfile.open(package(run)) as tar:
        names = tar.getnames()
    assert names == ["model.txt"]


# ------------------------------------------------------------------ station resolver


def _index():
    from src.serving.stations import StationIndex

    return StationIndex(
        names_to_codes={"vienna": ["K08"], "metro center": ["A01", "C01"]},
        code_to_name={"K08": "Vienna", "A01": "Metro Center", "C01": "Metro Center"},
        code_to_platforms={
            "K08": ["PF_K08_C"],
            "A01": ["PF_A01_1"],
            "C01": ["PF_C01_1"],
        },
    )


def _schedule(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "origin",
            "destination",
            "hour",
            "sched_sec",
            "n_segments",
            "stop_span",
            "trips",
            "route_id",
            "direction_id",
            "first_leg_to",
        ],
    )


def test_station_names_normalise():
    """'metro center', 'Metro Center' and 'METRO  CENTER' are the same station."""
    from src.serving.stations import normalise

    assert normalise("Metro Center") == normalise("metro  center") == "metro center"
    # Punctuation must not defeat a lookup — L'Enfant is the real case.
    assert normalise("L'Enfant Plaza") == "l enfant plaza"


def test_unknown_station_suggests_rather_than_crashes():
    from src.serving.stations import StationError

    with pytest.raises(StationError, match="did you mean Vienna"):
        _index().candidates("Viena")


def test_transfer_station_resolves_via_the_connected_platform():
    """Metro Center has two codes; only one is connected to the destination."""
    from src.serving.stations import ANY_HOUR, resolve_journey

    schedule = _schedule(
        [("PF_A01_1", "PF_K08_C", ANY_HOUR, 900, 6, 6, 20, "RED", 0, "PF_A02_1")]
    )
    out = resolve_journey("Metro Center", "Vienna", _index(), schedule)
    assert out["origin_stop_id"] == "PF_A01_1"
    assert out["scheduled_total_sec"] == 900
    # The first leg is what the recent-conditions lookup is keyed on.
    assert out["first_leg_to"] == "PF_A02_1"


def test_genuinely_ambiguous_journey_is_refused_not_guessed():
    """Both platforms connected means we cannot know which the rider meant."""
    from src.serving.stations import ANY_HOUR, StationError, resolve_journey

    schedule = _schedule(
        [
            ("PF_A01_1", "PF_K08_C", ANY_HOUR, 900, 6, 6, 20, "RED", 0, "PF_A02_1"),
            ("PF_C01_1", "PF_K08_C", ANY_HOUR, 700, 5, 5, 20, "BLUE", 0, "PF_C02_1"),
        ]
    )
    with pytest.raises(StationError, match="ambiguous"):
        resolve_journey("Metro Center", "Vienna", _index(), schedule)


def test_unconnected_stations_are_refused():
    """No single train runs between them — the model cannot answer transfers."""
    from src.serving.stations import StationError, resolve_journey

    with pytest.raises(StationError, match="no scheduled trip"):
        resolve_journey("Vienna", "Metro Center", _index(), _schedule([]))


# ------------------------------------------------ split gate: blocking vs advisory


def _split_frame(days: int, unseen_segment: bool = False):
    """A frame spanning `days` service dates, optionally with a cold-start segment."""
    # Realistically dense: a real service day is ~33,000 rows, so one cold-start segment
    # is a fraction of a percent. A sparse fixture makes a single unseen segment look
    # like 3.45% and blocks for the right reason on the wrong data.
    rows = []
    for day in range(days):
        date = f"2026-08-{7 + day:02d}"
        for hour in range(6, 22):
            for segment in range(20):
                rows.append(
                    {
                        "service_date": date,
                        "actual_departure_ts": datetime(
                            2026, 8, 7 + day, hour, segment % 60, tzinfo=UTC
                        ),
                        "from_stop_id": f"PF_A{segment:02d}_C",
                        "to_stop_id": f"PF_A{segment + 1:02d}_C",
                        "local_hour": hour,
                        "actual_duration_sec": 120,
                    }
                )
    if unseen_segment:
        # One rare segment appearing only on the final day, so it lands in validation.
        rows.append(
            {
                "service_date": f"2026-08-{6 + days:02d}",
                "actual_departure_ts": datetime(2026, 8, 6 + days, 23, tzinfo=UTC),
                "from_stop_id": "PF_Z99_C",
                "to_stop_id": "PF_Z98_C",
                "local_hour": 23,
                "actual_duration_sec": 120,
            }
        )
    return pd.DataFrame(rows)


def test_too_few_days_blocks():
    from src.features.split import temporal_split

    _, _, report = temporal_split(_split_frame(days=5))
    assert not report.is_trustworthy
    assert any("service day(s) available" in w for w in report.blocking)


def test_small_cold_start_tail_is_advisory_not_blocking():
    """0.14% unseen segments held trustworthy at False forever. It must not block."""
    from src.features.split import temporal_split

    _, _, report = temporal_split(_split_frame(days=20, unseen_segment=True))

    unseen = [w for w in report.warnings if "never in training" in w]
    if unseen:  # only meaningful if the fixture actually produced a cold-start segment
        assert unseen[0] not in report.blocking, "a tiny cold-start tail must not block"
    assert report.is_trustworthy, report.warnings


def test_advisory_warnings_are_still_reported():
    """The change is what disqualifies, NOT what gets said. Nothing may be hidden."""
    from src.features.split import temporal_split

    _, _, report = temporal_split(_split_frame(days=20, unseen_segment=True))
    metadata = report.as_metadata()

    # Present in the metadata that travels into metrics.json and the registry manifest.
    assert "warnings" in metadata and "blocking_warnings" in metadata
    assert len(metadata["warnings"]) >= len(metadata["blocking_warnings"])
    if report.warnings:
        assert any(w in report.format() for w in report.warnings)


def test_report_header_keys_off_blocking_only():
    """A 0.14% note must not print 'NOT YET MEANINGFUL' — readers learn to skim it."""
    from src.features.split import temporal_split

    _, _, blocked = temporal_split(_split_frame(days=5))
    assert "NOT YET MEANINGFUL" in blocked.format()

    _, _, ok = temporal_split(_split_frame(days=20, unseen_segment=True))
    assert "NOT YET MEANINGFUL" not in ok.format()


def test_repackaging_different_contents_yields_a_different_key(tmp_path):
    """v1-v4 all resolved to one S3 file because the key ignored contents.

    The key must change when the bytes change, or a registry version can be silently
    replaced by a later packaging of the same training run.
    """
    import hashlib
    from datetime import UTC, datetime

    from src.models.artifacts import new_run_dir, package

    def key_for(run, contents: str) -> str:
        (run / "model.txt").write_text(contents)
        archive = package(run, output=run / f"{contents}.tar.gz")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()[:8]
        return f"models/journey_duration/{run.name}/{digest}/model.tar.gz"

    run = new_run_dir(tmp_path, datetime(2026, 8, 23, 9, 0, tzinfo=UTC))
    first = key_for(run, "packaging-one")
    second = key_for(run, "packaging-two")

    assert first != second, "same run, different bytes, must not collide"
    assert run.name in first and run.name in second, "run id stays traceable in the key"


# ------------------------------------------------- serving: thin-support warnings

from src.serving.inference import (  # noqa: E402 - grouped with its own tests
    MIN_TRAINING_SUPPORT,
    _training_support,
)

# Real counts from the 17-date run, so the numbers in these assertions mean something.
SUPPORT = {"1": 430821, "4": 342507, "17": 78664, "24": 18182, "28": 6359, "32": 98}


def test_support_is_exact_when_the_length_was_trained():
    assert _training_support(4, SUPPORT) == 342507


def test_support_interpolates_between_trained_lengths():
    """26 segments is not a trained length but sits between 24 and 28.

    REGRESSION: an exact-key lookup reported "no training journeys of 26 segments" for a
    journey the model predicts well — Shady Grove to Glenmont, 61.35 min against a
    62-minute schedule. Training covers a discrete set of lengths; requests are
    continuous, so most real journeys fell through the exact lookup and warned wrongly.
    """
    # The weaker of the two neighbours, not the nearer one: 28 has less evidence.
    assert _training_support(26, SUPPORT) == 6359


def test_beyond_the_largest_trained_length_is_genuine_extrapolation():
    """Nothing to interpolate between past the end — that is a different claim."""
    assert _training_support(40, SUPPORT) is None


def test_thinly_supported_length_is_still_flagged():
    """n=32 has 98 training journeys. Trained on is not the same as supported."""
    assert _training_support(32, SUPPORT) == 98
    assert _training_support(32, SUPPORT) < MIN_TRAINING_SUPPORT


def test_missing_support_map_reports_unknown():
    """Manifests predating per-length counts have nothing to reason from."""
    assert _training_support(8, {}) is None


def test_coverage_interpolates_like_support():
    """The quantile model's achieved coverage is per-length and also discrete.

    Reported coverage must be the WEAKER neighbour: claiming 78.7% for a 26-segment
    journey when the neighbouring length achieves 62.3% would overstate it
    exactly where it is weakest.
    """
    from src.serving.inference import _bracketed

    coverage = {"1": 80.2, "24": 64.2, "28": 62.3}
    assert _bracketed(26, coverage) == 62.3
    assert _bracketed(24, coverage) == 64.2
    assert _bracketed(40, coverage) is None


# ----------------------------------------------------------------- run comparison


def _manifest(run_id, boundary, *, by_length=None, trustworthy=True, target="j_sec"):
    """The slice of a manifest `compare_runs` reads. Not the whole thing."""
    return {
        "run_id": run_id,
        "model_name": "journey_duration",
        "target": target,
        "trustworthy": trustworthy,
        "training_data": {
            "service_dates": ["2026-08-07", "2026-08-19"],
            "split": {
                "boundary_utc": boundary,
                "train_rows": 10,
                "validation_rows": 5,
                "validation_days": ["2026-08-19"],
            },
        },
        "best_iteration": 100,
        "git": {"commit": "abcdef1234", "dirty": False},
        "headline_metrics": {"by_length": by_length or []},
    }


def test_rescoring_refuses_a_model_that_trained_on_the_holdout():
    """The whole point of the leakage guard: a later boundary means it may have seen
    these rows, and grading it on them is a memory test, not a comparison."""
    from src.models.compare_runs import _leakage_free

    older = _manifest("old", "2026-08-17T18:00:00+00:00")
    newer = _manifest("new", "2026-08-20T19:34:00+00:00")

    assert _leakage_free(older, newer)[0] is True
    allowed, reason = _leakage_free(newer, older)
    assert allowed is False
    assert "may have trained on these rows" in reason


def test_runs_without_a_recorded_boundary_are_never_rescored():
    """Absent provenance is not permission — it cannot be proven leakage-free."""
    from src.models.compare_runs import _leakage_free

    blank = {"training_data": {"split": {}}}
    allowed, reason = _leakage_free(blank, _manifest("h", "2026-08-20T19:34:00+00:00"))
    assert allowed is False
    assert "boundary not recorded" in reason


def test_differing_validation_windows_are_detected():
    """A moved split boundary is exactly what makes two runs' metrics incomparable."""
    from src.models.compare_runs import comparable

    a = _manifest("a", "2026-08-17T18:00:00+00:00")
    b = _manifest("b", "2026-08-20T19:34:00+00:00")
    assert comparable([a, a]) is True
    assert comparable([a, b]) is False


def test_mae_reads_both_manifest_shapes_onto_one_axis():
    """The two families record journey MAE under different keys; both must land on the
    same axis or the comparison silently drops one of them."""
    from src.models.compare_runs import mae_by_length

    journey = _manifest(
        "j", "x", by_length=[{"segments": 4, "mae_journey_model": 54.1}]
    )
    segment = {
        "headline_metrics": {"journey_level": [{"segments": 4, "mae_sec": 61.9}]}
    }
    assert mae_by_length(journey) == {4: 54.1}
    assert mae_by_length(segment) == {4: 61.9}
    assert mae_by_length({"headline_metrics": {}}) == {}


def test_interrupted_runs_are_skipped_not_compared(tmp_path):
    """Training writes the manifest last, so a manifest-less directory is a run that
    died partway — comparing it would invent a result it never produced."""
    from src.models.compare_runs import load_runs

    runs = tmp_path / "runs"
    (runs / "2026-08-20T10-00-00Z").mkdir(parents=True)
    (runs / "2026-08-20T10-00-00Z" / "manifest.json").write_text(
        json.dumps(_manifest("2026-08-20T10-00-00Z", "2026-08-17T18:00:00+00:00"))
    )
    (runs / "2026-08-21T10-00-00Z").mkdir(parents=True)  # no manifest

    loaded = load_runs(tmp_path)
    assert [m["run_id"] for _, m in loaded] == ["2026-08-20T10-00-00Z"]


# ----------------------------------------------------------------- mlflow projection


def test_metrics_flatten_from_both_manifest_shapes():
    """Both model families must land on the same metric names, or the UI shows two
    disjoint sets of runs that cannot be charted against each other."""
    from src.models.mlflow_sync import _flatten_metrics

    journey = _flatten_metrics(
        {
            "headline_metrics": {
                "by_length": [
                    {"segments": 4, "journeys": 10, "mae_journey_model": 54.1},
                    {"segments": 17, "journeys": 30, "mae_journey_model": 143.7},
                ]
            }
        }
    )
    segment = _flatten_metrics(
        {"headline_metrics": {"journey_level": [{"segments": 4, "mae_sec": 61.9}]}}
    )
    assert journey["mae_segments_04"] == 54.1
    assert segment["mae_segments_04"] == 61.9

    # Journey-count weighted, not a flat mean: 10 journeys at 54.1 and 30 at 143.7.
    assert journey["mae_weighted"] == pytest.approx((54.1 * 10 + 143.7 * 30) / 40)


def test_metric_names_survive_prose_baseline_labels():
    """The segment model's baseline labels are prose — 'segment x hour median (fitted
    on train)'. MLflow rejects most of those characters."""
    from src.models.mlflow_sync import _flatten_metrics

    metrics = _flatten_metrics(
        {"headline_metrics": {"segment_mae": {"model (bias-calibrated)": 24.9}}}
    )
    name = next(k for k in metrics if k.startswith("segment_mae."))
    assert set(name) <= set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
    )
    assert metrics[name] == 24.9


def test_the_split_window_is_carried_as_a_tag():
    """Two runs' metrics are only comparable if graded on the same window, so the
    window has to be visible in the UI beside the numbers."""
    from src.models.mlflow_sync import _tags

    tags = _tags(
        {
            "run_id": "r",
            "model_name": "journey_duration",
            "target": "j",
            "trustworthy": False,
            "training_data": {
                "service_dates": ["2026-08-07", "2026-08-19"],
                "split": {
                    "boundary_utc": "2026-08-17T18:00:00+00:00",
                    "validation_days": ["2026-08-17", "2026-08-19"],
                },
            },
            "git": {"commit": "abc123", "dirty": True},
        }
    )
    assert tags["metro_pulse.split_boundary"] == "2026-08-17T18:00:00+00:00"
    assert tags["metro_pulse.validation_days"] == "2026-08-17..2026-08-19"
    # A provisional score must never read as a validated one, in any surface.
    assert tags["metro_pulse.trustworthy"] == "False"


# ----------------------------------------------------------------- registry metrics


def test_model_quality_weights_by_journey_count():
    """Journeys are wildly uneven across lengths, so an unweighted mean would let the
    32-segment bucket — 118 training examples — drag the headline around."""
    from src.serving.register import model_quality

    report = model_quality(
        {
            "run_id": "r",
            "trustworthy": True,
            "target": "journey_duration_sec",
            "best_iteration": 10,
            "git": {},
            "training_data": {
                "validation_rows": 40,
                "service_dates": [],
                "split": {},
            },
            "headline_metrics": {
                "by_length": [
                    {"segments": 1, "journeys": 30, "mae_journey_model": 20.0},
                    {"segments": 32, "journeys": 10, "mae_journey_model": 600.0},
                ]
            },
        }
    )
    metrics = report["regression_metrics"]
    assert metrics["mae"]["value"] == pytest.approx((20.0 * 30 + 600.0 * 10) / 40)
    # Per-length kept alongside: an aggregate hides a regression at one horizon.
    assert metrics["mae_segments_01"]["value"] == 20.0
    assert metrics["mae_segments_32"]["value"] == 600.0
    # The flag rides along, outside the AWS schema, so nothing reads a provisional
    # score as a validated one.
    assert report["metro_pulse"]["trustworthy"] is True


def test_model_quality_survives_a_manifest_with_no_lengths():
    """The segment model's manifest has no `by_length`; registering must not explode."""
    from src.serving.register import model_quality

    report = model_quality(
        {
            "run_id": "r",
            "trustworthy": False,
            "target": "actual_duration_sec",
            "best_iteration": 1,
            "git": {},
            "training_data": {"validation_rows": 0, "service_dates": [], "split": {}},
            "headline_metrics": {},
        }
    )
    assert report["regression_metrics"] == {}
    assert report["metro_pulse"]["trustworthy"] is False
