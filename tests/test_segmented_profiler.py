"""客群画像的固定规则、参考人口、dtype 和局部状态回归。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsDataProfiler, profile_stats
from mars.core.constants import DIVISION_EPSILON, METRIC_EPSILON
from mars.feature import MarsNativeBinner
from mars.feature._scopes import resolve_scopes


def _population(dtype: pl.DataType = pl.Float64) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(["seg1", "seg2", "seg3"]):
        for group in [1, 2, 10]:
            for index in range(8):
                rows.append({
                    "segment": segment, "group": group,
                    "x": float(index + segment_index * 20 + group),
                    "category": "a" if index < (6 if group == 1 else 2 + segment_index) else "b",
                })
    frame = pl.DataFrame(rows).with_columns(pl.col("x").cast(dtype))
    return pl.concat([frame.slice(0, 13), frame.slice(13, 22), frame.slice(35)], rechunk=False)


def _polars(table: Any) -> pl.DataFrame:
    return table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("scope", ["shared", "per_segment"])
def test_segmented_stats_keep_original_values_and_dtypes(
    backend: str, dtype: pl.DataType, scope: str,
) -> None:
    frame = _population(dtype)
    reference = frame.filter(pl.col("group") == 1)
    current = frame if backend == "polars" else frame.to_pandas()
    benchmark = reference if backend == "polars" else reference.to_pandas()
    report = profile_stats(
        current, features=["x", "category"], metrics=["missing", "mean", "psi", "schema", "unseen"],
        segment_col="segment", group_col="group", benchmark_df=benchmark, binning_scope=scope,
    )
    assert isinstance(report.overview_table, pl.DataFrame if backend == "polars" else pd.DataFrame)
    assert report._scopes.height == 4
    for value in [None, "seg1", "seg2", "seg3"]:
        scope_id = "all" if value is None else resolve_scopes(report._scopes, [value], False)[0]
        population = frame if value is None else frame.filter(pl.col("segment") == value)
        local_benchmark = reference if value is None else reference.filter(pl.col("segment") == value)
        expected = profile_stats(
            population if backend == "polars" else population.to_pandas(),
            features=["x", "category"], metrics=["missing", "mean", "psi", "schema", "unseen"],
            group_col="group", benchmark_df=local_benchmark if backend == "polars" else local_benchmark.to_pandas(),
        )
        for metric in ["missing", "mean", "schema", "unseen"]:
            actual_tables = {**report.dq_tables, **report.stats_tables, **report.comparison_tables}
            expected_tables = {**expected.dq_tables, **expected.stats_tables, **expected.comparison_tables}
            actual = _polars(actual_tables[metric]).filter(pl.col("scope_id") == scope_id).drop("scope_id")
            assert_frame_equal(actual, _polars(expected_tables[metric]))
        if scope == "per_segment" or value is None:
            actual = _polars(report.stats_tables["psi"]).filter(pl.col("scope_id") == scope_id).drop("scope_id")
            assert_frame_equal(actual, _polars(expected.stats_tables["psi"]), rtol=1e-10, atol=1e-10)
    assert _polars(report.overview_table).filter(pl.col("feature") == "x")["dtype"].unique().to_list() == [str(dtype)]


def test_shared_category_psi_uses_each_segments_expected_counts() -> None:
    frame = _population()
    benchmark = frame.filter(pl.col("group") == 1)
    report = profile_stats(
        frame, features=["category"], metrics=["psi"], segment_col="segment",
        group_col="group", benchmark_df=benchmark,
    )
    for segment_index, segment in enumerate(["seg1", "seg2", "seg3"]):
        scope_id = resolve_scopes(report._scopes, [segment], False)[0]
        table = report.stats_tables["psi"].filter(pl.col("scope_id") == scope_id)
        expected = [6 / (8 + DIVISION_EPSILON), 2 / (8 + DIVISION_EPSILON)]
        actual = [(2 + segment_index) / (8 + DIVISION_EPSILON), (6 - segment_index) / (8 + DIVISION_EPSILON)]
        manual = sum((a - e) * math.log(a / (e + METRIC_EPSILON)) for a, e in zip(actual, expected))
        assert table["2"].item() == pytest.approx(manual, rel=1e-9, abs=1e-9)
        assert table["1"].item() == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("scope", ["shared", "per_segment"])
def test_missing_segment_benchmark_keeps_stats_and_null_psi(scope: str) -> None:
    frame = _population()
    benchmark = frame.filter((pl.col("group") == 1) & (pl.col("segment") != "seg2"))
    report = profile_stats(
        frame, features=["x", "category"], metrics=["mean", "psi", "schema", "unseen"],
        segment_col="segment", group_col="group", benchmark_df=benchmark, binning_scope=scope,
    )
    scope_id = resolve_scopes(report._scopes, ["seg2"], False)[0]
    psi = report.stats_tables["psi"].filter(pl.col("scope_id") == scope_id)
    assert psi.schema["total"] == pl.Float64
    assert psi["total"].null_count() == 2
    states = _polars(report._calculation_status).filter((pl.col("scope_id") == scope_id) & (pl.col("metric") == "psi"))
    assert states["status"].to_list() == ["not_computed", "not_computed"]
    assert states["reason"].to_list() == ["missing_segment_benchmark"] * 2
    mean = report.stats_tables["mean"].filter((pl.col("scope_id") == scope_id) & (pl.col("feature") == "x"))
    assert mean["total"].item() == pytest.approx(frame.filter(pl.col("segment") == "seg2")["x"].mean())
    schema = report.comparison_tables["schema"].filter(pl.col("scope_id") == scope_id)
    assert schema["status"].to_list() == ["matched", "matched"]
    unseen = report.comparison_tables["unseen"].filter((pl.col("scope_id") == scope_id) & (pl.col("feature") == "category"))
    assert unseen["status"].item() == "no_reference_values"


def test_no_benchmark_fixes_the_overall_first_group_for_every_segment() -> None:
    frame = _population().filter(~((pl.col("segment") == "seg2") & (pl.col("group") == 1)))
    report = profile_stats(frame, features=["x"], metrics=["psi"], segment_col="segment", group_col="group")
    scope_id = resolve_scopes(report._scopes, ["seg2"], False)[0]
    assert report.stats_tables["psi"].filter(pl.col("scope_id") == scope_id)["total"].item() is None
    status = _polars(report._calculation_status).filter(pl.col("scope_id") == scope_id)
    assert status["reason"].to_list() == ["missing_first_group"]
    references = report.report_meta["scope_references"]
    assert [reference["reference_group"] for reference in references] == [1, 1, 1, 1]
    expected = profile_stats(frame, features=["x"], metrics=["psi"], group_col="group")
    actual = report.stats_tables["psi"].filter(pl.col("scope_id") == "all").drop("scope_id")
    assert_frame_equal(actual, expected.stats_tables["psi"])


def test_explicit_group_named_month_keeps_priority_over_dates() -> None:
    frame = _population().rename({"group": "month"}).with_columns(
        pl.lit(datetime(2026, 10, 1)).alias("date"),
    )
    report = profile_stats(
        frame, features=["x"], metrics=["mean", "psi"], segment_col="segment",
        group_col="month", time_col="date", time_grain="day",
    )
    mean = report.stats_tables["mean"]
    assert all(group in mean.columns for group in ["1", "2", "10"])
    assert all(not column.startswith("2026") for column in mean.columns)


def test_shared_psi_fits_once_and_statistics_do_not_depend_on_binning_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _population()
    calls: list[int] = []
    original = MarsNativeBinner.fit

    def counted(self: MarsNativeBinner, X: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(X.height)
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(MarsNativeBinner, "fit", counted)
    profiler = MarsDataProfiler(psi_batch_size=1)
    shared = profiler.generate_profile(
        frame, metrics=["missing", "mean", "psi"], features=["x", "category"],
        segment_col="segment", group_col="group", benchmark_df=frame,
    )
    assert calls == [frame.height]
    independent = MarsDataProfiler(binning_scope="per_segment").generate_profile(
        frame, metrics=["missing", "mean", "psi"], features=["x", "category"],
        segment_col="segment", group_col="group", benchmark_df=frame,
    )
    assert_frame_equal(shared.overview_table, independent.overview_table)
    assert_frame_equal(shared.stats_tables["mean"], independent.stats_tables["mean"])
    second = profiler.generate_profile(frame.filter(pl.col("segment") != "seg3"), metrics=["missing"], segment_col="segment")
    assert "seg3" not in second._scopes["display_name"].to_list()
    assert "segment" not in second.overview_table["feature"].to_list()


def test_segment_roles_and_nonapplicable_modes_are_rejected() -> None:
    frame = _population()
    with pytest.raises(ValueError, match="requires.*segment_col"):
        profile_stats(frame, metrics=["psi"], binning_scope="per_segment")
    with pytest.raises(ValueError, match="requires.*metric 'psi'"):
        profile_stats(frame, metrics=["mean"], segment_col="segment", binning_scope="per_segment")
    with pytest.raises(ValueError, match="explicit feature"):
        profile_stats(frame, features=["segment"], metrics=["missing"], segment_col="segment")
    with pytest.raises(ValueError, match="group_col"):
        profile_stats(frame, metrics=["missing"], segment_col="segment", group_col="segment")
    with pytest.raises(ValueError, match="missing segment_col"):
        profile_stats(frame, metrics=["psi"], segment_col="segment", benchmark_df=frame.drop("segment"))


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
def test_empty_current_retains_dtype_and_does_not_fit(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _population(pl.Float32)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Empty current population must not fit")

    monkeypatch.setattr(MarsNativeBinner, "fit", forbidden)
    report = profile_stats(frame.head(0), metrics=["psi", "mean"], features=["x"], segment_col="segment", benchmark_df=frame, binning_scope=mode)
    assert report._scopes["scope_id"].to_list() == ["all"]
    assert report.stats_tables["psi"]["dtype"].item() == "Float32"
    assert report.stats_tables["psi"]["total"].item() is None
