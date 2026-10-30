"""画像 PSI 按特征、期间和固定参考人口判断适用性的回归。"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsDataProfiler, profile_stats
from mars.compute import to_polars_frame
from mars.feature import MarsNativeBinner


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("batch", [1, 2, 16])
def test_local_empty_psi_population_keeps_null_and_local_status(mode: str, batch: int) -> None:
    frame = pl.DataFrame([
        {"s": segment, "g": group, "x": float(i), "z": float(i)}
        for segment in ("seg1", "seg2") for group in ("01", "02") for i in range(20)
    ])
    current = frame.with_columns(
        pl.when((pl.col("s") == "seg2") & (pl.col("g") == "02"))
        .then(None).otherwise(pl.col("x")).alias("x"),
    )
    report = MarsDataProfiler(psi_batch_size=batch, binning_scope=mode).generate_profile(
        current, features=["x", "z"], metrics=["psi"], segment_col="s", group_col="g",
        benchmark_df=frame.filter(pl.col("g") == "01"), psi_include_missing=False,
    )
    result = report.get_table("stats.psi", features="x", segments=["seg2"], include_all=False)
    assert result.schema["02"] == pl.Float64
    assert result["02"].item() is None
    assert result["01"].item() == pytest.approx(0, abs=1e-10)
    assert result["total"].item() == pytest.approx(0, abs=1e-10)
    status = report.get_table("calculation_status", features="x", segments=["seg2"], include_all=False)
    assert status.select("group", "metric", "status", "reason").to_dicts() == [{
        "group": "02", "metric": "psi", "status": "not_computed",
        "reason": "no_effective_current_population",
    }]


def _frame(value: Any) -> pl.DataFrame:
    """统一已聚合输出的后端，不转换原始计算人口。"""
    return to_polars_frame(value)


def _status(report: Any, **kwargs: Any) -> pl.DataFrame:
    """按稳定业务身份比较异常状态，正常报告不要求虚构状态表。"""
    if "calculation_status" not in report.describe()["tables"]:
        return pl.DataFrame(schema={
            "feature": pl.String, "group": pl.String, "metric": pl.String,
            "status": pl.String, "reason": pl.String,
        })
    result = _frame(report.get_table("calculation_status", **kwargs))
    keys = [column for column in ["scope_id", "feature", "group", "metric"] if column in result.columns]
    assert result.select(keys).unique().height == result.height
    return result.sort(keys)


def _oracle(actual: list[int], expected: list[int]) -> float | None:
    """固定箱的小频数 oracle；先判人口，再沿用正式分母与平滑常量。"""
    if sum(actual) == 0 or sum(expected) == 0:
        return None
    actual_probs = [count / (sum(actual) + 1e-9) if count else 1e-6 for count in actual]
    expected_probs = [count / (sum(expected) + 1e-9) if count else 1e-6 for count in expected]
    return sum(
        (a - e) * math.log(a / (e + 1e-6))
        for a, e in zip(actual_probs, expected_probs)
    )


@pytest.mark.parametrize("mode", ["plain", "shared", "per_segment"])
@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_psi_business_keys_do_not_depend_on_batch_neighbours_or_feature_order(
    mode: str, backend: str,
) -> None:
    frame = pl.DataFrame([
        {"s": segment, "g": group, "x": float(i), "z": float(i),
         "w": None if group == "01" else float(i)}
        for segment in ("seg1", "seg2") for group in ("01", "02") for i in range(20)
    ])
    benchmark = frame.filter(pl.col("g") == "01")
    current = frame.with_columns(
        pl.when((pl.col("s") == "seg2") & (pl.col("g") == "02"))
        .then(None).otherwise(pl.col("x")).alias("x"),
    )
    selection: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if mode == "plain":
        current = current.filter(pl.col("s") == "seg2")
        benchmark = benchmark.filter(pl.col("s") == "seg2")
    else:
        options["segment_col"] = "s"
        selection = {"segments": ["seg2"], "include_all": False}
    if backend == "pandas":
        current_input, benchmark_input = current.to_pandas(), benchmark.to_pandas()
    else:
        current_input, benchmark_input = current, benchmark
    expected_values: pl.DataFrame | None = None
    expected_status: pl.DataFrame | None = None
    for batch in (1, 2, 16):
        for features in (["x"], ["x", "z"], ["z", "x"], ["w", "z", "x"]):
            report = MarsDataProfiler(psi_batch_size=batch, binning_scope="shared" if mode == "plain" else mode).generate_profile(
                current_input, features=list(features), metrics=["psi"], group_col="g",
                benchmark_df=benchmark_input, **options,
            )
            values = _frame(report.get_table("stats.psi", features="x", **selection))
            state = _status(report, features="x", **selection)
            assert values["02"].to_list() == [None]
            assert values["01"].item() == pytest.approx(0, abs=1e-10)
            assert values["total"].item() == pytest.approx(0, abs=1e-10)
            assert values.schema["02"] == pl.Float64
            assert state["group"].to_list() == ["02"]
            assert state["reason"].to_list() == ["no_effective_current_population"]
            if expected_values is None:
                expected_values, expected_status = values, state
            else:
                assert_frame_equal(values, expected_values)
                assert expected_status is not None
                assert_frame_equal(state, expected_status)
            if "z" in features:
                z = _frame(report.get_table("stats.psi", features="z", **selection))
                assert z.select("01", "02", "total").row(0) == pytest.approx([0, 0, 0], abs=1e-10)
                assert _status(report, features="z", **selection).is_empty()
            if mode != "plain":
                valid = _frame(report.get_table("stats.psi", features="x", segments=["seg1"], include_all=False))
                assert valid.select("01", "02", "total").row(0) == pytest.approx([0, 0, 0], abs=1e-10)
                assert _status(report, features="x", segments=["seg1"], include_all=False).is_empty()
                overall = _frame(report.get_table("stats.psi", features="x", include_all=True, filters={"scope_id": "all"}))
                assert overall.select("01", "02", "total").row(0) == pytest.approx([0, 0, 0], abs=1e-10)
                assert _status(report, features="x", include_all=True, filters={"scope_id": "all"}).is_empty()


@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.parametrize("all_features", [False, True])
def test_empty_groups_and_entire_current_feature_keep_float_identity(
    all_empty: bool, all_features: bool,
) -> None:
    values = [None] * 30 if all_empty else [float(i) for i in range(10)] + [None] * 20
    frame = pl.DataFrame({
        "g": ["01"] * 10 + ["02"] * 10 + ["03"] * 10,
        "x": pl.Series(values, dtype=pl.Float64),
        "z": pl.Series(values if all_features else list(range(10)) * 3, dtype=pl.Float64),
    })
    benchmark = pl.DataFrame({"x": [float(i) for i in range(10)], "z": [float(i) for i in range(10)]})
    report = profile_stats(frame, features=["x", "z"], metrics=["psi"], group_col="g", benchmark_df=benchmark)
    result = report.get_table("stats.psi", features="x")
    assert result.columns == ["feature", "display_name", "dtype", "data_source", "01", "02", "03", "total"]
    for group in ("01", "02", "03", "total"):
        assert result.schema[group] == pl.Float64
        assert (result[group].item() is None) == (all_empty or group in ("02", "03"))
    status = _status(report, features="x")
    assert set(status["group"]) == ({"01", "02", "03", "Total"} if all_empty else {"02", "03"})
    assert set(status["reason"]) == {"no_effective_current_population"}
    if not all_features:
        assert _status(report, features="z").is_empty()


@pytest.mark.parametrize("mode", ["plain", "shared", "per_segment"])
@pytest.mark.parametrize("include_missing", [False, True])
def test_first_group_reference_cannot_skip_empty_feature_to_later_period(
    mode: str, include_missing: bool,
) -> None:
    frame = pl.DataFrame({
        "g": ["01"] * 20 + ["02"] * 20,
        "x": pl.Series([None] * 20 + [float(i) for i in range(20)], dtype=pl.Float64),
        "z": [float(i) for i in range(20)] * 2, "s": ["seg1"] * 40,
    })
    options: dict[str, Any] = {} if mode == "plain" else {"segment_col": "s"}
    report = MarsDataProfiler(psi_batch_size=2, binning_scope="shared" if mode == "plain" else mode).generate_profile(
        frame, features=["x", "z"], metrics=["psi"], group_col="g",
        psi_include_missing=include_missing, **options,
    )
    values = report.get_table("stats.psi", features="x")
    if include_missing:
        assert values["01"].to_list() == pytest.approx([0] * values.height, abs=1e-10)
        assert values["02"].to_list() == pytest.approx([24.6613305226] * values.height, abs=1e-9)
        assert values["total"].to_list() == pytest.approx([5.4228698670] * values.height, abs=1e-9)
        assert _status(report, features="x").is_empty()
    else:
        assert values.select("01", "02", "total").null_count().row(0) == (values.height,) * 3
        state = _status(report, features="x")
        assert state.height > 0
        assert set(state["status"]) == {"not_computed"}
        assert set(state["reason"]) == {"no_effective_reference_population"}
    assert report.get_table("stats.psi", features="z")["total"].to_list() == pytest.approx([0] * values.height, abs=1e-10)


@pytest.mark.parametrize("include_missing", [False, True])
def test_shared_local_excluded_reference_is_not_a_fit_failure(include_missing: bool) -> None:
    benchmark = pl.DataFrame({
        "s": ["seg1"] * 20 + ["seg2"] * 20,
        "x": pl.Series([float(i) for i in range(20)] + [None] * 20, dtype=pl.Float64),
    })
    current = pl.DataFrame({
        "s": ["seg1"] * 40 + ["seg2"] * 40, "g": (["01"] * 20 + ["02"] * 20) * 2,
        "x": [float(i) for i in range(20)] * 4,
    })
    report = profile_stats(current, features=["x"], metrics=["psi"], segment_col="s", group_col="g",
                           benchmark_df=benchmark, binning_scope="shared", psi_include_missing=include_missing)
    values = report.get_table("stats.psi", segments=["seg2"], include_all=False)
    if include_missing:
        assert all(value is not None and math.isfinite(value) for value in values.select("01", "02", "total").row(0))
        assert _status(report, segments=["seg2"], include_all=False).is_empty()
    else:
        assert values.select("01", "02", "total").row(0) == (None, None, None)
        state = _status(report, segments=["seg2"], include_all=False)
        assert set(state["status"]) == {"not_computed"}
        assert set(state["reason"]) == {"no_effective_reference_population"}
    assert _status(report, segments=["seg1"], include_all=False).is_empty()


def test_total_psi_uses_pooled_counts_instead_of_mean_period_psi() -> None:
    frame = pl.DataFrame({
        "g": ["01"] * 10 + ["02"] * 5 + ["03"] * 20,
        "x": pl.Series([0.0] * 9 + [1.0] + [None] * 5 + [0.0] + [1.0] * 19, dtype=pl.Float64),
        "z": [0.0, 1.0] * 17 + [0.0],
    })
    benchmark = pl.DataFrame({"x": [0.0, 1.0] * 5, "z": [0.0, 1.0] * 5})
    report = MarsDataProfiler(psi_n_bins=2, psi_bin_method="uniform", psi_merge_small_bins=False).generate_profile(
        frame, features=["x", "z"], metrics=["psi"], group_col="g", benchmark_df=benchmark,
    )
    values = report.get_table("stats.psi", features="x")
    period01, period03 = _oracle([9, 1], [5, 5]), _oracle([1, 19], [5, 5])
    pooled = _oracle([10, 20], [5, 5])
    assert values["01"].item() == pytest.approx(period01, abs=1e-10)
    assert values["02"].item() is None
    assert values["03"].item() == pytest.approx(period03, abs=1e-10)
    assert values["total"].item() == pytest.approx(pooled, abs=1e-10)
    assert period01 is not None and period03 is not None and pooled is not None
    assert abs(pooled - (period01 + period03) / 2) > 0.1


@pytest.mark.parametrize("include_missing", [False, True])
@pytest.mark.parametrize("include_special", [False, True])
def test_missing_special_switches_restore_only_their_applicable_groups(
    include_missing: bool, include_special: bool,
) -> None:
    frame = pl.DataFrame({
        "g": ["zero"] * 10 + ["missing"] * 10 + ["special"] * 10 + ["mixed"] * 10,
        "x": pl.Series([0.0] * 10 + [None] * 10 + [-999.0] * 10 + [None, -999.0] * 5, dtype=pl.Float64),
    })
    report = profile_stats(frame, features=["x"], metrics=["psi"], group_col="g",
                           benchmark_df=pl.DataFrame({"x": [0.0, 1.0] * 5}), special_values=[-999.0],
                           psi_include_missing=include_missing, psi_include_special=include_special)
    values = report.get_table("stats.psi")
    expected_null = {
        "zero": False, "missing": not include_missing, "special": not include_special,
        "mixed": not (include_missing or include_special), "total": False,
    }
    for group, null in expected_null.items():
        value = values[group].item()
        assert (value is None) == null
        if value is not None:
            assert math.isfinite(value)
    status = _status(report)
    assert set(status["group"]) == {group for group, null in expected_null.items() if null}


@pytest.mark.parametrize("kind", ["numeric", "string", "null", "date"])
def test_group_labels_status_and_multichunk_inputs_keep_the_same_identity(kind: str) -> None:
    groups: list[Any] = {
        "numeric": [1, 2, 3], "string": ["01", "02", "03"],
        "null": ["null", None, "(空分组)"],
        "date": [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)],
    }[kind]
    full = pl.DataFrame({
        "g": [group for group in groups for _ in range(10)],
        "x": pl.Series([float(i) for i in range(10)] + [None] * 10 + [float(i) for i in range(10)], dtype=pl.Float64),
        "z": [float(i) for i in range(10)] * 3,
    })
    current = pl.concat([full.head(7), full.slice(7, 16), full.slice(23)], rechunk=False)
    benchmark = pl.concat([full.head(3), full.slice(3, 7)], rechunk=False).select("x", "z")
    before_current, before_benchmark = current.clone(), benchmark.clone()
    chunks = {name: [data[column].n_chunks() for column in data.columns] for name, data in [("current", current), ("benchmark", benchmark)]}
    options: dict[str, Any] = {"time_col": "g", "time_grain": "month"} if kind == "date" else {"group_col": "g"}
    actual = profile_stats(current, features=["x", "z"], metrics=["psi"], benchmark_df=benchmark, **options)
    expected = profile_stats(current.rechunk(), features=["x", "z"], metrics=["psi"], benchmark_df=benchmark.rechunk(), **options)
    for name in actual.describe()["tables"]:
        assert_frame_equal(actual.get_table(name), expected.get_table(name))
    label = {"numeric": "2", "string": "02", "null": "(空分组)#", "date": "202602"}[kind]
    assert actual.get_table("stats.psi", features="x")[label].item() is None
    assert _status(actual)["group"].to_list() == [label]
    assert_frame_equal(current, before_current)
    assert_frame_equal(benchmark, before_benchmark)
    assert chunks == {name: [data[column].n_chunks() for column in data.columns] for name, data in [("current", current), ("benchmark", benchmark)]}


@pytest.mark.parametrize("category", [False, True])
def test_current_constant_and_numeric_zero_are_valid_distributions(category: bool) -> None:
    frame = pl.DataFrame({"g": ["01"] * 10 + ["02"] * 10,
                          "x": ["a"] * 20 if category else [0.0] * 20})
    reference = pl.DataFrame({"x": ["a", "b"] * 5 if category else [0.0, 1.0] * 5})
    report = MarsDataProfiler(psi_n_bins=2, psi_bin_method="uniform").generate_profile(
        frame, features=["x"], metrics=["psi"], group_col="g", benchmark_df=reference,
    )
    expected = _oracle([20, 0], [5, 5])
    assert report.get_table("stats.psi")["total"].item() == pytest.approx(expected, abs=1e-10)
    assert _status(report).is_empty()


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
def test_population_checks_reuse_actual_fit_population_without_refitting_groups(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame([
        {"s": segment, "g": group, "x": float(i), "z": float(i)}
        for segment in ("seg1", "seg2") for group in ("01", "02") for i in range(10)
    ])
    benchmark = frame.filter(pl.col("g") == "01")
    current = frame.with_columns(pl.when(pl.col("g") == "02").then(None).otherwise(pl.col("x")).alias("x"))
    original_fit = MarsNativeBinner.fit
    calls: list[int] = []

    def tracked_fit(self: MarsNativeBinner, data: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(data.height)
        return original_fit(self, data, *args, **kwargs)

    monkeypatch.setattr(MarsNativeBinner, "fit", tracked_fit)
    report = MarsDataProfiler(binning_scope=mode, psi_batch_size=1).generate_profile(
        current, features=["x", "z"], metrics=["psi"], segment_col="s", group_col="g",
        benchmark_df=benchmark,
    )
    assert calls == ([20] if mode == "shared" else [20, 10, 10])
    assert report.get_table("stats.psi", features="x")["02"].null_count() == 3


@pytest.mark.parametrize("side", ["valid", "current", "reference"])
def test_ungrouped_benchmark_population_and_total_status(side: str) -> None:
    frame = pl.DataFrame({"s": ["seg1"] * 20 + ["seg2"] * 20,
                          "x": [float(i) for i in range(20)] * 2})
    benchmark = frame.clone()
    if side == "current":
        frame = frame.with_columns(pl.when(pl.col("s") == "seg2").then(None).otherwise(pl.col("x")).alias("x"))
    elif side == "reference":
        benchmark = benchmark.with_columns(pl.when(pl.col("s") == "seg2").then(None).otherwise(pl.col("x")).alias("x"))
    report = profile_stats(frame, features=["x"], metrics=["psi"], segment_col="s", benchmark_df=benchmark)
    values = report.get_table("stats.psi", segments=["seg2"], include_all=False)
    if side == "valid":
        assert values["total"].item() == pytest.approx(0, abs=1e-10)
        assert _status(report).is_empty()
    else:
        assert values["total"].item() is None
        status = _status(report, segments=["seg2"], include_all=False)
        assert set(status["group"]) <= {"*", "Total"}
        assert status["reason"].to_list() == [f"no_effective_{side}_population"]
        assert status["status"].to_list() == ["not_computed"]


@pytest.mark.parametrize("no_current_population", [False, True])
def test_plain_ungrouped_psi_uses_benchmark_and_typed_total(no_current_population: bool) -> None:
    benchmark = pl.DataFrame({"x": [0.0, 1.0] * 5})
    current = pl.DataFrame({
        "x": pl.Series([None] * 10 if no_current_population else [0.0] * 10, dtype=pl.Float64),
    })
    report = MarsDataProfiler(psi_n_bins=2, psi_bin_method="uniform").generate_profile(
        current, features=["x"], metrics=["psi"], benchmark_df=benchmark,
    )
    values = report.get_table("stats.psi")
    assert values.columns == ["feature", "display_name", "dtype", "data_source", "total"]
    assert values.schema["total"] == pl.Float64
    state = _status(report)
    if no_current_population:
        assert values["total"].item() is None
        assert state.select("feature", "group", "metric", "status", "reason").to_dicts() == [{
            "feature": "x", "group": "Total", "metric": "psi", "status": "not_computed",
            "reason": "no_effective_current_population",
        }]
    else:
        assert values["total"].item() == pytest.approx(_oracle([10, 0], [5, 5]), abs=1e-10)
        assert state.is_empty()
