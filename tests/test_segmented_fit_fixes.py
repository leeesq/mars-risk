"""拟合人口诊断与筛选器运行状态隔离的数值回归。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import profile_stats
from mars.feature import MarsStatsSelector
from mars.reporting import load_report


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("missing", [False, True])
def test_prefitted_psi_rejects_degenerate_actual_fit_population(
    mode: str, missing: bool,
) -> None:
    benchmark = pl.DataFrame({
        "x": pl.Series([None if missing else 0.0] * 10, dtype=pl.Float64),
        "segment": ["seg1"] * 10,
    })
    frame = pl.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "x": pl.Series([None if missing else 0.0] * 10 + [1.0] * 10, dtype=pl.Float64),
        "segment": ["seg1"] * 20,
    })
    report = profile_stats(
        frame, features=["x"], metrics=["psi", "mean"], group_col="group",
        benchmark_df=benchmark, segment_col="segment", binning_scope=mode,
        psi_include_missing=missing,
    )
    for group in ("A", "B", "total"):
        assert report.stats_tables["psi"][group].null_count() == 2
    statuses = report.get_table("calculation_status").filter(pl.col("metric") == "psi")
    assert statuses.height == 2
    expected = "All values are missing or special." if missing else "Degenerate feature: single unique value."
    assert statuses["reason"].to_list() == [expected] * 2


def _selection_frame() -> pl.DataFrame:
    """构造两个客群、两期且稳定分布的筛选输入。"""
    return pl.DataFrame([
        {"segment": segment, "group": group, "x": float(i % 2), "y": i % 2}
        for segment in ("a", "b") for group in ("01", "02") for i in range(20)
    ])


def _selector(**overrides: Any) -> MarsStatsSelector:
    """固定阈值，仅让 PSI 识别参考组残留造成的名单变化。"""
    options: dict[str, Any] = dict(
        selection_scope="per_segment", binning_scope="shared", skip_fine_scan=True,
        rough_iv_thr=-1, rough_lift_thr=1e6, psi_thr=0.05,
        rc_thr=None, corr_thr=None, n_jobs=1, rough_binning_params={"n_bins": 2},
    )
    options.update(overrides)
    return MarsStatsSelector(**options)


def test_reused_selector_does_not_keep_segment_reference_in_global_fit() -> None:
    frame = _selection_frame()
    reused = _selector().fit(
        frame, target="y", features=["x"], segment_col="segment", group_col="group",
    )
    reused.set_params(selection_scope="global")
    current = frame.with_columns(pl.col("group").replace({"01": "03", "02": "04"}))
    reused.fit(current, target="y", features=["x"], group_col="group")
    fresh = _selector(selection_scope="global").fit(
        current, target="y", features=["x"], group_col="group",
    )
    assert reused.selected_features_ == fresh.selected_features_ == ["x"]
    assert_frame_equal(reused.get_report(), fresh.get_report())
    assert reused._stability_report is not None and fresh._stability_report is not None
    assert_frame_equal(reused._stability_report.get_table("summary"), fresh._stability_report.get_table("summary"))


@pytest.mark.parametrize("dtype", [pl.Null, pl.Float32, pl.Float64])
@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_null_fit_feature_does_not_poison_valid_psi_or_other_metrics(
    dtype: pl.DataType, mode: str, backend: str,
) -> None:
    frame = pl.DataFrame({
        "segment": ["seg1"] * 20, "group": ["A"] * 10 + ["B"] * 10,
        "x": pl.Series([None] * 20, dtype=dtype), "valid": [0.0, 1.0] * 10,
    })
    benchmark = frame.filter(pl.col("group") == "A")
    current: pl.DataFrame | pd.DataFrame = frame if backend == "polars" else frame.to_pandas()
    reference: pl.DataFrame | pd.DataFrame = benchmark if backend == "polars" else benchmark.to_pandas()
    report = profile_stats(
        current, features=["x", "valid"], metrics=["psi", "mean", "missing", "schema", "unseen"],
        segment_col="segment", group_col="group", benchmark_df=reference,
        binning_scope=mode, psi_include_missing=True,
    )
    psi = report.get_table("stats.psi")
    if isinstance(psi, pd.DataFrame):
        psi = pl.from_pandas(psi)
    assert psi.filter(pl.col("feature") == "x")["total"].null_count() == 2
    assert psi.filter(pl.col("feature") == "valid")["total"].to_list() == pytest.approx([0, 0], abs=1e-10)
    missing = report.get_table("dq.missing")
    if isinstance(missing, pd.DataFrame):
        missing = pl.from_pandas(missing)
    assert missing.filter(pl.col("feature") == "x")["total"].to_list() == [1.0, 1.0]


@pytest.mark.parametrize("include_missing", [False, True])
@pytest.mark.parametrize("include_special", [False, True])
@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("excluded_role", ["special_values", "missing_values"])
def test_special_only_fit_is_unavailable_regardless_of_distribution_switches(
    include_missing: bool, include_special: bool, mode: str, excluded_role: str,
) -> None:
    frame = pl.DataFrame({
        "segment": ["seg1"] * 20, "group": ["A"] * 10 + ["B"] * 10,
        "x": [-999.0] * 10 + [1.0] * 10,
    })
    report = profile_stats(
        frame, features=["x"], metrics=["psi"], group_col="group", segment_col="segment",
        benchmark_df=frame.head(10), binning_scope=mode,
        psi_include_missing=include_missing, psi_include_special=include_special,
        **{excluded_role: [-999]},
    )
    assert report.stats_tables["psi"]["total"].null_count() == 2
    assert set(report.get_table("calculation_status")["reason"]) == {"All values are missing or special."}


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
def test_shared_fit_uses_overall_population_instead_of_local_constant_reference(mode: str) -> None:
    frame = pl.DataFrame({
        "segment": ["seg1"] * 20 + ["seg2"] * 20,
        "group": (["A"] * 10 + ["B"] * 10) * 2,
        "x": [0.0] * 10 + [1.0] * 10 + [1.0] * 10 + [0.0] * 10,
    })
    report = profile_stats(
        frame, features=["x"], metrics=["psi"], group_col="group", segment_col="segment",
        benchmark_df=frame.filter(pl.col("group") == "A"), binning_scope=mode,
    )
    psi = report.get_table("stats.psi", segments=["seg1"], include_all=False)
    if mode == "shared":
        assert psi["A"].item() == pytest.approx(0, abs=1e-10)
        assert psi["B"].item() > 1
        assert "calculation_status" not in report.describe()["tables"]
    else:
        assert psi["total"].item() is None
        assert report.stats_tables["psi"].filter(pl.col("scope_id") == "all")["total"].item() is not None


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("constant_current", [False, True])
def test_internal_first_group_reference_keeps_old_no_benchmark_behavior(
    mode: str, constant_current: bool,
) -> None:
    frame = pl.DataFrame({
        "segment": ["seg1"] * 20, "group": ["A"] * 10 + ["B"] * 10,
        "x": [0.0] * 10 + [0.0 if constant_current else 1.0] * 10,
    })
    plain = profile_stats(frame, features=["x"], metrics=["psi"], group_col="group")
    segmented = profile_stats(
        frame, features=["x"], metrics=["psi"], group_col="group",
        segment_col="segment", binning_scope=mode,
    )
    actual = segmented.get_table("stats.psi", segments=["seg1"], include_all=False).drop("scope_id")
    assert_frame_equal(actual, plain.stats_tables["psi"])
    if constant_current:
        assert actual["B"].item() == pytest.approx(0, abs=1e-10)
    else:
        assert actual["B"].item() > 1


def test_degenerate_psi_status_and_values_survive_report_snapshot(tmp_path: Path) -> None:
    frame = pl.DataFrame({
        "segment": ["seg1"] * 20, "group": ["A"] * 10 + ["B"] * 10,
        "x": [0.0] * 10 + [1.0] * 10,
    })
    report = profile_stats(
        frame, features=["x"], metrics=["mean", "psi"], group_col="group",
        segment_col="segment", benchmark_df=frame.head(10),
    )
    path = tmp_path / "degenerate.marsreport"
    report.save(path, overwrite=True)
    restored = load_report(path)
    for name in ("stats.psi", "calculation_status", "stats.mean"):
        assert_frame_equal(
            restored.get_table(name, segments=["seg1"], include_all=False),
            report.get_table(name, segments=["seg1"], include_all=False),
        )
    assert restored.describe()["parameters"]["diagnostics"] == report.report_meta["diagnostics"]


@pytest.mark.parametrize("first_scope,second_scope", [
    ("global", "per_segment"), ("per_segment", "global"), ("per_segment", "per_segment"),
])
@pytest.mark.parametrize("benchmark_transition", ["add", "remove", "none"])
@pytest.mark.parametrize("binning_scope", ["shared", "per_segment"])
def test_selector_reuse_matches_fresh_public_decisions_and_stability(
    first_scope: str, second_scope: str, benchmark_transition: str, binning_scope: str,
) -> None:
    frame = _selection_frame().with_columns(pl.col("x").alias("copy"))
    first_binning = binning_scope if first_scope == "per_segment" else "shared"
    second_binning = binning_scope if second_scope == "per_segment" else "shared"
    reused = _selector(selection_scope=first_scope, binning_scope=first_binning, rc_thr=-1, corr_thr=0.9)
    first_kwargs: dict[str, Any] = {"segment_col": "segment"} if first_scope == "per_segment" else {}
    if benchmark_transition == "remove":
        first_kwargs["benchmark_df"] = frame.filter(pl.col("group") == "01")
    reused.fit(frame, target="y", features=["x", "copy"], group_col="group", **first_kwargs)
    current = frame.with_columns(
        pl.col("group").replace({"01": "03", "02": "04"}),
        pl.col("segment").replace({"a": "c", "b": "d"}),
    )
    second_kwargs: dict[str, Any] = {"segment_col": "segment"} if second_scope == "per_segment" else {}
    if benchmark_transition == "add":
        second_kwargs["benchmark_df"] = current.filter(pl.col("group") == "03")
    reused.set_params(selection_scope=second_scope, binning_scope=second_binning)
    reused.fit(current, target="y", features=["x", "copy"], group_col="group", **second_kwargs)
    fresh = _selector(selection_scope=second_scope, binning_scope=second_binning, rc_thr=-1, corr_thr=0.9).fit(
        current, target="y", features=["x", "copy"], group_col="group", **second_kwargs,
    )
    assert reused.selected_features_ == fresh.selected_features_
    assert reused.selected_features_by_segment_ == fresh.selected_features_by_segment_
    assert_frame_equal(reused.get_report(), fresh.get_report())
    report_kwargs = {key: value for key, value in second_kwargs.items() if key == "benchmark_df"}
    reused_report = reused.get_binning_report(current, **report_kwargs)
    fresh_report = fresh.get_binning_report(current, **report_kwargs)
    assert_frame_equal(reused_report.get_table("summary"), fresh_report.get_table("summary"))
    assert_frame_equal(
        reused_report.get_table("risk_corr_reference"), fresh_report.get_table("risk_corr_reference"),
    )
    for table in ("pairs", "correlation_decisions", "funnel"):
        assert_frame_equal(reused.get_correlation_report().get_table(table), fresh.get_correlation_report().get_table(table))
    assert reused._funnel_stats == fresh._funnel_stats


def test_failed_public_fit_invalidates_previous_results_then_recovers() -> None:
    frame = _selection_frame()
    reused = _selector().fit(
        frame, target="y", features=["x"], segment_col="segment", group_col="group",
    )
    with pytest.raises(ValueError, match="Segment column"):
        reused.fit(frame, target="y", features=["x"], segment_col="absent", group_col="group")
    for method in (reused.get_report, reused.get_correlation_report):
        with pytest.raises(ValueError, match="not fitted"):
            method()
    reused.set_params(selection_scope="global")
    current = frame.with_columns(pl.col("group").replace({"01": "03", "02": "04"}))
    reused.fit(current, target="y", features=["x"], group_col="group")
    fresh = _selector(selection_scope="global").fit(current, target="y", features=["x"], group_col="group")
    assert reused.selected_features_ == fresh.selected_features_ == ["x"]
    assert_frame_equal(reused.get_report(), fresh.get_report())


@pytest.mark.parametrize("baseline", ["total", "first_group"])
def test_segmented_refit_validates_benchmark_with_current_rc_policy(baseline: str) -> None:
    frame = _selection_frame()
    reference = frame.filter(pl.col("group") == "01")
    reused = _selector(rc_thr=-1).fit(
        frame, target="y", features=["x"], segment_col="segment", group_col="group",
        benchmark_df=reference, risk_corr_baseline="benchmark",
    )
    # 本次 total/first_group RC 从当前样本取标签，外部分布基准允许不含 y。
    current_reference = reference.drop("y")
    reused.fit(
        frame, target="y", features=["x"], segment_col="segment", group_col="group",
        benchmark_df=current_reference, risk_corr_baseline=baseline,
    )
    fresh = _selector(rc_thr=-1).fit(
        frame, target="y", features=["x"], segment_col="segment", group_col="group",
        benchmark_df=current_reference, risk_corr_baseline=baseline,
    )
    assert reused.selected_features_ == fresh.selected_features_ == ["x"]
    assert reused.selected_features_by_segment_ == fresh.selected_features_by_segment_
    assert_frame_equal(reused.get_report(), fresh.get_report())
