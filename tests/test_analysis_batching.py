"""有界计算与既有统计口径的回归验证，重性能测量单独运行。"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from inspect import signature
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsBinEvaluator, MarsDataProfiler, profile_stats
from mars.analysis._profiling.context import build_run_context
from mars.analysis._profiling.pivot import generate_pivot_report, generate_pivot_reports
from mars.analysis._profiling.types import ProfileComputeOptions

# 旧版 Polars 使用 rtol/atol，按实际签名选择参数，保持同一浮点容差。
_FRAME_TOLERANCES = (
    {"rel_tol": 1e-8, "abs_tol": 1e-8}
    if "rel_tol" in signature(assert_frame_equal).parameters
    else {"rtol": 1e-8, "atol": 1e-8}
)


def sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            **{f"x{i}": [None, -999.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0] * 3 for i in range(5)},
            "y": [0, 1, None, 1, 0, 1, 0, 1] * 3,
            "weight": [0.5, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0] * 3,
            "amount": [0.0, 100.0, None, 200.0, -1.0, 50.0, 100.0, 10.0] * 3,
            "g": ["a"] * 8 + ["b"] * 8 + ["c"] * 8,
            "day": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(24)],
        }
    )


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("target", ["y", None, "unobserved"])
def test_evaluation_all_tables_match_across_batches(
    pandas: bool, benchmark: bool, target: str | None
) -> None:
    frame = sample()
    if target == "unobserved":
        frame = frame.with_columns(pl.lit(None).cast(pl.Int32).alias("unobserved"))
    features = ["x4", "x1", "x3", "x0", "x2"]
    data = frame.to_pandas() if pandas else frame
    baseline = frame.head(16).to_pandas() if pandas else frame.head(16)
    evaluator = MarsBinEvaluator(
        binner_params={"n_bins": 6, "special_values": [-999.0], "remove_empty_bins": False}
    )
    options = dict(
        features=features,
        target=target,
        group_col="g",
        time_col="day",
        weights_col="weight",
        amount_col="amount",
        psi_include_missing=True,
        psi_include_special=True,
        benchmark_df=baseline if benchmark else None,
        risk_corr_baseline="benchmark" if benchmark and target == "y" else "total",
        feature_start_aware_reference=not benchmark,
    )
    expected = evaluator.evaluate(data, batch_size=5, **options).report
    actual = evaluator.evaluate(data, batch_size=2, **options).report
    assert set(actual._query_tables()) == set(expected._query_tables())
    for name in actual._query_tables():
        left, right = actual.get_table(name), expected.get_table(name)
        if pandas:
            left, right = pl.from_pandas(left), pl.from_pandas(right)
        if name == "missing_by_day":
            assert_frame_equal(left, right)
            description = actual.describe()["tables"][name]
            assert description["grain"] == "feature (days in columns)"
            assert description["fields"]["total"]["unit"] == "ratio"
        sort = [c for c in ["feature", "y", "g", "bin_index", "date"] if c in left.columns]
        if not sort:
            sort = left.columns[:1]
        assert_frame_equal(
            left.sort(sort), right.sort(sort), check_row_order=True, **_FRAME_TOLERANCES
        )


def test_profiler_joint_aggregation_matches_original_per_metric() -> None:
    frame = sample().with_columns(pl.Series("category", ["a", "b", None] * 8))
    features = ["x0", "x1", "x2", "x3", "x4", "category"]
    metrics = [
        "missing",
        "zeros",
        "unique",
        "mode",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "p25",
        "p75",
        "skew",
        "kurtosis",
    ]
    profiler = MarsDataProfiler(
        missing_values=[-1.0], special_values=[-999.0], overview_batch_size=2
    )
    report = profiler.generate_profile(
        frame, features=features, group_col="g", metrics=metrics, enable_sparkline=False
    )
    context = build_run_context(
        frame, features=features, group_col="g", time_col=None, time_grain=None
    )
    options = ProfileComputeOptions(
        missing_values=[-1.0],
        special_values=[-999.0],
        psi_n_bins=5,
        psi_bin_method="quantile",
        psi_remove_empty_bins=True,
        psi_merge_small_bins=True,
        psi_min_bin_size=0.05,
        psi_cv_ignore_threshold=0.001,
        psi_batch_size=2,
        overview_batch_size=2,
        sparkline_bins=5,
        sparkline_sample_size=100,
        psi_include_missing=False,
        psi_include_special=False,
        categorical_features=[],
        diagnostics=[],
    )
    for metric in metrics:
        prefix = "dq" if metric in {"missing", "zeros", "unique", "mode"} else "stats"
        expected = generate_pivot_report(context, options, metric)
        assert_frame_equal(report.get_table(f"{prefix}.{metric}", columns=expected.columns), expected, **_FRAME_TOLERANCES)


@pytest.mark.parametrize("grouped", [False, True])
def test_profiler_keeps_float32_total_dtype(grouped: bool) -> None:
    frame = pl.DataFrame(
        {"x": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32), "g": ["a", "a", "b", "b"]}
    )
    report = MarsDataProfiler().generate_profile(
        frame,
        features=["x"],
        metrics=["mean", "std", "min", "max"],
        group_col="g" if grouped else None,
        enable_sparkline=False,
    )
    assert all(table.schema["total"] == pl.Float32 for table in report.stats_tables.values())


@pytest.mark.parametrize("batch", [1, 2, 16, 100])
def test_profiler_overview_mixed_float_widths_across_feature_batches(batch: int) -> None:
    """各批可具有不同物理浮点类型，overview 的 Float64 公共合同仍一致。"""
    frame = pl.DataFrame({
        **{f"f32_{i}": pl.Series([1.0, None, 3.0], dtype=pl.Float32) for i in range(2)},
        **{f"f64_{i}": [2.0, 4.0, None] for i in range(3)},
    })
    expected = MarsDataProfiler(overview_batch_size=100).generate_profile(
        frame, metrics=["mean", "psi"], enable_sparkline=False,
    )
    actual = MarsDataProfiler(overview_batch_size=batch).generate_profile(
        frame, metrics=["mean", "psi"], enable_sparkline=False,
    )
    for name in actual._query_tables():
        assert_frame_equal(actual.get_table(name), expected.get_table(name))
    assert actual.overview_table.schema["mean"] == pl.Float64
    assert actual.overview_table.filter(pl.col("feature").str.starts_with("f32"))["dtype"].to_list() == ["Float32"] * 2
    assert frame.schema["f32_0"] == pl.Float32


def test_batch_parameter_is_validated() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        MarsBinEvaluator().evaluate(sample(), target="y", batch_size=0)


def _profile_options() -> ProfileComputeOptions:
    """提供小批宽，覆盖末批和 overview 缓存复用。"""
    return ProfileComputeOptions(
        missing_values=[-1.0], special_values=[-999.0], psi_n_bins=3,
        psi_bin_method="quantile", psi_remove_empty_bins=True, psi_merge_small_bins=True,
        psi_min_bin_size=0.05, psi_cv_ignore_threshold=0.001, psi_batch_size=2,
        overview_batch_size=2, sparkline_bins=5, sparkline_sample_size=100,
        psi_include_missing=False, psi_include_special=False, categorical_features=[],
        diagnostics=[],
    )


def _assert_multichunk_profile_contract() -> None:
    """在导入前固定单线程的子进程内核验多 chunk 与同值 rechunk 的完整合同。"""
    assert os.environ["POLARS_MAX_THREADS"] == "1"
    assert pl.thread_pool_size() == 1
    full = pl.DataFrame(
        {
            "x32": pl.Series([1.0, None, -999.0, 0.0, 3.0, 4.0, -1.0, 6.0], dtype=pl.Float32),
            "x64": [0.0, 1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0],
            "category": ["a", "b", None, "a", "c", "a", "b", "c"],
            "category2": ["u", "v", "u", None, "w", "u", "v", "w"],
            "null_num": pl.Series([None] * 8, dtype=pl.Float64),
            "null_str": pl.Series([None] * 8, dtype=pl.String),
            "g": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "single": ["only"] * 8,
            "nullable_group": [None, "null", "(空分组)", "normal"] * 2,
            "day": [datetime(2026, 1, 1) + timedelta(days=i // 2) for i in range(8)],
        }
    )
    full = full.with_columns(pl.col("category").cast(pl.Categorical).alias("encoded_category"))
    frame = pl.concat([full.slice(0, 2), full.slice(2, 5), full.slice(7, 1)], rechunk=False)
    before = frame.clone()
    chunks = {c: [len(chunk) for chunk in frame[c].get_chunks()] for c in frame.columns}
    assert all(frame[c].n_chunks() > 1 for c in frame.columns)
    assert chunks["x32"] == [2, 5, 1]
    scopes = [
        ["x32"], ["x64", "x32"], ["category"], ["category2", "category"],
        ["x32", "category", "x64"], ["null_str", "null_num"],
        ["encoded_category"],
    ]
    metric_sets = [["psi"], ["missing", "zeros", "mean", "std", "psi"], ["mean", "std"], ["zeros"]]
    dimensions: list[dict[str, Any]] = [
        {}, {"group_col": "single"}, {"group_col": "g"},
        {"time_col": "day", "time_grain": "day"},
        {"group_col": "nullable_group"},
    ]
    count = 0
    for features in scopes:
        for metrics in metric_sets:
            for dimension in dimensions:
                for benchmark in [False, True]:
                    kwargs = {
                        "features": features, "metrics": metrics, **dimension,
                        "benchmark_df": full.select(features) if benchmark else None,
                        "enable_sparkline": False,
                    }
                    profiler = MarsDataProfiler(overview_batch_size=2, psi_batch_size=2)
                    actual = profiler.generate_profile(frame, **kwargs)
                    expected = profiler.generate_profile(frame.rechunk(), **kwargs)
                    assert list(actual._query_tables()) == list(expected._query_tables())
                    for name in actual._query_tables():
                        left, right = actual.get_table(name), expected.get_table(name)
                        assert_frame_equal(left, right, **_FRAME_TOLERANCES)
                        assert actual.describe()["tables"][name] == expected.describe()["tables"][name]
                    assert set(actual.overview_table["feature"].to_list()) == set(features)
                    if "psi" in metrics and not benchmark and not dimension:
                        psi = actual.stats_tables["psi"]
                        assert psi.schema["total"] == pl.Float64
                        assert psi["total"].null_count() == len(features)
                        status = actual.get_table("calculation_status").filter(pl.col("metric") == "psi")
                        assert status.height == len(features)
                        assert set(status["status"].to_list()) == {"not_computed"}
                        assert set(status["reason"].to_list()) == {"no_group_or_benchmark"}
                    count += 1

    # 绕开 overview 缓存分别核验单指标与联合聚合，纯常量路径不能依赖其他指标撑成单行。
    for features in [["category"], ["category", "category2"], ["x32", "category"]]:
        for group in [None, "single", "g", "nullable_group"]:
            context = build_run_context(frame, features=features, group_col=group, time_col=None, time_grain=None)
            metrics = ["mean", "std", "zeros", "psi"]
            joint = generate_pivot_reports(context, _profile_options(), metrics, pl.DataFrame())
            for metric in metrics:
                expected = generate_pivot_report(context, _profile_options(), metric)
                assert_frame_equal(joint[metric], expected, **_FRAME_TOLERANCES)

    # 空表不存在分组值，但每个请求特征仍须有 typed total 和 overview 身份。
    for features in scopes:
        for dimension in dimensions:
            actual = profile_stats(frame.head(0), features=features, metrics=["zeros", "mean", "std", "psi"], **dimension)
            assert set(actual.overview_table["feature"].to_list()) == set(features)
            assert actual.stats_tables["psi"].height == len(features)
            assert actual.stats_tables["psi"].schema["total"] == pl.Float64
            assert actual.stats_tables["psi"]["total"].null_count() == len(features)
    public = profile_stats(frame, features=["x32"], metrics=["psi"])
    assert public.overview_table["feature"].to_list() == ["x32"]
    nullable = profile_stats(
        frame, features=["x64"], metrics=["mean", "psi"],
        group_col="nullable_group", benchmark_df=full.select("x64"),
    )
    assert nullable.stats_tables["mean"]["(空分组)#"].to_list() == [2.0]
    assert nullable.stats_tables["mean"]["null"].to_list() == [3.0]
    for label, group_value in [("(空分组)#", None), ("null", "null"), ("(空分组)", "(空分组)")]:
        mask = pl.col("nullable_group").is_null() if group_value is None else pl.col("nullable_group") == group_value
        subset = profile_stats(frame.filter(mask), features=["x64"], metrics=["psi"], benchmark_df=full.select("x64"))
        assert nullable.stats_tables["psi"][label].to_list() == pytest.approx(subset.stats_tables["psi"]["total"].to_list())
    assert_frame_equal(frame, before)
    assert {c: [len(chunk) for chunk in frame[c].get_chunks()] for c in frame.columns} == chunks
    print(f"multichunk contract: {count} paired public calls; Polars {pl.__version__}")


def test_profiler_multichunk_contract_in_fresh_single_thread_process() -> None:
    """线程环境必须在子进程首次导入 Polars 前设置，不能复用 pytest 的线程池。"""
    env = {**os.environ, "POLARS_MAX_THREADS": "1"}
    code = (
        "import runpy; "
        f"runpy.run_path({str(Path(__file__).resolve())!r})['_assert_multichunk_profile_contract']()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "280 paired public calls" in result.stdout


def test_psi_only_keeps_identity_without_generic_metric_aggregation(monkeypatch: pytest.MonkeyPatch) -> None:
    """PSI 无参考时按 typed 结果构造，不能偷偷恢复通用 literal 透视或全表 rechunk。"""
    import mars.analysis._profiling.metrics as metrics_module
    import mars.analysis._profiling.pivot as pivot_module

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("PSI-only must not aggregate placeholder metrics or rechunk input")

    frame = pl.concat([pl.DataFrame({"x": [1.0]}), pl.DataFrame({"x": [2.0, 3.0]})], rechunk=False)
    monkeypatch.setattr(metrics_module, "metric_expr", forbidden)
    monkeypatch.setattr(pivot_module, "metric_expr", forbidden)
    monkeypatch.setattr(pl.DataFrame, "rechunk", forbidden)
    report = profile_stats(frame, features=["x"], metrics=["psi"])
    assert report.overview_table["feature"].to_list() == ["x"]
    assert report.stats_tables["psi"].select("total").schema == {"total": pl.Float64}
    assert report.stats_tables["psi"]["total"].to_list() == [None]
