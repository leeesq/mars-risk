"""有界计算与既有统计口径的回归验证，重性能测量单独运行。"""

from __future__ import annotations

from datetime import datetime, timedelta
from inspect import signature

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsBinEvaluator, MarsDataProfiler
from mars.analysis._profiling.context import build_run_context
from mars.analysis._profiling.pivot import generate_pivot_report
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
        assert_frame_equal(report.get_table(f"{prefix}.{metric}"), expected, **_FRAME_TOLERANCES)


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


def test_batch_parameter_is_validated() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        MarsBinEvaluator().evaluate(sample(), target="y", batch_size=0)
