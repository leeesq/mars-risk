from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mars.reporting as reporting_package
from mars.analysis import MarsDataProfiler, profile_stats
from mars.feature import MarsNativeBinner


def _manual_psi(actual: list[float], expected: list[float]) -> float:
    """按 MARS PSI 定义计算测试期望值。"""
    return sum(
        (actual_prob - expected_prob)
        * math.log(actual_prob / (expected_prob + 1e-6))
        for actual_prob, expected_prob in zip(actual, expected)
    )


def test_profiler_returns_pandas_tables_for_pandas_input(sample_credit_pd: pd.DataFrame) -> None:
    profiler = MarsDataProfiler(missing_values=[-999])

    report = profiler.generate_profile(
        sample_credit_pd,
        metrics=["missing", "zeros", "mean", "psi"],
        group_col="month",
        enable_sparkline=False,
    )

    assert isinstance(report.overview_table, pd.DataFrame)
    assert isinstance(report.dq_tables["missing"], pd.DataFrame)
    assert isinstance(report.stats_tables["mean"], pd.DataFrame)
    assert not report.overview_table.empty


def test_evaluation_templates_are_packaged() -> None:
    template_dir = Path(reporting_package.__file__).resolve().parent / "template"
    linux_template = template_dir.joinpath("mars_bin_report_linux.xlsx")
    win_mac_template = template_dir.joinpath("mars_bin_report_win_mac.xlsx")

    assert linux_template.is_file()
    assert win_mac_template.is_file()


def test_profile_report_show_overview_does_not_mutate_polars_storage(sample_credit_df: pl.DataFrame) -> None:
    profiler = MarsDataProfiler(missing_values=[-999])

    report = profiler.generate_profile(
        sample_credit_df,
        metrics=["missing", "zeros", "mean", "psi"],
        group_col="month",
        enable_sparkline=False,
    )

    styler = report.show_overview(features=["income"])

    assert isinstance(styler.data, pd.DataFrame)
    assert isinstance(report.overview_table, pl.DataFrame)
    assert "income" in report.overview_table["feature"].to_list()


def test_profiler_sparkline_uses_explicit_runtime_params(
    sample_credit_df: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pl, "thread_pool_size", lambda: 1)

    profiler = MarsDataProfiler(missing_values=[-999])
    report = profiler.generate_profile(
        sample_credit_df,
        metrics=["missing", "mean"],
        group_col="month",
        enable_sparkline=True,
        sparkline_bins=6,
        sparkline_sample_size=50,
    )

    overview = report.overview_table
    assert "distribution" in overview.columns
    assert not overview.is_empty()


def test_generate_profile_uses_default_full_metrics(sample_credit_df: pl.DataFrame) -> None:
    report = MarsDataProfiler(missing_values=[-999]).generate_profile(
        sample_credit_df,
        features=["income"],
        group_col="month",
        enable_sparkline=False,
    )

    assert {"missing", "zeros", "unique", "mode"}.issubset(report.dq_tables)
    assert {"mode_rate", "mode_value"}.issubset(report.overview_table.columns)
    assert {"psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"}.issubset(
        report.stats_tables
    )


def test_profile_stats_returns_lightweight_report_with_requested_metrics(sample_credit_df: pl.DataFrame) -> None:
    report = profile_stats(
        sample_credit_df,
        metrics=["missing", "mean"],
        features=["income", "utilization"],
        group_col="month",
    )

    assert isinstance(report.overview_table, pl.DataFrame)
    assert set(report.dq_tables) == {"missing"}
    assert set(report.stats_tables) == {"mean"}
    assert "utilization" in report.overview_table["feature"].to_list()


def test_profile_stats_preserves_pandas_output_contract(sample_credit_pd: pd.DataFrame) -> None:
    report = profile_stats(
        sample_credit_pd,
        metrics=["missing", "mean"],
        features=["income"],
        group_col="month",
    )

    assert isinstance(report.overview_table, pd.DataFrame)
    assert isinstance(report.dq_tables["missing"], pd.DataFrame)
    assert isinstance(report.stats_tables["mean"], pd.DataFrame)


def test_profiler_psi_scope_uses_explicit_params() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01"] * 8 + ["2024-02"] * 8,
            "score": [
                -999.0,
                0.10,
                0.20,
                0.30,
                0.70,
                0.80,
                0.90,
                1.00,
                -999.0,
                -999.0,
                -999.0,
                None,
                0.70,
                0.80,
                0.90,
                1.00,
            ],
        }
    )
    profiler = MarsDataProfiler(missing_values=[-999], special_values=[-999])

    base_report = profiler.generate_profile(
        df,
        metrics=["psi"],
        features=["score"],
        group_col="month",
        enable_sparkline=False,
        psi_include_missing=False,
        psi_include_special=False,
    )
    scoped_report = profiler.generate_profile(
        df,
        metrics=["psi"],
        features=["score"],
        group_col="month",
        enable_sparkline=False,
        psi_include_missing=True,
        psi_include_special=True,
    )

    base_psi = base_report.stats_tables["psi"].filter(pl.col("feature") == "score")["2024-02"][0]
    scoped_psi = scoped_report.stats_tables["psi"].filter(pl.col("feature") == "score")["2024-02"][0]
    assert scoped_psi != pytest.approx(base_psi)

    quick_report = profile_stats(
        df,
        metrics=["psi"],
        features=["score"],
        group_col="month",
        missing_values=[-999],
        special_values=[-999],
        psi_include_missing=True,
        psi_include_special=True,
    )
    quick_psi = quick_report.stats_tables["psi"].filter(pl.col("feature") == "score")["2024-02"][0]
    assert quick_psi == pytest.approx(scoped_psi)


def test_profiler_external_benchmark_drives_group_and_total_psi() -> None:
    benchmark_df = pl.DataFrame({"x": [0.0] * 8 + [1.0] * 2})
    current_df = pl.DataFrame(
        {
            "month": ["2024-01"] * 10 + ["2024-02"] * 10,
            "x": [0.0] * 8 + [1.0] * 2 + [0.0] * 2 + [1.0] * 8,
        }
    )

    report = MarsDataProfiler(psi_merge_small_bins=False).generate_profile(
        current_df,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark_df,
        group_col="month",
        enable_sparkline=False,
    )

    psi_row = report.stats_tables["psi"].row(0, named=True)
    assert psi_row["2024-01"] == pytest.approx(0.0, abs=1e-12)
    assert psi_row["2024-02"] == pytest.approx(_manual_psi([0.2, 0.8], [0.8, 0.2]))
    assert psi_row["total"] == pytest.approx(_manual_psi([0.5, 0.5], [0.8, 0.2]))
    assert psi_row["group_mean"] == pytest.approx(
        (psi_row["2024-01"] + psi_row["2024-02"]) / 2
    )


def test_profiler_external_benchmark_supports_total_only_psi(tmp_path: Path) -> None:
    benchmark_df = pl.DataFrame({"x": [0.0] * 8 + [1.0] * 2})
    current_df = pl.DataFrame({"x": [0.0] * 2 + [1.0] * 8})

    report = profile_stats(
        current_df,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark_df,
    )

    psi_table = report.stats_tables["psi"]
    assert psi_table.columns == ["feature", "dtype", "total"]
    assert psi_table["total"][0] == pytest.approx(
        _manual_psi([0.2, 0.8], [0.8, 0.2])
    )
    assert "Trend Analysis: psi" in report.show_trend("psi").to_html()

    output_path = tmp_path / "benchmark-profile.xlsx"
    report.write_excel(str(output_path))
    assert output_path.stat().st_size > 0


def test_profiler_external_benchmark_uses_union_of_observed_bins() -> None:
    benchmark_df = pl.DataFrame({"segment": ["A"] * 5 + ["B"] * 5})
    current_df = pl.DataFrame({"segment": ["C"] * 10})

    report = MarsDataProfiler(psi_n_bins=2).generate_profile(
        current_df,
        metrics=["psi"],
        features=["segment"],
        benchmark_df=benchmark_df,
        enable_sparkline=False,
    )

    expected = _manual_psi(
        [1e-6, 1e-6, 1.0],
        [0.5, 0.5, 1e-6],
    )
    assert report.stats_tables["psi"]["total"][0] == pytest.approx(expected)


def test_profiler_external_benchmark_is_native_binner_fit_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mars.analysis._profiling import psi as psi_module

    observed_fit_values: list[list[float]] = []
    original_fit = MarsNativeBinner.fit

    def spy_fit(
        self: MarsNativeBinner,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any] | None = None,
        *,
        features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MarsNativeBinner:
        """记录 NativeBinner 实际接收的拟合数据。"""
        X_pl = X if isinstance(X, pl.DataFrame) else pl.from_pandas(X)
        observed_fit_values.append(X_pl.get_column("x").to_list())
        return original_fit(
            self,
            X,
            y,
            features=features,
            cat_features=cat_features,
        )

    monkeypatch.setattr(psi_module.MarsNativeBinner, "fit", spy_fit)
    benchmark_df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    current_df = pl.DataFrame({"x": [100.0, 101.0, 102.0, 103.0]})

    profile_stats(
        current_df,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark_df,
    )

    assert observed_fit_values == [[0.0, 1.0, 2.0, 3.0]]


def test_profiler_external_benchmark_does_not_change_other_metrics() -> None:
    current_df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "x": [1.0, None, 3.0, 4.0],
        }
    )
    benchmark_df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    profiler = MarsDataProfiler()

    base_report = profiler.generate_profile(
        current_df,
        metrics=["missing", "mean"],
        features=["x"],
        group_col="month",
        enable_sparkline=False,
    )
    benchmark_report = profiler.generate_profile(
        current_df,
        metrics=["missing", "mean"],
        features=["x"],
        benchmark_df=benchmark_df,
        group_col="month",
        enable_sparkline=False,
    )

    assert base_report.overview_table.equals(benchmark_report.overview_table)
    assert base_report.dq_tables["missing"].equals(
        benchmark_report.dq_tables["missing"]
    )
    assert base_report.stats_tables["mean"].equals(
        benchmark_report.stats_tables["mean"]
    )


def test_profiler_ignores_invalid_benchmark_when_psi_is_not_requested() -> None:
    report = profile_stats(
        pl.DataFrame({"x": [1.0, 2.0, 3.0]}),
        metrics=["mean"],
        features=["x"],
        benchmark_df=pl.DataFrame(),
    )

    assert report.stats_tables["mean"]["total"][0] == pytest.approx(2.0)


@pytest.mark.parametrize("current_kind", ["polars", "pandas"])
@pytest.mark.parametrize("benchmark_kind", ["polars", "pandas"])
def test_profiler_external_benchmark_preserves_current_output_type(
    current_kind: str,
    benchmark_kind: str,
) -> None:
    current_pl = pl.DataFrame({"x": [0.0, 0.0, 1.0, 1.0]})
    benchmark_pl = pl.DataFrame({"x": [0.0, 0.0, 0.0, 1.0]})
    current = current_pl if current_kind == "polars" else current_pl.to_pandas()
    benchmark = benchmark_pl if benchmark_kind == "polars" else benchmark_pl.to_pandas()

    report = profile_stats(
        current,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark,
    )

    expected_type = pl.DataFrame if current_kind == "polars" else pd.DataFrame
    assert isinstance(report.stats_tables["psi"], expected_type)


@pytest.mark.parametrize(
    ("benchmark_df", "error_match"),
    [
        (pl.DataFrame(schema={"x": pl.Float64}), "at least one row"),
        (pl.DataFrame({"other": [1.0, 2.0]}), "missing active PSI features"),
        (pl.DataFrame({"x": ["a", "b"]}), "incompatible dtypes"),
    ],
)
def test_profiler_external_benchmark_rejects_invalid_frames(
    benchmark_df: pl.DataFrame,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        profile_stats(
            pl.DataFrame({"x": [1.0, 2.0]}),
            metrics=["psi"],
            features=["x"],
            benchmark_df=benchmark_df,
        )


def test_profiler_external_benchmark_rejects_features_without_included_bins() -> None:
    current_df = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Float64})
    benchmark_df = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Float64})

    with pytest.raises(ValueError, match="at least one included bin"):
        profile_stats(
            current_df,
            metrics=["psi"],
            features=["x"],
            benchmark_df=benchmark_df,
        )


def test_profile_stats_and_profiler_share_external_benchmark_contract() -> None:
    current_df = pl.DataFrame({"x": [0.0, 1.0, 1.0, 1.0]})
    benchmark_df = pl.DataFrame({"x": [0.0, 0.0, 0.0, 1.0]})

    quick_report = profile_stats(
        current_df,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark_df,
    )
    class_report = MarsDataProfiler().generate_profile(
        current_df,
        metrics=["psi"],
        features=["x"],
        benchmark_df=benchmark_df,
        enable_sparkline=False,
    )

    assert quick_report.stats_tables["psi"].equals(class_report.stats_tables["psi"])


def test_profiler_public_api_replaces_sampling_with_external_benchmark() -> None:
    profile_stats_params = inspect.signature(profile_stats).parameters
    generate_profile_params = inspect.signature(MarsDataProfiler.generate_profile).parameters

    assert "benchmark_df" in profile_stats_params
    assert "benchmark_df" in generate_profile_params
    assert "categorical_features" in profile_stats_params
    assert "categorical_features" in generate_profile_params
    assert "sample_frac" not in profile_stats_params
    assert "sample_frac" not in generate_profile_params


def test_profiler_categorical_psi_uses_native_binner_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mars.analysis._profiling import psi as psi_module

    observed_bins: list[int] = []
    def fake_calc_psi_from_stats(
        *,
        stats_df: pl.LazyFrame,
        unique_bins_skel: pl.LazyFrame,
        group_col: str,
        baseline_group: object,
        include_missing: bool,
        include_special: bool,
    ) -> pl.LazyFrame:
        skeleton = unique_bins_skel.collect()
        observed_bins.extend(skeleton["bin_id"].to_list())
        groups = stats_df.select(group_col).unique()
        features = skeleton.lazy().select("feat_idx").unique()
        return groups.join(features, how="cross").with_columns(
            [
                pl.lit(0.0).alias("total"),
                pl.lit(0.0).alias("psi"),
            ]
        )

    monkeypatch.setattr(psi_module, "calc_psi_from_stats", fake_calc_psi_from_stats)

    df = pl.DataFrame(
        {
            "month": ["2024-01"] * 14 + ["2024-02"] * 14,
            "segment": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "D",
                "D",
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "D",
                "D",
                "D",
                "D",
                "D",
                "D",
            ],
        }
    )

    report = MarsDataProfiler(psi_n_bins=2).generate_profile(
        df,
        metrics=["psi"],
        features=["segment"],
        group_col="month",
        enable_sparkline=False,
    )

    assert set(observed_bins) >= {0, 1, -2}
    assert "segment" in report.stats_tables["psi"]["feature"].to_list()


def test_profiler_categorical_psi_scope_uses_missing_and_special_bins() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01"] * 8 + ["2024-02"] * 8,
            "segment": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "SPECIAL",
                None,
                "A",
                "B",
                "C",
                "C",
                "SPECIAL",
                "SPECIAL",
                None,
                None,
            ],
        }
    )

    profiler = MarsDataProfiler(special_values=["SPECIAL"])
    base_report = profiler.generate_profile(
        df,
        metrics=["psi"],
        features=["segment"],
        group_col="month",
        enable_sparkline=False,
        psi_include_missing=False,
        psi_include_special=False,
    )
    scoped_report = profiler.generate_profile(
        df,
        metrics=["psi"],
        features=["segment"],
        group_col="month",
        enable_sparkline=False,
        psi_include_missing=True,
        psi_include_special=True,
    )

    base_psi = base_report.stats_tables["psi"].filter(pl.col("feature") == "segment")["2024-02"][0]
    scoped_psi = scoped_report.stats_tables["psi"].filter(pl.col("feature") == "segment")["2024-02"][0]
    assert scoped_psi != pytest.approx(base_psi)


def test_profiler_psi_binner_uses_stable_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from mars.analysis._profiling import psi as psi_module

    captured_params: list[dict[str, object]] = []

    class SpyNativeBinner:
        def __init__(self, **kwargs: object) -> None:
            captured_params.append(kwargs)
            self.features: list[str] = []
            self.bin_cuts_: dict[str, list[float]] = {}

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> SpyNativeBinner:
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
            features: list[str] | None = None,
            return_type: str = "index",
            lazy: bool = False,
        ) -> pl.DataFrame | pl.LazyFrame:
            result = df.with_columns(
                [
                    pl.when(pl.col(feature) < 3)
                    .then(pl.lit(0))
                    .otherwise(pl.lit(1))
                    .cast(pl.Int16)
                    .alias(f"{feature}_bin")
                    for feature in self.features
                ]
            ).select([col for col in df.columns if col not in self.features] + [f"{feature}_bin" for feature in self.features])
            return result.lazy() if lazy else result

    monkeypatch.setattr(psi_module, "MarsNativeBinner", SpyNativeBinner)

    df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    profile_stats(df, metrics=["psi"], features=["score"], group_col="month")

    assert captured_params
    assert captured_params[0]["remove_empty_bins"] is True
    assert captured_params[0]["merge_small_bins"] is True
    assert captured_params[0]["min_bin_size"] == pytest.approx(0.02)


def test_profiler_psi_binner_accepts_legacy_style_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from mars.analysis._profiling import psi as psi_module

    captured_params: list[dict[str, object]] = []

    class SpyNativeBinner:
        def __init__(self, **kwargs: object) -> None:
            captured_params.append(kwargs)
            self.features: list[str] = []
            self.bin_cuts_: dict[str, list[float]] = {}

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> SpyNativeBinner:
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
            features: list[str] | None = None,
            return_type: str = "index",
            lazy: bool = False,
        ) -> pl.DataFrame | pl.LazyFrame:
            result = df.with_columns(
                [pl.lit(0, dtype=pl.Int16).alias(f"{feature}_bin") for feature in self.features]
            ).select([col for col in df.columns if col not in self.features] + [f"{feature}_bin" for feature in self.features])
            return result.lazy() if lazy else result

    monkeypatch.setattr(psi_module, "MarsNativeBinner", SpyNativeBinner)

    df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    profile_stats(
        df,
        metrics=["psi"],
        features=["score"],
        group_col="month",
        psi_remove_empty_bins=False,
        psi_merge_small_bins=False,
        psi_min_bin_size=0.0,
    )

    assert captured_params
    assert captured_params[0]["remove_empty_bins"] is False
    assert captured_params[0]["merge_small_bins"] is False
    assert captured_params[0]["min_bin_size"] == pytest.approx(0.0)


def test_profiler_psi_uses_observed_bins_for_numeric_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mars.analysis._profiling import psi as psi_module

    observed_bins: list[int] = []

    class SparseNativeBinner:
        def __init__(self, **kwargs: object) -> None:
            self.features: list[str] = []
            self.bin_cuts_: dict[str, list[float]] = {}

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> SparseNativeBinner:
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
            features: list[str] | None = None,
            return_type: str = "index",
            lazy: bool = False,
        ) -> pl.DataFrame | pl.LazyFrame:
            result = df.with_columns(
                [
                    pl.when(pl.col(feature) < 3)
                    .then(pl.lit(0))
                    .otherwise(pl.lit(2))
                    .cast(pl.Int16)
                    .alias(f"{feature}_bin")
                    for feature in self.features
                ]
            ).select([col for col in df.columns if col not in self.features] + [f"{feature}_bin" for feature in self.features])
            return result.lazy() if lazy else result

    def fake_calc_psi_from_stats(
        *,
        stats_df: pl.LazyFrame,
        unique_bins_skel: pl.LazyFrame,
        group_col: str,
        baseline_group: object,
        include_missing: bool,
        include_special: bool,
    ) -> pl.LazyFrame:
        skeleton = unique_bins_skel.collect()
        observed_bins.extend(skeleton["bin_id"].to_list())
        groups = stats_df.select(group_col).unique()
        features = skeleton.lazy().select("feat_idx").unique()
        return groups.join(features, how="cross").with_columns(
            [
                pl.lit(0.0).alias("total"),
                pl.lit(0.0).alias("psi"),
            ]
        )

    monkeypatch.setattr(psi_module, "MarsNativeBinner", SparseNativeBinner)
    monkeypatch.setattr(psi_module, "calc_psi_from_stats", fake_calc_psi_from_stats)

    df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = profile_stats(
        df,
        metrics=["psi"],
        features=["score"],
        group_col="month",
    )

    assert set(observed_bins) == {0, 2}
    assert set(report.stats_tables["psi"].columns) >= {
        "feature",
        "dtype",
        "total",
        "group_mean",
        "group_var",
        "group_cv",
    }


def test_profiler_reuses_instance_without_runtime_state(sample_credit_pd: pd.DataFrame) -> None:
    profiler = MarsDataProfiler(missing_values=[-999])
    first = profiler.generate_profile(
        sample_credit_pd,
        metrics=["missing", "mean"],
        features=["income"],
        group_col="month",
        enable_sparkline=False,
    )
    second_df = sample_credit_pd.rename(columns={"income": "debt"})
    second = profiler.generate_profile(
        second_df,
        metrics=["missing", "mean"],
        features=["debt"],
        group_col="month",
        enable_sparkline=False,
    )

    assert set(first.overview_table["feature"]) == {"income"}
    assert set(second.overview_table["feature"]) == {"debt"}
    assert not hasattr(profiler, "features")
    assert not hasattr(profiler, "df")
    assert not hasattr(profiler, "_dtype_map")


def test_profiler_handles_notebook_synthetic_stability_metrics() -> None:
    rng = np.random.default_rng(2026)
    rows = 240
    month_idx = np.arange(rows) // 80
    months = np.array(["2024-01", "2024-02", "2024-03"])[month_idx]
    stable = rng.normal(loc=0.0, scale=1.0, size=rows)
    drift = rng.normal(loc=month_idx * 0.45, scale=1.0, size=rows)
    zeros = np.where(rng.random(rows) < 0.35, 0.0, rng.normal(size=rows))
    skew = rng.lognormal(mean=0.2, sigma=0.8, size=rows)
    missing_values = rng.normal(size=rows).astype(object)
    missing_values[::13] = None
    missing_values[::17] = -999.0

    df = pl.DataFrame(
        {
            "month": months.tolist(),
            "stable": stable,
            "drift": drift,
            "zeros": zeros,
            "skew": skew,
            "missing_feature": missing_values.tolist(),
        }
    )

    profiler = MarsDataProfiler(missing_values=[-999])
    report = profiler.generate_profile(
        df,
        metrics=["missing", "zeros", "unique", "mode", "mean", "min", "max", "skew", "psi"],
        features=["stable", "drift", "zeros", "skew", "missing_feature"],
        group_col="month",
        enable_sparkline=False,
    )

    assert set(report.dq_tables) == {"missing", "zeros", "unique", "mode"}
    assert {"mean", "min", "max", "skew", "psi"}.issubset(report.stats_tables)
    assert set(report.overview_table["feature"].to_list()) == {
        "stable",
        "drift",
        "zeros",
        "skew",
        "missing_feature",
    }
    assert "drift" in report.stats_tables["psi"]["feature"].to_list()


def test_profiler_rejects_legacy_top1_metric(sample_credit_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="mode"):
        profile_stats(
            sample_credit_df,
            metrics=["top1"],
            features=["income"],
            group_col="month",
        )


def test_profile_schema_comparison_uses_business_column_union() -> None:
    current = pl.DataFrame(
        {
            "matched": [1, 2],
            "width_change": pl.Series([1, 2], dtype=pl.Int32),
            "current_only": [True, False],
        }
    )
    benchmark = pl.DataFrame(
        {
            "matched": [3, 4],
            "width_change": pl.Series([3, 4], dtype=pl.Int64),
            "benchmark_only": ["a", "b"],
        }
    )

    report = profile_stats(current, metrics=["schema"], benchmark_df=benchmark)
    status = {
        row["feature"]: row["status"]
        for row in report.comparison_tables["schema"].to_dicts()
    }

    assert status == {
        "matched": "matched",
        "width_change": "compatible_change",
        "current_only": "current_only",
        "benchmark_only": "benchmark_only",
    }
    assert report.report_meta["benchmark_row_count"] == 2
    assert report.report_meta["diagnostics"] == []


def test_profile_unseen_excludes_missing_special_and_supports_groups() -> None:
    benchmark = pl.DataFrame({"category_code": [1, 2, -999, None]})
    current = pl.DataFrame(
        {
            "category_code": [1, 3, -999, None],
            "month": ["2026-01", "2026-01", "2026-02", "2026-02"],
        }
    )

    report = profile_stats(
        current,
        metrics=["unseen"],
        benchmark_df=benchmark,
        categorical_features=["category_code"],
        special_values=[-999],
        group_col="month",
    )
    row = report.comparison_tables["unseen"].row(0, named=True)

    assert row["benchmark_unique_count"] == 2
    assert row["valid_count"] == 2
    assert row["unseen_count"] == 1
    assert row["unseen_unique_count"] == 1
    assert row["total"] == pytest.approx(0.5)
    assert row["2026-01"] == pytest.approx(0.5)
    assert row["2026-02"] is None


def test_profile_comparison_output_type_and_report_exports(tmp_path: Path) -> None:
    current = pd.DataFrame({"category": ["a", "new"]})
    benchmark = pl.DataFrame({"category": ["a", "b"]})

    report = profile_stats(
        current,
        metrics=["schema", "unseen"],
        benchmark_df=benchmark,
    )
    html_path = tmp_path / "profile.html"
    excel_path = tmp_path / "profile.xlsx"
    report.write_html(str(html_path), report_name="<unsafe>")
    report.write_excel(str(excel_path))
    html = html_path.read_text(encoding="utf-8")

    assert isinstance(report.comparison_tables["schema"], pd.DataFrame)
    assert report.get_profile_data()._fields == (
        "overview",
        "dq_trends",
        "stats_trends",
        "comparisons",
    )
    assert "&lt;unsafe&gt;" in html
    assert "<unsafe>" not in html
    assert "Search all tables" in html
    assert "Comparisons" in html
    assert excel_path.stat().st_size > 0
    assert {"Metadata", "Compare_Schema", "Compare_Unseen"}.issubset(
        pd.ExcelFile(excel_path).sheet_names
    )


def test_profile_comparison_requires_benchmark_and_other_metrics_ignore_it() -> None:
    current = pl.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="require `benchmark_df`"):
        profile_stats(current, metrics=["schema"])

    report = profile_stats(
        current,
        metrics=["mean"],
        benchmark_df=object(),  # benchmark must remain untouched for non-comparison metrics
    )
    assert "mean" in report.stats_tables
