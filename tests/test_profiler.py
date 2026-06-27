from importlib import resources

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.analysis import MarsDataProfiler, profile_stats


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
    template_dir = resources.files("mars.reporting").joinpath("template")
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

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> "SpyNativeBinner":
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
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

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> "SpyNativeBinner":
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
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

        def fit(self, df: pl.DataFrame, features: list[str] | None = None) -> "SparseNativeBinner":
            self.features = list(features or [])
            self.bin_cuts_ = {feature: [] for feature in self.features}
            return self

        def transform(
            self,
            df: pl.DataFrame,
            *,
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
        sample_frac=0.5,
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
