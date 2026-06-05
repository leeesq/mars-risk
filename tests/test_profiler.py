from importlib import resources

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.analysis import MarsDataProfiler, profile_stats


def test_profiler_returns_pandas_tables_for_pandas_input(sample_credit_pd):
    profiler = MarsDataProfiler(
        missing_values=[-999],
        config=None,
    )

    report = profiler.generate_profile(
        sample_credit_pd,
        group_col="month",
        config_overrides={
            "dq_metrics": ["missing", "zeros"],
            "stat_metrics": ["mean", "psi"],
            "enable_sparkline": False,
        },
    )

    assert isinstance(report.overview_table, pd.DataFrame)
    assert isinstance(report.dq_tables["missing"], pd.DataFrame)
    assert isinstance(report.stats_tables["mean"], pd.DataFrame)
    assert not report.overview_table.empty


def test_evaluation_templates_are_packaged():
    linux_template = resources.files("mars.analysis").joinpath("mars_bin_report_linux.xlsx")
    win_mac_template = resources.files("mars.analysis").joinpath("mars_bin_report_win_mac.xlsx")

    assert linux_template.is_file()
    assert win_mac_template.is_file()


def test_profile_report_show_overview_does_not_mutate_polars_storage(sample_credit_df):
    profiler = MarsDataProfiler(
        missing_values=[-999],
        config=None,
    )

    report = profiler.generate_profile(
        sample_credit_df,
        group_col="month",
        config_overrides={
            "dq_metrics": ["missing", "zeros"],
            "stat_metrics": ["mean", "psi"],
            "enable_sparkline": False,
        },
    )

    styler = report.show_overview(features=["income"])

    assert isinstance(styler.data, pd.DataFrame)
    assert isinstance(report.overview_table, pl.DataFrame)
    assert "income" in report.overview_table["feature"].to_list()


def test_profiler_sparkline_works_when_polars_reports_single_worker(sample_credit_df, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pl, "thread_pool_size", lambda: 1)

    profiler = MarsDataProfiler(
        missing_values=[-999],
        config=None,
    )

    report = profiler.generate_profile(
        sample_credit_df,
        group_col="month",
        config_overrides={
            "dq_metrics": ["missing"],
            "stat_metrics": ["mean"],
            "enable_sparkline": True,
        },
    )

    assert not report.overview_table.is_empty()


def test_profile_stats_returns_lightweight_report_with_requested_metrics(sample_credit_df):
    report = profile_stats(
        sample_credit_df,
        metrics=["missing", "mean"],
        features=["income", "utilization"],
        group_col="month",
    )

    assert isinstance(report.overview_table, pl.DataFrame)
    assert "missing" in report.dq_tables
    assert "mean" in report.stats_tables
    assert "utilization" in report.overview_table["feature"].to_list()


def test_profile_stats_preserves_pandas_output_contract(sample_credit_pd):
    report = profile_stats(
        sample_credit_pd,
        metrics=["missing", "mean"],
        features=["income"],
        group_col="month",
    )

    assert isinstance(report.overview_table, pd.DataFrame)
    assert isinstance(report.dq_tables["missing"], pd.DataFrame)
    assert isinstance(report.stats_tables["mean"], pd.DataFrame)


def test_profiler_can_reuse_instance_across_dataframes_without_sample_state(sample_credit_pd):
    profiler = MarsDataProfiler(missing_values=[-999])
    first = profiler.generate_profile(
        sample_credit_pd,
        features=["income"],
        group_col="month",
        sample_frac=0.5,
        config_overrides={"enable_sparkline": False},
    )
    second_df = sample_credit_pd.rename(columns={"income": "debt"})
    second = profiler.generate_profile(
        second_df,
        features=["debt"],
        group_col="month",
        config_overrides={"enable_sparkline": False},
    )

    assert set(first.overview_table["feature"]) == {"income"}
    assert set(second.overview_table["feature"]) == {"debt"}
    assert profiler.features == []


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
        features=["stable", "drift", "zeros", "skew", "missing_feature"],
        group_col="month",
        config_overrides={
            "enable_sparkline": False,
            "dq_metrics": ["missing", "zeros", "unique", "top1"],
            "stat_metrics": ["mean", "min", "max", "skew", "psi"],
        },
    )

    assert set(report.dq_tables) == {"missing", "zeros", "unique", "top1"}
    assert {"mean", "min", "max", "skew", "psi"}.issubset(report.stats_tables)
    assert set(report.overview_table["feature"].to_list()) == {
        "stable",
        "drift",
        "zeros",
        "skew",
        "missing_feature",
    }
    assert "drift" in report.stats_tables["psi"]["feature"].to_list()
