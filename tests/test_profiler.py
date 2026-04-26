from importlib import resources

import pandas as pd
import polars as pl

from mars.analysis import MarsDataProfiler


def test_profiler_returns_pandas_tables_for_pandas_input(sample_credit_pd):
    profiler = MarsDataProfiler(
        sample_credit_pd,
        missing_values=[-999],
        config=None,
    )

    report = profiler.generate_profile(
        profile_by="month",
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
        sample_credit_df,
        missing_values=[-999],
        config=None,
    )

    report = profiler.generate_profile(
        profile_by="month",
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
