from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mars.analysis import MarsBinEvaluator, profile_risk
from mars.feature import MarsNativeBinner


def test_evaluator_accepts_deprecated_bining_type_with_warning(sample_credit_df):
    with pytest.warns(FutureWarning, match="bining_type"):
        evaluator = MarsBinEvaluator(
            target="target",
            bining_type="native",
            method="quantile",
            n_bins=3,
        )

    report = evaluator.evaluate(sample_credit_df, features=["income", "utilization"], profile_by="month")

    assert evaluator.binning_type == "native"
    assert evaluator.bining_type == "native"
    assert report.summary_table.height == 2


def test_profile_risk_returns_report_and_evaluator_for_pandas_input(sample_credit_pd):
    report, evaluator = profile_risk(
        sample_credit_pd,
        target="target",
        features=["income", "utilization"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    assert isinstance(report.summary_table, pd.DataFrame)
    assert isinstance(evaluator.binner, MarsNativeBinner)
    assert set(report.summary_table["feature"]) == {"income", "utilization"}


def test_profile_risk_summary_is_consistent_between_polars_and_pandas(sample_credit_df, sample_credit_pd):
    report_pl, _ = profile_risk(
        sample_credit_df,
        target="target",
        features=["income", "utilization"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )
    report_pd, _ = profile_risk(
        sample_credit_pd,
        target="target",
        features=["income", "utilization"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    summary_pl = report_pl.summary_table.to_pandas().sort_values("feature").reset_index(drop=True)
    summary_pd = report_pd.summary_table.sort_values("feature").reset_index(drop=True)

    pd.testing.assert_frame_equal(summary_pl, summary_pd, check_dtype=False, check_like=False)


def test_profile_risk_multi_target_keeps_pandas_return_type(sample_credit_pd):
    df = sample_credit_pd.copy()
    df["target_alt"] = (df["utilization"] >= 0.45).astype(int)

    report, evaluator = profile_risk(
        df,
        target=["target", "target_alt"],
        features=["income", "utilization"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    assert isinstance(report.summary_table, pd.DataFrame)
    assert isinstance(report.detail_table, pd.DataFrame)
    assert set(report.summary_table["target"]) == {"target", "target_alt"}
    assert isinstance(evaluator.binner, MarsNativeBinner)


def test_profile_risk_without_target_returns_distribution_only_metrics(sample_credit_df):
    monitor_df = sample_credit_df.select(["month", "income", "utilization", "segment"])

    report, evaluator = profile_risk(
        monitor_df,
        target=None,
        features=["income", "utilization", "segment"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    summary = report.summary_table

    assert evaluator.has_target_ is False
    assert "psi" in report.trend_tables
    assert summary.select(pl.col("iv").is_null().all()).item()


def test_evaluation_report_can_write_excel(sample_credit_df, caplog):
    report, _ = profile_risk(
        sample_credit_df,
        target="target",
        features=["income"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    artifacts_dir = Path(__file__).resolve().parent / "_artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / "evaluation_report.xlsx"
    if output_path.exists():
        output_path.unlink()

    try:
        with caplog.at_level("INFO", logger="mars"):
            report.write_excel(str(output_path), engine="openpyxl")
        assert output_path.exists()
        assert any("导出成功" in message or "Exported" in message for message in caplog.messages)
    finally:
        if output_path.exists():
            output_path.unlink()
        if artifacts_dir.exists() and not any(artifacts_dir.iterdir()):
            artifacts_dir.rmdir()


def test_show_summary_uses_pandas_view_without_mutating_polars_report(sample_credit_df):
    report, _ = profile_risk(
        sample_credit_df,
        target="target",
        features=["income", "utilization"],
        profile_by="month",
        plot=False,
        binning_type="native",
        n_bins=3,
        binner_kwargs={"method": "quantile"},
    )

    styler = report.show_summary(features=["income"])

    assert isinstance(styler.data, pd.DataFrame)
    assert isinstance(report.summary_table, pl.DataFrame)
    assert set(report.summary_table["feature"].to_list()) == {"income", "utilization"}
