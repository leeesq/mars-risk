import pandas as pd
import polars as pl
import pytest

import mars
import mars.monitoring as monitoring_module
from mars.feature import MarsLiteOptBinner, MarsNativeBinner
from mars.monitoring import (
    MarsMonitor,
    MarsMonitoringAlertConfig,
    MarsMonitoringAlerter,
    MarsMonitoringReport,
    generate_monitoring_alert,
)


def _trend_value_columns(table: pl.DataFrame | pd.DataFrame) -> list[str]:
    metadata_cols = {"feature", "dtype", "bin_index", "bin_label", "bin_type", "Total"}
    return [col for col in table.columns if col not in metadata_cols]


def test_monitoring_public_exports() -> None:
    assert "MarsMonitor" in monitoring_module.__all__
    assert "MarsMonitoringReport" in monitoring_module.__all__
    assert "MarsMonitoringAlerter" in monitoring_module.__all__
    assert "generate_monitoring_alert" in monitoring_module.__all__
    assert "MarsMonitor" in mars.__all__
    assert "generate_monitoring_alert" in mars.__all__


def _make_partial_target_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "month": [
                "2024-01",
                "2024-01",
                "2024-01",
                "2024-01",
                "2024-02",
                "2024-02",
                "2024-02",
                "2024-02",
                "2024-03",
                "2024-03",
                "2024-03",
                "2024-03",
            ],
            "score": [0.10, 0.20, 0.82, 0.91, 0.18, 0.24, 0.72, 0.88, 0.16, 0.30, 0.76, 0.93],
            "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58, None, None, 0.56, 0.63],
            "target": [0, 0, 1, 1, False, 0, True, 1, None, None, None, None],
        }
    )


def test_monitor_handles_unobserved_latest_target_values() -> None:
    df = _make_partial_target_df()
    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
    ).monitor(
        df,
        features=["score", "utilization"],
        target="target",
        group_col="month",
    )

    assert isinstance(report, MarsMonitoringReport)
    assert report.features == ["score", "utilization"]
    assert report.target == "target"
    assert "bad_rate" in report.bin_stat_trend_tables
    assert {"mean", "min", "max", "median"}.issubset(report.bin_stat_table.columns)

    group_col = report.metadata["group_col"]
    latest_detail = report.detail_table.filter(
        (pl.col(group_col) == "2024-03")
        & (pl.col("feature") == "score")
        & (pl.col("bin_index") != 9999)
    )
    assert latest_detail.select(pl.col("count").sum()).item() == 4
    assert latest_detail.select(pl.col("observed_count").sum()).item() == 0
    assert latest_detail.select(pl.col("bad_rate").is_null().all()).item()

    target_observation = report.target_observation_table
    assert target_observation is not None
    latest_observation = target_observation.filter(pl.col(group_col) == "2024-03")
    assert latest_observation.select("target_observed_count").item() == 0
    assert latest_observation.select("target_unobserved_count").item() == 4
    assert latest_observation.select("target_observed_rate").item() == 0
    assert latest_observation.select(pl.col("bad_rate_observed").is_null()).item()


def test_monitor_supports_lite_opt_binning_with_observed_target_subset() -> None:
    df = _make_partial_target_df()

    report = MarsMonitor(
        binning_type="lite_opt",
        binner_params={
            "n_bins": 2,
            "n_prebins": 6,
            "monotonic_trend": "ascending",
        },
    ).monitor(
        df,
        features=["score"],
        target="target",
        group_col="month",
    )

    assert isinstance(report.binner, MarsLiteOptBinner)
    assert report.target_observation_table is not None
    assert report.target_observation_table.filter(
        pl.col(report.metadata["group_col"]) == "2024-03"
    ).select("target_observed_rate").item() == 0
    assert report.binner.fitted_trends_["score"] == "ascending"


def test_monitor_supports_month_grain_alias_without_group_warning(caplog) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    score = [idx / 40 for idx in range(40)]
    df = pl.DataFrame(
        {
            "apply_dt": dates.strftime("%Y-%m-%d").to_list(),
            "score": score,
            "target": [int(value > 0.5) for value in score],
        }
    )

    with caplog.at_level("WARNING", logger="mars"):
        report = MarsMonitor(
            binner_params={"method": "quantile", "n_bins": 2},
        ).monitor(
            df,
            features=["score"],
            target="target",
            time_col="apply_dt",
            time_grain="1m",
    )

    assert report.metadata["trend_column_order"] == "asc"
    assert _trend_value_columns(report.trend_tables["psi"]) == ["202401", "202402"]
    assert report.trend_tables["psi"].height == 1
    assert not caplog.messages


def test_monitor_preserves_pandas_output_contract(sample_credit_df: pl.DataFrame) -> None:
    df = sample_credit_df.select(["month", "income", "utilization", "target"]).to_pandas()

    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 3},
    ).monitor(
        df,
        features=["income", "utilization"],
        target="target",
        group_col="month",
    )

    assert isinstance(report.summary_table, pd.DataFrame)
    assert isinstance(report.detail_table, pd.DataFrame)
    assert isinstance(report.bin_stat_table, pd.DataFrame)


def test_monitor_rejects_dirty_target_values() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "score": [0.1, 0.8, 0.2, 0.9],
            "target": ["0", "1", "pending", None],
        }
    )

    with pytest.raises(ValueError, match="0/1/True/False/null"):
        MarsMonitor(binner_params={"method": "quantile", "n_bins": 2}).monitor(
            df,
            features=["score"],
            target="target",
            group_col="month",
        )


def test_monitor_without_target_returns_distribution_only_metrics(sample_credit_df: pl.DataFrame) -> None:
    monitor_df = sample_credit_df.select(["month", "income", "utilization"])

    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 3},
    ).monitor(
        monitor_df,
        features=["income", "utilization"],
        target=None,
        group_col="month",
    )

    assert report.target is None
    assert report.target_observation_table is None
    assert report.summary_table.select(pl.col("iv").is_null().all()).item()
    assert report.detail_table.select(pl.col("observed_count").is_null().all()).item()
    assert "psi" in report.trend_tables


def test_monitor_supports_categorical_features() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "segment": ["new", "vip", "new", "repeat"],
            "target": [0, 1, 0, 1],
        }
    )

    report = MarsMonitor().monitor(
        df,
        features=["segment"],
        target="target",
        group_col="month",
    )

    assert report.bin_stat_table.select(pl.col("count").sum()).item() == 4
    assert report.bin_stat_table.select(pl.col("bad_rate").is_not_null().any()).item()
    assert report.bin_stat_table.select(pl.col("mean").is_null().all()).item()


def test_monitor_psi_include_missing_changes_psi_scope() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01"] * 6 + ["2024-02"] * 6,
            "score": [None, 0.1, 0.2, 0.8, 0.9, 1.0, None, None, None, 0.8, 0.9, 1.0],
            "target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    base_report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
        psi_include_missing=False,
    ).monitor(
        df,
        features=["score"],
        target="target",
        group_col="month",
    )
    missing_report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
        psi_include_missing=True,
    ).monitor(
        df,
        features=["score"],
        target="target",
        group_col="month",
    )

    base_psi = base_report.summary_table.select("psi_max").item()
    missing_psi = missing_report.summary_table.select("psi_max").item()
    assert missing_psi != pytest.approx(base_psi)


def test_monitor_method_level_psi_scope_overrides_instance_default() -> None:
    df = pl.DataFrame(
        {
            "month": ["2024-01"] * 6 + ["2024-02"] * 6,
            "score": [None, 0.1, 0.2, 0.8, 0.9, 1.0, None, None, None, 0.8, 0.9, 1.0],
            "target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
        psi_include_missing=False,
        psi_include_special=False,
    ).monitor(
        df,
        features=["score"],
        target="target",
        group_col="month",
        psi_include_missing=True,
        psi_include_special=True,
    )

    assert report.metadata["psi_include_missing"] is True
    assert report.metadata["psi_include_special"] is True


def test_monitor_trend_column_order_defaults_to_ascending() -> None:
    df = _make_partial_target_df()

    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
    ).monitor(
        df,
        features=["score", "utilization"],
        target="target",
        group_col="month",
    )

    assert report.metadata["trend_column_order"] == "asc"
    assert report.metadata["trend_value_columns"] == ["2024-01", "2024-02", "2024-03"]
    assert _trend_value_columns(report.trend_tables["psi"]) == ["2024-01", "2024-02", "2024-03"]
    assert _trend_value_columns(report.bin_stat_trend_tables["pct"]) == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]


def test_monitor_trend_column_order_desc_reorders_trend_tables() -> None:
    df = _make_partial_target_df()

    report = MarsMonitor(
        binner_params={"method": "quantile", "n_bins": 2},
    ).monitor(
        df,
        features=["score", "utilization"],
        target="target",
        group_col="month",
        trend_column_order="desc",
    )

    assert report.metadata["trend_column_order"] == "desc"
    assert report.metadata["trend_value_columns"] == ["2024-03", "2024-02", "2024-01"]
    assert _trend_value_columns(report.trend_tables["psi"]) == ["2024-03", "2024-02", "2024-01"]
    assert report.trend_tables["psi"].columns[-1] == "Total"
    assert _trend_value_columns(report.bin_stat_trend_tables["pct"]) == [
        "2024-03",
        "2024-02",
        "2024-01",
    ]


def test_monitor_rejects_invalid_trend_column_order() -> None:
    df = _make_partial_target_df()

    with pytest.raises(ValueError, match="trend_column_order"):
        MarsMonitor(
            binner_params={"method": "quantile", "n_bins": 2},
        ).monitor(
            df,
            features=["score"],
            target="target",
            group_col="month",
            trend_column_order="latest",
        )


def _make_alert_report() -> MarsMonitoringReport:
    summary = pl.DataFrame(
        {
            "feature": ["score", "utilization"],
            "psi_max": [0.30, 0.12],
            "missing_min": [0.00, 0.00],
            "missing_max": [0.12, 0.04],
            "rc_min": [0.50, 0.75],
        }
    )
    pct_trend = pl.DataFrame(
        {
            "feature": ["score", "score", "utilization"],
            "bin_index": [0, 1, 0],
            "bin_label": ["low", "high", "low"],
            "bin_type": ["正常组", "正常组", "正常组"],
            "dtype": ["Float64", "Float64", "Float64"],
            "2024-01": [0.80, 0.20, 0.50],
            "2024-02": [0.20, 0.80, 0.62],
        }
    )
    mean_trend = pl.DataFrame(
        {
            "feature": ["score", "score"],
            "bin_index": [0, 1],
            "bin_label": ["low", "high"],
            "bin_type": ["正常组", "正常组"],
            "dtype": ["Float64", "Float64"],
            "2024-01": [0.20, 0.80],
            "2024-02": [0.20, 0.80],
        }
    )
    bad_rate_trend = pl.DataFrame(
        {
            "feature": ["score", "utilization"],
            "bin_index": [0, 0],
            "bin_label": ["low", "low"],
            "bin_type": ["正常组", "正常组"],
            "dtype": ["Float64", "Float64"],
            "2024-01": [0.10, 0.20],
            "2024-02": [0.24, 0.25],
        }
    )
    target_observation = pl.DataFrame(
        {
            "month": ["Total", "2024-01", "2024-02"],
            "sample_count": [200, 100, 100],
            "target_observed_count": [140, 100, 40],
            "target_unobserved_count": [60, 0, 60],
            "bad": [50, 30, 20],
            "target_observed_rate": [0.70, 1.00, 0.40],
            "bad_rate_observed": [0.36, 0.30, 0.50],
        }
    )
    return MarsMonitoringReport(
        summary_table=summary,
        detail_table=pl.DataFrame(),
        trend_tables={
            "psi": pl.DataFrame(
                {
                    "feature": ["score", "utilization"],
                    "dtype": ["Float64", "Float64"],
                    "2024-01": [0.0, 0.0],
                    "2024-02": [0.30, 0.12],
                    "Total": [0.30, 0.12],
                }
            ),
            "missing": pl.DataFrame(
                {
                    "feature": ["score", "utilization"],
                    "dtype": ["Float64", "Float64"],
                    "2024-01": [0.0, 0.0],
                    "2024-02": [0.12, 0.04],
                    "Total": [0.12, 0.04],
                }
            ),
            "bad_rate": pl.DataFrame(
                {
                    "feature": ["score", "utilization"],
                    "dtype": ["Float64", "Float64"],
                    "2024-01": [0.10, 0.20],
                    "2024-02": [0.24, 0.25],
                    "Total": [0.17, 0.22],
                }
            ),
        },
        missing_by_day_table=None,
        bin_stat_table=pl.DataFrame(),
        bin_stat_trend_tables={
            "pct": pct_trend,
            "mean": mean_trend,
            "bad_rate": bad_rate_trend,
        },
        target_observation_table=target_observation,
        binner=MarsNativeBinner(),
        features=["score", "utilization"],
        target="target",
        metadata={"group_col": "month"},
    )


def _make_desc_order_alert_report(trend_column_order: str) -> MarsMonitoringReport:
    if trend_column_order == "desc":
        value_columns = ["2024-10", "2024-2"]
    else:
        value_columns = ["2024-2", "2024-10"]

    pct_trend = pl.DataFrame(
        {
            "feature": ["score", "score"],
            "bin_index": [0, 1],
            "bin_label": ["low", "high"],
            "bin_type": ["正常组", "正常组"],
            "dtype": ["Float64", "Float64"],
            "2024-10": [0.20, 0.80],
            "2024-2": [0.80, 0.20],
        }
    ).select(["feature", "bin_index", "bin_label", "bin_type", "dtype", *value_columns])
    mean_trend = pl.DataFrame(
        {
            "feature": ["score", "score"],
            "bin_index": [0, 1],
            "bin_label": ["low", "high"],
            "bin_type": ["正常组", "正常组"],
            "dtype": ["Float64", "Float64"],
            "2024-10": [0.20, 0.80],
            "2024-2": [0.20, 0.80],
        }
    ).select(["feature", "bin_index", "bin_label", "bin_type", "dtype", *value_columns])
    target_observation = pl.DataFrame(
        {
            "month": ["Total", "2024-2", "2024-10"],
            "sample_count": [300, 100, 200],
            "target_observed_count": [160, 40, 120],
            "target_unobserved_count": [140, 60, 80],
            "bad": [48, 16, 32],
            "target_observed_rate": [0.53, 0.40, 0.60],
            "bad_rate_observed": [0.30, 0.40, 0.27],
        }
    )
    return MarsMonitoringReport(
        summary_table=pl.DataFrame({"feature": ["score"], "psi_max": [0.0]}),
        detail_table=pl.DataFrame(),
        trend_tables={},
        missing_by_day_table=None,
        bin_stat_table=pl.DataFrame(),
        bin_stat_trend_tables={"pct": pct_trend, "mean": mean_trend},
        target_observation_table=target_observation,
        binner=MarsNativeBinner(),
        features=["score"],
        target="target",
        metadata={
            "group_col": "month",
            "trend_column_order": trend_column_order,
            "trend_value_columns": value_columns,
        },
    )


def test_generate_monitoring_alert_orders_text_by_priority() -> None:
    report = _make_alert_report()

    text = generate_monitoring_alert(
        report,
        score_key="score",
        model_features=["utilization"],
    )

    assert "MARS 监控报警摘要" in text
    assert text.index("\n严重：") < text.index("\n警告：")
    assert "模型分 `score` PSI" in text
    assert "分箱占比变化" in text
    assert "target 表现覆盖率" in text


def test_monitoring_alerter_class_matches_function_entry() -> None:
    report = _make_alert_report()

    function_text = generate_monitoring_alert(
        report,
        score_key="score",
        model_features=["utilization"],
    )
    class_text = MarsMonitoringAlerter().generate(
        report,
        score_key="score",
        model_features=["utilization"],
    )

    assert function_text == class_text


def test_monitoring_alert_skips_missing_tables_and_features() -> None:
    report = _make_alert_report()
    report.bin_stat_trend_tables.pop("mean")
    report.trend_tables = {}

    text = generate_monitoring_alert(
        report,
        score_key="missing_score",
        model_features=["utilization"],
    )

    assert "数据跳过" in text
    assert "missing_score" in text
    assert "缺少 pct 或 mean" in text


def test_monitoring_alert_config_changes_severity() -> None:
    report = _make_alert_report()
    config = MarsMonitoringAlertConfig(psi_critical=0.10)

    text = generate_monitoring_alert(
        report,
        score_key="score",
        model_features=["utilization"],
        config=config,
    )

    assert "严重" in text
    assert "特征 `utilization` PSI 最大值达到 0.1200" in text


def test_monitoring_alert_uses_report_trend_order_for_latest_group() -> None:
    config = MarsMonitoringAlertConfig(score_mean_relative_delta_warn=0.01)

    for trend_column_order in ["asc", "desc"]:
        report = _make_desc_order_alert_report(trend_column_order)
        text = generate_monitoring_alert(
            report,
            score_key="score",
            model_features=[],
            config=config,
        )

        assert "基准=0.3200，最新=0.6800" in text
        assert "最新分组 `2024-10` target 表现覆盖率为 0.6000" in text
