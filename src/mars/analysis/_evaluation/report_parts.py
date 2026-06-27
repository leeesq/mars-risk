"""分箱评估报告局部表构造。"""

from __future__ import annotations

import polars as pl

from mars.compute import (
    bad_rate_agg_expr,
    bin_missing_rate_expr,
    normalized_auc_expr,
    observed_auc_agg_expr,
    observed_iv_agg_expr,
    observed_ks_agg_expr,
    observed_lift_max_agg_expr,
    observed_sum_agg_expr,
)

TREND_METRICS: tuple[str, ...] = (
    "psi",
    "auc",
    "ks",
    "iv",
    "missing",
    "lift",
    "bad_rate",
    "risk_corr",
)


def _sort_trend_columns(pivot_df: pl.DataFrame) -> pl.DataFrame:
    """把 Total 趋势列固定到最右侧。"""
    cols: list[str] = [col for col in pivot_df.columns if col not in ["feature", "dtype"]]
    sorted_cols: list[str] = sorted([col for col in cols if col != "Total"])
    if "Total" in cols:
        sorted_cols.append("Total")
    return pivot_df.select(["feature", "dtype", *sorted_cols])


def _aggregate_metric_source(
    stats_long: pl.DataFrame,
    *,
    group_col: str,
    metric: str,
    missing_bin_index: int,
) -> pl.DataFrame:
    """按趋势指标选择分箱长表聚合表达式。"""
    if metric == "bad_rate":
        agg_expr = bad_rate_agg_expr()
    elif metric == "missing":
        agg_expr = bin_missing_rate_expr(missing_bin_index=missing_bin_index)
    elif metric == "lift":
        agg_expr = observed_lift_max_agg_expr(output_col=metric)
    elif metric == "ks":
        agg_expr = observed_ks_agg_expr(output_col=metric)
    elif metric == "auc":
        agg_expr = observed_auc_agg_expr(output_col=metric)
    elif metric == "iv":
        agg_expr = observed_iv_agg_expr(output_col=metric)
    else:
        agg_expr = observed_sum_agg_expr(f"{metric}_bin", output_col=metric)

    pivot_src: pl.DataFrame = (
        stats_long
        .group_by([group_col, "feature"])
        .agg(agg_expr.alias(metric))
    )
    if metric == "auc":
        pivot_src = pivot_src.with_columns(
            normalized_auc_expr(auc_col=metric, output_col=metric)
        )
    return pivot_src


def _build_psi_trend_source(
    *,
    monitor_group_level_metrics: pl.DataFrame,
    monitoring_total: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """合并分组和 Total 的 PSI 趋势来源。"""
    psi_group_src: pl.DataFrame = monitor_group_level_metrics.filter(pl.col(group_col) != "Total")
    psi_total_src: pl.DataFrame = (
        monitoring_total
        .group_by(["feature", group_col])
        .agg(pl.col("psi_bin").sum().alias("psi"))
    )
    return pl.concat(
        [
            psi_group_src.select(["feature", group_col, "psi"]),
            psi_total_src.select(["feature", group_col, "psi"]),
        ],
        how="vertical_relaxed",
    )


def build_binning_trend_tables(
    *,
    stats_long: pl.DataFrame,
    risk_corr_long: pl.DataFrame,
    monitor_group_level_metrics: pl.DataFrame,
    monitoring_total: pl.DataFrame,
    group_col: str,
    missing_bin_index: int,
) -> dict[str, pl.DataFrame]:
    """构造分箱报告趋势表。

    Parameters
    ----------
    stats_long : pl.DataFrame
        分箱明细长表。
    risk_corr_long : pl.DataFrame
        分组级 RC 长表。
    monitor_group_level_metrics : pl.DataFrame
        分组级监控指标表。
    monitoring_total : pl.DataFrame
        Total 维度监控指标表。
    group_col : str
        趋势分组列。
    missing_bin_index : int
        缺失箱索引。

    Returns
    -------
    dict[str, pl.DataFrame]
        指标名到趋势透视表的映射。
    """
    trend_tables: dict[str, pl.DataFrame] = {}
    for metric in TREND_METRICS:
        if metric == "risk_corr":
            pivot_src = risk_corr_long
        elif metric == "psi":
            pivot_src = _build_psi_trend_source(
                monitor_group_level_metrics=monitor_group_level_metrics,
                monitoring_total=monitoring_total,
                group_col=group_col,
            )
        else:
            pivot_src = _aggregate_metric_source(
                stats_long,
                group_col=group_col,
                metric=metric,
                missing_bin_index=missing_bin_index,
            )

        pivot_df: pl.DataFrame = (
            pivot_src
            .pivot(index="feature", on=group_col, values=metric)
            .sort("feature")
            .with_columns(pl.lit("Float64").alias("dtype"))
        )
        trend_tables[metric] = _sort_trend_columns(pivot_df)
    return trend_tables
