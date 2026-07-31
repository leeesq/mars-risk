"""分箱评估报告局部表构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from mars._compat import polars_is_in
from mars.analysis._evaluation.references import build_risk_corr_long
from mars.compute import (
    amount_distribution_exprs,
    amount_metric_exprs,
    bad_rate_agg_expr,
    bad_rate_expr,
    bin_missing_rate_expr,
    distribution_rate_expr,
    normalized_auc_expr,
    observed_auc_agg_expr,
    observed_iv_agg_expr,
    observed_ks_agg_expr,
    observed_lift_max_agg_expr,
    observed_lift_min_agg_expr,
    observed_sum_agg_expr,
    ordered_count_metric_exprs,
)
from mars.feature.binning.base import MarsBinnerBase

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


@dataclass(frozen=True)
class BinningReportParts:
    """保存报告对象创建前的 Polars 中间表。"""

    summary_table: pl.DataFrame
    detail_table: pl.DataFrame
    trend_tables: dict[str, pl.DataFrame]
    risk_corr_reference_table: pl.DataFrame


def build_bin_label_map(
    stats_long: pl.DataFrame,
    *,
    binner: MarsBinnerBase,
) -> pl.DataFrame:
    """构建明细报告使用的特征和分箱索引到标签的映射。"""
    map_rows: list[dict[str, Any]] = []
    features = set(stats_long["feature"].unique().to_list())

    for feature, mapping in binner.bin_mappings_.items():
        if feature not in features:
            continue

        for bin_index, label in mapping.items():
            try:
                map_rows.append(
                    {
                        "feature": feature,
                        "bin_index": int(bin_index),
                        "bin_label": str(label),
                    }
                )
            except (ValueError, TypeError):
                continue

    map_schema = {
        "feature": pl.String,
        "bin_index": pl.Int16,
        "bin_label": pl.String,
    }
    if not map_rows:
        return pl.DataFrame([], schema=map_schema)
    return pl.DataFrame(map_rows, schema=map_schema)


def _build_detail_base(
    *,
    stats_long: pl.DataFrame,
    metrics_total: pl.DataFrame,
    group_col: str,
    binner: MarsBinnerBase,
) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    """补充分箱标签、趋势方向和明细排序键。"""
    map_df = build_bin_label_map(stats_long, binner=binner)
    detail_base = (
        stats_long
        .join(map_df, on=["feature", "bin_index"], how="left")
        .with_columns(pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)))
    )
    amount_detail_cols = (
        ["tot_amt", "good_amt", "bad_amt", "avg_amt", "amt_bad_rate", "lift_amt"]
        if {"tot_amt", "good_amt", "bad_amt"}.issubset(detail_base.columns)
        else []
    )

    trend_source = (
        metrics_total
        .lazy()
        .filter(pl.col("bin_index") >= 0)
        .sort(["feature", "bin_index"])
        .select(["feature", "woe"])
    )
    trend_shape_df = MarsBinnerBase._build_trend_shape_frame(
        trend_source.group_by("feature").agg(pl.col("woe")).collect(),
        trend_col_name="trend",
    )
    detail_base = detail_base.join(trend_shape_df, on="feature", how="left")

    detail_table = (
        detail_base
        .with_columns([
            pl.when(pl.col("bin_index") >= 0)
            .then(0)
            .otherwise(1)
            .cast(pl.Int32)
            .alias("_sort_group"),
            pl.when(pl.col("bin_index") >= 0)
            .then(pl.col("bin_index").cast(pl.Int32))
            .when(pl.col("bin_index") == -1)
            .then(10000)
            .when(pl.col("bin_index") == -2)
            .then(10001)
            .otherwise(20000 + pl.col("bin_index").abs().cast(pl.Int32))
            .alias("_sort_idx"),
        ])
        .sort(["feature", group_col, "_sort_group", "_sort_idx"])
    )
    return detail_table, amount_detail_cols, trend_shape_df


def _append_detail_derived_columns(
    detail_table: pl.DataFrame,
    *,
    group_col: str,
) -> pl.DataFrame:
    """补充明细表展示需要的累计列、占比和箱类型。"""
    return (
        detail_table
        .with_columns([
            *ordered_count_metric_exprs(["feature", group_col]),
            (pl.col("observed_count") - pl.col("bad"))
            .cum_sum()
            .over(["feature", group_col])
            .alias("cum_good"),
            distribution_rate_expr(
                numerator_col="count",
                denominator_col="total_count",
                output_col="pct",
            ),
            pl.col("bin_index").max().over(["feature", group_col]).alias("bin_index_max"),
        ])
        .with_columns([
            pl.when((pl.col("bin_index") == pl.col("bin_index_max")) | (pl.col("bin_index") == 0))
            .then(pl.lit("首尾组"))
            .when(pl.col("bin_index") == -1)
            .then(pl.lit("空值组"))
            .when(pl.col("bin_index") == -2)
            .then(pl.lit("其他组"))
            .when(pl.col("bin_index") <= -3)
            .then(pl.lit("特殊组"))
            .otherwise(pl.lit("正常组"))
            .alias("bin_type")
        ])
    )


def _build_total_detail_rows(
    *,
    stats_long: pl.DataFrame,
    group_col: str,
    amount_detail_cols: list[str],
    trend_shape_df: pl.DataFrame,
) -> pl.DataFrame:
    """为每个 feature/group 构造用于展示的 Total 行。"""
    total_rows = (
        stats_long
        .group_by(["feature", group_col])
        .agg([
            pl.col("count").sum().alias("count"),
            pl.col("observed_count").sum().alias("observed_count"),
            pl.col("bad").sum().alias("bad"),
            pl.col("iv_bin").sum().alias("iv_bin"),
            pl.col("psi_bin").sum().alias("psi_bin"),
            pl.col("auc_bin").sum().alias("auc_bin"),
            pl.col("ks_bin").max().alias("ks_bin"),
            pl.col("lift").max().alias("lift"),
            pl.col("count").sum().alias("total_count"),
        ])
        .with_columns([
            (pl.col("observed_count") - pl.col("bad")).alias("good"),
            bad_rate_expr(),
            pl.lit(1.0).alias("pct"),
            pl.col("count").alias("cum_count"),
            pl.col("observed_count").alias("cum_observed_count"),
            pl.col("bad").alias("cum_bad"),
            bad_rate_expr(output_col="cum_bad_rate"),
            normalized_auc_expr(auc_col="auc_bin", output_col="auc_bin"),
            pl.lit(9999).cast(pl.Int16).alias("bin_index"),
            pl.lit("Total").alias("bin_label"),
            pl.lit("汇总组").alias("bin_type"),
            pl.lit(2).cast(pl.Int32).alias("_sort_group"),
            pl.lit(0).cast(pl.Int32).alias("_sort_idx"),
        ])
    )

    if amount_detail_cols:
        amount_totals = (
            stats_long
            .group_by(["feature", group_col])
            .agg([
                pl.col("count").sum().alias("count"),
                pl.col("tot_amt").sum().alias("tot_amt"),
                pl.col("good_amt").sum().alias("good_amt"),
                pl.col("bad_amt").sum().alias("bad_amt"),
            ])
            .with_columns(amount_distribution_exprs(["feature", group_col]))
            .with_columns(amount_metric_exprs())
            .select(["feature", group_col, *amount_detail_cols])
        )
        total_rows = total_rows.join(amount_totals, on=["feature", group_col], how="left")
    return total_rows.join(trend_shape_df, on="feature", how="left")


def build_binning_detail_table(
    *,
    stats_long: pl.DataFrame,
    metrics_total: pl.DataFrame,
    group_col: str,
    binner: MarsBinnerBase,
    target_name: str,
    feature_source_map: dict[str, str] | None,
) -> pl.DataFrame:
    """构造包含普通箱、特殊箱和 Total 行的分箱明细表。"""
    detail_table, amount_detail_cols, trend_shape_df = _build_detail_base(
        stats_long=stats_long,
        metrics_total=metrics_total,
        group_col=group_col,
        binner=binner,
    )
    detail_table = _append_detail_derived_columns(detail_table, group_col=group_col)
    total_rows = _build_total_detail_rows(
        stats_long=stats_long,
        group_col=group_col,
        amount_detail_cols=amount_detail_cols,
        trend_shape_df=trend_shape_df,
    )

    detail_cols = [
        "feature",
        group_col,
        "bin_index",
        "bin_label",
        "_sort_group",
        "_sort_idx",
        "count",
        "observed_count",
        "pct",
        "bad",
        "good",
        "bad_rate",
        "lift",
        "trend",
        "cum_count",
        "cum_observed_count",
        "cum_bad",
        "cum_bad_rate",
        "psi_bin",
        "ks_bin",
        "auc_bin",
        "iv_bin",
        "total_count",
        "bin_type",
    ]
    if amount_detail_cols:
        detail_cols.extend(amount_detail_cols)

    detail_table = (
        pl.concat([
            detail_table.select(detail_cols),
            total_rows.select(detail_cols),
        ])
        .sort(["feature", group_col, "_sort_group", "_sort_idx"])
        .select([
            pl.lit(target_name).alias("y"),
            "feature",
            "trend",
            group_col,
            "bin_index",
            "bin_label",
            "count",
            "observed_count",
            "bad",
            "good",
            "pct",
            "bad_rate",
            "lift",
            "cum_count",
            "cum_observed_count",
            "cum_bad",
            "cum_bad_rate",
            "psi_bin",
            "ks_bin",
            "auc_bin",
            "iv_bin",
            "total_count",
            "bin_type",
            *amount_detail_cols,
        ])
    )

    if not feature_source_map:
        return detail_table

    source_df = pl.DataFrame({
        "feature": list(feature_source_map.keys()),
        "data_source": [feature_source_map[feature] for feature in feature_source_map],
    })
    return detail_table.join(source_df, on="feature", how="left").with_columns(
        pl.col("data_source").fill_null("UNMAPPED")
    )


def _merge_feature_frame(
    default_df: pl.DataFrame,
    override_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """按 feature 维度用覆盖表替换默认表中的同名特征记录。"""
    if override_df is None or override_df.is_empty():
        return default_df

    override_features = override_df.get_column("feature").unique().to_list()
    retained_default = default_df.filter(
        ~polars_is_in(pl.col("feature"), pl.Series(override_features))
    )
    return pl.concat([retained_default, override_df], how="vertical_relaxed")


def build_binning_summary_table(
    *,
    stats_long: pl.DataFrame,
    metrics_groups: pl.DataFrame,
    metrics_total: pl.DataFrame,
    monotonicity_df: pl.DataFrame,
    group_col: str,
    risk_corr_long: pl.DataFrame,
    feature_source_map: dict[str, str] | None,
    monitor_metrics_groups: pl.DataFrame | None,
    monitor_metrics_total: pl.DataFrame | None,
    missing_bin_index: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """构造 summary 表和趋势表所需的监控指标来源。"""
    monitoring_groups = _merge_feature_frame(metrics_groups, monitor_metrics_groups)
    monitoring_total = _merge_feature_frame(metrics_total, monitor_metrics_total)
    group_level_metrics = (
        metrics_groups
        .group_by(["feature", group_col])
        .agg([
            observed_iv_agg_expr(),
            observed_auc_agg_expr(),
            bin_missing_rate_expr(missing_bin_index=missing_bin_index),
            observed_lift_max_agg_expr(),
        ])
        .with_columns(normalized_auc_expr())
    )
    monitor_group_level_metrics = (
        monitoring_groups
        .group_by(["feature", group_col])
        .agg(pl.col("psi_bin").sum().alias("psi"))
    )
    group_level_metrics = (
        group_level_metrics
        .join(monitor_group_level_metrics, on=["feature", group_col], how="left")
        .join(risk_corr_long, on=["feature", group_col], how="left")
    )

    total_missing_metrics = (
        stats_long
        .group_by("feature")
        .agg(bin_missing_rate_expr(missing_bin_index=missing_bin_index))
    )
    total_real_bin_lift_metrics = (
        metrics_total
        .filter(pl.col("bin_index") >= 0)
        .group_by("feature")
        .agg([
            observed_lift_min_agg_expr(output_col="lift_min"),
            observed_lift_max_agg_expr(output_col="lift_max"),
        ])
    )
    total_metrics_agg = (
        metrics_total
        .group_by("feature")
        .agg([
            observed_iv_agg_expr(),
            observed_ks_agg_expr(),
            observed_auc_agg_expr(),
        ])
        .with_columns(normalized_auc_expr())
    )

    if not group_level_metrics.is_empty():
        summary_audit = (
            group_level_metrics
            .group_by("feature")
            .agg([
                pl.col("psi").max().fill_null(0.0).alias("psi_max"),
                pl.col("risk_corr").min().fill_null(1.0).alias("rc_min"),
                pl.col("missing").min().alias("missing_min"),
                pl.col("missing").max().alias("missing_max"),
            ])
        )
    else:
        summary_audit = pl.DataFrame({
            "feature": total_metrics_agg["feature"],
            "psi_max": [0.0] * len(total_metrics_agg),
            "rc_min": [1.0] * len(total_metrics_agg),
            "missing_min": [0.0] * len(total_metrics_agg),
            "missing_max": [0.0] * len(total_metrics_agg),
        })

    summary_df = (
        total_metrics_agg
        .join(summary_audit, on="feature", how="left")
        .join(total_missing_metrics, on="feature", how="left")
        .join(total_real_bin_lift_metrics, on="feature", how="left")
        .join(monotonicity_df, on="feature", how="left")
        .with_columns([
            pl.col("psi_max").fill_null(0.0),
            pl.col("rc_min").fill_null(1.0),
            pl.col("missing").fill_null(0.0),
            pl.col("missing_min").fill_null(0.0),
            pl.col("missing_max").fill_null(0.0),
            pl.col("mono").fill_null(1.0),
        ])
        .sort(["iv", "rc_min"], descending=[True, True])
        .select([
            "feature",
            "iv",
            "ks",
            "auc",
            "psi_max",
            "rc_min",
            "lift_min",
            "lift_max",
            "missing",
            "missing_min",
            "missing_max",
            "mono",
        ])
    )

    if feature_source_map:
        source_df = pl.DataFrame({
            "feature": list(feature_source_map.keys()),
            "data_source": [feature_source_map[feature] for feature in feature_source_map],
        })
        summary_df = summary_df.join(source_df, on="feature", how="left").with_columns(
            pl.col("data_source").fill_null("UNMAPPED")
        )
        summary_df = summary_df.select(
            ["feature", "data_source"]
            + [col for col in summary_df.columns if col not in {"feature", "data_source"}]
        )
    return summary_df, monitor_group_level_metrics, monitoring_total


def build_binning_report_parts(
    *,
    stats_long: pl.DataFrame,
    metrics_groups: pl.DataFrame,
    metrics_total: pl.DataFrame,
    group_col: str,
    monotonicity_df: pl.DataFrame,
    binner: MarsBinnerBase,
    target_name: str,
    feature_source_map: dict[str, str] | None,
    risk_corr_reference_table: pl.DataFrame,
    monitor_metrics_groups: pl.DataFrame | None,
    monitor_metrics_total: pl.DataFrame | None,
) -> BinningReportParts:
    """构造分箱报告所需的 summary、detail、trend 和 RC reference 表。"""
    detail_table = build_binning_detail_table(
        stats_long=stats_long,
        metrics_total=metrics_total,
        group_col=group_col,
        binner=binner,
        target_name=target_name,
        feature_source_map=feature_source_map,
    )
    baseline_df = risk_corr_reference_table.select(["feature", "bin_index", "base_br"])
    monitoring_groups = _merge_feature_frame(metrics_groups, monitor_metrics_groups)
    monitoring_total = _merge_feature_frame(metrics_total, monitor_metrics_total)
    all_metrics_for_corr = pl.concat([
        monitoring_groups.select(["feature", group_col, "bin_index", "bad_rate", "observed_count"]),
        monitoring_total.select(["feature", group_col, "bin_index", "bad_rate", "observed_count"]),
    ])
    risk_corr_long = build_risk_corr_long(
        all_metrics_for_corr,
        baseline_df,
        group_col=group_col,
    )
    summary_table, monitor_group_level_metrics, monitoring_total = build_binning_summary_table(
        stats_long=stats_long,
        metrics_groups=metrics_groups,
        metrics_total=metrics_total,
        monotonicity_df=monotonicity_df,
        group_col=group_col,
        risk_corr_long=risk_corr_long,
        feature_source_map=feature_source_map,
        monitor_metrics_groups=monitor_metrics_groups,
        monitor_metrics_total=monitor_metrics_total,
        missing_bin_index=MarsBinnerBase.IDX_MISSING,
    )
    trend_tables = build_binning_trend_tables(
        stats_long=stats_long,
        risk_corr_long=risk_corr_long,
        monitor_group_level_metrics=monitor_group_level_metrics,
        monitoring_total=monitoring_total,
        group_col=group_col,
        missing_bin_index=MarsBinnerBase.IDX_MISSING,
    )
    return BinningReportParts(
        summary_table=summary_table,
        detail_table=detail_table,
        trend_tables=trend_tables,
        risk_corr_reference_table=risk_corr_reference_table,
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
