"""数据画像指标表达式与 overview 计算。"""

from __future__ import annotations

from typing import Any

import polars as pl

from mars._compat import polars_is_in
from mars.analysis._profiling.types import (
    COMPARISON_METRICS,
    DEFAULT_DQ_METRICS,
    DEFAULT_PROFILE_METRICS,
    DEFAULT_STAT_METRICS,
    ProfileComputeOptions,
    ProfileMetricSelection,
    ProfileRunContext,
)
from mars.compute import (
    filter_compatible_values,
    is_numeric_dtype,
    missing_rate_expr,
    values_to_exclude,
)


def normalize_profile_metrics(metrics: list[str] | None, *, require_metrics: bool) -> ProfileMetricSelection:
    """校验并拆分画像指标。"""
    if require_metrics and not metrics:
        raise ValueError("`metrics` must contain at least one metric name.")

    requested_metrics = list(metrics or DEFAULT_PROFILE_METRICS)
    dq_supported = set(DEFAULT_DQ_METRICS)
    stat_supported = set(DEFAULT_STAT_METRICS)

    dq_metrics: list[str] = []
    stat_metrics: list[str] = []
    unknown_metrics: list[str] = []
    comparison_metrics: list[str] = []
    for metric in requested_metrics:
        metric_name = str(metric)
        if metric_name in dq_supported:
            if metric_name not in dq_metrics:
                dq_metrics.append(metric_name)
        elif metric_name in stat_supported:
            if metric_name not in stat_metrics:
                stat_metrics.append(metric_name)
        elif metric_name in COMPARISON_METRICS:
            if metric_name not in comparison_metrics:
                comparison_metrics.append(metric_name)
        else:
            unknown_metrics.append(metric_name)

    if unknown_metrics:
        supported = sorted(dq_supported | stat_supported | set(COMPARISON_METRICS))
        raise ValueError(f"Unknown metrics: {unknown_metrics}. Supported metrics: {supported}")

    return ProfileMetricSelection(
        dq_metrics=dq_metrics,
        stat_metrics=stat_metrics,
        comparison_metrics=comparison_metrics,
    )


def is_numeric_feature(context: ProfileRunContext, col: str) -> bool:
    """判断字段是否为数值类型。"""
    return is_numeric_dtype(context.dtype_map.get(col))


def feature_dtypes(context: ProfileRunContext) -> pl.DataFrame:
    """返回字段类型表。"""
    rows = {
        "feature": list(context.dtype_map),
        "dtype": [str(dtype) for dtype in context.dtype_map.values()],
    }
    return pl.DataFrame(rows)


def valid_missing_values(context: ProfileRunContext, col: str, options: ProfileComputeOptions) -> list[Any]:
    """返回与字段类型兼容的自定义缺失值。"""
    return filter_compatible_values(context.dtype_map.get(col), options.missing_values)


def excluded_values(context: ProfileRunContext, col: str, options: ProfileComputeOptions) -> list[Any]:
    """返回统计指标计算时需要排除的缺失值和特殊值。"""
    return values_to_exclude(
        context.dtype_map.get(col),
        missing_values=options.missing_values,
        special_values=options.special_values,
    )


def metric_expr(context: ProfileRunContext, options: ProfileComputeOptions, col: str, metric: str) -> pl.Expr:
    """生成单字段单指标表达式。"""
    raw_col = pl.col(col)
    is_num = is_numeric_feature(context, col)
    col_dtype = context.dtype_map.get(col)

    if metric == "missing":
        return missing_rate_expr(raw_col, dtype=col_dtype, missing_values=options.missing_values)
    if metric == "zeros":
        return (raw_col == 0).sum() / pl.len() if is_num else pl.lit(0, dtype=pl.UInt32)
    if metric == "unique":
        return raw_col.n_unique() / pl.len()
    if metric == "mode":
        return raw_col.value_counts(sort=True).first().struct.field("count") / pl.len()

    if not is_num:
        return pl.lit(None, dtype=pl.Float64)

    keep_mask = raw_col.is_not_null()
    if col_dtype in [pl.Float32, pl.Float64]:
        keep_mask &= raw_col.is_not_nan()

    exclude_vals = excluded_values(context, col, options)
    if exclude_vals:
        keep_mask &= ~polars_is_in(raw_col, pl.Series(exclude_vals))

    clean_col = raw_col.filter(keep_mask)
    mapper: dict[str, pl.Expr] = {
        "mean": clean_col.mean(),
        "median": clean_col.median(),
        "sum": clean_col.sum(),
        "std": clean_col.std(),
        "min": clean_col.min(),
        "max": clean_col.max(),
        "p25": clean_col.quantile(0.25),
        "p75": clean_col.quantile(0.75),
        "skew": clean_col.skew(),
        "kurtosis": clean_col.kurtosis(),
    }
    return mapper.get(metric, pl.lit(None, dtype=pl.Float64))


def overview_exprs(
    context: ProfileRunContext,
    selection: ProfileMetricSelection,
    options: ProfileComputeOptions,
    col: str,
) -> list[pl.Expr]:
    """为 overview 生成单字段全量表达式。"""
    total_len = pl.len()
    exprs: list[pl.Expr] = []

    if "missing" in selection.dq_metrics:
        exprs.append(
            missing_rate_expr(
                col,
                dtype=context.dtype_map.get(col),
                missing_values=options.missing_values,
            ).alias("missing_rate")
        )
    if "zeros" in selection.dq_metrics:
        zeros_expr = (pl.col(col) == 0).sum() if is_numeric_feature(context, col) else pl.lit(0, dtype=pl.UInt32)
        exprs.append((zeros_expr / total_len).alias("zeros_rate"))
    if "unique" in selection.dq_metrics:
        unique_expr = pl.col(col).approx_n_unique() if context.df.height > 1_000_000 else pl.col(col).n_unique()
        exprs.append((unique_expr / total_len).alias("unique_rate"))
    if "mode" in selection.dq_metrics:
        mode_struct = pl.col(col).value_counts(sort=True).first()
        exprs.append((mode_struct.struct.field("count") / total_len).alias("mode_rate"))
        exprs.append(mode_struct.struct.field(col).cast(pl.Utf8).alias("mode_value"))

    if is_numeric_feature(context, col):
        for metric in selection.stat_metrics:
            if metric != "psi":
                exprs.append(metric_expr(context, options, col, metric).alias(metric))
    else:
        null_value = pl.lit(None, dtype=pl.Float64)
        for metric in selection.stat_metrics:
            if metric != "psi":
                exprs.append(null_value.alias(metric))
    return exprs


def analyze_columns(
    context: ProfileRunContext,
    selection: ProfileMetricSelection,
    options: ProfileComputeOptions,
) -> pl.DataFrame:
    """批量向量化计算 overview 指标。"""
    if not context.features:
        return pl.DataFrame()

    all_batches: list[pl.DataFrame] = []
    for start in range(0, len(context.features), options.overview_batch_size):
        batch_cols = context.features[start : start + options.overview_batch_size]
        all_exprs: list[pl.Expr] = []
        for col in batch_cols:
            for expr in overview_exprs(context, selection, options, col):
                metric_name = expr.meta.output_name()
                all_exprs.append(expr.alias(f"{col}:::{metric_name}"))

        batch_raw = context.df.select(all_exprs)
        batch_long = batch_raw.unpivot(variable_name="temp_id", value_name="value")
        batch_pivoted = (
            batch_long.with_columns(
                pl.col("temp_id")
                .str.split_exact(":::", 1)
                .struct.rename_fields(["feature", "metric"])
                .alias("meta")
            )
            .unnest("meta")
            .pivot(on="metric", index="feature", values="value", aggregate_function="first")
        )
        all_batches.append(batch_pivoted)

    pivoted = pl.concat(all_batches)
    cols_to_cast = [col for col in pivoted.columns if col not in ["feature", "mode_value"]]
    if cols_to_cast:
        pivoted = pivoted.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in cols_to_cast])
    return pivoted


def calculate_overview(
    context: ProfileRunContext,
    selection: ProfileMetricSelection,
    options: ProfileComputeOptions,
    sparkline_df: pl.DataFrame,
) -> pl.DataFrame:
    """计算 overview 宽表并整理列顺序。"""
    stats = analyze_columns(context, selection, options)
    stats = stats.join(feature_dtypes(context), on="feature", how="left")

    if not sparkline_df.is_empty():
        stats = stats.join(sparkline_df, on="feature", how="left")

    ideal_order = [
        "feature",
        "dtype",
        "distribution",
        "missing_rate",
        "zeros_rate",
        "unique_rate",
        "mode_rate",
        "mode_value",
        *selection.stat_metrics,
    ]
    final_cols: list[str] = []
    seen: set[str] = set()
    for col in ideal_order:
        if col in stats.columns and col not in seen:
            final_cols.append(col)
            seen.add(col)

    remaining_cols = [col for col in stats.columns if col not in seen]
    return stats.select(final_cols + remaining_cols).sort(["dtype", "feature"])
