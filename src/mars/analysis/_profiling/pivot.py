"""数据画像趋势透视表。"""

from __future__ import annotations

import polars as pl

from mars.analysis._profiling.metrics import feature_dtypes, metric_expr
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.core.constants import DIVISION_EPSILON


def generate_pivot_report(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    metric: str,
) -> pl.DataFrame:
    """生成指定指标的分组趋势透视表。"""
    target_cols = [col for col in context.features if col != context.group_col]
    if not target_cols:
        return pl.DataFrame()

    total_exprs = [metric_expr(context, options, col, metric).alias(col) for col in target_cols]
    total_df = context.working_df.select(total_exprs).transpose(
        include_header=True,
        header_name="feature",
        column_names=["total"],
    )
    base_df = total_df.join(feature_dtypes(context), on="feature", how="left")

    # 没有分组列时，直接返回基础宽表
    if context.group_col is None:
        return base_df.select(["feature", "dtype", "total"]).sort(["dtype", "feature"])

    agg_exprs = [metric_expr(context, options, col, metric).alias(col) for col in target_cols]
    grouped = (
        context.working_df.group_by(context.group_col)
        .agg(agg_exprs)
        .sort(context.group_col)
        .with_columns(pl.col(context.group_col).cast(pl.String))
    )
    pivot_df = grouped.transpose(
        include_header=True,
        header_name="feature",
        column_names=context.group_col,
    )
    result = base_df.join(pivot_df, on="feature", how="left")
    fixed_cols = {"feature", "dtype", "total"}
    group_cols = [col for col in result.columns if col not in fixed_cols]
    return result.select(["feature", "dtype", *group_cols, "total"]).sort(["dtype", "feature"])


def add_stability_metrics(df: pl.DataFrame, exclude_cols: list[str]) -> pl.DataFrame:
    """为趋势宽表增加分组均值、方差和变异系数。"""
    if df.is_empty():
        return df

    calc_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32]
    ]
    if not calc_cols:
        return df

    return (
        df.with_columns(pl.concat_list(calc_cols).alias("_tmp"))
        .with_columns(
            [
                pl.col("_tmp").list.mean().fill_null(0).alias("group_mean"),
                pl.col("_tmp").list.var().fill_null(0).alias("group_var"),
                (
                    pl.col("_tmp").list.std()
                    / (pl.col("_tmp").list.mean().abs() + DIVISION_EPSILON)
                )
                .fill_null(0)
                .alias("group_cv"),
            ]
        )
        .drop("_tmp")
        .select(["feature", "dtype", *calc_cols, "total", "group_mean", "group_var", "group_cv"])
    )

