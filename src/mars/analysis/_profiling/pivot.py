"""数据画像趋势透视表。"""

from __future__ import annotations

import polars as pl

from mars.analysis._profiling.metrics import feature_dtypes, metric_expr
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext


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


def generate_pivot_reports(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    metrics: list[str],
    overview: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """在有界特征批次内联合聚合趋势，并复用口径一致的 overview 全量统计。"""
    target_cols = [col for col in context.features if col != context.group_col]
    if not target_cols:
        return {metric: pl.DataFrame() for metric in metrics}
    frames: dict[str, list[pl.DataFrame]] = {metric: [] for metric in metrics}
    for start in range(0, len(target_cols), options.overview_batch_size):
        batch_cols = target_cols[start : start + options.overview_batch_size]
        aliases = {(col, metric): f"__metric_{i}_{j}"
                   for i, col in enumerate(batch_cols) for j, metric in enumerate(metrics)}
        total_exprs: list[pl.Expr] = []
        cached: dict[str, str] = {}
        for metric in metrics:
            column = f"{metric}_rate" if metric in {"missing", "zeros", "unique", "mode"} else metric
            # 百万行以上 overview unique 使用历史近似算法，趋势仍需精确计算。
            if column in overview.columns and not (metric == "unique" and context.df.height > 1_000_000):
                cached[metric] = column
            else:
                total_exprs.extend(metric_expr(context, options, col, metric).alias(aliases[col, metric])
                                   for col in batch_cols)
        totals = context.working_df.select(total_exprs) if total_exprs else pl.DataFrame()
        grouped: pl.DataFrame | None = None
        if context.group_col:
            grouped = (
                context.working_df.group_by(context.group_col)
                .agg([metric_expr(context, options, col, metric).alias(aliases[col, metric])
                      for col in batch_cols for metric in metrics])
                .sort(context.group_col)
                .with_columns(pl.col(context.group_col).cast(pl.String))
            )
        for metric in metrics:
            if metric in cached:
                # overview 为展示统一到 Float64；恢复历史趋势 transpose 的实际公共 dtype。
                if grouped is not None and not grouped.is_empty():
                    dtype_probe = grouped.select(
                        [pl.col(aliases[col, metric]) for col in batch_cols]
                    ).head(1).transpose()
                else:
                    dtype_probe = context.working_df.head(0).select(
                        [metric_expr(context, options, col, metric).alias(col) for col in batch_cols]
                    ).transpose()
                total_dtype = dtype_probe.dtypes[0]
                base = overview.filter(pl.col("feature").is_in(batch_cols)).select(
                    "feature", "dtype", pl.col(cached[metric]).cast(total_dtype).alias("total"),
                )
            else:
                base = totals.select([pl.col(aliases[col, metric]).alias(col) for col in batch_cols]).transpose(
                    include_header=True, header_name="feature", column_names=["total"],
                ).join(feature_dtypes(context), on="feature", how="left")
            if grouped is not None:
                pivot = grouped.select(
                    pl.col(context.group_col),
                    *[pl.col(aliases[col, metric]).alias(col) for col in batch_cols],
                ).transpose(include_header=True, header_name="feature", column_names=context.group_col)
                base = base.join(pivot, on="feature", how="left")
            group_cols = [c for c in base.columns if c not in {"feature", "dtype", "total"}]
            frames[metric].append(base.select("feature", "dtype", *group_cols, "total"))
    return {metric: pl.concat(parts, how="vertical_relaxed").sort(["dtype", "feature"])
            for metric, parts in frames.items()}
