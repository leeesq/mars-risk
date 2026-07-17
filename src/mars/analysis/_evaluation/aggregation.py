"""分箱评估聚合与基准分布构造。"""

from __future__ import annotations

import pandas as pd
import polars as pl

from mars.compute import (
    amount_stats_agg_exprs,
    binary_stats_agg_exprs,
    build_missing_by_period_stats,
    expected_dist_expr,
)
from mars.utils.date import MarsDate
from mars.utils.logger import logger


def aggregate_basic_stats(
    df_binned: pl.DataFrame,
    *,
    group_col: str,
    features: list[str],
    target_col: str,
    weights_col: str | None,
    amount_col: str | None = None,
    batch_size: int = 500,
) -> pl.DataFrame:
    """把分箱索引表聚合为分组、特征、箱粒度的统计长表。"""
    theoretical_bin_cols = [f"{feature}_bin" for feature in features]
    existing_cols = set(df_binned.columns)
    bin_cols = [col for col in theoretical_bin_cols if col in existing_cols]
    missing_cols = set(theoretical_bin_cols) - set(bin_cols)
    if missing_cols:
        logger.warning(
            "%s features were not binned and will be skipped in evaluation. All missing: %s",
            len(missing_cols),
            sorted(missing_cols),
        )
    if not bin_cols:
        raise ValueError("No valid binned columns found in dataframe. Check your binner fit results.")

    index_cols = [group_col, target_col]
    if weights_col:
        index_cols.append(weights_col)
    if amount_col:
        index_cols.append(amount_col)

    agg_exprs = binary_stats_agg_exprs(target_col, weight_col=weights_col)
    if amount_col:
        agg_exprs += amount_stats_agg_exprs(target_col, amount_col)

    result_frames: list[pl.DataFrame] = []
    for start in range(0, len(bin_cols), batch_size):
        batch_bins = bin_cols[start : start + batch_size]
        batch_res = (
            df_binned
            .lazy()
            .select(
                [pl.col(col).cast(pl.Int16) for col in batch_bins]
                + [pl.col(col) for col in index_cols]
            )
            .unpivot(
                index=index_cols,
                on=batch_bins,
                variable_name="feature_bin",
                value_name="bin_index",
            )
            .with_columns(pl.col("feature_bin").str.replace(r"_bin$", "").alias("feature"))
            .group_by([group_col, "feature", "bin_index"])
            .agg(agg_exprs)
            .collect(engine="streaming")
        )
        result_frames.append(batch_res)

    if not result_frames:
        return pl.DataFrame()
    return pl.concat(result_frames)


def rollup_total_stats(stats_df: pl.DataFrame, *, group_col: str) -> pl.DataFrame:
    """把分组级统计表汇总为 Total 面板统计表。"""
    agg_exprs: list[pl.Expr] = [
        pl.col("count").sum().alias("count"),
        pl.col("observed_count").sum().alias("observed_count"),
        pl.col("bad").sum().alias("bad"),
    ]
    for amount_metric in ["tot_amt", "good_amt", "bad_amt"]:
        if amount_metric in stats_df.columns:
            agg_exprs.append(pl.col(amount_metric).sum().alias(amount_metric))
    return (
        stats_df
        .group_by(["feature", "bin_index"])
        .agg(agg_exprs)
        .with_columns(pl.lit("Total").alias(group_col))
    )


def get_benchmark_dist(
    *,
    group_stats_raw: pl.DataFrame,
    benchmark_binned: pl.DataFrame | None,
    group_col: str,
    features: list[str],
    weights_col: str | None,
) -> pl.DataFrame:
    """构造 PSI expected distribution。"""
    if benchmark_binned is not None:
        bin_cols = [
            f"{feature}_bin"
            for feature in features
            if f"{feature}_bin" in benchmark_binned.columns
        ]
        if not bin_cols:
            return pl.DataFrame(schema={"feature": pl.String, "bin_index": pl.Int16, "expected_dist": pl.Float64})

        if weights_col and weights_col in benchmark_binned.columns:
            idx_cols = [weights_col]
            agg_expr = pl.col(weights_col).cast(pl.Float64).sum().alias("expected_count")
        else:
            idx_cols = []
            agg_expr = pl.len().cast(pl.Float64).alias("expected_count")

        return (
            benchmark_binned
            .select(bin_cols + idx_cols)
            .unpivot(index=idx_cols, on=bin_cols, variable_name="feat_bin", value_name="bin_index")
            .with_columns(pl.col("feat_bin").str.replace(r"_bin$", "").alias("feature"))
            .group_by(["feature", "bin_index"])
            .agg(agg_expr)
            .with_columns(
                expected_dist_expr()
            )
            .select(["feature", "bin_index", "expected_dist"])
        )

    min_group = group_stats_raw.select(pl.col(group_col).min()).item()
    logger.debug("[BASELINE] Using earliest group '%s' as baseline (from stats cache).", min_group)
    return (
        group_stats_raw
        .filter(pl.col(group_col) == min_group)
        .group_by(["feature", "bin_index"])
        .agg(pl.col("count").sum().alias("expected_count"))
        .with_columns(
            expected_dist_expr()
        )
        .select(["feature", "bin_index", "expected_dist"])
    )


def build_missing_by_day_table(
    *,
    df: pl.DataFrame,
    features: list[str],
    dt_col: str | None,
    output_kind: str,
    missing_values: list[object] | None,
) -> pl.DataFrame | pd.DataFrame | None:
    """构建按日缺失率趋势表，失败时降级为不输出该附表。"""
    if not dt_col or dt_col not in df.columns:
        return None

    try:
        working_df = df.with_columns(MarsDate.from_grain(dt_col, "day").alias("__mars_missing_day"))
        missing_table = build_missing_by_period_stats(
            working_df,
            features=features,
            period_col="__mars_missing_day",
            missing_values=missing_values,
        )
        if missing_table is None:
            return None
        if output_kind == "pandas" and isinstance(missing_table, pl.DataFrame):
            return missing_table.to_pandas()
        if output_kind == "polars" and isinstance(missing_table, pd.DataFrame):
            return pl.from_pandas(missing_table)
        return missing_table
    except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
        logger.warning("Missing-by-day trend generation skipped due to error: %s", exc)
        return None
