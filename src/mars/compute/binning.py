"""分箱指标表达式工厂。"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from mars.core.constants import METRIC_EPSILON


def _valid_amount_value_expr(amount_col: str) -> pl.Expr:
    """构造可参与金额统计的金额值表达式。"""
    return (
        pl.when(pl.col(amount_col).is_not_null() & (pl.col(amount_col) >= 0))
        .then(pl.col(amount_col).cast(pl.Float64))
        .otherwise(0.0)
    )


def binary_count_expr(
    *,
    weight_col: str | None = None,
    output_col: str = "count",
) -> pl.Expr:
    """构造分箱总样本数聚合表达式。"""
    if weight_col:
        return pl.col(weight_col).cast(pl.Float64).sum().alias(output_col)
    return pl.len().cast(pl.Float64).alias(output_col)


def binary_observed_count_expr(
    target_col: str,
    *,
    weight_col: str | None = None,
    output_col: str = "observed_count",
) -> pl.Expr:
    """构造参与坏率计算的观测样本数聚合表达式。"""
    if weight_col:
        return (
            pl.when(pl.col(target_col).is_not_null())
            .then(pl.col(weight_col).cast(pl.Float64))
            .otherwise(0.0)
            .sum()
            .alias(output_col)
        )
    return pl.col(target_col).is_not_null().sum().cast(pl.Float64).alias(output_col)


def binary_bad_expr(
    target_col: str,
    *,
    weight_col: str | None = None,
    output_col: str = "bad",
) -> pl.Expr:
    """构造坏样本数聚合表达式。"""
    if weight_col:
        return (
            (
                pl.col(target_col).fill_null(0).cast(pl.Float64)
                * pl.col(weight_col).cast(pl.Float64)
            )
            .sum()
            .alias(output_col)
        )
    return pl.col(target_col).fill_null(0).cast(pl.Float64).sum().alias(output_col)


def binary_stats_agg_exprs(
    target_col: str,
    *,
    weight_col: str | None = None,
    count_col: str = "count",
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
) -> list[pl.Expr]:
    """构造分箱基础统计聚合表达式列表。"""
    return [
        binary_count_expr(weight_col=weight_col, output_col=count_col),
        binary_observed_count_expr(
            target_col,
            weight_col=weight_col,
            output_col=observed_count_col,
        ),
        binary_bad_expr(
            target_col,
            weight_col=weight_col,
            output_col=bad_col,
        ),
    ]


def total_amount_expr(
    amount_col: str,
    *,
    output_col: str = "tot_amt",
) -> pl.Expr:
    """构造总金额聚合表达式。"""
    return _valid_amount_value_expr(amount_col).sum().alias(output_col)


def good_amount_expr(
    target_col: str,
    amount_col: str,
    *,
    output_col: str = "good_amt",
) -> pl.Expr:
    """构造好样本金額聚合表达式。"""
    valid_amount_expr = _valid_amount_value_expr(amount_col)
    return (
        pl.when(pl.col(target_col) == 0)
        .then(valid_amount_expr)
        .otherwise(0.0)
        .sum()
        .alias(output_col)
    )


def bad_amount_expr(
    target_col: str,
    amount_col: str,
    *,
    output_col: str = "bad_amt",
) -> pl.Expr:
    """构造坏样本金額聚合表达式。"""
    valid_amount_expr = _valid_amount_value_expr(amount_col)
    return (
        pl.when(pl.col(target_col) == 1)
        .then(valid_amount_expr)
        .otherwise(0.0)
        .sum()
        .alias(output_col)
    )


def amount_stats_agg_exprs(
    target_col: str,
    amount_col: str,
    *,
    total_amount_col: str = "tot_amt",
    good_amount_col: str = "good_amt",
    bad_amount_col: str = "bad_amt",
) -> list[pl.Expr]:
    """构造金额统计聚合表达式列表。"""
    return [
        total_amount_expr(amount_col, output_col=total_amount_col),
        good_amount_expr(
            target_col,
            amount_col,
            output_col=good_amount_col,
        ),
        bad_amount_expr(
            target_col,
            amount_col,
            output_col=bad_amount_col,
        ),
    ]


def binary_distribution_exprs(
    group_keys: Sequence[str],
    *,
    count_col: str = "count",
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
    total_count_col: str = "total_count",
    total_observed_col: str = "total_observed",
    total_bad_col: str = "total_bad",
    total_good_col: str = "total_good",
    good_col: str = "good",
) -> list[pl.Expr]:
    """构造分箱分布级中间量表达式列表。"""
    partitions = list(group_keys)
    good_value = pl.col(observed_count_col) - pl.col(bad_col)
    return [
        good_value.alias(good_col),
        pl.col(count_col).sum().over(partitions).alias(total_count_col),
        pl.col(observed_count_col).sum().over(partitions).alias(total_observed_col),
        pl.col(bad_col).sum().over(partitions).alias(total_bad_col),
        good_value.sum().over(partitions).alias(total_good_col),
    ]


def amount_distribution_exprs(
    group_keys: Sequence[str],
    *,
    total_amount_col: str = "tot_amt",
    good_amount_col: str = "good_amt",
    bad_amount_col: str = "bad_amt",
    observed_amount_col: str = "observed_amt",
    total_observed_amount_col: str = "total_observed_amt",
    total_bad_amount_col: str = "total_bad_amt",
) -> list[pl.Expr]:
    """构造金额口径的中间量表达式列表。"""
    partitions = list(group_keys)
    observed_amount_expr = pl.col(good_amount_col) + pl.col(bad_amount_col)
    return [
        observed_amount_expr.alias(observed_amount_col),
        observed_amount_expr.sum().over(partitions).alias(total_observed_amount_col),
        pl.col(bad_amount_col).sum().over(partitions).alias(total_bad_amount_col),
    ]


def binary_metric_exprs(
    *,
    count_col: str = "count",
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
    total_count_col: str = "total_count",
    total_observed_col: str = "total_observed",
    total_bad_col: str = "total_bad",
    total_good_col: str = "total_good",
    actual_dist_col: str = "actual_dist",
    bad_dist_col: str = "bad_dist",
    good_dist_col: str = "good_dist",
    bad_rate_col: str = "bad_rate",
    lift_col: str = "lift",
    woe_col: str = "woe",
    iv_bin_col: str = "iv_bin",
    epsilon: float = METRIC_EPSILON,
) -> list[pl.Expr]:
    """构造分箱指标表达式列表。"""
    bad_dist_expr = pl.col(bad_col) / (pl.col(total_bad_col) + epsilon)
    good_dist_expr = (
        (pl.col(observed_count_col) - pl.col(bad_col))
        / (pl.col(total_good_col) + epsilon)
    )
    current_bad_rate_expr = (
        pl.when(pl.col(observed_count_col) > 0)
        .then(pl.col(bad_col) / pl.col(observed_count_col))
        .otherwise(None)
    )
    woe_expr = ((bad_dist_expr + epsilon) / (good_dist_expr + epsilon)).log()
    return [
        ((pl.col(count_col) + epsilon) / (pl.col(total_count_col) + epsilon)).alias(
            actual_dist_col,
        ),
        bad_dist_expr.alias(bad_dist_col),
        good_dist_expr.alias(good_dist_col),
        current_bad_rate_expr.alias(bad_rate_col),
        (
            pl.when(pl.col(total_observed_col) > 0)
            .then(
                current_bad_rate_expr
                / (
                    (pl.col(total_bad_col) + epsilon)
                    / (pl.col(total_observed_col) + epsilon)
                )
            )
            .otherwise(None)
            .alias(lift_col)
        ),
        woe_expr.cast(pl.Float32).alias(woe_col),
        (
            pl.when(pl.col(total_observed_col) > 0)
            .then((bad_dist_expr - good_dist_expr) * woe_expr)
            .otherwise(None)
            .cast(pl.Float32)
            .alias(iv_bin_col)
        ),
    ]


def amount_metric_exprs(
    *,
    count_col: str = "count",
    total_amount_col: str = "tot_amt",
    observed_amount_col: str = "observed_amt",
    total_observed_amount_col: str = "total_observed_amt",
    bad_amount_col: str = "bad_amt",
    total_bad_amount_col: str = "total_bad_amt",
    average_amount_col: str = "avg_amt",
    amount_bad_rate_col: str = "amt_bad_rate",
    amount_lift_col: str = "lift_amt",
    epsilon: float = METRIC_EPSILON,
) -> list[pl.Expr]:
    """构造金额口径指标表达式列表。"""
    amount_bad_rate_expr = (
        pl.when(pl.col(observed_amount_col) > 0)
        .then(pl.col(bad_amount_col) / pl.col(observed_amount_col))
        .otherwise(None)
    )
    return [
        (
            pl.when(pl.col(count_col) > 0)
            .then(pl.col(total_amount_col) / pl.col(count_col))
            .otherwise(None)
            .alias(average_amount_col)
        ),
        amount_bad_rate_expr.alias(amount_bad_rate_col),
        (
            pl.when(pl.col(total_observed_amount_col) > 0)
            .then(
                amount_bad_rate_expr
                / (
                    (pl.col(total_bad_amount_col) + epsilon)
                    / (pl.col(total_observed_amount_col) + epsilon)
                )
            )
            .otherwise(None)
            .alias(amount_lift_col)
        ),
    ]


def ordered_binary_metric_exprs(
    group_keys: Sequence[str],
    *,
    total_observed_col: str = "total_observed",
    bad_dist_col: str = "bad_dist",
    good_dist_col: str = "good_dist",
    cum_bad_dist_col: str = "cum_bad_dist",
    cum_good_dist_col: str = "cum_good_dist",
    ks_bin_col: str = "ks_bin",
    auc_bin_col: str = "auc_bin",
) -> list[pl.Expr]:
    """构造需要高层先排序后再计算的有序指标表达式列表。"""
    partitions = list(group_keys)
    cum_bad_expr = pl.col(bad_dist_col).cum_sum().over(partitions)
    cum_good_expr = pl.col(good_dist_col).cum_sum().over(partitions)
    prev_bad_expr = cum_bad_expr.shift(1, fill_value=0.0).over(partitions)
    prev_good_expr = cum_good_expr.shift(1, fill_value=0.0).over(partitions)
    return [
        cum_bad_expr.alias(cum_bad_dist_col),
        cum_good_expr.alias(cum_good_dist_col),
        (
            pl.when(pl.col(total_observed_col) > 0)
            .then((cum_bad_expr - cum_good_expr).abs() * 100)
            .otherwise(None)
            .alias(ks_bin_col)
        ),
        (
            pl.when(pl.col(total_observed_col) > 0)
            .then(
                (cum_good_expr - prev_good_expr)
                * (cum_bad_expr + prev_bad_expr)
                / 2
            )
            .otherwise(None)
            .alias(auc_bin_col)
        ),
    ]


def bad_rate_expr(
    *,
    bad_col: str = "bad",
    observed_count_col: str = "observed_count",
    output_col: str = "bad_rate",
) -> pl.Expr:
    """构造坏率表达式。"""
    return (
        pl.when(pl.col(observed_count_col) > 0)
        .then(pl.col(bad_col) / pl.col(observed_count_col))
        .otherwise(None)
        .alias(output_col)
    )


__all__ = [
    "amount_distribution_exprs",
    "amount_metric_exprs",
    "amount_stats_agg_exprs",
    "bad_amount_expr",
    "bad_rate_expr",
    "binary_bad_expr",
    "binary_count_expr",
    "binary_distribution_exprs",
    "binary_metric_exprs",
    "binary_observed_count_expr",
    "binary_stats_agg_exprs",
    "good_amount_expr",
    "ordered_binary_metric_exprs",
    "total_amount_expr",
]
