"""分箱指标的纯 Polars 表达式工厂。"""

from __future__ import annotations

from typing import Literal, Sequence, cast

import polars as pl

from mars.core.constants import DIVISION_EPSILON, METRIC_EPSILON

OrderedMetricSortBy = Literal["woe", "bin_index"]


def normalize_ordered_metric_sort_by(value: str | None) -> OrderedMetricSortBy:
    """校验并规范化 KS/AUC 有序指标排序口径。"""
    normalized = "woe" if value is None else str(value).strip().lower()
    if normalized not in {"woe", "bin_index"}:
        raise ValueError(
            "ordered_metric_sort_by must be one of {'woe', 'bin_index'}, "
            f"got {value!r}."
        )
    return cast(OrderedMetricSortBy, normalized)


def _valid_amount_value_expr(amount_col: str) -> pl.Expr:
    """构造可参与金额统计的非负金额表达式。"""
    return (
        pl.when(pl.col(amount_col).is_not_null() & (pl.col(amount_col) >= 0))
        .then(pl.col(amount_col).cast(pl.Float64))
        .otherwise(0.0)
    )


def _good_value_expr(*, observed_count_col: str, bad_col: str) -> pl.Expr:
    """构造未命名的好样本数表达式。"""
    return pl.col(observed_count_col) - pl.col(bad_col)


def _bad_rate_value_expr(*, bad_col: str, observed_count_col: str) -> pl.Expr:
    """构造未命名的坏样本率表达式。"""
    return (
        pl.when(pl.col(observed_count_col) > 0)
        .then(pl.col(bad_col) / pl.col(observed_count_col))
        .otherwise(None)
    )


def _bad_dist_value_expr(*, bad_col: str, total_bad_col: str, epsilon: float) -> pl.Expr:
    """构造未命名的坏样本分布表达式。"""
    return pl.col(bad_col) / (pl.col(total_bad_col) + epsilon)


def _good_dist_value_expr(
    *,
    observed_count_col: str,
    bad_col: str,
    total_good_col: str,
    epsilon: float,
) -> pl.Expr:
    """构造未命名的好样本分布表达式。"""
    return _good_value_expr(
        observed_count_col=observed_count_col,
        bad_col=bad_col,
    ) / (pl.col(total_good_col) + epsilon)


def _woe_from_dist_values_expr(
    *,
    bad_dist_expr: pl.Expr,
    good_dist_expr: pl.Expr,
    epsilon: float,
) -> pl.Expr:
    """构造基于分布表达式的未命名 WOE 表达式。"""
    return ((bad_dist_expr + epsilon) / (good_dist_expr + epsilon)).log()


def _woe_value_expr(*, bad_dist_col: str, good_dist_col: str, epsilon: float) -> pl.Expr:
    """构造未命名的 WOE 表达式。"""
    return _woe_from_dist_values_expr(
        bad_dist_expr=pl.col(bad_dist_col),
        good_dist_expr=pl.col(good_dist_col),
        epsilon=epsilon,
    )


def _lift_value_expr(
    *,
    bad_col: str,
    observed_count_col: str,
    total_bad_col: str,
    total_observed_col: str,
    epsilon: float,
) -> pl.Expr:
    """构造未命名的坏率 Lift 表达式。"""
    current_bad_rate = _bad_rate_value_expr(
        bad_col=bad_col,
        observed_count_col=observed_count_col,
    )
    total_bad_rate = (pl.col(total_bad_col) + epsilon) / (
        pl.col(total_observed_col) + epsilon
    )
    return (
        pl.when(pl.col(total_observed_col) > 0)
        .then(current_bad_rate / total_bad_rate)
        .otherwise(None)
    )


def _iv_bin_value_expr(
    *,
    bad_dist_expr: pl.Expr,
    good_dist_expr: pl.Expr,
    total_observed_col: str,
    epsilon: float,
) -> pl.Expr:
    """构造未命名的单箱 IV 贡献表达式。"""
    woe_value = _woe_from_dist_values_expr(
        bad_dist_expr=bad_dist_expr,
        good_dist_expr=good_dist_expr,
        epsilon=epsilon,
    )
    return (
        pl.when(pl.col(total_observed_col) > 0)
        .then((bad_dist_expr - good_dist_expr) * woe_value)
        .otherwise(None)
    )


def _cum_bad_rate_value_expr(
    *,
    cum_bad_expr: pl.Expr,
    cum_observed_count_expr: pl.Expr,
) -> pl.Expr:
    """构造未命名的累计坏率表达式。"""
    return (
        pl.when(cum_observed_count_expr > 0)
        .then(cum_bad_expr / cum_observed_count_expr)
        .otherwise(None)
    )


def _ks_bin_value_expr(
    *,
    cum_bad_dist_expr: pl.Expr,
    cum_good_dist_expr: pl.Expr,
    total_observed_col: str,
) -> pl.Expr:
    """构造未命名的单箱 KS 表达式。"""
    return (
        pl.when(pl.col(total_observed_col) > 0)
        .then((cum_bad_dist_expr - cum_good_dist_expr).abs() * 100)
        .otherwise(None)
    )


def _auc_bin_value_expr(
    *,
    cum_bad_dist_expr: pl.Expr,
    cum_good_dist_expr: pl.Expr,
    prev_bad_dist_expr: pl.Expr,
    prev_good_dist_expr: pl.Expr,
    total_observed_col: str,
) -> pl.Expr:
    """构造未命名的单箱 AUC 梯形贡献表达式。"""
    return (
        pl.when(pl.col(total_observed_col) > 0)
        .then(
            (cum_good_dist_expr - prev_good_dist_expr)
            * (cum_bad_dist_expr + prev_bad_dist_expr)
            / 2
        )
        .otherwise(None)
    )


def _amount_bad_rate_value_expr(
    *,
    bad_amount_col: str,
    observed_amount_col: str,
) -> pl.Expr:
    """构造未命名的金额坏率表达式。"""
    return (
        pl.when(pl.col(observed_amount_col) > 0)
        .then(pl.col(bad_amount_col) / pl.col(observed_amount_col))
        .otherwise(None)
    )


def _amount_lift_value_expr(
    *,
    bad_amount_col: str,
    observed_amount_col: str,
    total_bad_amount_col: str,
    total_observed_amount_col: str,
    epsilon: float,
) -> pl.Expr:
    """构造未命名的金额坏率 Lift 表达式。"""
    amount_bad_rate = _amount_bad_rate_value_expr(
        bad_amount_col=bad_amount_col,
        observed_amount_col=observed_amount_col,
    )
    total_amount_bad_rate = (pl.col(total_bad_amount_col) + epsilon) / (
        pl.col(total_observed_amount_col) + epsilon
    )
    return (
        pl.when(pl.col(total_observed_amount_col) > 0)
        .then(amount_bad_rate / total_amount_bad_rate)
        .otherwise(None)
    )


def _observed_metric_agg_expr(
    metric_expr: pl.Expr,
    *,
    observed_count_col: str,
    output_col: str,
) -> pl.Expr:
    """按观察样本是否存在包装聚合指标表达式。"""
    return (
        pl.when(pl.col(observed_count_col).sum() > 0)
        .then(metric_expr)
        .otherwise(None)
        .alias(output_col)
    )


def binary_count_expr(*, weight_col: str | None = None, output_col: str = "count") -> pl.Expr:
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
    """构造参与坏率计算的观察样本数聚合表达式。"""
    if weight_col:
        return (
            pl.when(pl.col(target_col).is_not_null())
            .then(pl.col(weight_col).cast(pl.Float64))
            .otherwise(0.0)
            .sum()
            .alias(output_col)
        )
    return pl.col(target_col).is_not_null().sum().cast(pl.Float64).alias(output_col)


def binary_unobserved_count_expr(
    target_col: str,
    *,
    output_col: str = "target_unobserved_count",
) -> pl.Expr:
    """构造目标缺失样本数聚合表达式。"""
    return pl.col(target_col).is_null().sum().cast(pl.Float64).alias(output_col)


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


def good_expr(
    *,
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
    output_col: str = "good",
) -> pl.Expr:
    """构造好样本数表达式。"""
    return _good_value_expr(
        observed_count_col=observed_count_col,
        bad_col=bad_col,
    ).alias(output_col)


def ratio_expr(*, numerator_col: str, denominator_col: str, output_col: str) -> pl.Expr:
    """构造安全比率表达式。"""
    return (
        pl.when(pl.col(denominator_col) > 0)
        .then(pl.col(numerator_col) / pl.col(denominator_col))
        .otherwise(None)
        .alias(output_col)
    )


def distribution_rate_expr(
    *,
    numerator_col: str,
    denominator_col: str,
    output_col: str,
    epsilon: float = DIVISION_EPSILON,
) -> pl.Expr:
    """构造带平滑分母的分布占比表达式。"""
    return (pl.col(numerator_col) / (pl.col(denominator_col) + epsilon)).alias(output_col)


def global_distribution_expr(
    *,
    count_col: str,
    output_col: str,
    epsilon: float = DIVISION_EPSILON,
) -> pl.Expr:
    """构造全局分布占比表达式。"""
    return (pl.col(count_col) / (pl.col(count_col).sum() + epsilon)).alias(output_col)


def partition_distribution_expr(
    group_keys: Sequence[str],
    *,
    count_col: str,
    output_col: str,
    epsilon: float = DIVISION_EPSILON,
) -> pl.Expr:
    """构造分区内分布占比表达式。"""
    return (
        pl.col(count_col)
        / (pl.col(count_col).sum().over(list(group_keys)) + epsilon)
    ).alias(output_col)


def total_count_expr(
    group_keys: Sequence[str],
    *,
    count_col: str = "count",
    output_col: str = "total_count",
) -> pl.Expr:
    """构造分组内总样本数窗口表达式。"""
    return pl.col(count_col).sum().over(list(group_keys)).alias(output_col)


def total_observed_expr(
    group_keys: Sequence[str],
    *,
    observed_count_col: str = "observed_count",
    output_col: str = "total_observed",
) -> pl.Expr:
    """构造分组内观察样本数窗口表达式。"""
    return pl.col(observed_count_col).sum().over(list(group_keys)).alias(output_col)


def total_bad_expr(
    group_keys: Sequence[str],
    *,
    bad_col: str = "bad",
    output_col: str = "total_bad",
) -> pl.Expr:
    """构造分组内坏样本数窗口表达式。"""
    return pl.col(bad_col).sum().over(list(group_keys)).alias(output_col)


def total_good_expr(
    group_keys: Sequence[str],
    *,
    good_col: str = "good",
    output_col: str = "total_good",
) -> pl.Expr:
    """构造分组内好样本数窗口表达式。"""
    return pl.col(good_col).sum().over(list(group_keys)).alias(output_col)


def actual_dist_expr(
    *,
    count_col: str = "count",
    total_count_col: str = "total_count",
    output_col: str = "actual_dist",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造样本占比分布表达式。"""
    return ((pl.col(count_col) + epsilon) / (pl.col(total_count_col) + epsilon)).alias(
        output_col,
    )


def bad_dist_expr(
    *,
    bad_col: str = "bad",
    total_bad_col: str = "total_bad",
    output_col: str = "bad_dist",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造坏样本分布表达式。"""
    return _bad_dist_value_expr(
        bad_col=bad_col,
        total_bad_col=total_bad_col,
        epsilon=epsilon,
    ).alias(output_col)


def good_dist_expr(
    *,
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
    total_good_col: str = "total_good",
    output_col: str = "good_dist",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造好样本分布表达式。"""
    return _good_dist_value_expr(
        observed_count_col=observed_count_col,
        bad_col=bad_col,
        total_good_col=total_good_col,
        epsilon=epsilon,
    ).alias(output_col)


def bad_rate_expr(
    *,
    bad_col: str = "bad",
    observed_count_col: str = "observed_count",
    output_col: str = "bad_rate",
) -> pl.Expr:
    """构造坏率表达式。"""
    return _bad_rate_value_expr(
        bad_col=bad_col,
        observed_count_col=observed_count_col,
    ).alias(output_col)


def bad_rate_agg_expr(
    *,
    bad_col: str = "bad",
    observed_count_col: str = "observed_count",
    output_col: str = "bad_rate",
) -> pl.Expr:
    """构造分组聚合坏率表达式。"""
    return _observed_metric_agg_expr(
        pl.col(bad_col).sum() / pl.col(observed_count_col).sum(),
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def lift_expr(
    *,
    bad_col: str = "bad",
    observed_count_col: str = "observed_count",
    total_bad_col: str = "total_bad",
    total_observed_col: str = "total_observed",
    output_col: str = "lift",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造坏率 Lift 表达式。"""
    return _lift_value_expr(
        bad_col=bad_col,
        observed_count_col=observed_count_col,
        total_bad_col=total_bad_col,
        total_observed_col=total_observed_col,
        epsilon=epsilon,
    ).alias(output_col)


def woe_expr(
    *,
    bad_dist_col: str = "bad_dist",
    good_dist_col: str = "good_dist",
    output_col: str = "woe",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造 WOE 表达式。"""
    return _woe_value_expr(
        bad_dist_col=bad_dist_col,
        good_dist_col=good_dist_col,
        epsilon=epsilon,
    ).cast(pl.Float32).alias(output_col)


def iv_bin_expr(
    *,
    bad_dist_col: str = "bad_dist",
    good_dist_col: str = "good_dist",
    total_observed_col: str = "total_observed",
    output_col: str = "iv_bin",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造单箱 IV 贡献表达式。"""
    return _iv_bin_value_expr(
        bad_dist_expr=pl.col(bad_dist_col),
        good_dist_expr=pl.col(good_dist_col),
        total_observed_col=total_observed_col,
        epsilon=epsilon,
    ).cast(pl.Float32).alias(output_col)


def cum_count_expr(
    group_keys: Sequence[str],
    *,
    count_col: str = "count",
    output_col: str = "cum_count",
) -> pl.Expr:
    """构造累计样本数表达式。"""
    return pl.col(count_col).cum_sum().over(list(group_keys)).alias(output_col)


def cum_observed_count_expr(
    group_keys: Sequence[str],
    *,
    observed_count_col: str = "observed_count",
    output_col: str = "cum_observed_count",
) -> pl.Expr:
    """构造累计观察样本数表达式。"""
    return pl.col(observed_count_col).cum_sum().over(list(group_keys)).alias(output_col)


def cum_bad_expr(
    group_keys: Sequence[str],
    *,
    bad_col: str = "bad",
    output_col: str = "cum_bad",
) -> pl.Expr:
    """构造累计坏样本数表达式。"""
    return pl.col(bad_col).cum_sum().over(list(group_keys)).alias(output_col)


def cum_bad_rate_expr(
    *,
    cum_bad_col: str = "cum_bad",
    cum_observed_count_col: str = "cum_observed_count",
    output_col: str = "cum_bad_rate",
) -> pl.Expr:
    """构造累计坏率表达式。"""
    return _cum_bad_rate_value_expr(
        cum_bad_expr=pl.col(cum_bad_col),
        cum_observed_count_expr=pl.col(cum_observed_count_col),
    ).alias(output_col)


def cum_bad_dist_expr(
    group_keys: Sequence[str],
    *,
    bad_dist_col: str = "bad_dist",
    output_col: str = "cum_bad_dist",
) -> pl.Expr:
    """构造累计坏样本分布表达式。"""
    return pl.col(bad_dist_col).cum_sum().over(list(group_keys)).alias(output_col)


def cum_good_dist_expr(
    group_keys: Sequence[str],
    *,
    good_dist_col: str = "good_dist",
    output_col: str = "cum_good_dist",
) -> pl.Expr:
    """构造累计好样本分布表达式。"""
    return pl.col(good_dist_col).cum_sum().over(list(group_keys)).alias(output_col)


def ks_bin_expr(
    *,
    cum_bad_dist_col: str = "cum_bad_dist",
    cum_good_dist_col: str = "cum_good_dist",
    total_observed_col: str = "total_observed",
    output_col: str = "ks_bin",
) -> pl.Expr:
    """构造单箱 KS 表达式。"""
    return _ks_bin_value_expr(
        cum_bad_dist_expr=pl.col(cum_bad_dist_col),
        cum_good_dist_expr=pl.col(cum_good_dist_col),
        total_observed_col=total_observed_col,
    ).alias(output_col)


def auc_bin_expr(
    group_keys: Sequence[str],
    *,
    cum_bad_dist_col: str = "cum_bad_dist",
    cum_good_dist_col: str = "cum_good_dist",
    total_observed_col: str = "total_observed",
    output_col: str = "auc_bin",
) -> pl.Expr:
    """构造单箱 AUC 梯形贡献表达式。"""
    partitions = list(group_keys)
    cum_bad_dist = pl.col(cum_bad_dist_col)
    cum_good_dist = pl.col(cum_good_dist_col)
    prev_bad_dist = cum_bad_dist.shift(1, fill_value=0.0).over(partitions)
    prev_good_dist = cum_good_dist.shift(1, fill_value=0.0).over(partitions)
    return _auc_bin_value_expr(
        cum_bad_dist_expr=cum_bad_dist,
        cum_good_dist_expr=cum_good_dist,
        prev_bad_dist_expr=prev_bad_dist,
        prev_good_dist_expr=prev_good_dist,
        total_observed_col=total_observed_col,
    ).alias(output_col)


def normalized_auc_expr(*, auc_col: str = "auc", output_col: str = "auc") -> pl.Expr:
    """构造方向修正后的 AUC 表达式。"""
    return (
        pl.when(pl.col(auc_col) < 0.5)
        .then(pl.lit(1.0) - pl.col(auc_col))
        .otherwise(pl.col(auc_col))
        .alias(output_col)
    )


def bin_missing_rate_expr(
    *,
    bin_index_col: str = "bin_index",
    count_col: str = "count",
    missing_bin_index: int = -1,
    output_col: str = "missing",
    epsilon: float = DIVISION_EPSILON,
) -> pl.Expr:
    """构造缺失箱样本占比聚合表达式。"""
    return (
        pl.when(pl.col(bin_index_col) == missing_bin_index)
        .then(pl.col(count_col))
        .otherwise(0)
        .sum()
        / (pl.col(count_col).sum() + epsilon)
    ).alias(output_col)


def total_amount_expr(amount_col: str, *, output_col: str = "tot_amt") -> pl.Expr:
    """构造总金额聚合表达式。"""
    return _valid_amount_value_expr(amount_col).sum().alias(output_col)


def good_amount_expr(target_col: str, amount_col: str, *, output_col: str = "good_amt") -> pl.Expr:
    """构造好样本金额聚合表达式。"""
    valid_amount_expr = _valid_amount_value_expr(amount_col)
    return (
        pl.when(pl.col(target_col) == 0)
        .then(valid_amount_expr)
        .otherwise(0.0)
        .sum()
        .alias(output_col)
    )


def bad_amount_expr(target_col: str, amount_col: str, *, output_col: str = "bad_amt") -> pl.Expr:
    """构造坏样本金额聚合表达式。"""
    valid_amount_expr = _valid_amount_value_expr(amount_col)
    return (
        pl.when(pl.col(target_col) == 1)
        .then(valid_amount_expr)
        .otherwise(0.0)
        .sum()
        .alias(output_col)
    )


def observed_amount_expr(
    *,
    good_amount_col: str = "good_amt",
    bad_amount_col: str = "bad_amt",
    output_col: str = "observed_amt",
) -> pl.Expr:
    """构造参与金额坏率计算的观察金额表达式。"""
    return (pl.col(good_amount_col) + pl.col(bad_amount_col)).alias(output_col)


def total_observed_amount_expr(
    group_keys: Sequence[str],
    *,
    observed_amount_col: str = "observed_amt",
    output_col: str = "total_observed_amt",
) -> pl.Expr:
    """构造分组内观察金额窗口表达式。"""
    return pl.col(observed_amount_col).sum().over(list(group_keys)).alias(output_col)


def total_bad_amount_expr(
    group_keys: Sequence[str],
    *,
    bad_amount_col: str = "bad_amt",
    output_col: str = "total_bad_amt",
) -> pl.Expr:
    """构造分组内坏样本金额窗口表达式。"""
    return pl.col(bad_amount_col).sum().over(list(group_keys)).alias(output_col)


def avg_amount_expr(
    *,
    total_amount_col: str = "tot_amt",
    count_col: str = "count",
    output_col: str = "avg_amt",
) -> pl.Expr:
    """构造平均金额表达式。"""
    return (
        pl.when(pl.col(count_col) > 0)
        .then(pl.col(total_amount_col) / pl.col(count_col))
        .otherwise(None)
        .alias(output_col)
    )


def amount_bad_rate_expr(
    *,
    bad_amount_col: str = "bad_amt",
    observed_amount_col: str = "observed_amt",
    output_col: str = "amt_bad_rate",
) -> pl.Expr:
    """构造金额坏率表达式。"""
    return _amount_bad_rate_value_expr(
        bad_amount_col=bad_amount_col,
        observed_amount_col=observed_amount_col,
    ).alias(output_col)


def amount_lift_expr(
    *,
    bad_amount_col: str = "bad_amt",
    observed_amount_col: str = "observed_amt",
    total_bad_amount_col: str = "total_bad_amt",
    total_observed_amount_col: str = "total_observed_amt",
    output_col: str = "lift_amt",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造金额坏率 Lift 表达式。"""
    return _amount_lift_value_expr(
        bad_amount_col=bad_amount_col,
        observed_amount_col=observed_amount_col,
        total_bad_amount_col=total_bad_amount_col,
        total_observed_amount_col=total_observed_amount_col,
        epsilon=epsilon,
    ).alias(output_col)


def observed_sum_agg_expr(
    metric_col: str,
    *,
    observed_count_col: str = "observed_count",
    output_col: str | None = None,
) -> pl.Expr:
    """构造观察样本存在时才求和的聚合表达式。"""
    return _observed_metric_agg_expr(
        pl.col(metric_col).sum(),
        observed_count_col=observed_count_col,
        output_col=output_col or metric_col,
    )


def observed_max_agg_expr(
    metric_col: str,
    *,
    observed_count_col: str = "observed_count",
    output_col: str | None = None,
) -> pl.Expr:
    """构造观察样本存在时才取最大值的聚合表达式。"""
    return _observed_metric_agg_expr(
        pl.col(metric_col).max(),
        observed_count_col=observed_count_col,
        output_col=output_col or metric_col,
    )


def observed_min_agg_expr(
    metric_col: str,
    *,
    observed_count_col: str = "observed_count",
    output_col: str | None = None,
) -> pl.Expr:
    """构造观察样本存在时才取最小值的聚合表达式。"""
    return _observed_metric_agg_expr(
        pl.col(metric_col).min(),
        observed_count_col=observed_count_col,
        output_col=output_col or metric_col,
    )


def observed_auc_agg_expr(
    *,
    auc_bin_col: str = "auc_bin",
    observed_count_col: str = "observed_count",
    output_col: str = "auc",
) -> pl.Expr:
    """构造观察样本存在时的 AUC 聚合表达式。"""
    return observed_sum_agg_expr(
        auc_bin_col,
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def observed_iv_agg_expr(
    *,
    iv_bin_col: str = "iv_bin",
    observed_count_col: str = "observed_count",
    output_col: str = "iv",
) -> pl.Expr:
    """构造观察样本存在时的 IV 聚合表达式。"""
    return observed_sum_agg_expr(
        iv_bin_col,
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def observed_ks_agg_expr(
    *,
    ks_bin_col: str = "ks_bin",
    observed_count_col: str = "observed_count",
    output_col: str = "ks",
) -> pl.Expr:
    """构造观察样本存在时的 KS 聚合表达式。"""
    return observed_max_agg_expr(
        ks_bin_col,
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def observed_lift_max_agg_expr(
    *,
    lift_col: str = "lift",
    observed_count_col: str = "observed_count",
    output_col: str = "lift",
) -> pl.Expr:
    """构造观察样本存在时的 Lift 最大值聚合表达式。"""
    return observed_max_agg_expr(
        lift_col,
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def observed_lift_min_agg_expr(
    *,
    lift_col: str = "lift",
    observed_count_col: str = "observed_count",
    output_col: str = "lift_min",
) -> pl.Expr:
    """构造观察样本存在时的 Lift 最小值聚合表达式。"""
    return observed_min_agg_expr(
        lift_col,
        observed_count_col=observed_count_col,
        output_col=output_col,
    )


def expected_dist_expr(
    *,
    expected_count_col: str = "expected_count",
    output_col: str = "expected_dist",
    epsilon: float = 0.0,
) -> pl.Expr:
    """构造按特征归一化的 PSI 期望分布表达式。"""
    return partition_distribution_expr(
        ["feature"],
        count_col=expected_count_col,
        output_col=output_col,
        epsilon=epsilon,
    )


def binary_stats_agg_exprs(
    target_col: str,
    *,
    weight_col: str | None = None,
    count_col: str = "count",
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
) -> list[pl.Expr]:
    """组合分箱基础统计聚合表达式。"""
    return [
        binary_count_expr(weight_col=weight_col, output_col=count_col),
        binary_observed_count_expr(
            target_col,
            weight_col=weight_col,
            output_col=observed_count_col,
        ),
        binary_bad_expr(target_col, weight_col=weight_col, output_col=bad_col),
    ]


def amount_stats_agg_exprs(
    target_col: str,
    amount_col: str,
    *,
    total_amount_col: str = "tot_amt",
    good_amount_col: str = "good_amt",
    bad_amount_col: str = "bad_amt",
) -> list[pl.Expr]:
    """组合金额统计聚合表达式。"""
    return [
        total_amount_expr(amount_col, output_col=total_amount_col),
        good_amount_expr(target_col, amount_col, output_col=good_amount_col),
        bad_amount_expr(target_col, amount_col, output_col=bad_amount_col),
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
    """组合分箱分布级中间量表达式。"""
    good_value = _good_value_expr(
        observed_count_col=observed_count_col,
        bad_col=bad_col,
    )
    return [
        good_value.alias(good_col),
        total_count_expr(group_keys, count_col=count_col, output_col=total_count_col),
        total_observed_expr(
            group_keys,
            observed_count_col=observed_count_col,
            output_col=total_observed_col,
        ),
        total_bad_expr(group_keys, bad_col=bad_col, output_col=total_bad_col),
        good_value.sum().over(list(group_keys)).alias(total_good_col),
    ]


def amount_distribution_exprs(
    group_keys: Sequence[str],
    *,
    good_amount_col: str = "good_amt",
    bad_amount_col: str = "bad_amt",
    observed_amount_col: str = "observed_amt",
    total_observed_amount_col: str = "total_observed_amt",
    total_bad_amount_col: str = "total_bad_amt",
) -> list[pl.Expr]:
    """组合金额口径中间量表达式。"""
    observed_amount_value = pl.col(good_amount_col) + pl.col(bad_amount_col)
    return [
        observed_amount_value.alias(observed_amount_col),
        observed_amount_value.sum().over(list(group_keys)).alias(total_observed_amount_col),
        total_bad_amount_expr(
            group_keys,
            bad_amount_col=bad_amount_col,
            output_col=total_bad_amount_col,
        ),
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
    """组合分箱指标表达式。"""
    bad_dist_value = _bad_dist_value_expr(
        bad_col=bad_col,
        total_bad_col=total_bad_col,
        epsilon=epsilon,
    )
    good_dist_value = _good_dist_value_expr(
        observed_count_col=observed_count_col,
        bad_col=bad_col,
        total_good_col=total_good_col,
        epsilon=epsilon,
    )
    woe_value = _woe_from_dist_values_expr(
        bad_dist_expr=bad_dist_value,
        good_dist_expr=good_dist_value,
        epsilon=epsilon,
    )
    return [
        actual_dist_expr(
            count_col=count_col,
            total_count_col=total_count_col,
            output_col=actual_dist_col,
            epsilon=epsilon,
        ),
        bad_dist_value.alias(bad_dist_col),
        good_dist_value.alias(good_dist_col),
        bad_rate_expr(
            bad_col=bad_col,
            observed_count_col=observed_count_col,
            output_col=bad_rate_col,
        ),
        _lift_value_expr(
            bad_col=bad_col,
            observed_count_col=observed_count_col,
            total_bad_col=total_bad_col,
            total_observed_col=total_observed_col,
            epsilon=epsilon,
        ).alias(lift_col),
        woe_value.cast(pl.Float32).alias(woe_col),
        _iv_bin_value_expr(
            bad_dist_expr=bad_dist_value,
            good_dist_expr=good_dist_value,
            total_observed_col=total_observed_col,
            epsilon=epsilon,
        ).cast(pl.Float32).alias(iv_bin_col),
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
    """组合金额口径指标表达式。"""
    return [
        avg_amount_expr(
            total_amount_col=total_amount_col,
            count_col=count_col,
            output_col=average_amount_col,
        ),
        amount_bad_rate_expr(
            bad_amount_col=bad_amount_col,
            observed_amount_col=observed_amount_col,
            output_col=amount_bad_rate_col,
        ),
        amount_lift_expr(
            bad_amount_col=bad_amount_col,
            observed_amount_col=observed_amount_col,
            total_bad_amount_col=total_bad_amount_col,
            total_observed_amount_col=total_observed_amount_col,
            output_col=amount_lift_col,
            epsilon=epsilon,
        ),
    ]


def ordered_count_metric_exprs(
    group_keys: Sequence[str],
    *,
    count_col: str = "count",
    observed_count_col: str = "observed_count",
    bad_col: str = "bad",
    cum_count_col: str = "cum_count",
    cum_observed_count_col: str = "cum_observed_count",
    cum_bad_col: str = "cum_bad",
    cum_bad_rate_col: str = "cum_bad_rate",
) -> list[pl.Expr]:
    """组合需要排序后计算的累计件数口径表达式。"""
    partitions = list(group_keys)
    cum_observed = pl.col(observed_count_col).cum_sum().over(partitions)
    cum_bad = pl.col(bad_col).cum_sum().over(partitions)
    return [
        cum_count_expr(group_keys, count_col=count_col, output_col=cum_count_col),
        cum_observed_count_expr(
            group_keys,
            observed_count_col=observed_count_col,
            output_col=cum_observed_count_col,
        ),
        cum_bad_expr(group_keys, bad_col=bad_col, output_col=cum_bad_col),
        _cum_bad_rate_value_expr(
            cum_bad_expr=cum_bad,
            cum_observed_count_expr=cum_observed,
        ).alias(cum_bad_rate_col),
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
    """组合需要排序后计算的 KS/AUC 表达式。"""
    partitions = list(group_keys)
    cum_bad_dist = pl.col(bad_dist_col).cum_sum().over(partitions)
    cum_good_dist = pl.col(good_dist_col).cum_sum().over(partitions)
    # Polars 1.8 不允许在窗口表达式外再套一层 shift().over()。排序后的前一累计值
    # 等价于当前累计值减去当前箱分布，且能保留单次 with_columns 的 public 用法。
    prev_bad_dist = cum_bad_dist - pl.col(bad_dist_col)
    prev_good_dist = cum_good_dist - pl.col(good_dist_col)
    return [
        cum_bad_dist_expr(
            group_keys,
            bad_dist_col=bad_dist_col,
            output_col=cum_bad_dist_col,
        ),
        cum_good_dist_expr(
            group_keys,
            good_dist_col=good_dist_col,
            output_col=cum_good_dist_col,
        ),
        _ks_bin_value_expr(
            cum_bad_dist_expr=cum_bad_dist,
            cum_good_dist_expr=cum_good_dist,
            total_observed_col=total_observed_col,
        ).alias(ks_bin_col),
        _auc_bin_value_expr(
            cum_bad_dist_expr=cum_bad_dist,
            cum_good_dist_expr=cum_good_dist,
            prev_bad_dist_expr=prev_bad_dist,
            prev_good_dist_expr=prev_good_dist,
            total_observed_col=total_observed_col,
        ).alias(auc_bin_col),
    ]


__all__ = [
    "actual_dist_expr",
    "amount_bad_rate_expr",
    "amount_distribution_exprs",
    "amount_lift_expr",
    "amount_metric_exprs",
    "amount_stats_agg_exprs",
    "auc_bin_expr",
    "avg_amount_expr",
    "bad_amount_expr",
    "bad_dist_expr",
    "bad_rate_agg_expr",
    "bad_rate_expr",
    "bin_missing_rate_expr",
    "binary_bad_expr",
    "binary_count_expr",
    "binary_distribution_exprs",
    "binary_metric_exprs",
    "binary_observed_count_expr",
    "binary_stats_agg_exprs",
    "binary_unobserved_count_expr",
    "cum_bad_dist_expr",
    "cum_bad_expr",
    "cum_bad_rate_expr",
    "cum_count_expr",
    "cum_good_dist_expr",
    "cum_observed_count_expr",
    "distribution_rate_expr",
    "expected_dist_expr",
    "global_distribution_expr",
    "good_amount_expr",
    "good_dist_expr",
    "good_expr",
    "iv_bin_expr",
    "ks_bin_expr",
    "lift_expr",
    "normalized_auc_expr",
    "observed_amount_expr",
    "observed_auc_agg_expr",
    "observed_iv_agg_expr",
    "observed_ks_agg_expr",
    "observed_lift_max_agg_expr",
    "observed_lift_min_agg_expr",
    "observed_max_agg_expr",
    "observed_min_agg_expr",
    "observed_sum_agg_expr",
    "OrderedMetricSortBy",
    "normalize_ordered_metric_sort_by",
    "ordered_binary_metric_exprs",
    "ordered_count_metric_exprs",
    "partition_distribution_expr",
    "ratio_expr",
    "total_amount_expr",
    "total_bad_amount_expr",
    "total_bad_expr",
    "total_count_expr",
    "total_good_expr",
    "total_observed_amount_expr",
    "total_observed_expr",
    "woe_expr",
]
