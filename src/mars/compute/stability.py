"""稳定性相关的纯 Polars 表达式工厂。"""

from __future__ import annotations

from typing import Literal, Sequence, cast

import polars as pl

from mars.core.constants import DIVISION_EPSILON, METRIC_EPSILON

RiskCorrBaseline = Literal["total", "first_group", "benchmark"]
RiskCorrMethod = Literal["pearson", "spearman"]


def psi_valid_condition(
    bin_expr: pl.Expr,
    *,
    include_missing: bool,
    include_special: bool,
) -> pl.Expr:
    """构造基于 Int16 分箱索引的 PSI 有效箱筛选条件。"""
    condition = pl.lit(True)
    if not include_missing:
        condition &= bin_expr != -1
    if not include_special:
        condition &= bin_expr > -3
    return condition


def psi_total_count_expr(
    group_keys: Sequence[str],
    valid_condition: pl.Expr,
    *,
    count_col: str = "count",
    output_col: str = "total_count_psi",
) -> pl.Expr:
    """构造 PSI 有效样本总数窗口表达式。"""
    return pl.col(count_col).filter(valid_condition).sum().over(list(group_keys)).alias(output_col)


def psi_total_expected_expr(
    group_keys: Sequence[str],
    valid_condition: pl.Expr,
    *,
    expected_dist_col: str = "expected_dist",
    output_col: str = "total_expected_dist_psi",
) -> pl.Expr:
    """构造 PSI 有效期望分布总量窗口表达式。"""
    return (
        pl.col(expected_dist_col)
        .filter(valid_condition)
        .sum()
        .over(list(group_keys))
        .alias(output_col)
    )


def psi_actual_prob_expr(
    *,
    count_col: str = "count",
    total_count_col: str = "total_count_psi",
    output_col: str = "act_prob_clean",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造 PSI 实际分布概率表达式。"""
    return (pl.col(count_col) / (pl.col(total_count_col) + epsilon)).alias(output_col)


def psi_expected_prob_expr(
    *,
    expected_dist_col: str = "expected_dist",
    total_expected_dist_col: str = "total_expected_dist_psi",
    output_col: str = "exp_prob_clean",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造 PSI 期望分布概率表达式。"""
    return (
        pl.col(expected_dist_col) / (pl.col(total_expected_dist_col) + epsilon)
    ).alias(output_col)


def _psi_partition_prob_value_expr(
    partition_keys: Sequence[str],
    *,
    count_col: str,
    epsilon: float,
) -> pl.Expr:
    """构造未命名的分区概率表达式。"""
    partitions = list(partition_keys)
    return pl.col(count_col) / (pl.col(count_col).sum().over(partitions) + epsilon)


def psi_partition_prob_expr(
    partition_keys: Sequence[str],
    *,
    count_col: str = "len",
    output_col: str,
    epsilon: float = DIVISION_EPSILON,
) -> pl.Expr:
    """构造指定分区内的 PSI 分布概率表达式。"""
    return _psi_partition_prob_value_expr(
        partition_keys,
        count_col=count_col,
        epsilon=epsilon,
    ).alias(output_col)


def psi_contribution_expr(
    actual_expr: pl.Expr,
    expected_expr: pl.Expr,
    *,
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造单个分箱的 PSI 贡献表达式。"""
    return (actual_expr - expected_expr) * (actual_expr / (expected_expr + epsilon)).log()


def psi_bin_expr(
    valid_condition: pl.Expr,
    *,
    actual_prob_col: str = "act_prob_clean",
    expected_prob_col: str = "exp_prob_clean",
    output_col: str = "psi_bin",
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造单箱 PSI 表达式。"""
    return (
        pl.when(valid_condition)
        .then(
            psi_contribution_expr(
                pl.col(actual_prob_col),
                pl.col(expected_prob_col),
                epsilon=epsilon,
            ),
        )
        .otherwise(None)
        .alias(output_col)
    )


def psi_exprs(
    group_keys: Sequence[str],
    *,
    bin_col: str = "bin_index",
    count_col: str = "count",
    expected_dist_col: str = "expected_dist",
    total_count_col: str = "total_count_psi",
    total_expected_dist_col: str = "total_expected_dist_psi",
    actual_prob_col: str = "act_prob_clean",
    expected_prob_col: str = "exp_prob_clean",
    output_col: str = "psi_bin",
    include_missing: bool = True,
    include_special: bool = True,
    epsilon: float = METRIC_EPSILON,
) -> list[pl.Expr]:
    """组合 PSI 支撑列与分箱贡献表达式。"""
    valid_condition = psi_valid_condition(
        pl.col(bin_col),
        include_missing=include_missing,
        include_special=include_special,
    )
    partitions = list(group_keys)
    total_count = pl.col(count_col).filter(valid_condition).sum().over(partitions)
    total_expected = (
        pl.col(expected_dist_col)
        .filter(valid_condition)
        .sum()
        .over(partitions)
    )
    actual_prob_raw = pl.col(count_col) / (total_count + epsilon)
    expected_prob_raw = pl.col(expected_dist_col) / (total_expected + epsilon)
    return [
        total_count.alias(total_count_col),
        total_expected.alias(total_expected_dist_col),
        actual_prob_raw.alias(actual_prob_col),
        expected_prob_raw.alias(expected_prob_col),
        (
            pl.when(valid_condition)
            .then(
                psi_contribution_expr(
                    actual_prob_raw,
                    expected_prob_raw,
                    epsilon=epsilon,
                ),
            )
            .otherwise(None)
            .alias(output_col)
        ),
    ]


def normalize_risk_corr_baseline(value: str | None) -> RiskCorrBaseline:
    """标准化并校验 RiskCorr 基准模式。"""
    normalized = str(value or "total").strip().lower()
    valid_modes = {"total", "first_group", "benchmark"}
    if normalized not in valid_modes:
        raise ValueError(
            "risk_corr_baseline must be one of {'total', 'first_group', 'benchmark'}, "
            f"got {value!r}.",
        )
    return cast(RiskCorrBaseline, normalized)


def risk_corr_expr(
    *,
    bad_rate_col: str = "bad_rate",
    reference_col: str = "base_br",
    observed_count_col: str = "observed_count",
    method: RiskCorrMethod = "spearman",
    output_col: str = "risk_corr",
) -> pl.Expr:
    """构造分组级 RC 聚合表达式。"""
    return (
        pl.when(pl.col(observed_count_col).sum() <= 0)
        .then(pl.lit(None).cast(pl.Float64))
        .when(pl.len() > 1)
        .then(pl.corr(bad_rate_col, reference_col, method=method))
        .otherwise(pl.lit(1.0))
        .fill_nan(1.0)
        .alias(output_col)
    )


__all__ = [
    "RiskCorrBaseline",
    "RiskCorrMethod",
    "normalize_risk_corr_baseline",
    "psi_actual_prob_expr",
    "psi_bin_expr",
    "psi_contribution_expr",
    "psi_expected_prob_expr",
    "psi_exprs",
    "psi_partition_prob_expr",
    "psi_total_count_expr",
    "psi_total_expected_expr",
    "psi_valid_condition",
    "risk_corr_expr",
]
