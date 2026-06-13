"""分析与监控共用的 PSI/稳定性算子。"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from mars.core.constants import METRIC_EPSILON


def psi_valid_condition(
    bin_expr: pl.Expr,
    *,
    include_missing: bool,
    include_special: bool,
    is_numeric_bin: bool = True,
    special_values: Sequence[Any] | None = None,
) -> pl.Expr:
    """构造参与 PSI 计算的分箱筛选条件。"""
    condition = pl.lit(True)
    if is_numeric_bin:
        if not include_missing:
            condition &= bin_expr != -1
        if not include_special:
            condition &= bin_expr > -3
        return condition

    if not include_missing:
        condition &= (bin_expr != "Missing") & bin_expr.is_not_null()
    if not include_special and special_values:
        special_str_list = [str(value) for value in special_values]
        condition &= ~bin_expr.is_in(special_str_list)
    return condition


def psi_contribution_expr(
    actual_expr: pl.Expr,
    expected_expr: pl.Expr,
    *,
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造单个分箱的 PSI 贡献表达式。"""
    return (actual_expr - expected_expr) * (actual_expr / (expected_expr + epsilon)).log()


def with_psi_from_counts(
    df: pl.DataFrame,
    *,
    group_col: str,
    feature_col: str = "feature",
    bin_col: str = "bin_index",
    count_col: str = "count",
    expected_dist_col: str = "expected_dist",
    output_col: str = "psi_bin",
    include_missing: bool = True,
    include_special: bool = True,
    epsilon: float = METRIC_EPSILON,
) -> pl.DataFrame:
    """基于分组计数与期望分布计算分箱级 PSI。"""
    valid_condition = psi_valid_condition(
        pl.col(bin_col),
        include_missing=include_missing,
        include_special=include_special,
        is_numeric_bin=True,
    )
    with_totals = df.with_columns(
        [
            pl.col(count_col)
            .filter(valid_condition)
            .sum()
            .over([group_col, feature_col])
            .alias("total_count_psi"),
            pl.col(expected_dist_col)
            .filter(valid_condition)
            .sum()
            .over([group_col, feature_col])
            .alias("total_expected_dist_psi"),
        ]
    )
    return (
        with_totals.with_columns(
            [
                (pl.col(count_col) / (pl.col("total_count_psi") + epsilon)).alias("act_prob_clean"),
                (
                    pl.col(expected_dist_col)
                    / (pl.col("total_expected_dist_psi") + epsilon)
                ).alias("exp_prob_clean"),
            ]
        )
        .with_columns(
            pl.when(valid_condition)
            .then(
                psi_contribution_expr(
                    pl.col("act_prob_clean"),
                    pl.col("exp_prob_clean"),
                    epsilon=epsilon,
                )
            )
            .otherwise(None)
            .alias(output_col)
        )
    )
