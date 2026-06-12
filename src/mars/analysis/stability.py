"""PSI 与稳定性指标的共享计算工具。"""

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
    """
    构建 PSI 计算时需要保留的分箱条件。

    Parameters
    ----------
    bin_expr : pl.Expr
        分箱编号或分箱标签表达式。
    include_missing : bool
        是否保留缺失值箱。
    include_special : bool
        是否保留特殊值箱。
    is_numeric_bin : bool
        分箱是否使用 MARS 数值索引约定。
    special_values : Sequence[Any] | None
        类别分箱场景下的业务特殊值。

    Returns
    -------
    polars.Expr
        可直接用于 ``filter`` 或 ``when`` 的布尔表达式。
    """
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
    """
    构建单箱 PSI 贡献表达式。

    Parameters
    ----------
    actual_expr : pl.Expr
        当前分布占比表达式。
    expected_expr : pl.Expr
        基准分布占比表达式。
    epsilon : float
        平滑系数，避免除零和对数异常。

    Returns
    -------
    polars.Expr
        ``(A - E) * log(A / E)`` 的 Polars 表达式。
    """
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
    """
    基于聚合 count 和 expected distribution 计算单箱 PSI。

    Parameters
    ----------
    df : pl.DataFrame
        已包含当前分箱样本量和基准分布的长表。
    group_col : str
        分组列名。
    feature_col : str
        特征列名。
    bin_col : str
        分箱编号列名。
    count_col : str
        当前样本量列名。
    expected_dist_col : str
        基准分布占比列名。
    output_col : str
        输出 PSI 贡献列名。
    include_missing : bool
        是否纳入缺失值箱。
    include_special : bool
        是否纳入特殊值箱。
    epsilon : float
        平滑系数。

    Returns
    -------
    pl.DataFrame
        增加 PSI 专用分布列和单箱 PSI 贡献列后的表。
    """
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
