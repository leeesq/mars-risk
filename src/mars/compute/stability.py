"""稳定性相关表达式工厂。"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import polars as pl

from mars.core.constants import METRIC_EPSILON

RiskCorrBaseline = Literal["total", "first_group", "benchmark"]


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
        condition &= ~bin_expr.is_in(pl.Series(special_str_list).implode())
    return condition


def psi_contribution_expr(
    actual_expr: pl.Expr,
    expected_expr: pl.Expr,
    *,
    epsilon: float = METRIC_EPSILON,
) -> pl.Expr:
    """构造单个分箱的 PSI 贡献表达式。"""
    return (actual_expr - expected_expr) * (actual_expr / (expected_expr + epsilon)).log()


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
    """构造 PSI 支撑列与分箱贡献表达式列表。"""
    partitions = list(group_keys)
    valid_condition = psi_valid_condition(
        pl.col(bin_col),
        include_missing=include_missing,
        include_special=include_special,
        is_numeric_bin=True,
    )
    total_count_expr = (
        pl.col(count_col)
        .filter(valid_condition)
        .sum()
        .over(partitions)
    )
    total_expected_expr = (
        pl.col(expected_dist_col)
        .filter(valid_condition)
        .sum()
        .over(partitions)
    )
    actual_prob_expr = pl.col(count_col) / (total_count_expr + epsilon)
    expected_prob_expr = pl.col(expected_dist_col) / (total_expected_expr + epsilon)
    return [
        total_count_expr.alias(total_count_col),
        total_expected_expr.alias(total_expected_dist_col),
        actual_prob_expr.alias(actual_prob_col),
        expected_prob_expr.alias(expected_prob_col),
        (
            pl.when(valid_condition)
            .then(
                psi_contribution_expr(
                    actual_prob_expr,
                    expected_prob_expr,
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
    return normalized  # type: ignore[return-value]


def risk_corr_expr(
    *,
    bad_rate_col: str = "bad_rate",
    reference_col: str = "base_br",
    observed_count_col: str = "observed_count",
    output_col: str = "risk_corr",
) -> pl.Expr:
    """构造分组级 RC 聚合表达式。"""
    return (
        pl.when(pl.col(observed_count_col).sum() <= 0)
        .then(pl.lit(None).cast(pl.Float64))
        .when(pl.len() > 1)
        .then(pl.corr(bad_rate_col, reference_col, method="spearman"))
        .otherwise(pl.lit(1.0))
        .fill_nan(1.0)
        .alias(output_col)
    )


__all__ = [
    "RiskCorrBaseline",
    "normalize_risk_corr_baseline",
    "psi_contribution_expr",
    "psi_exprs",
    "psi_valid_condition",
    "risk_corr_expr",
]
