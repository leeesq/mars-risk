"""共享缺失语义与缺失统计算子。"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import polars as pl

_NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}
_FLOAT_DTYPES = {pl.Float32, pl.Float64}
_STRING_DTYPES = {pl.String}


def is_numeric_dtype(dtype: pl.DataType | None) -> bool:
    """判断 dtype 是否属于支持的数值类型。"""
    return dtype in _NUMERIC_DTYPES


def is_float_dtype(dtype: pl.DataType | None) -> bool:
    """判断 dtype 是否属于支持的浮点类型。"""
    return dtype in _FLOAT_DTYPES


def filter_compatible_values(
    dtype: pl.DataType | None,
    values: Sequence[Any] | None,
) -> list[Any]:
    """仅保留可与当前列 dtype 安全比较的取值。"""
    if not values:
        return []

    is_num = is_numeric_dtype(dtype)
    is_str = dtype in _STRING_DTYPES
    valid_values: list[Any] = []
    for value in values:
        if is_num and isinstance(value, (int, float)) and not isinstance(value, bool):
            valid_values.append(value)
        elif is_str and isinstance(value, str):
            valid_values.append(value)
    return valid_values


def values_to_exclude(
    dtype: pl.DataType | None,
    *,
    missing_values: Sequence[Any] | None = None,
    special_values: Sequence[Any] | None = None,
) -> list[Any]:
    """返回缺失值与特殊值的类型安全并集。"""
    candidates = list(missing_values or []) + list(special_values or [])
    return filter_compatible_values(dtype, candidates)


def missing_condition_expr(
    column: str | pl.Expr,
    *,
    dtype: pl.DataType | None = None,
    missing_values: Sequence[Any] | None = None,
) -> pl.Expr:
    """为列构造共享缺失判断表达式。"""
    expr = pl.col(column) if isinstance(column, str) else column
    condition = expr.is_null()

    if is_float_dtype(dtype):
        condition |= expr.is_nan()

    valid_missing = filter_compatible_values(dtype, missing_values)
    if valid_missing:
        condition |= expr.is_in(valid_missing)

    return condition


def missing_rate_expr(
    column: str | pl.Expr,
    *,
    dtype: pl.DataType | None = None,
    missing_values: Sequence[Any] | None = None,
) -> pl.Expr:
    """基于共享缺失语义构造缺失率表达式。"""
    return (
        missing_condition_expr(
            column,
            dtype=dtype,
            missing_values=missing_values,
        ).sum()
        / pl.len()
    )


def _feature_dtype_frame(
    schema: dict[str, pl.DataType],
    features: Iterable[str],
) -> pl.DataFrame:
    """为选中特征构造 feature 到 dtype 的映射表。"""
    return pl.DataFrame(
        {
            "feature": list(features),
            "dtype": [str(schema[feature]) for feature in features],
        }
    )


def build_missing_by_period_stats(
    df: pl.DataFrame,
    *,
    features: Sequence[str],
    period_col: str,
    missing_values: Sequence[Any] | None = None,
) -> pl.DataFrame:
    """使用共享表达式按时间维度构造缺失率宽表。"""
    if period_col not in df.columns:
        raise ValueError(f"Column '{period_col}' not found.")

    target_features = [feature for feature in features if feature in df.columns and feature != period_col]
    if not target_features:
        return pl.DataFrame()

    schema = df.schema
    total_exprs = [
        missing_rate_expr(
            feature,
            dtype=schema.get(feature),
            missing_values=missing_values,
        ).alias(feature)
        for feature in target_features
    ]
    total_df = (
        df.select(total_exprs)
        .transpose(include_header=True, header_name="feature", column_names=["total"])
    )
    base_df = total_df.join(_feature_dtype_frame(schema, target_features), on="feature", how="left")

    grouped = (
        df.group_by(period_col)
        .agg(total_exprs)
        .sort(period_col)
        .with_columns(pl.col(period_col).cast(pl.String))
    )
    pivot_df = grouped.transpose(
        include_header=True,
        header_name="feature",
        column_names=period_col,
    )
    result = base_df.join(pivot_df, on="feature", how="left")

    fixed_cols = {"feature", "dtype", "total"}
    group_cols = [col for col in result.columns if col not in fixed_cols]
    return result.select(["feature", "dtype"] + group_cols + ["total"]).sort(["dtype", "feature"])
