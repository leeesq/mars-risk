"""数据画像 current/benchmark schema 与未见类别对比。"""

from __future__ import annotations

from typing import Any

import polars as pl

from mars._compat import polars_is_in
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.compute import filter_compatible_values, is_numeric_dtype

_UNSEEN_META_COLUMNS = {
    "feature",
    "current_dtype",
    "benchmark_dtype",
    "status",
    "reason",
    "benchmark_unique_count",
    "valid_count",
    "unseen_count",
    "unseen_unique_count",
    "total",
}


def _dtype_family(dtype: pl.DataType | None) -> str:
    """将 Polars dtype 收口到 comparison 兼容族。"""
    if dtype is None:
        return "missing"
    if is_numeric_dtype(dtype):
        return "numeric"
    dtype_name = str(dtype).lower()
    if any(token in dtype_name for token in ("string", "utf8", "categorical", "enum", "bool")):
        return "categorical"
    if any(token in dtype_name for token in ("date", "datetime", "duration", "time")):
        return "temporal"
    return dtype_name


def _dtypes_comparable(current: pl.DataType, benchmark: pl.DataType) -> bool:
    """判断两侧 dtype 是否允许进行值级比较。"""
    return _dtype_family(current) == _dtype_family(benchmark)


def build_schema_comparison(
    current_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    features: list[str],
) -> pl.DataFrame:
    """构造两侧特征存在性、dtype 漂移和可比较性表。"""
    rows: list[dict[str, Any]] = []
    for feature in features:
        current_dtype = current_df.schema.get(feature)
        benchmark_dtype = benchmark_df.schema.get(feature)
        if current_dtype is None:
            status = "benchmark_only"
            comparable = False
            reason = "feature is absent from current data"
        elif benchmark_dtype is None:
            status = "current_only"
            comparable = False
            reason = "feature is absent from benchmark data"
        elif current_dtype == benchmark_dtype:
            status = "matched"
            comparable = True
            reason = None
        elif _dtypes_comparable(current_dtype, benchmark_dtype):
            status = "compatible_change"
            comparable = True
            reason = "dtype changed within a comparable family"
        else:
            status = "incompatible_change"
            comparable = False
            reason = "dtype families are not comparable"
        rows.append(
            {
                "feature": feature,
                "current_dtype": str(current_dtype) if current_dtype is not None else None,
                "benchmark_dtype": (
                    str(benchmark_dtype) if benchmark_dtype is not None else None
                ),
                "status": status,
                "comparable": comparable,
                "reason": reason,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "feature": pl.Utf8,
            "current_dtype": pl.Utf8,
            "benchmark_dtype": pl.Utf8,
            "status": pl.Utf8,
            "comparable": pl.Boolean,
            "reason": pl.Utf8,
        },
    )


def _valid_value_expr(
    feature: str,
    dtype: pl.DataType,
    options: ProfileComputeOptions,
) -> pl.Expr:
    """构造 unseen 分母使用的有效值条件。"""
    column = pl.col(feature)
    condition = column.is_not_null()
    if dtype in [pl.Float32, pl.Float64]:
        condition &= column.is_not_nan()
    excluded_values = filter_compatible_values(
        dtype,
        [*options.missing_values, *options.special_values],
    )
    if excluded_values:
        condition &= ~polars_is_in(column, pl.Series(excluded_values))
    return condition


def _normalized_value_expr(feature: str, dtype: pl.DataType) -> pl.Expr:
    """将同一 dtype 族规范化为可跨宽度比较的值。"""
    if is_numeric_dtype(dtype):
        return pl.col(feature).cast(pl.Float64).alias("_value")
    return pl.col(feature).cast(pl.Utf8).alias("_value")


def _is_categorical_feature(
    feature: str,
    current_dtype: pl.DataType | None,
    benchmark_dtype: pl.DataType | None,
    categorical_features: set[str],
) -> bool:
    """根据 dtype 与显式声明判断 unseen 适用性。"""
    if feature in categorical_features:
        return True
    families = {
        _dtype_family(dtype)
        for dtype in (current_dtype, benchmark_dtype)
        if dtype is not None
    }
    return bool(families) and families <= {"categorical"}


def _safe_group_label(value: Any) -> str:
    """生成不会覆盖 unseen 固定字段的分组列名。"""
    label = "null" if value is None else str(value)
    return f"group:{label}" if label in _UNSEEN_META_COLUMNS else label


def _unique_group_labels(values: list[Any]) -> list[str]:
    """Return stable group labels without duplicate output columns."""
    labels: list[str] = []
    counts: dict[str, int] = {}
    for value in values:
        base = _safe_group_label(value)
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        labels.append(base if occurrence == 0 else f"{base}#{occurrence + 1}")
    return labels


def build_unseen_comparison(
    context: ProfileRunContext,
    benchmark_df: pl.DataFrame,
    features: list[str],
    options: ProfileComputeOptions,
) -> pl.DataFrame:
    """计算 benchmark 类别集合之外的 current 样本占比。"""
    categorical_features = set(options.categorical_features)
    group_col = context.group_col
    group_values: list[Any] = []
    group_columns: list[str] = []
    if group_col is not None:
        group_values = sorted(
            context.working_df.get_column(group_col).unique().to_list(),
            key=lambda value: str(value),
        )
        group_columns = _unique_group_labels(group_values)
    group_label_map = dict(zip(group_values, group_columns))

    rows: list[dict[str, Any]] = []
    for feature in features:
        current_dtype = context.df.schema.get(feature)
        benchmark_dtype = benchmark_df.schema.get(feature)
        row: dict[str, Any] = {
            "feature": feature,
            "current_dtype": str(current_dtype) if current_dtype is not None else None,
            "benchmark_dtype": str(benchmark_dtype) if benchmark_dtype is not None else None,
            "status": "comparable",
            "reason": None,
            "benchmark_unique_count": None,
            "valid_count": None,
            "unseen_count": None,
            "unseen_unique_count": None,
            "total": None,
        }
        row.update({column: None for column in group_columns})

        if current_dtype is None:
            row.update(status="benchmark_only", reason="feature is absent from current data")
        elif benchmark_dtype is None:
            row.update(status="current_only", reason="feature is absent from benchmark data")
        elif not _is_categorical_feature(
            feature,
            current_dtype,
            benchmark_dtype,
            categorical_features,
        ):
            row.update(status="not_applicable", reason="feature is not categorical")
        elif not _dtypes_comparable(current_dtype, benchmark_dtype):
            row.update(status="incompatible_dtype", reason="dtype families are not comparable")
        else:
            benchmark_values = (
                benchmark_df
                .filter(_valid_value_expr(feature, benchmark_dtype, options))
                .select(_normalized_value_expr(feature, benchmark_dtype))
                .get_column("_value")
                .unique()
            )
            row["benchmark_unique_count"] = int(benchmark_values.len())
            if benchmark_values.len() == 0:
                row.update(status="no_reference_values", reason="benchmark has no valid values")
            else:
                current_values = (
                    context.working_df
                    .filter(_valid_value_expr(feature, current_dtype, options))
                    .select(
                        [
                            _normalized_value_expr(feature, current_dtype),
                            *(
                                [pl.col(group_col)]
                                if group_col is not None
                                else []
                            ),
                        ]
                    )
                    .with_columns(
                        (~polars_is_in(pl.col("_value"), benchmark_values)).alias("_unseen")
                    )
                )
                valid_count = current_values.height
                row["valid_count"] = int(valid_count)
                if valid_count == 0:
                    row.update(status="no_current_values", reason="current has no valid values")
                else:
                    unseen_values = current_values.filter(pl.col("_unseen"))
                    unseen_count = unseen_values.height
                    row["unseen_count"] = int(unseen_count)
                    row["unseen_unique_count"] = int(
                        unseen_values.get_column("_value").n_unique()
                    )
                    row["total"] = float(unseen_count / valid_count)
                    if group_col is not None:
                        grouped = (
                            current_values
                            .group_by(group_col)
                            .agg(
                                [
                                    pl.len().alias("_valid_count"),
                                    pl.col("_unseen").sum().alias("_unseen_count"),
                                ]
                            )
                        )
                        group_rate_map = {
                            group_label_map[group_row[group_col]]: (
                                float(group_row["_unseen_count"])
                                / float(group_row["_valid_count"])
                            )
                            for group_row in grouped.to_dicts()
                        }
                        row.update(group_rate_map)

        rows.append(row)

    base_columns = [
        "feature",
        "current_dtype",
        "benchmark_dtype",
        "status",
        "reason",
        "benchmark_unique_count",
        "valid_count",
        "unseen_count",
        "unseen_unique_count",
        *group_columns,
        "total",
    ]
    schema: dict[str, pl.DataType] = {
        "feature": pl.Utf8,
        "current_dtype": pl.Utf8,
        "benchmark_dtype": pl.Utf8,
        "status": pl.Utf8,
        "reason": pl.Utf8,
        "benchmark_unique_count": pl.Int64,
        "valid_count": pl.Int64,
        "unseen_count": pl.Int64,
        "unseen_unique_count": pl.Int64,
        **{column: pl.Float64 for column in group_columns},
        "total": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).select(base_columns)
