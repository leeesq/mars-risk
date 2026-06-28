"""数据画像运行上下文构建。"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import polars as pl

from mars.analysis._profiling.types import FrameDtype, ProfileRunContext
from mars.utils.date import MarsDate
from mars.utils.logger import logger


def normalize_include_dtypes(include_dtypes: FrameDtype | list[FrameDtype] | None) -> list[Any] | None:
    """把 Python 原生类型转换为 Polars dtype selector 可识别的类型。"""
    if include_dtypes is None:
        return None

    raw_dtypes = include_dtypes if isinstance(include_dtypes, list) else [include_dtypes]
    target_dtypes: list[Any] = []
    for dtype in raw_dtypes:
        if dtype is int:
            target_dtypes.extend(
                [
                    cast(pl.DataType, pl.Int8),
                    cast(pl.DataType, pl.Int16),
                    cast(pl.DataType, pl.Int32),
                    cast(pl.DataType, pl.Int64),
                    cast(pl.DataType, pl.UInt8),
                    cast(pl.DataType, pl.UInt16),
                    cast(pl.DataType, pl.UInt32),
                    cast(pl.DataType, pl.UInt64),
                ]
            )
        elif dtype is float:
            target_dtypes.extend([cast(pl.DataType, pl.Float32), cast(pl.DataType, pl.Float64)])
        elif dtype is str:
            target_dtypes.append(pl.String)
        elif dtype is bool:
            target_dtypes.append(pl.Boolean)
        elif dtype is list:
            target_dtypes.append(pl.List)
        else:
            target_dtypes.append(dtype)
    return target_dtypes


def prepare_profile_data(
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    *,
    ensure_polars: Any,
    features: list[str] | None,
    exclude_features: list[str] | None,
    include_dtypes: FrameDtype | list[FrameDtype] | None,
    sample_frac: float | None,
) -> tuple[pl.DataFrame, list[str]]:
    """准备单次画像运行使用的数据副本和特征范围。"""
    df_pl = ensure_polars(df)
    if isinstance(df_pl, pl.LazyFrame):
        df_pl = df_pl.collect()

    if sample_frac is not None:
        if not 0 < sample_frac < 1:
            raise ValueError("`sample_frac` must be in (0, 1).")
        logger.warning("[SAMPLE] Data is sampled (frac=%s). Metrics are estimates.", sample_frac)
        df_pl = df_pl.sample(fraction=sample_frac, shuffle=True)

    candidates = list(features) if features else list(df_pl.columns)
    if exclude_features:
        exclude_set = set(exclude_features)
        candidates = [col for col in candidates if col not in exclude_set]

    target_dtypes = normalize_include_dtypes(include_dtypes)
    if target_dtypes:
        import polars.selectors as cs

        try:
            dtype_selector = cs.by_dtype(target_dtypes)
            candidates = df_pl.select(pl.col(candidates)).select(dtype_selector).columns
        except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            logger.warning(
                "Type filtering failed: %s. Falling back to direct schema filtering.",
                exc,
            )
            candidates = [col for col in candidates if df_pl.schema[col] in target_dtypes]

    if not candidates and (features or exclude_features or include_dtypes):
        raise ValueError("No features selected after filtering.")

    return df_pl, candidates


def resolve_profile_group(
    *,
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
) -> str | None:
    """把分组列和时间粒度参数解析为内部趋势维度。"""
    if group_col:
        return group_col
    if time_col:
        return time_grain or "month"
    return None


def build_run_context(
    df: pl.DataFrame,
    *,
    features: list[str],
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
) -> ProfileRunContext:
    """构造单次画像上下文。"""
    profile_by = resolve_profile_group(
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
    )
    working_df = df
    effective_group_col = profile_by

    if time_col and profile_by is not None and MarsDate.is_time_grain(profile_by):
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in DataFrame.")

        temp_group_col = f"_mars_auto_{profile_by}"
        working_df = df.with_columns(MarsDate.from_grain(time_col, profile_by).alias(temp_group_col))
        effective_group_col = temp_group_col
    elif time_col is None and profile_by is not None and profile_by not in df.columns:
        raise ValueError(f"Column '{profile_by}' not found. Did you forget to set `time_col`?")

    return ProfileRunContext(
        df=df,
        working_df=working_df,
        features=features,
        dtype_map=dict(df.schema),
        group_col=effective_group_col,
    )
