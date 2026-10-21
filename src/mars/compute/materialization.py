"""DataFrame 物化与类型恢复的共享工具。"""

from __future__ import annotations

from typing import Any, Union

import pandas as pd
import polars as pl

FrameLike = Union[pd.DataFrame, pl.DataFrame]


def is_polars_dataframe(df: Any) -> bool:
    """判断对象是否为立即执行的 Polars DataFrame。"""
    return isinstance(df, pl.DataFrame)


def to_pandas_frame(df: FrameLike) -> pd.DataFrame:
    """返回 Pandas 数据框副本，Polars 输入先转换；其他类型抛出 TypeError。"""
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def to_polars_frame(df: FrameLike) -> pl.DataFrame:
    """返回 Polars 数据框副本，Pandas 输入先转换；其他类型抛出 TypeError。"""
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def restore_frame_type(df: FrameLike, prefer_polars: bool) -> FrameLike:
    """按 prefer_polars 返回指定数据框类型，类型已匹配时直接返回原对象。"""
    if prefer_polars:
        if isinstance(df, pl.DataFrame):
            return df
        return pl.from_pandas(df)
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


def to_pandas_table(table: Any) -> pd.DataFrame:
    """将报告表转为 Pandas 数据框；None 返回空表，已有 Pandas 表返回副本。"""
    if table is None:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, pl.DataFrame):
        return table.to_pandas()
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame(table)
