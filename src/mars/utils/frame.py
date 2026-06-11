"""DataFrame 输入输出转换工具。"""

from __future__ import annotations

from typing import Any, Union

import pandas as pd
import polars as pl

FrameLike = Union[pd.DataFrame, pl.DataFrame]


def is_polars_dataframe(df: Any) -> bool:
    """
    判断对象是否为 Polars eager DataFrame。

    Parameters
    ----------
    df : Any
        待检查对象。

    Returns
    -------
    bool
        若对象是 ``polars.DataFrame``，返回 ``True``。

    Examples
    --------
    >>> import polars as pl
    >>> is_polars_dataframe(pl.DataFrame({"x": [1]}))
    True
    """
    return isinstance(df, pl.DataFrame)


def to_pandas_frame(df: FrameLike) -> pd.DataFrame:
    """
    将 Pandas 或 Polars DataFrame 转为 Pandas 副本。

    Parameters
    ----------
    df : FrameLike
        输入数据框。

    Returns
    -------
    pandas.DataFrame
        Pandas 数据框副本。

    Raises
    ------
    TypeError
        输入类型不是 Pandas 或 Polars DataFrame 时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> to_pandas_frame(pl.DataFrame({"x": [1]})).shape
    (1, 1)
    """
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def to_polars_frame(df: FrameLike) -> pl.DataFrame:
    """
    将 Pandas 或 Polars DataFrame 转为 Polars 副本。

    Parameters
    ----------
    df : FrameLike
        输入数据框。

    Returns
    -------
    polars.DataFrame
        Polars eager 数据框副本。

    Raises
    ------
    TypeError
        输入类型不是 Pandas 或 Polars DataFrame 时抛出。

    Examples
    --------
    >>> import pandas as pd
    >>> to_polars_frame(pd.DataFrame({"x": [1]})).shape
    (1, 1)
    """
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def restore_frame_type(df: FrameLike, prefer_polars: bool) -> FrameLike:
    """
    按调用方偏好的数据引擎恢复输出类型。

    Parameters
    ----------
    df : FrameLike
        内部处理后的数据框。
    prefer_polars : bool
        是否优先返回 Polars DataFrame。

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        与输入链路一致的数据框。

    Examples
    --------
    >>> import pandas as pd
    >>> restore_frame_type(pd.DataFrame({"x": [1]}), prefer_polars=True).shape
    (1, 1)
    """
    if prefer_polars:
        if isinstance(df, pl.DataFrame):
            return df
        return pl.from_pandas(df)
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


def to_pandas_table(table: Any) -> pd.DataFrame:
    """
    将报表表对象安全转成 Pandas DataFrame。

    Parameters
    ----------
    table : Any
        可能为 ``None``、Pandas、Polars 或可被 Pandas 构造的数据对象。

    Returns
    -------
    pandas.DataFrame
        统一后的 Pandas 表；空输入返回空表。

    Examples
    --------
    >>> to_pandas_table(None).empty
    True
    """
    if table is None:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, pl.DataFrame):
        return table.to_pandas()
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame(table)
