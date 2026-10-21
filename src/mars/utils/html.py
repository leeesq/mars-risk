"""HTML 报告渲染的通用格式化工具。"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd


def escape_html_value(value: Any) -> str:
    """
    按 HTML 文本和属性上下文转义任意值。

    Parameters
    ----------
    value : Any
        待转义对象。

    Returns
    -------
    str
        已转义字符串。
    """
    return html.escape("" if value is None else str(value), quote=True)


def is_missing_html_value(value: Any) -> bool:
    """
    判断单元格值在 HTML 表格中是否应按缺失展示。

    Parameters
    ----------
    value : Any
        待判断值。

    Returns
    -------
    bool
        ``None`` 或 ``pd.isna`` 判定为缺失的标量返回 ``True``。
        无法归约为布尔值的数组等对象返回 ``False``。
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_html_value(
    value: Any,
    *,
    as_percent: bool = False,
    precision: int = 2,
    missing_text: str = "",
    compact_float: bool = False,
) -> str:
    """
    按数值、日期、百分比和缺失值语义格式化 HTML 单元格文本。

    Parameters
    ----------
    value : Any
        待格式化值。
    as_percent : bool
        是否将数值乘以 100 并添加百分号，默认 ``False``。整数 0 和 1 也按比例处理。
    precision : int
        普通数值和百分比的小数位数，默认 2；紧凑模式的普通浮点数固定保留 4 位。
    missing_text : str
        缺失值和非有限浮点数的替代文本，默认空字符串。
    compact_float : bool
        是否启用建模报告格式，默认 ``False``。启用后整数加千分位，
        绝对值至少为 1,000 的浮点数加千分位并保留 2 位小数。

    Returns
    -------
    str
        格式化后的展示文本。
    """
    if is_missing_html_value(value):
        return missing_text
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.to_datetime(value).strftime("%Y-%m-%d"))
    # 整数形式的比例仍须先按比例展示，0/1 不能退化为普通数值。
    if as_percent and isinstance(value, (np.integer, int, np.floating, float)) and not isinstance(value, (np.bool_, bool)):
        num = float(value)
        if not np.isfinite(num):
            return missing_text
        return f"{num * 100:.{precision}f}%" if not compact_float else f"{num:.{precision}%}"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        if compact_float:
            return f"{int(value):,}"
        return f"{float(value):.{precision}f}"
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        num = float(value)
        if not np.isfinite(num):
            return missing_text
        if compact_float and abs(num) >= 1000:
            return f"{num:,.2f}"
        return f"{num:.{precision if not compact_float else 4}f}"
    return str(value)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 MultiIndex 列名压平成前端表格可展示的字符串列名。

    Parameters
    ----------
    df : pd.DataFrame
        待处理表。

    Returns
    -------
    pd.DataFrame
        列名已压平的副本。
    """
    flat = df.copy()
    flat.columns = [
        " | ".join(str(part) for part in col if str(part) not in {"", "nan"})
        if isinstance(col, tuple)
        else str(col)
        for col in flat.columns
    ]
    return flat
