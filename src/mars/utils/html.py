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
        缺失或不可展示数值返回 ``True``。
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
        是否按百分比展示。
    precision : int
        小数位数。
    missing_text : str
        缺失值展示文本。
    compact_float : bool
        是否使用建模报告中更紧凑的浮点格式。

    Returns
    -------
    str
        格式化后的展示文本。
    """
    if is_missing_html_value(value):
        return missing_text
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.to_datetime(value).strftime("%Y-%m-%d"))
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        if compact_float:
            return f"{int(value):,}"
        return f"{float(value):.{precision}f}"
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        num = float(value)
        if not np.isfinite(num):
            return missing_text
        if as_percent:
            return f"{num * 100:.{precision}f}%" if not compact_float else f"{num:.{precision}%}"
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
