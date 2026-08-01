"""MARS 跨 Python 与 Polars 版本的内部兼容工具。"""

from __future__ import annotations

import re
from typing import Any, Callable, cast

import polars as pl


def _polars_version() -> tuple[int, int]:
    """提取 Polars 主次版本；无法解析时按现代版本处理。"""
    match = re.match(r"^(\d+)\.(\d+)", pl.__version__)
    if match is None:
        return (999, 999)
    return (int(match.group(1)), int(match.group(2)))


def polars_is_in(value: Any, candidates: Any) -> Any:
    """按 Polars 版本构造不触发形状错误或弃用警告的成员判断。"""
    if _polars_version() < (1, 9):
        return value.is_in(candidates)
    return value.is_in(candidates.implode())


def collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    """使用当前 Polars 版本支持的 streaming 参数物化 LazyFrame。"""
    collect = cast(Callable[..., pl.DataFrame], frame.collect)
    if _polars_version() < (1, 25):
        return collect(streaming=True)
    return collect(engine="streaming")


def remove_suffix(value: str, suffix: str) -> str:
    """兼容 Python 3.8 地移除已存在的字符串后缀。"""
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def pandas_styler_map(
    styler: Any,
    function: Callable[[Any], str],
    *,
    subset: Any,
) -> Any:
    """在 Pandas 2.0 的 ``applymap`` 与现代 ``map`` 间统一元素样式调用。"""
    map_method = getattr(styler, "map", None)
    if map_method is None:
        map_method = styler.applymap
    return map_method(function, subset=subset)
