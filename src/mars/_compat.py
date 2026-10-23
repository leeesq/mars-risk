"""MARS 跨 Python 与 Polars 版本的内部兼容工具。"""

from __future__ import annotations

import re
from inspect import signature
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


def _left_join_preserving_order(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    left_on: str,
    right_on: str,
    order_column: str,
) -> Any:
    """保留左表样本顺序；旧 Polars 用调用方已避让的内部行身份恢复顺序。"""
    # 按实际能力分流，不能向 1.8.2 传入不存在的 maintain_order 参数。
    if "maintain_order" in signature(type(left).join).parameters:
        return left.join(
            right, left_on=left_on, right_on=right_on, how="left", maintain_order="left",
        )
    return (
        left.with_row_index(order_column)
        .join(right, left_on=left_on, right_on=right_on, how="left")
        .sort(order_column)
        .drop(order_column)
    )


def remove_suffix(value: str, suffix: str) -> str:
    """兼容 Python 3.8 地移除已存在的字符串后缀。"""
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _left_join_nulls(
    left: pl.DataFrame, right: pl.DataFrame, *, on: list[str],
) -> pl.DataFrame:
    """兼容各 Polars 版本，把空分组键视为相等并保留左表行序。"""
    kwargs = {"join_nulls" if _polars_version() < (1, 24) else "nulls_equal": True}
    return (
        left.with_row_index("__mars_join_order")
        .join(right, on=on, how="left", **kwargs)
        .sort("__mars_join_order")
        .drop("__mars_join_order")
    )


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
