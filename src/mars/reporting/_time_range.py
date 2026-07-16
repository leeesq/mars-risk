"""风险趋势图时间范围契约工具。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

TimeRange: TypeAlias = tuple[str, str]

_INVALID_TIME_VALUES = frozenset({"", "<na>", "nan", "nat", "none"})


def normalize_time_range(values: tuple[object, object] | None) -> TimeRange:
    """校验并规范化风险趋势图使用的时间范围。"""
    if values is None or len(values) != 2:
        raise ValueError(
            "Risk trend plotting requires a valid time_range derived from `time_col`."
        )

    normalized_values = tuple(str(value).strip() for value in values)
    if any(value.lower() in _INVALID_TIME_VALUES for value in normalized_values):
        raise ValueError(
            "Risk trend plotting requires a valid time_range derived from `time_col`."
        )

    return normalized_values[0], normalized_values[1]


def resolve_report_time_range(
    *,
    report_meta: Mapping[str, Any] | None,
    dt_col: str | None,
) -> TimeRange:
    """从报告元数据解析风险趋势图的时间范围。"""
    metadata = report_meta or {}
    time_col = metadata.get("dt_col") or dt_col
    if not time_col:
        raise ValueError(
            "Risk trend plotting requires `time_col`; `group_col` cannot provide "
            "the chart time range."
        )

    try:
        return normalize_time_range((metadata.get("start_dt"), metadata.get("end_dt")))
    except ValueError as exc:
        raise ValueError(
            f"Risk trend plotting requires valid time values in `time_col={time_col!r}`."
        ) from exc
