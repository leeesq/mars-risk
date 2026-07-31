"""风险趋势图时间范围契约工具。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Tuple

TimeRange = Tuple[str, str]

_INVALID_TIME_VALUES = frozenset({"", "<na>", "nan", "nat", "none"})


def _normalize_time_value(value: object) -> str:
    """将可解析的时间值截断为日，保留旧式非日期标签。"""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    raw_value = str(value).strip()
    if raw_value.lower() in _INVALID_TIME_VALUES:
        raise ValueError(
            "Risk trend plotting requires a valid time_range derived from `time_col`."
        )

    iso_value = raw_value.replace("Z", "+00:00") if raw_value.endswith("Z") else raw_value
    try:
        return datetime.fromisoformat(iso_value).date().isoformat()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw_value, fmt).date().isoformat()
        except ValueError:
            continue
    return raw_value


def normalize_time_range(values: tuple[object, object] | None) -> TimeRange:
    """校验并规范化风险趋势图使用的时间范围。"""
    if values is None or len(values) != 2:
        raise ValueError(
            "Risk trend plotting requires a valid time_range derived from `time_col`."
        )

    return _normalize_time_value(values[0]), _normalize_time_value(values[1])


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
