"""报告与消费者共用的有效 JSON 值编码。"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

JSON_RULES = {
    "non_finite": "tagged float: {$mars: float, value: nan/inf/-inf}",
    "date": "ISO-8601",
    "null": "missing; calculation state is separate",
}


def json_safe(value: Any) -> Any:
    """保留非有限浮点类型；普通字符串标记、缺失及零互不混淆。"""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else np.datetime_as_string(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "$mars": "float",
            "value": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf",
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings.")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported report value type: {type(value).__name__}.")


def encode(value: Any) -> str:
    """输出紧凑的标准 JSON；预算按最终 Unicode 字符计算。"""
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def decode_json_value(value: Any) -> Any:
    """只恢复报告的受限浮点标签；递归复制容器，不恢复任意 Python 对象。"""
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("$mars") == "float":
            float_values = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
            label = value.get("value")
            if (
                set(value) != {"$mars", "value"}
                or not isinstance(label, str)
                or label not in float_values
            ):
                raise ValueError("Invalid report tagged float value.")
            return float_values[label]
        return {key: decode_json_value(item) for key, item in value.items()}
    return value


def table_rows(frame: pl.DataFrame | pd.DataFrame) -> list[dict[str, Any]]:
    """只转换已分页证据；Arrow 保留 null/NaN 及日期，避免旧 Polars 时区转换崩溃。"""
    rows: list[dict[str, Any]] = (
        frame.to_arrow().to_pylist()
        if isinstance(frame, pl.DataFrame)
        else frame.to_dict("records")
    )
    return rows
