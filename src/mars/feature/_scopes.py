"""客群的稳定身份、人口匹配和角色校验；不持有原始宽表。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import polars as pl

from mars.feature.binning._json_codec import decode_json_value, encode_json_value


def _scalar(value: Any) -> Any:
    """统一缺失值并拒绝不能作为客群身份的复合值。"""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, (str, bool, int, float, date, datetime, Decimal)):
        raise ValueError(f"Unsupported segment scalar type: {type(value).__name__}.")
    if isinstance(value, (float, Decimal)) and not math.isfinite(value):
        raise ValueError("segment_col must not contain infinite values.")
    return value


def _encoded(value: Any) -> str:
    """复用正式规则的受限编码，保留日期、数字和文本的身份。"""
    return json.dumps(encode_json_value(_scalar(value)), ensure_ascii=False, sort_keys=True)


def build_scopes(
    frame: pl.DataFrame, segment_col: str, benchmark: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """从客群列的不同值建立稳定目录；基准列须具有相同逻辑类型。"""
    if segment_col not in frame.columns:
        raise ValueError(f"segment_col {segment_col!r} is missing from input.")
    dtype = frame.schema[segment_col]
    if dtype.base_type() in (pl.Object, pl.List, pl.Array, pl.Struct, pl.Binary):
        raise ValueError(f"Unsupported segment_col dtype: {dtype}.")
    if benchmark is not None:
        if segment_col not in benchmark.columns:
            raise ValueError(f"benchmark_df is missing segment_col {segment_col!r}.")
        if benchmark.schema[segment_col] != dtype:
            raise ValueError("benchmark_df segment_col dtype must match the current input.")
        # 不仅校验当前人口，也拒绝基准中非法的身份。
        for value in benchmark[segment_col].unique().to_list():
            _scalar(value)
    values = sorted({_encoded(v) for v in frame[segment_col].unique().to_list()})
    rows: list[dict[str, Any]] = [{
        "scope_id": "all", "scope_kind": "all", "segment_col": segment_col,
        "segment_dtype": str(dtype), "segment_value": None, "display_name": "All",
        "scope_order": 0, "status": "computed", "reason": None,
    }]
    for order, encoded in enumerate(values, 1):
        value = decode_json_value(json.loads(encoded))
        identity = json.dumps([segment_col, str(dtype), encoded], ensure_ascii=False)
        rows.append({
            "scope_id": "seg_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "scope_kind": "segment", "segment_col": segment_col,
            "segment_dtype": str(dtype), "segment_value": encoded,
            "display_name": "Missing" if value is None else str(value),
            "scope_order": order, "status": "computed", "reason": None,
        })
    return pl.DataFrame(rows, schema={
        "scope_id": pl.String, "scope_kind": pl.String, "segment_col": pl.String,
        "segment_dtype": pl.String, "segment_value": pl.String, "display_name": pl.String,
        "scope_order": pl.Int64, "status": pl.String, "reason": pl.String,
    })


def scope_filter(frame: pl.DataFrame, scopes: pl.DataFrame, scope_id: str) -> pl.DataFrame:
    """筛选一个人口，保留原始类型、chunk 和相对行序；整体不复制宽表。"""
    rows = scopes.filter(pl.col("scope_id") == scope_id).to_dicts()
    if len(rows) != 1:
        raise ValueError(f"Unknown scope_id: {scope_id!r}.")
    row = rows[0]
    if row["scope_kind"] == "all":
        return frame
    column = row["segment_col"]
    value = decode_json_value(json.loads(row["segment_value"]))
    expr = pl.col(column)
    if value is None:
        missing = expr.is_null()
        if frame.schema[column].is_float():
            missing = missing | expr.is_nan()
        return frame.filter(missing)
    return frame.filter(expr == pl.lit(value))


def resolve_scopes(
    scopes: pl.DataFrame, segments: list[Any] | None = None, include_all: bool = True,
) -> list[str]:
    """按业务原值选择客群，整体身份由独立开关控制，不使用字符串哨兵。"""
    if type(include_all) is not bool or segments is not None and not isinstance(segments, list):
        raise ValueError("segments must be a list or None; include_all must be bool.")
    selected = ["all"] if include_all else []
    for value in segments or []:
        encoded = _encoded(value)
        matches = scopes.filter(
            (pl.col("scope_kind") == "segment") & (pl.col("segment_value") == encoded)
        )["scope_id"].to_list()
        if len(matches) != 1:
            raise ValueError(f"Unknown segment value: {value!r}.")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def validate_segment_roles(
    segment_col: str, features: list[str] | None, roles: dict[str, Any],
) -> None:
    """拒绝客群列与显式特征、目标、分组、时间、权重或金额角色重叠。"""
    if features is not None and segment_col in features:
        raise ValueError("segment_col must not also be an explicit feature.")
    for role, value in roles.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if segment_col in values:
            raise ValueError(f"segment_col conflicts with {role}: {segment_col!r}.")
