"""统一业务元数据与旧来源参数的薄适配。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Union

import pandas as pd
import polars as pl

from ._serialization import encode, json_safe

FeatureMetadata = Union[Dict[str, Dict[str, Any]], pd.DataFrame, pl.DataFrame]
FeatureFrame = Union[pd.DataFrame, pl.DataFrame]
_FIELDS = {"display_name", "description", "data_source", "unit"}
FEATURE_IDENTITY_COLUMNS = ("feature", "display_name", "dtype", "data_source")
_METADATA_COLUMNS = frozenset({
    "scope_id", "scope_kind", "segment_col", "segment_dtype", "segment_value",
    "scope_order",
    *FEATURE_IDENTITY_COLUMNS, "target", "y", "status", "reason", "metric", "group",
    "current_dtype", "benchmark_dtype", "distribution", "mode_value", "is_categorical",
    "bin_index", "bin_index_max", "bin_label", "bin_type", "source", "trend",
    "feature_a", "feature_b", "peer_feature", "trigger_feature",
})


def is_metadata_column(column: str) -> bool:
    """识别宽表的身份与描述字段，避免把业务字典误当作时间段或指标。"""
    return column in _METADATA_COLUMNS


def capture_feature_dtypes(frame: pl.DataFrame, features: list[str]) -> dict[str, str]:
    """从变换前输入捕获原始逻辑类型，沿用项目的 Polars 类型名称。"""
    return {
        feature: str(frame.schema[feature])
        for feature in features if feature in frame.schema
    }


def feature_type_catalog(
    tables: dict[str, FeatureFrame], report_type: str, types: dict[str, str] | None = None,
) -> dict[str, str]:
    """只复用可靠的原字段类型；旧分箱趋势的聚合 Float64 不构成原始 schema。"""
    catalog = dict(types or {})
    reliable = {"overview", "comparison.schema"} if report_type == "MarsProfileReport" else {"summary"}
    for name in reliable:
        frame = tables.get(name)
        if frame is None or "feature" not in frame.columns:
            continue
        column = "current_dtype" if name == "comparison.schema" else "dtype"
        if column not in frame.columns:
            continue
        rows = frame.select("feature", column).to_dicts() if isinstance(frame, pl.DataFrame) else frame[["feature", column]].to_dict("records")
        for row in rows:
            feature, dtype = row["feature"], row[column]
            if isinstance(feature, str) and isinstance(dtype, str):
                if catalog.get(feature) not in (None, dtype):
                    raise ValueError(f"Conflicting original dtype for feature {feature!r}.")
                catalog[feature] = dtype
    return catalog


def normalize_feature_frame(
    frame: FeatureFrame, metadata: dict[str, dict[str, Any]], types: dict[str, str],
    *, discard_aggregate_dtype: bool = False,
) -> FeatureFrame:
    """幂等补全短身份列，保留索引、指标类型、行粒度与顺序；非空矛盾明确拒绝。"""
    if "feature" not in frame.columns:
        return frame.clone() if isinstance(frame, pl.DataFrame) else frame.copy()
    identities = frame["feature"].to_list() if isinstance(frame, pl.DataFrame) else frame["feature"].tolist()
    values = {
        "display_name": [metadata.get(f, {}).get("display_name") for f in identities],
        "dtype": [types.get(f) for f in identities],
        "data_source": [metadata.get(f, {}).get("data_source") or "UNMAPPED" for f in identities],
    }
    result = frame.clone() if isinstance(frame, pl.DataFrame) else frame.copy()
    for column, expected in values.items():
        if column in frame.columns:
            existing = frame[column].to_list() if isinstance(frame, pl.DataFrame) else frame[column].tolist()
            for index, (actual, canonical) in enumerate(zip(existing, expected)):
                absent = actual is None or actual is pd.NA or isinstance(actual, float) and pd.isna(actual)
                # nullable 字符串的 pd.NA 比较结果仍为 pd.NA，必须先结束缺失分支。
                if absent:
                    continue
                if column == "dtype" and discard_aggregate_dtype and actual == "Float64":
                    continue
                if column == "data_source" and actual == "UNMAPPED":
                    continue
                if canonical is not None and actual != canonical:
                    raise ValueError(f"Conflicting {column} for feature {identities[index]!r}.")
                if canonical is None:
                    expected[index] = actual
        unique: dict[str, str] = {}
        for feature, value in zip(identities, expected):
            if value is not None:
                if unique.get(feature) not in (None, value):
                    raise ValueError(f"Conflicting {column} for feature {feature!r}.")
                unique[feature] = value
        if isinstance(result, pl.DataFrame):
            result = result.with_columns(pl.Series(column, expected, dtype=pl.String))
        else:
            result[column] = expected if len(expected) else pd.Series([], index=result.index, dtype="object")
    prefix = [c for c in ("target", "y") if c in result.columns and list(frame.columns).index(c) < list(frame.columns).index("feature")]
    ordering = [*prefix, *FEATURE_IDENTITY_COLUMNS, *[c for c in result.columns if c not in {*prefix, *FEATURE_IDENTITY_COLUMNS}]]
    return result.select(ordering) if isinstance(result, pl.DataFrame) else result.loc[:, ordering]


def resolve_feature_identity_metadata(
    tables: dict[str, FeatureFrame], metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """将已有短身份列合入唯一目录；已登记非空值最可信，矛盾记录不能被展示覆盖。"""
    result = deepcopy(metadata)
    for frame in tables.values():
        fields = [c for c in ("display_name", "data_source") if c in frame.columns]
        if "feature" not in frame.columns or not fields:
            continue
        columns = ["feature", *fields]
        rows = frame.select(columns).unique().to_dicts() if isinstance(frame, pl.DataFrame) else frame[columns].drop_duplicates().to_dict("records")
        for row in rows:
            feature = row["feature"]
            if not isinstance(feature, str):
                continue
            for field in fields:
                value = row[field]
                if not isinstance(value, str) or field == "data_source" and value == "UNMAPPED":
                    continue
                entry = result.setdefault(feature, {})
                if entry.get(field) not in (None, value):
                    raise ValueError(f"Conflicting {field} for feature {feature!r}.")
                entry[field] = value
    return result


def _null_group_label(values: list[Any]) -> str:
    """为宽趋势选择不与真实业务文字冲突的空组列名，保留既有 null 列约定。"""
    occupied = {str(value) for value in values if not pd.isna(value)}
    label = "null"
    if label in occupied:
        label = "(空分组)"
        while label in occupied:
            label += "#"
    return label


def validate_group_labels(frame: pl.DataFrame, group_col: str | None) -> None:
    """拒绝与宽表身份列冲突的组值，避免 pivot 或补列覆盖业务维度。"""
    if group_col is None:
        return
    labels = frame[group_col].cast(pl.String).drop_nulls().unique().to_list()
    collisions = [value for value in labels if is_metadata_column(value)]
    if collisions:
        raise ValueError(f"Group values conflict with reserved report columns: {collisions}.")


def normalize_metadata(
    metadata: FeatureMetadata | None,
    features: list[str],
    legacy: dict[str, Any] | None = None,
    *,
    legacy_direction: str = "source_to_features",
) -> dict[str, dict[str, Any]]:
    """校验完整字典后按分析特征裁剪；来源冲突绝不覆盖。"""
    if isinstance(metadata, (pd.DataFrame, pl.DataFrame)):
        if "feature" not in metadata.columns:
            raise ValueError("feature_metadata DataFrame requires a 'feature' identifier column.")
        rows = (
            metadata.to_dicts()
            if isinstance(metadata, pl.DataFrame)
            else metadata.to_dict("records")
        )
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            feature = row.pop("feature")
            if not isinstance(feature, str) or feature in records:
                raise ValueError(f"Duplicate or invalid feature_metadata identifier: {feature!r}.")
            records[feature] = row
    elif metadata is None:
        records = {}
    elif isinstance(metadata, dict):
        records = deepcopy(metadata)
    else:
        raise ValueError("feature_metadata must be a dictionary, Pandas or Polars DataFrame.")
    for feature, entry in records.items():
        if not isinstance(feature, str) or not isinstance(entry, dict):
            raise ValueError("feature_metadata requires string identifiers and dictionary records.")
        if set(entry) - _FIELDS:
            raise ValueError(
                f"Unknown feature_metadata fields for {feature}: {set(entry) - _FIELDS}."
            )
        for field, value in entry.items():
            if value is None or value is pd.NA or isinstance(value, float) and pd.isna(value):
                entry[field] = None
            elif not isinstance(value, str):
                raise ValueError(
                    f"feature_metadata[{feature!r}][{field!r}] must be a string or null."
                )
    if legacy is not None:
        if not isinstance(legacy, dict):
            raise ValueError("feature_data_source must be a dictionary.")
        for key, value in legacy.items():
            if not isinstance(key, str):
                raise ValueError("feature_data_source keys must be strings.")
            if legacy_direction == "feature_to_source":
                pairs = [(key, value)]
            else:
                if not isinstance(value, list) or any(not isinstance(f, str) for f in value):
                    raise ValueError(
                        "feature_data_source requires source -> list of feature identifiers."
                    )
                if len(set(value)) != len(value):
                    raise ValueError(f"Duplicate feature_data_source records in {key!r}.")
                if set(value) - set(features):
                    raise ValueError(
                        "feature_data_source contains features outside the active feature set."
                    )
                pairs = [(f, key) for f in value]
            for feature, source in pairs:
                if not isinstance(source, str):
                    raise ValueError("feature_data_source values must be strings.")
                if source == "UNMAPPED":
                    continue
                entry = records.setdefault(feature, {})
                if entry.get("data_source") not in (None, source):
                    raise ValueError(f"Conflicting data_source for feature {feature!r}.")
                entry["data_source"] = source
    return {feature: records.get(feature, {}) for feature in features}


def normalize_business_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """只接受明确的 JSON 结构；标签业务定义按原始目标字段登记。"""
    if context is not None and not isinstance(context, dict):
        raise ValueError("business_context must be a dictionary.")
    result = deepcopy(context or {})
    labels = result.get("labels", {})
    if not isinstance(labels, dict) or any(not isinstance(v, dict) for v in labels.values()):
        raise ValueError("business_context.labels must map target identifiers to dictionaries.")

    # 业务上下文不能偷偷携带可执行对象或非标准容器。
    def validate(value: Any) -> None:
        """递归约束用户输入为 JSON 原语和 ISO 可编码日期。"""
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str) or key == "$mars":
                    raise ValueError("business_context keys must be strings; '$mars' is reserved.")
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif isinstance(value, tuple):
            raise ValueError("business_context accepts JSON lists, not tuples.")
        else:
            json_safe(value)

    validate(result)
    return result


def _excel_json_rows(values: dict[str, Any]) -> pd.DataFrame:
    """将长 JSON 分成有顺序的片段，避免 Excel 32767 个 UTF-16 单元限制。"""
    rows: list[dict[str, Any]] = []
    for key, value in values.items():
        serialized = encode(value)
        # 每个 Unicode 字符最多占两个 UTF-16 单元，16000 字符片段安全容纳补充平面字符。
        chunks = [serialized[offset:offset + 16000] for offset in range(0, len(serialized), 16000)]
        for index, chunk in enumerate(chunks, start=1):
            rows.append({"key": key, "value": chunk, "part": index, "parts": len(chunks)})
    return pd.DataFrame(rows, columns=["key", "value", "part", "parts"])


def export_semantics(description: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """所有报告导出复用公共 describe；长定义集中存放，不重复到统计行。"""
    metadata = description.get("feature_metadata", {})
    types = description.get("parameters", {}).get("feature_dtypes", {})
    return {
        "FeatureMetadata": pd.DataFrame(
            [{"feature": feature, **entry, "dtype": types.get(feature)} for feature, entry in metadata.items()],
            columns=["feature", "display_name", "dtype", "data_source", "description", "unit"],
        ),
        "BusinessContext": _excel_json_rows(description.get("business_context", {})),
        "ReportSemantics": _excel_json_rows({
            key: value for key, value in description.items()
            if key not in {"feature_metadata", "business_context"}
        }),
    }


def export_report_semantics(report: Any) -> dict[str, pd.DataFrame]:
    """通过公共目录导出元信息和紧凑状态表，沿用领域报告的统计表排版。"""
    description = report.describe()
    frames = export_semantics(description)
    if "calculation_status" in description["tables"]:
        state = report.get_table("calculation_status")
        frames["CalculationStatus"] = (
            state.to_pandas() if isinstance(state, pl.DataFrame) else state
        )
    return frames
