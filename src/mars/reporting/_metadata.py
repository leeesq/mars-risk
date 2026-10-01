"""统一业务元数据与旧来源参数的薄适配。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Union

import pandas as pd
import polars as pl

from ._serialization import encode, json_safe

FeatureMetadata = Union[Dict[str, Dict[str, Any]], pd.DataFrame, pl.DataFrame]
_FIELDS = {"display_name", "description", "data_source", "unit"}


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


def export_semantics(description: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """所有报告导出复用公共 describe；长定义集中存放，不重复到统计行。"""
    metadata = description.get("feature_metadata", {})
    return {
        "FeatureMetadata": pd.DataFrame(
            [{"feature": feature, **entry} for feature, entry in metadata.items()],
            columns=["feature", "display_name", "description", "data_source", "unit"],
        ),
        "BusinessContext": pd.DataFrame(
            [
                {"key": key, "value": encode(value)}
                for key, value in description.get("business_context", {}).items()
            ],
            columns=["key", "value"],
        ),
        "ReportSemantics": pd.DataFrame(
            [
                {"key": key, "value": encode(value)}
                for key, value in description.items()
                if key not in {"feature_metadata", "business_context"}
            ],
            columns=["key", "value"],
        ),
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
