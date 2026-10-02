"""新统计结果复用既有快照、元数据和存储契约的内部工厂。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import polars as pl

from ._artifact import ReportSnapshot
from ._metadata import FeatureMetadata, normalize_business_context, normalize_metadata
from ._semantics import _definition


def _result_report(
    report_type: str,
    tables: dict[str, pl.DataFrame],
    parameters: dict[str, Any],
    features: list[str],
    metadata: FeatureMetadata | None,
    context: dict[str, Any] | None,
    *,
    roles: dict[str, dict[str, str]] | None = None,
    scopes: dict[str, list[str]] | None = None,
    scope_roles: dict[str, str] | None = None,
    definitions: dict[str, str] | None = None,
    grains: dict[str, str] | None = None,
) -> ReportSnapshot:
    """组装版本化语义目录；表本身保持列式存储，不经 JSON 中转。"""
    catalog: dict[str, Any] = {}
    public_functions = {
        "correlation": [
            "mars.reporting.get_correlation_matrix",
            "mars.reporting.get_related_features",
            "mars.reporting.show_correlation_matrix",
        ],
        "score_cross": [
            "mars.analysis.get_score_bin_definitions",
            "mars.analysis.get_score_cell",
            "mars.analysis.show_score_matrix",
            "mars.analysis.evaluate_score_policy",
            "mars.analysis.write_score_cross_html",
        ],
    }.get(report_type, [])
    for name, frame in tables.items():
        fields: dict[str, Any] = {}
        for column, dtype in frame.schema.items():
            meaning = _definition(column, report_type=report_type, table=name)
            unit = (
                "count"
                if column.endswith("count")
                else "fraction"
                if any(word in column for word in ("rate", "share", "coverage", "delta", "ci_"))
                else "dimensionless"
                if "correlation" in column or "lift" in column
                else "unknown"
            )
            fields[column] = {
                "dtype": "Enum" if isinstance(dtype, pl.Enum) else str(dtype),
                "unit": meaning["unit"] if meaning["unit"] != "unknown" else unit,
                "meaning": (definitions or {}).get(column, meaning["meaning"]),
                "null": "unavailable; inspect status and report parameters",
            }
        catalog[name] = {
            "rows": frame.height,
            "grain": (grains or {}).get(name, name),
            "schema_version": 1,
            "fields": fields,
        }
        if roles is not None and name in roles:
            catalog[name]["feature_roles"] = roles[name]
        elif scopes is not None and name in scopes:
            catalog[name]["feature_scope"] = scopes[name]
            catalog[name]["feature_scope_roles"] = scope_roles or {}
        elif "feature" in frame.columns:
            catalog[name]["feature_roles"] = {"feature": "feature"}
    return ReportSnapshot(
        tables,
        {
            "report_id": str(uuid4()),
            "report_type": report_type,
            "format_version": 1,
            "tables": catalog,
            "parameters": parameters,
            "feature_metadata": normalize_metadata(metadata, features),
            "business_context": {
                "dataset_id": "unknown",
                "sample_unit": "unknown",
                "label_definition": "unknown",
                "observation_window": "unknown",
                **normalize_business_context(context),
            },
            "source": {
                "dataset_id": (context or {}).get("dataset_id", "unknown"),
                "computation": report_type,
            },
            "public_functions": public_functions,
            "saved_operations": "public functions accept load_report ReportSnapshot; no source samples or fitted objects required",
            "operations": [
                "describe",
                "get_table",
                "query_page",
                "get_feature",
                "search_features",
                "to_ai_context",
                "save",
                "show_table",
                "write_excel",
                "write_html",
            ],
            "limitations": [
                "Only saved aggregate evidence; no causal or deployment conclusion.",
                "Null is unavailable, not numeric zero. See status and denominators.",
            ],
        },
    )
