"""Notebook 摘要只消费身份和 PSI，不物化完整分箱明细。"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsBinningReport, _binning_html


def _frame(data: dict[str, Any], backend: str) -> pd.DataFrame | pl.DataFrame:
    """建立指定后端，保持测试数据和业务含义一致。"""
    frame = pd.DataFrame(data)
    return pl.from_pandas(frame) if backend == "polars" else frame


def _forbid_conversion(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """摘要计数不能依赖完整表或跨后端转换。"""
    raise AssertionError("Notebook summary must aggregate in the original backend")


def _assert_counts(rendered: str, feature_count: int, psi_count: int) -> None:
    """同时核对英文身份计数和任一标签 PSI 告警数。"""
    assert re.search(rf"Features:</strong>\s*{feature_count}<", rendered)
    assert re.search(
        rf"PSI &gt; 0.25 .*?<span[^>]*>{psi_count}</span>", rendered,
    )


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("psi_state", ["computed", "missing", "absent"])
def test_notebook_summary_never_reads_detail_when_summary_has_identities(
    backend: str, psi_state: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """双标签和重复中文名仍只计英文身份；缺失或未计算 PSI 不读取明细。"""
    summary: dict[str, Any] = {
        "feature": ["income", "debt", "score"] * 2,
        "target": ["bad"] * 3 + ["later"] * 3,
        "iv": [0.1] * 6,
        **{f"unused_{index}": [index] * 6 for index in range(12)},
    }
    if psi_state != "absent":
        summary["psi_max"] = (
            [0.3, 0.25, float("nan"), 0.4, 0.25, None]
            if psi_state == "computed" else [None] * 6
        )
    report = MarsBinningReport(
        _frame(summary, backend), {},
        _frame({"feature": ["detail_only"], "count": [999]}, backend),
        feature_metadata={
            feature: {"display_name": "重复中文名"}
            for feature in ("income", "debt", "score")
        },
    )

    def forbid_detail(self: MarsBinningReport) -> pd.DataFrame | pl.DataFrame:
        raise AssertionError("Summary identities must avoid even reading detail_table")

    monkeypatch.setattr(MarsBinningReport, "detail_table", property(forbid_detail))
    monkeypatch.setattr(_binning_html, "to_pandas_frame", _forbid_conversion)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", _forbid_conversion)
    _assert_counts(report._repr_html_(), 3, int(psi_state == "computed"))


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("summary_state", ["empty", "no_feature", "null_feature"])
def test_notebook_summary_fallback_only_aggregates_detail_feature(
    backend: str, summary_state: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要无身份时，宽明细只允许 feature 投影与原后端去重标量聚合。"""
    summaries: dict[str, dict[str, Any]] = {
        "empty": {"feature": [], "psi_max": []},
        "no_feature": {"iv": [0.1]},
        "null_feature": {"feature": [None], "psi_max": [0.9]},
    }
    report = MarsBinningReport(
        _frame(summaries[summary_state], backend), {},
        _frame({
            "feature": ["income", "debt", "income", None] * 25,
            **{f"unused_{index}": [index] * 100 for index in range(32)},
        }, backend),
    )
    detail = report.detail_table
    projections: list[str] = []
    pandas_getitem = pd.DataFrame.__getitem__
    polars_select = pl.DataFrame.select

    def limited_getitem(self: pd.DataFrame, key: Any) -> Any:
        if self is detail:
            assert key == "feature", "Fallback must not copy or read other detail columns"
            projections.append(key)
        return pandas_getitem(self, key)

    def limited_select(self: pl.DataFrame, *exprs: Any, **named_exprs: Any) -> pl.DataFrame:
        if self is detail:
            assert not named_exprs and len(exprs) == 1
            assert exprs[0].meta.root_names() == ["feature"]
            projections.append("feature")
        result = polars_select(self, *exprs, **named_exprs)
        if self is detail:
            assert result.shape == (1, 1), "Only the aggregated count may be materialized"
        return result

    monkeypatch.setattr(pd.DataFrame, "__getitem__", limited_getitem)
    monkeypatch.setattr(pl.DataFrame, "select", limited_select)
    monkeypatch.setattr(_binning_html, "to_pandas_frame", _forbid_conversion)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", _forbid_conversion)
    _assert_counts(report._repr_html_(), 2, 0)
    assert projections == ["feature"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_notebook_summary_uses_trusted_catalog_without_evidence_and_does_not_cache(
    backend: str,
) -> None:
    """无明细时沿用可信元数据；后续替换源摘要立即反映新的身份和 PSI。"""
    report = MarsBinningReport(
        _frame({}, backend),
        {"psi": _frame({"feature": ["income", "debt"], "Total": [0.1, 0.2]}, backend)},
        _frame({}, backend),
        feature_metadata={
            "income": {"display_name": "收入"}, "debt": {"display_name": "负债"},
        },
    )
    _assert_counts(report._repr_html_(), 2, 0)
    report._summary = _frame({"feature": ["score"], "psi_max": [0.26]}, backend)
    _assert_counts(report._repr_html_(), 1, 1)
