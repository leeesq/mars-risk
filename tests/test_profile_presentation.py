"""画像比较表的真实 Notebook 渲染与现有导出契约。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from mars.analysis import profile_stats
from mars.compute import to_pandas_frame
from mars.reporting import MarsProfileReport


def _comparison_report(backend: str, grouped: bool) -> MarsProfileReport:
    """构造可比较、不可比较、缺字段和全空类别的真实比较结果。"""
    current = pl.DataFrame(
        {
            "category": ["甲", "新类别", "甲", None],
            "number": [1, 2, 3, 4],
            "empty_category": pl.Series([None] * 4, dtype=pl.Utf8),
            "current_only": [True, False, True, False],
            "month": ["开发", "观察", "开发", "观察"],
        }
    )
    benchmark = pl.DataFrame(
        {
            "category": ["甲", "乙"],
            "number": ["one", "two"],
            "empty_category": pl.Series([None, None], dtype=pl.Utf8),
            "benchmark_only": [1, 2],
            "month": ["开发", "开发"],
        }
    )
    return profile_stats(
        current.to_pandas() if backend == "pandas" else current,
        metrics=["schema", "unseen"],
        benchmark_df=benchmark.to_pandas() if backend == "pandas" else benchmark,
        categorical_features=["number"],
        group_col="month" if grouped else None,
    )


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("metric", ["schema", "unseen"])
@pytest.mark.parametrize("explicit", [False, True])
def test_comparison_notebook_renders_native_text_and_numbers(
    backend: str, grouped: bool, metric: str, explicit: bool, tmp_path: Path
) -> None:
    """原生比较值在 Notebook、HTML 和 Excel 中保持一致。"""
    report = _comparison_report(backend, grouped)
    before = to_pandas_frame(report.get_table(f"comparison.{metric}"))
    options: dict[str, Any] = {"sort_by": "feature", "sort_ascending": True} if explicit else {}
    styled = report.show_trend(metric, **options)
    html = styled.to_html()
    assert "String" in html and "Int64" in html
    assert "incompatible" in html and "dtype families are not comparable" in html
    assert "current_only" in html and "benchmark_only" in html
    assert_frame_equal(before, to_pandas_frame(report.get_table(f"comparison.{metric}")))
    displayed = styled.data.set_index("feature").loc[before["feature"], before.columns[1:]]
    assert_frame_equal(displayed.reset_index(), before, check_dtype=True)
    if explicit:
        assert styled.data["feature"].tolist() == sorted(before["feature"].tolist())
    elif metric == "schema":
        assert styled.data["feature"].tolist() == sorted(before["feature"].tolist(), reverse=True)
    else:
        totals = styled.data["total"].dropna().tolist()
        assert totals == sorted(totals, reverse=True)
        assert styled.data.set_index("feature").loc["category", "total"] == pytest.approx(1 / 3)
    if metric == "unseen":
        assert "no_reference_values" in html
        assert "background-color:" in html and "0.33" in html
    path = tmp_path / "comparison.html"
    report.write_html(str(path))
    exported = path.read_text(encoding="utf-8")
    assert "dtype families are not comparable" in exported
    excel = tmp_path / "comparison.xlsx"
    report.write_excel(str(excel))
    actual = pd.read_excel(excel, sheet_name=f"Compare_{metric.capitalize()}")
    assert_frame_equal(actual, before, check_dtype=False)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("metric", ["schema", "unseen"])
def test_comparison_empty_selection_and_unknown_sort_are_explicit(
    backend: str, metric: str
) -> None:
    """空结果仍能渲染，用户显式请求不存在的排序列清楚失败。"""
    report = _comparison_report(backend, grouped=False)
    styled = report.show_trend(metric, limit=0)
    assert styled.data.empty
    assert "<table" in styled.to_html() and "status" in styled.to_html()
    for column in ("not_a_column", "total" if metric == "schema" else "not_a_column"):
        with pytest.raises(ValueError, match="Unknown columns"):
            report.show_trend(metric, sort_by=column)


def test_shared_styler_keeps_nullable_numeric_boolean_and_all_empty_columns() -> None:
    """显式渐变子集不能把文本、Boolean 或全空列转成数字。"""
    report = MarsProfileReport(pd.DataFrame(), dq_tables={}, stats_tables={})
    frame = pd.DataFrame(
        {
            "feature": ["first", "second"],
            "current_dtype": ["Int64", "Int64"],
            "status": ["comparable", "not_computed"],
            "reason": [None, "no_target"],
            "count": pd.Series([1, None], dtype="Int64"),
            "rate": pd.Series([0.25, None], dtype="Float64"),
            "all_empty": pd.Series([None, None], dtype="Float64"),
            "comparable": pd.Series([True, None], dtype="boolean"),
        }
    )
    styled = report._get_styler(
        frame, title="Nullable comparison", cmap="Blues",
        subset_cols=list(frame.columns),
    )
    html = styled.to_html()
    assert "25.00%" in html and "1.00" in html
    assert "Int64" in html and "not_computed" in html and "no_target" in html
    assert "True" in html and "background-color:" in html
    assert_frame_equal(styled.data, frame)
    styled._compute()
    assert {column for _, column in styled.ctx} == {4, 5}


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("metric", ["schema", "unseen"])
def test_explicit_comparison_sort_list_remains_supported(backend: str, metric: str) -> None:
    """合法多列排序与明确错误沿用公共查询契约。"""
    report = _comparison_report(backend, grouped=True)
    styled = report.show_trend(metric, sort_by=["feature"], sort_ascending=True)
    assert styled.data["feature"].tolist() == sorted(styled.data["feature"].tolist())
    assert "<table" in styled.to_html()
    with pytest.raises(ValueError, match="Unknown columns"):
        report.show_trend(metric, sort_by=["feature", "not_a_column"])


def test_existing_profile_trends_keep_default_total_order_and_percentage_format() -> None:
    """普通数值趋势保留按 total 降序与比例格式。"""
    report = MarsProfileReport(
        pd.DataFrame({"feature": ["income", "debt"], "missing_rate": [0.1, 0.4]}),
        dq_tables={
            "missing": pd.DataFrame(
                {"feature": ["income", "debt"], "total": [0.1, 0.4], "period": [0.2, 0.3]}
            )
        },
        stats_tables={},
    )
    styled = report.show_trend("missing")
    assert styled.data["feature"].tolist() == ["debt", "income"]
    assert "40.00%" in styled.to_html() and "10.00%" in styled.to_html()
    assert "background-color:" in report.show_overview().to_html()
