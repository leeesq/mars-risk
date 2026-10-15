"""Grouped Pivot 的真实空组与文字组身份、数值及源报告回归。"""

from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.analysis import profile_risk
from mars.compute import to_pandas_frame
from mars.core.constants import DIVISION_EPSILON
from mars.reporting import MarsBinningReport
from mars.reporting._binning_excel import _BinningExcelWriter


def _null_group_report(backend: str, *, ks_method: str = "binned") -> MarsBinningReport:
    """用不同人数、坏样本与日期的五个真实业务组生成双标签原生报告。"""
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate((None, "null", "None", "nan", "normal")):
        for bin_index, value in enumerate((0.0, 10.0)):
            count = group_index + bin_index + 2
            for index in range(count):
                rows.append({
                    "income": value,
                    "debt": 10.0 - value,
                    "bad": int(index < group_index + bin_index),
                    "later": int(index < bin_index + 1),
                    "group": group,
                    "dt": f"2024-01-{group_index * 2 + bin_index + 1:02d}",
                })
    frame = pl.DataFrame(rows)
    return profile_risk(
        frame.to_pandas() if backend == "pandas" else frame,
        target=["bad", "later"], features=["income", "debt"],
        group_col="group", time_col="dt", n_bins=2, method="uniform",
        ks_method=ks_method,
        ordered_metric_sort_by="bin_index",
        feature_metadata={
            "income": {"display_name": "家庭收入", "data_source": "申请"},
            "debt": {"display_name": "负债余额", "data_source": "征信"},
        },
    ).report


class _PivotCells(HTMLParser):
    """只读取实际透视行及其列值，避免用页面包含文字代替数值验收。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.target: str | None = None
        self.row: dict[str, Any] | None = None
        self.cell: dict[str, str | None] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and "mars-pivot-table" in (attributes.get("class") or "").split():
            self.target = str(attributes["id"])[len("mars-pivot-"):]
            self.rows[self.target] = []
        elif tag == "tr" and self.target and attributes.get("data-feature") and not attributes.get("data-role"):
            self.row = {"feature": attributes["data-feature"], "cells": {}, "text": ""}
        elif tag == "td" and self.row is not None:
            self.cell = attributes
            if "data-col" in attributes:
                self.row["cells"][attributes["data-col"]] = float(str(attributes["data-sort-value"]))

    def handle_data(self, data: str) -> None:
        if self.row is not None and self.cell is not None and "mars-pivot-bin" in (self.cell.get("class") or ""):
            self.row["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows[str(self.target)].append(self.row)
            self.row = None
        elif tag == "table":
            self.target = None


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("ks_method", ["raw", "binned"])
def test_grouped_pivot_keeps_null_and_literal_groups_with_conserved_totals(
    backend: str, ks_method: str, tmp_path: Path,
) -> None:
    report = _null_group_report(backend, ks_method=ks_method)
    before = deepcopy(report.describe())
    original_tables = {
        name: to_pandas_frame(report.get_table(name)) for name in report.describe()["tables"]
    }
    original_objects = (report.summary_table, report.detail_table, report.trend_tables)
    detail = to_pandas_frame(report.detail_table)
    group_col = report.detail_group_col
    assert group_col is not None
    assert detail[group_col].isna().any()
    assert set(detail[group_col].dropna()) == {"null", "None", "nan", "normal", "Total"}
    # 宽趋势和 Excel 使用独立空组展示列，不能将它匹配到文字 null。
    exported, additional = _BinningExcelWriter(report)._prepare_excel_export()
    trend = additional["最终KS趋势"].set_index("feature")
    for feature in ("income", "debt"):
        for group in (None, "null", "None", "nan", "normal", "Total"):
            mask = exported[group_col].isna() if group is None else exported[group_col].eq(group)
            values = exported.loc[mask & exported["feature"].eq(feature) & exported["y"].eq("bad")]
            label = "(空分组)" if group is None else group
            expected_ks = values["ks_bin"].max()
            if pd.isna(expected_ks):
                assert pd.isna(trend.loc[feature, label])
            else:
                assert trend.loc[feature, label] == expected_ks
    output = tmp_path / "null-groups.html"
    # 默认 include_detail/include_charts 均开启，真实原生图与 Grouped Pivot 一起导出。
    report.write_html(str(output))
    parser = _PivotCells()
    parser.feed(output.read_text(encoding="utf-8"))
    assert set(parser.rows) == {"bad", "later"}
    for target, rows in parser.rows.items():
        assert len(rows) == 4
        for row in rows:
            candidates = detail.loc[
                (detail["feature"] == row["feature"])
                & (detail["y"] == target)
                & (detail["bin_label"] == row["text"])
            ]
            assert len(candidates) == 6
            assert set(key.split("|", 1)[1] for key in row["cells"]) == {
                "Total", "(空分组)", "null", "None", "nan", "normal",
            }
            non_total = candidates.loc[~candidates[group_col].eq("Total").fillna(False)]
            total_count = non_total["count"].sum()
            total_bad = non_total["bad"].sum()
            assert row["cells"]["total_count|Total"] == total_count
            assert row["cells"]["bad_count|Total"] == total_bad
            for _, expected in candidates.iterrows():
                label = "(空分组)" if pd.isna(expected[group_col]) else expected[group_col]
                count, bad = expected["count"], expected["bad"]
                assert row["cells"][f"total_count|{label}"] == count
                assert row["cells"][f"bad_count|{label}"] == bad
                assert row["cells"][f"risk|{label}"] == pytest.approx(bad / (count + DIVISION_EPSILON))
            assert sum(row["cells"][f"total_count|{label}"] for label in ("(空分组)", "null", "None", "nan", "normal")) == total_count
    assert all(
        original is actual for original, actual in zip(
            original_objects, (report.summary_table, report.detail_table, report.trend_tables),
        )
    )
    assert report.describe() == before
    for name in before["tables"]:
        pd.testing.assert_frame_equal(
            to_pandas_frame(report.get_table(name)), original_tables[name],
        )
