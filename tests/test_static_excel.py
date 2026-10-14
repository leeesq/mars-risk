"""既有 snapshot 静态 Excel 的当前数值、持久化与独立进程交付。"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import polars as pl
import pytest
from openpyxl import load_workbook

import mars
from mars.analysis import profile_risk
from mars.feature.binning import MarsNativeBinner
from mars.reporting import MarsBinningReport, ReportSnapshot, snapshot_report


def _assert_current_excel(path: Path, report: ReportSnapshot) -> None:
    """以不重算、不刷新方式逐表核对当前公共展示数据。"""
    workbook = load_workbook(path, data_only=True)
    try:
        for index, name in enumerate(report.describe()["tables"]):
            frame = report.show_table(name)
            sheet = workbook[f"{index:03d}_{name}"[:31]]
            records = list(sheet.values)
            assert records[0] == tuple(frame.columns)
            assert len(records) == len(frame) + 1
            for actual, expected in zip(records[1:], frame.itertuples(index=False, name=None)):
                for found, original in zip(actual, expected):
                    if pd.isna(original):
                        assert found is None
                    elif isinstance(original, float):
                        assert found == pytest.approx(original)
                    else:
                        assert found == original
    finally:
        workbook.close()
    # 静态交付所有当前值均是字面单元格，不依赖公式或透视缓存刷新。
    formulas = load_workbook(path, data_only=False)
    try:
        assert not any(cell.data_type == "f" for sheet in formulas for row in sheet for cell in row)
    finally:
        formulas.close()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_snapshot_excel_delivers_current_risk_values_after_new_process_load(
    backend: str, tmp_path: Path
) -> None:
    """静态文件和新进程文件都可直接读取本次风险值与中文类别。"""
    frame = pl.DataFrame(
        {
            "income": [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0],
            "category": ["甲", "甲", None, "乙", "乙", "乙", "甲", "乙"],
            "bad": [0, 0, 0, None, 1, 1, None, 1],
            "weight": [1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 3.0],
        }
    )
    report = profile_risk(
        frame.to_pandas() if backend == "pandas" else frame,
        features=["income", "category"], target="bad", weights_col="weight", n_bins=2,
        feature_metadata={"income": {"display_name": "月收入"}},
    ).report
    snapshot = snapshot_report(report)
    summary = snapshot.get_table("summary")
    rows: list[dict[str, Any]] = (
        summary.to_dict("records") if isinstance(summary, pd.DataFrame) else summary.to_dicts()
    )
    income = next(row for row in rows if row["feature"] == "income")
    assert math.isfinite(income["ks"]) and income["ks"] > 0
    assert math.isfinite(income["iv"]) and income["iv"] > 0
    output = tmp_path / "static.xlsx"
    snapshot.write_excel(str(output))
    _assert_current_excel(output, snapshot)
    archive = tmp_path / "risk.marsreport"
    report.save(archive)
    restored_output = tmp_path / "restored.xlsx"
    script = (
        "import sys; from mars.reporting import load_report; "
        "report=load_report(sys.argv[1]); report.write_excel(sys.argv[2]); "
        "print(report.report_id)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(mars.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", script, str(archive), str(restored_output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert completed.stdout.strip() == report.report_id
    _assert_current_excel(restored_output, snapshot)
    # 既有透视选项仍可写出原结构；这里只查 XML，不将它称为原生 Excel 刷新。
    pivot_output = tmp_path / "pivot.xlsx"
    report.write_excel(str(pivot_output), engine="openpyxl")
    with zipfile.ZipFile(pivot_output) as workbook_archive:
        pivots = [name for name in workbook_archive.namelist() if name.startswith("xl/pivotTables/") and name.endswith(".xml")]
        caches = [name for name in workbook_archive.namelist() if name.startswith("xl/pivotCache/pivotCacheDefinition") and name.endswith(".xml")]
        assert pivots and caches
        assert all(ElementTree.fromstring(workbook_archive.read(name)).attrib.get("refreshOnLoad") == "1" for name in caches)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_raw_and_binned_reuse_template_without_refit_and_keep_distinct_ks(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同模板两类 KS 不混写，模板表范围无示例行，最终值不依赖透视缓存。"""
    frame = pl.DataFrame({
        "x": list(range(12)) * 2,
        "category": ["甲", "乙"] * 12,
        "bad": [0, 1] * 6 + [None] * 12,
        "group": ["observed"] * 12 + ["undue"] * 12,
    })
    input_frame = frame.to_pandas() if backend == "pandas" else frame
    reports = {
        method: profile_risk(
            input_frame, target="bad", features=["x", "category"], group_col="group",
            n_bins=2, ks_method=method, ordered_metric_sort_by="bin_index",
            feature_metadata={"x": {"display_name": "连续字段", "data_source": "申请"}},
        ).report
        for method in ["binned", "raw"]
    }
    binned_ks = reports["binned"].get_table("summary", filters={"feature": "x"})
    raw_ks = reports["raw"].get_table("summary", filters={"feature": "x"})
    assert float(to_polars(binned_ks)["ks"][0]) != pytest.approx(float(to_polars(raw_ks)["ks"][0]))

    def forbid_refit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Excel 消费已经计算的报告，不得重新 fit。")

    monkeypatch.setattr(MarsNativeBinner, "fit", forbid_refit)
    for method, report in reports.items():
        output = tmp_path / f"{method}.xlsx"
        report.write_excel(str(output), engine="openpyxl")
        workbook = load_workbook(output, data_only=True)
        try:
            assert {"分组透视", "综合指标", "KS", "分组明细", "最终汇总", "最终KS趋势"} <= set(workbook.sheetnames)
            detail = workbook["分组明细"]
            native = to_polars(report.detail_table)
            assert detail.max_row == native.height + 1
            assert next(iter(detail.tables.values())).ref == f"A1:V{native.height + 1}"
            assert detail.freeze_panes == "A2"
            headers = list(next(detail.values))
            assert headers[:6] == ["y", "feature", "trend", "mars_group", "bin_index", "bin_label"]
            assert "display_name" not in headers and "dtype" not in headers and "data_source" not in headers
            actual = pl.DataFrame(list(detail.values)[1:], schema=headers, orient="row")
            assert actual["ks_bin"].to_list() == pytest.approx(native["ks_bin"].to_list(), nan_ok=True)
            if method == "raw":
                assert actual["ks"].to_list() == pytest.approx(native["ks"].to_list(), nan_ok=True)
                assert actual.filter((pl.col("feature") == "x") & (pl.col("mars_group") == "undue"))["ks"].null_count() > 0
                assert workbook["KS口径"]["B2"].value in {"raw", "binned"}
                assert "KS计算诊断" in workbook
            pivot = workbook["KS"]._pivots[0]
            assert pivot.cache.cacheFields[pivot.dataFields[0].fld].name == "ks"
            assert pivot.cache.cacheSource.worksheetSource.name == "detail"
            assert pivot.cache.refreshOnLoad and pivot.cache.recordCount == 0
            assert pivot.cache.records is None
            final = workbook["最终汇总"]
            final_headers = list(next(final.values))
            assert final_headers[:4] == ["feature", "display_name", "dtype", "data_source"]
            x_row = next(row for row in range(2, final.max_row + 1) if final.cell(row, 1).value == "x")
            assert final.cell(x_row, 2).value == "连续字段"
            assert final.cell(x_row, 4).value == "申请"
            assert final.freeze_panes == "A2" and final.auto_filter.ref is not None
            assert final.cell(x_row, final_headers.index("ks") + 1).value == pytest.approx(
                float(to_polars(report.summary_table).filter(pl.col("feature") == "x")["ks"][0]),
            )
        finally:
            workbook.close()


def to_polars(frame: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """测试中按真实原后端比较同一报告数值。"""
    return pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame


def test_raw_excel_rejects_missing_final_ks_instead_of_filling_binned(tmp_path: Path) -> None:
    """raw 最终值遗失时明确失败，不把尚存的分箱 KS 当作 raw 结果。"""
    report = profile_risk(
        pl.DataFrame({"x": list(range(12)), "bad": [0, 1] * 6}),
        target="bad", features=["x"], ks_method="raw", n_bins=2,
    ).report
    incomplete = MarsBinningReport(
        report.summary_table, report.trend_tables, report.detail_table.drop("ks"),
        detail_group_col=report.detail_group_col, report_meta=report.report_meta,
    )
    with pytest.raises(ValueError, match=r"detail\.ks.*ks_bin"):
        incomplete.write_excel(str(tmp_path / "invalid_raw.xlsx"), engine="openpyxl")
