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
from mars.reporting import ReportSnapshot, snapshot_report


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
