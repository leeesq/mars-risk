"""Reporting 层公开契约回归测试。"""

from __future__ import annotations

from importlib import resources

import polars as pl
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries

from mars.reporting import MarsBinningReport, MarsProfileReport
from mars.reporting._binning_excel import _BinningExcelWriter
from mars.reporting._binning_html import _BinningHtmlRenderer
from mars.reporting._binning_plot import _BinningPlotRenderer
from mars.reporting._profile_excel import _ProfileExcelWriter


def _sample_binning_report() -> MarsBinningReport:
    """构造覆盖 HTML、Excel 与绘图入口的最小分箱报告。"""
    summary = pl.DataFrame(
        {
            "feature": ["income"],
            "data_source": ["base"],
            "iv": [0.12],
            "ks": [18.0],
            "auc": [0.62],
            "psi_max": [0.03],
            "rc_min": [1.0],
            "lift_min": [0.8],
            "lift_max": [1.5],
            "missing": [0.1],
            "target": ["target"],
        },
    )
    detail = pl.DataFrame(
        {
            "y": ["target", "target", "target"],
            "feature": ["income", "income", "income"],
            "trend": ["ascending", "ascending", "ascending"],
            "mars_group": ["2024-01", "2024-01", "Total"],
            "bin_index": [0, 1, 9999],
            "bin_label": ["low", "high", "Total"],
            "count": [60.0, 40.0, 100.0],
            "observed_count": [60.0, 40.0, 100.0],
            "bad": [6.0, 8.0, 14.0],
            "good": [54.0, 32.0, 86.0],
            "pct": [0.6, 0.4, 1.0],
            "bad_rate": [0.1, 0.2, 0.14],
            "lift": [0.7, 1.4, 1.0],
            "cum_count": [60.0, 100.0, 100.0],
            "cum_observed_count": [60.0, 100.0, 100.0],
            "cum_bad": [6.0, 14.0, 14.0],
            "cum_bad_rate": [0.1, 0.14, 0.14],
            "psi_bin": [0.01, 0.02, 0.03],
            "ks_bin": [8.0, 18.0, 18.0],
            "auc_bin": [0.2, 0.42, 0.62],
            "iv_bin": [0.04, 0.08, 0.12],
            "total_count": [100.0, 100.0, 100.0],
            "bin_type": ["正常组", "正常组", "汇总组"],
            "data_source": ["base", "base", "base"],
        },
    )
    trend_tables = {
        "iv": pl.DataFrame({"feature": ["income"], "dtype": ["numeric"], "2024-01": [0.12], "Total": [0.12]}),
        "bad_rate": pl.DataFrame({"feature": ["income"], "dtype": ["numeric"], "2024-01": [0.14], "Total": [0.14]}),
    }
    return MarsBinningReport(
        summary_table=summary,
        trend_tables=trend_tables,
        detail_table=detail,
        feature_data_source={"income": "base"},
        report_meta={
            "targets": ["target"],
            "row_count": 100,
            "feature_count": 1,
            "profile_by_input": "mars_group",
            "group_count": 1,
            "event_rate_by_target": {"target": 0.14},
        },
    )


def test_binning_report_html_exports_public_sections(tmp_path) -> None:
    """HTML 导出需要保留公开区块和运行时入口。"""
    report = _sample_binning_report()
    output_path = tmp_path / "report.html"

    report.write_html(str(output_path), max_plots=0)

    html_text = output_path.read_text(encoding="utf-8")
    assert "<title>MARS Evaluation Report</title>" in html_text
    assert "Dataset Overview" in html_text
    assert "Summary" in html_text
    assert "Trend Tables" in html_text
    assert "Grouped Pivot" in html_text
    assert "Charts" in html_text
    assert "Global search across tables and charts" in html_text
    assert "marsQueueRefresh" in html_text


def test_report_objects_delegate_to_internal_renderers() -> None:
    """报告对象保持数据容器身份，不再通过 mixin 继承导出能力。"""
    assert _BinningExcelWriter not in MarsBinningReport.__mro__
    assert _BinningHtmlRenderer not in MarsBinningReport.__mro__
    assert _BinningPlotRenderer not in MarsBinningReport.__mro__
    assert _ProfileExcelWriter not in MarsProfileReport.__mro__

    binning_report = _sample_binning_report()
    profile_report = MarsProfileReport(
        overview=pl.DataFrame({"feature": ["age"], "missing_rate": [0.0]}),
        dq_tables={},
        stats_tables={},
    )

    assert callable(binning_report.write_excel)
    assert callable(binning_report.write_html)
    assert callable(binning_report.plot_risk_trends)
    assert callable(profile_report.write_excel)


def test_binning_report_excel_uses_template_columns_only(tmp_path) -> None:
    """Excel 明细导出只能写模板表头已有列。"""
    report = _sample_binning_report()
    output_path = tmp_path / "report.xlsx"

    report.write_excel(str(output_path), engine="openpyxl")

    template_path = resources.files("mars.reporting").joinpath(
        "template",
        "mars_bin_report_linux.xlsx",
    )
    template_workbook = load_workbook(template_path)
    output_workbook = load_workbook(output_path)
    try:
        template_sheet = template_workbook["分组明细"]
        output_sheet = output_workbook["分组明细"]
        template_ref = next(iter(template_sheet.tables.values())).ref
        output_ref = next(iter(output_sheet.tables.values())).ref
        _, _, template_max_col, _ = range_boundaries(template_ref)
        _, _, output_max_col, _ = range_boundaries(output_ref)
        template_headers = [
            template_sheet.cell(row=1, column=col_idx).value
            for col_idx in range(1, template_max_col + 1)
        ]
        output_headers = [
            output_sheet.cell(row=1, column=col_idx).value
            for col_idx in range(1, output_max_col + 1)
        ]
        assert output_headers == template_headers
        assert output_max_col == template_max_col
    finally:
        template_workbook.close()
        output_workbook.close()


def test_binning_report_plot_entry_uses_reporting_plotter(monkeypatch) -> None:
    """绘图公开入口必须继续由 report 对象统一转调。"""
    report = _sample_binning_report()
    captured: dict[str, object] = {}

    def fake_plot_feature_binning_risk_trend_batch(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "mars.reporting._binning_plot.MarsPlotter.plot_feature_binning_risk_trend_batch",
        fake_plot_feature_binning_risk_trend_batch,
    )

    report.plot_risk_trends(features="income", show_risk="both", max_plots=1)

    assert captured["features"] == ["income"]
    assert captured["show_risk"] == "both"
