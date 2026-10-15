"""分箱 HTML 搜索、明细开关与多标签计数回归。"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsBinningReport, load_report
from mars.reporting._binning_html import _BinningHtmlRenderer
from mars.reporting._binning_plot import _BinningPlotRenderer


def _report(*, backend: str = "polars", with_time: bool = True) -> MarsBinningReport:
    """建立重复中文名、特殊字符及两个标签的可信小报告。"""
    features = ["income", "income_other", "debt"]
    summary = pl.DataFrame(
        {
            "feature": features * 2,
            "target": ["bad"] * 3 + ["later"] * 3,
            "iv": [0.12] * 6,
            "ks": [18.0] * 6,
            "psi_max": [0.3, 0.25, float("nan"), 0.1, 0.25, None],
        }
    )
    rows: list[dict[str, Any]] = []
    for target in ("bad", "later"):
        for feature in features:
            for group in ("2024-01", "Total"):
                for index, count, bad in ((0, 60.0, 6.0), (1, 40.0, 8.0)):
                    rows.append(
                        {
                            "y": target,
                            "feature": feature,
                            "mars_group": group,
                            "bin_index": index,
                            "bin_label": "low" if index == 0 else "high",
                            "bin_type": "正常组",
                            "count": count,
                            "observed_count": count,
                            "bad": bad,
                            "good": count - bad,
                            "pct": count / 100.0,
                            "bad_rate": bad / count,
                            "lift": bad / count / 0.14,
                            "cum_count": count if index == 0 else 100.0,
                            "cum_bad": bad if index == 0 else 14.0,
                            "cum_bad_rate": 0.1 if index == 0 else 0.14,
                            "ks_bin": 8.0 if index == 0 else 18.0,
                        }
                    )
    detail = pl.DataFrame(rows)
    metadata = {
        "income": {"display_name": '家庭收入 "<>&', "data_source": "application"},
        "income_other": {"display_name": '家庭收入 "<>&', "data_source": "application"},
        "debt": {"display_name": "负债余额", "data_source": "credit"},
    }
    return MarsBinningReport(
        summary.to_pandas() if backend == "pandas" else summary,
        {},
        detail.to_pandas() if backend == "pandas" else detail,
        group_col="month",
        detail_group_col="mars_group",
        dt_col="biz_dt" if with_time else None,
        feature_metadata=metadata,
        report_meta={
            "targets": ["bad", "later"],
            **({"dt_col": "biz_dt", "start_dt": "2024-01-01", "end_dt": "2024-01-31"} if with_time else {}),
        },
    )


class _ChartCards(HTMLParser):
    """以 HTML 解析后的实际属性检查索引和稳定英文身份。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and attributes.get("class") == "mars-chart-card":
            self.cards.append(attributes)


@pytest.mark.parametrize("mode", ["inline", "asset"])
def test_chart_search_index_uses_registered_display_name(tmp_path: Path, mode: str) -> None:
    # 离线 PNG 验收使用无窗口后端，不依赖本机 Tk/桌面会话。
    matplotlib.use("Agg")
    report = _report()
    output = tmp_path / f"{mode}.html"
    report.write_html(str(output), chart_embed_mode=mode, include_detail=False)
    parser = _ChartCards()
    parser.feed(output.read_text(encoding="utf-8"))
    assert len(parser.cards) == 6
    for card in parser.cards:
        feature = card["data-feature"]
        assert feature in report.feature_metadata
        display_name = report.feature_metadata[feature]["display_name"]
        assert display_name.lower() in card["data-search-text"]
        assert feature in card["data-search-text"]
        assert card["data-target"] in card["data-search-text"]
        assert card["data-data-source"] in card["data-search-text"]
    if mode == "asset":
        assert 'data-src="asset_assets/' in output.read_text(encoding="utf-8")
        assert all("家庭收入" not in path.name for path in (tmp_path / "asset_assets").iterdir())


@pytest.mark.parametrize("include_detail", [True, False])
@pytest.mark.parametrize("include_charts", [True, False])
def test_html_detail_and_charts_switches_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_detail: bool,
    include_charts: bool,
) -> None:
    report = _report(with_time=include_charts)
    before = {name: frame.clone() for name, frame in report._query_tables().items()}
    description = report.describe()
    calls: list[bool] = []

    def chart_section(renderer: Any, **kwargs: Any) -> str:
        calls.append(True)
        assert kwargs["detail_pd"].shape[0] == before["detail"].height
        return '<details id="chart-section" class="mars-section"><summary>Charts</summary></details>'

    def reject_plot_context(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Pure table export must not resolve plotting time metadata.")

    monkeypatch.setattr(_BinningHtmlRenderer, "_build_chart_section_html", chart_section)
    monkeypatch.setattr(_BinningPlotRenderer, "_resolve_risk_trend_contexts", reject_plot_context)
    output = tmp_path / "switches.html"
    report.write_html(str(output), include_detail=include_detail, include_charts=include_charts)
    text = output.read_text(encoding="utf-8")
    assert ('id="pivot-section"' in text) is include_detail
    assert ('href="#pivot"' in text) is include_detail
    assert ('id="mars-pivot-target"' in text) is include_detail
    assert ('class="mars-pivot-view"' in text) is include_detail
    assert ('id="chart-section"' in text) is include_charts
    assert bool(calls) is include_charts
    assert report.describe() == description
    for name, frame in before.items():
        assert report.get_table(name).equals(frame)
    snapshot = tmp_path / "switches.marsreport"
    report.save(snapshot)
    loaded = load_report(snapshot)
    for name, frame in before.items():
        assert loaded.get_table(name).equals(frame)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("high_values", [[0.3, 0.1], [0.3, 0.4]])
def test_notebook_and_html_count_unique_features_across_targets(
    tmp_path: Path, backend: str, high_values: list[float]
) -> None:
    report = _report(backend=backend)
    if backend == "polars":
        report._summary = report.summary_table.with_columns(
            pl.when(pl.col("feature") == "income")
            .then(pl.when(pl.col("target") == "bad").then(high_values[0]).otherwise(high_values[1]))
            .otherwise(pl.col("psi_max"))
            .alias("psi_max")
        )
    else:
        report._summary.loc[report._summary["feature"] == "income", "psi_max"] = high_values
    notebook = report._repr_html_()
    assert re.search(r"Features:</strong>\s*3", notebook)
    assert "PSI > 0.25 的特征（任一标签触发）" in html.unescape(notebook)
    assert re.search(r"font-weight:bold;\">1</span>", notebook)
    output = tmp_path / "counts.html"
    report.write_html(str(output), include_charts=False, include_detail=False)
    text = output.read_text(encoding="utf-8")
    assert "Features: 3" in text
    assert "PSI &gt; 0.25 的特征（任一标签触发）: 1" in text


@pytest.mark.parametrize("fallback", ["detail", "metadata", "empty"])
def test_html_counts_handle_empty_summary_and_uncomputed_psi(
    tmp_path: Path, fallback: str
) -> None:
    report = _report()
    report._summary = report.summary_table.head(0).drop("psi_max")
    if fallback != "detail":
        report._detail = report.detail_table.head(0)
    if fallback == "empty":
        report.feature_metadata = {}
    expected = 0 if fallback == "empty" else 3
    notebook = report._repr_html_()
    assert re.search(rf"Features:</strong>\s*{expected}", notebook)
    assert re.search(r"font-weight:bold;\">0</span>", notebook)
    assert "color: green" not in notebook
    output = tmp_path / "empty.html"
    report.write_html(str(output), include_charts=False, include_detail=False)
    text = output.read_text(encoding="utf-8")
    assert f"Features: {expected}" in text
    assert "PSI &gt; 0.25 的特征（任一标签触发）: 0" in text


def test_notebook_count_preserves_nullable_psi_and_escaped_group() -> None:
    report = _report(backend="pandas")
    report._summary["psi_max"] = pd.Series([pd.NA, 0.25, pd.NA] * 2, dtype="Float64")
    report.group_col = '<script>"month"</script>'
    before = report._summary.copy(deep=True)
    notebook = report._repr_html_()
    assert re.search(r"font-weight:bold;\">0</span>", notebook)
    assert '<script>"month"</script>' not in notebook
    assert html.escape(report.group_col) in notebook
    pd.testing.assert_frame_equal(report._summary, before)


def test_html_count_without_psi_does_not_use_stale_metadata_count(tmp_path: Path) -> None:
    report = _report()
    report._summary = report.summary_table.drop("psi_max")
    report.report_meta["feature_count"] = 99
    before = report.describe()
    output = tmp_path / "without_psi.html"
    report.write_html(str(output), include_detail=False, include_charts=False)
    text = output.read_text(encoding="utf-8")
    assert "Features: 3" in text
    assert 'mars-kpi-label">Features</div><div class="mars-kpi-value">3</div>' in text
    assert "PSI &gt; 0.25 的特征（任一标签触发）: 0" in text
    assert report.describe() == before
