"""分箱 HTML 的页面导航、真实内容和多目标摘要回归。"""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsBinningReport
from mars.reporting.html_assets import build_html_runtime_script


class _BinningPages(HTMLParser):
    """提取实际生成的页面和导航，不假设两者使用相同的 key。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[dict[str, Any]] = []
        self.navigation: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "data-mars-view" in attributes:
            self.pages.append({
                "page": attributes["data-mars-view"],
                "tagName": tag.upper(),
                "open": "open" in attributes,
            })
        if "mars-page-nav" in (attributes.get("class") or "").split():
            self.navigation.append(str(attributes["data-page"]))


def _report(backend: str, missing: pd.DataFrame | None = None) -> MarsBinningReport:
    """构造两个特征和两个标签，覆盖重复汇总行及真实分组内容。"""
    summary = pd.DataFrame({
        "feature": ["income", "score", "income", "score"],
        "target": ["bad30", "bad30", "late60", "late60"],
        "iv": [0.12, 0.2, 0.1, 0.18],
    })
    detail = pd.DataFrame({
        "feature": ["income", "income"], "y": ["bad30", "bad30"],
        "mars_group": ["validation", "validation"], "bin_index": [0, 1],
        "bin_label": ["low", "high"], "count": [60.0, 40.0],
        "bad": [6.0, 8.0], "lift": [0.7, 1.4], "iv_bin": [0.04, 0.08],
    })
    trend = pd.DataFrame({"feature": ["income"], "validation": [0.12]})
    return MarsBinningReport(
        summary if backend == "pandas" else pl.from_pandas(summary),
        {"iv": trend if backend == "pandas" else pl.from_pandas(trend)},
        detail if backend == "pandas" else pl.from_pandas(detail),
        group_col="dataset", detail_group_col="mars_group", dt_col="application_date",
        missing_by_day_table=(
            missing if backend == "pandas" or missing is None else pl.from_pandas(missing)
        ),
        report_meta={"row_count": 100, "feature_count": 2, "targets": ["bad30", "late60"]},
        feature_metadata={"income": {"display_name": "申报收入", "description": "<script>unsafe</script>"}},
    )


def _export(report: MarsBinningReport, output: Path) -> tuple[str, _BinningPages]:
    """通过公共导出入口读取实际 HTML，不调用模板私有拼接方法。"""
    report.write_html(str(output), include_charts=False)
    exported = output.read_text(encoding="utf-8")
    pages = _BinningPages()
    pages.feed(exported)
    return exported, pages


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_binning_navigation_has_distinct_metadata_and_context_with_real_pivot(
    backend: str, tmp_path: Path,
) -> None:
    """元数据与数据口径各自可达，多标签不会重复计特征，透视页直接有内容。"""
    exported, parsed = _export(_report(backend), tmp_path / "binning.html")
    views = [page["page"] for page in parsed.pages]
    assert views[0] == "overview"
    assert views.count("semantics") == views.count("overview") == 1
    assert set(parsed.navigation) == set(views)
    assert '<div class="mars-pill">Features: 2</div>' in exported
    assert "Dataset Context" in exported
    assert "Interactive monitoring report" not in exported
    assert any(page["page"] == "pivot" and page["open"] for page in parsed.pages)
    assert "No grouped pivot data available" not in exported
    assert "validation" in exported and "low" in exported and "high" in exported
    assert 'class="dataframe mars-data-table mars-semantic-table"' in exported
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in exported
    assert "<script>unsafe</script>" not in exported


def test_feature_count_falls_back_to_detail_without_summary_rows(tmp_path: Path) -> None:
    """没有汇总行时，已有明细特征仍应计入报告头部。"""
    source = _report("polars")
    report = MarsBinningReport(
        source.summary_table.head(0), source.trend_tables, source.detail_table,
        group_col=source.group_col, detail_group_col=source.detail_group_col,
    )
    exported, _ = _export(report, tmp_path / "detail_only.html")
    assert '<div class="mars-pill">Features: 1</div>' in exported


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("populated", [None, False, True])
def test_daily_missing_navigation_only_appears_with_available_evidence(
    backend: str, populated: bool | None, tmp_path: Path,
) -> None:
    """未计算和零行日缺失表不提供空入口，已有数据保持可读。"""
    missing = None if populated is None else pd.DataFrame({
        "feature": ["income"] if populated else [],
        "2026-01-15": [0.125] if populated else [],
    })
    exported, parsed = _export(_report(backend, missing), tmp_path / "missing.html")
    assert ("missing-day" in parsed.navigation) is bool(populated)
    if populated:
        assert "12.50%" in exported
    else:
        assert 'id="missing-day-section"' not in exported
        assert "No daily missing-rate table is available" not in exported


_NAVIGATION_FIXTURE = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const make = spec => {
  const node = {...spec, active:false};
  node.dataset = {marsView:spec.page, page:spec.page};
  node.classList = {toggle:(_, active)=>{node.active=active;}};
  return node;
};
const pages=input.pages.map(make), navigation=input.navigation.map(page=>make({page}));
const location={hash:''}, history={
  pushState:(_,__,value)=>{location.hash=value;},
  replaceState:(_,__,value)=>{location.hash=value;}
};
const document={querySelectorAll:selector=>{
  if(selector==='[data-mars-view]') return pages;
  if(selector==='.mars-page-nav') return navigation;
  throw Error(selector);
}};
const context={document, location, history, marsState:{},
  marsQueueLayoutSync:()=>{}, marsScheduleViewportRefresh:()=>{}};
vm.createContext(context);
vm.runInContext(input.script, context);
const observations=[];
const observe=()=>observations.push({hash:location.hash,
  pages:pages.filter(page=>page.active).map(page=>({page:page.page, open:page.open})),
  navigation:navigation.filter(link=>link.active).map(link=>link.page)});
context.marsApplyPageFromHash(); observe();
context.marsNavigateTo('semantics'); observe();
context.marsNavigateTo('#semantics-section'); observe();
context.marsNavigateTo('pivot'); observe();
pages.find(page=>page.page==='pivot').open=false;
context.marsNavigateTo('semantics');
context.marsNavigateTo('#pivot-section'); observe();
process.stdout.write(JSON.stringify(observations));
"""


def test_exported_navigation_script_opens_metadata_and_reentered_pivot(tmp_path: Path) -> None:
    """执行实际导航脚本，验证元数据独立页、旧锚点和再次进入的透视内容。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装；实际页面导航脚本回归未执行。")
    _, parsed = _export(_report("polars"), tmp_path / "navigation.html")
    runtime = build_html_runtime_script([])
    # 仅隔离导航函数；不模拟筛选、图表和浏览器布局，真实浏览器另行验收。
    script = runtime[runtime.index("function marsAvailablePages()"):runtime.index("function marsResolveLocalScope(")]
    completed = subprocess.run(
        [node, "-e", _NAVIGATION_FIXTURE],
        input=json.dumps({"script": script, "pages": parsed.pages, "navigation": parsed.navigation}),
        text=True, capture_output=True, encoding="utf-8", check=True,
    )
    observations = json.loads(completed.stdout)
    for expected, actual in zip(["overview", "semantics", "semantics", "pivot", "pivot"], observations):
        assert actual == {
            "hash": f"#{expected}", "navigation": [expected],
            "pages": [{"page": expected, "open": True}],
        }
