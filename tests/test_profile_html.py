"""Profile 与 snapshot 生成脚本的最小 DOM 事件回归，不承担浏览器视觉验收。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.reporting import MarsProfileReport, snapshot_report
from mars.utils.html import format_html_value


class _ReportTables(HTMLParser):
    """从实际导出的表提取单元格，为最小 DOM fixture 提供真实文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self.table: dict[str, Any] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.in_body = False
        self.page: str | None = None
        self.row_attributes: list[dict[str, str | None]] = []
        self.cell_attributes: dict[str, str | None] = {}
        self.metric_buttons: list[dict[str, str | None]] = []
        self.metric_panels: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "metric-tab" in (attributes.get("class") or "").split():
            self.metric_buttons.append(attributes)
        if "metric-panel" in (attributes.get("class") or "").split():
            self.metric_panels.append(attributes)
        if tag == "div" and "page" in (attributes.get("class") or "").split():
            self.page = attributes["data-page"]
        elif tag == "table":
            self.table = {"id": attributes["id"], "page": self.page, "headers": [], "rows": [], "attributes": []}
        elif tag == "tbody":
            self.in_body = True
        elif tag == "tr" and self.table is not None:
            self.row = []
            self.row_attributes = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
            self.cell_attributes = attributes

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.row_attributes.append(self.cell_attributes)
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if self.in_body:
                self.table["rows"].append(self.row)
                self.table["attributes"].append(self.row_attributes)
            else:
                self.table["headers"] = self.row
            self.row = None
        elif tag == "tbody":
            self.in_body = False
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


_DOM_FIXTURE = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor(dataset = {}) { this.dataset = dataset; this.value = ''; this.listeners = {}; }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  setAttribute(name, value) { this[name] = value; }
  focus() { this.focused = true; }
  fire(name, key) { this.listeners[name]({target: this, key, preventDefault() {}}); }
}
const tables = input.tables.map(spec => {
  const table = {id: spec.id, page: spec.page};
  const body = {rows: spec.rows.map((cells, rowIndex) => ({
    get innerText() {
      const visible = !this.hidden && pages.find(page => page.dataset.page === table.page).active;
      return visible ? cells.join('\t') : '\n      ' + cells.join('\n      ') + '\n    ';
    },
    hidden: false, cells: cells.map((text, columnIndex) => ({innerText: text, textContent: text,
      dataset: {sortType: spec.attributes[rowIndex][columnIndex]['data-sort-type'], sortValue: spec.attributes[rowIndex][columnIndex]['data-sort-value']}}))
  }))};
  body.appendChild = row => { body.rows.splice(body.rows.indexOf(row), 1); body.rows.push(row); };
  table.tBodies = [body];
  table.headers = spec.headers.map(() => new Element());
  table.headers.forEach(header => {
    header.parentNode = {children: table.headers}; header.closest = () => table;
  });
  return table;
});
const global = new Element(), locals = tables.map(table => new Element({table: table.id}));
const pages = input.pages.map((name, index) => ({
  dataset: {page: name}, active: index === 0,
  classList: {toggle: (_, active) => { pages.find(page => page.dataset.page === name).active = active; }}
}));
const navigation = input.pages.map(page => new Element({page}));
const metricButtons = (input.metricButtons || []).map(attributes => {
  const element = new Element({metricGroup: attributes['data-metric-group'], metricPanel: attributes['data-metric-panel']});
  element.id = attributes.id; element['aria-selected'] = attributes['aria-selected']; return element;
});
const metricPanels = (input.metricPanels || []).map(attributes => {
  const element = new Element({metricGroup: attributes['data-metric-group'], metricPanel: attributes['data-metric-panel']});
  element.active = attributes.class.split(' ').includes('active');
  element.classList = {toggle: (_, active) => {element.active = active;}}; return element;
});
const selectors = {'.table-search': locals, '.report-table': tables,
  '.nav-button': navigation, '.page': pages, '.metric-tab': metricButtons, '.metric-panel': metricPanels,
  '.report-table th': tables.flatMap(table => table.headers)};
const document = {
  getElementById: id => id === 'global-search' ? global : tables.find(table => table.id === id),
  querySelectorAll: selector => { if (!(selector in selectors)) throw Error(selector); return selectors[selector]; }
};
vm.runInNewContext(input.script, {document, Map, Set, Number, String});
const observations = input.actions.map(action => {
  if (action.kind === 'global') { global.value = action.value; global.fire('input'); }
  if (action.kind === 'local') {
    const local = locals.find(input => input.dataset.table === action.table);
    local.value = action.value; local.fire('input');
  }
  if (action.kind === 'sort') tables.find(table => table.id === action.table).headers[action.column].fire('click');
  if (action.kind === 'navigation') navigation.find(button => button.dataset.page === action.value).fire('click');
  if (action.kind === 'metric') metricButtons.find(button => button.id === action.value).fire('click');
  if (action.kind === 'key') metricButtons.find(button => button.id === action.value).fire('keydown', action.key);
  return {
    rows: Object.fromEntries(tables.map(table => [table.id,
      table.tBodies[0].rows.filter(row => !row.hidden).map(row => row.cells.map(cell => cell.innerText))])),
    active: pages.filter(page => page.active).map(page => page.dataset.page),
    metrics: metricPanels.filter(panel => panel.active).map(panel => panel.dataset.metricPanel),
    selected: metricButtons.filter(button => button['aria-selected'] === 'true').map(button => button.id)
  };
});
process.stdout.write(JSON.stringify(observations));
"""


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_profile_and_snapshot_search_intersect_after_clear_navigation_and_sort(
    backend: str, snapshot: bool, tmp_path: Path
) -> None:
    """执行实际脚本事件，组合搜索与导航排序不会覆写另一条件。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装，最小 DOM 事件回归未运行；真实浏览器另行验收。")
    features = ["income", "debt", "income_aux", "中文收入"]
    table = pd.DataFrame({
        "feature": features, "value": [3, 2, 5, 1], "total": [3, 2, 5, 1],
        "note": ["plain", "another note", "a  b", "中文　收入"],
    })
    native = pl.from_pandas(table) if backend == "polars" else table
    report = MarsProfileReport(native, dq_tables={}, stats_tables={"mean": native})
    exported = snapshot_report(report) if snapshot else report
    path = tmp_path / "profile.html"
    exported.write_html(str(path))
    html = path.read_text(encoding="utf-8")
    parser = _ReportTables()
    parser.feed(html)
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert script is not None
    main = next(table for table in parser.tables if table["page"] == "Overview")
    other = next(table for table in parser.tables if table["page"] == "Stats")
    value_column = main["headers"].index("value")
    pages = list(dict.fromkeys(re.findall(r'data-page="([^"]+)"', html)))
    # Pandas 版本可能输出 ASCII 空格或 NBSP；按实际单元格文本保持字面匹配。
    literal_query: str = next(
        row[main["headers"].index("note")] for row in main["rows"] if row[0] == "income_aux"
    )
    actions: list[dict[str, Any]] = [
        {"kind": "navigation", "value": "Overview"},
        {"kind": "global", "value": "income"},
        {"kind": "local", "table": main["id"], "value": "debt"},
        {"kind": "local", "table": main["id"], "value": "DEBT"},
        {"kind": "local", "table": main["id"], "value": ""},
        {"kind": "local", "table": main["id"], "value": "debt"},
        {"kind": "global", "value": ""},
        {"kind": "global", "value": "INCOME"},
        {"kind": "local", "table": main["id"], "value": "no_matching_row"},
        {"kind": "sort", "table": main["id"], "column": value_column},
        {"kind": "navigation", "value": "Stats"},
        {"kind": "local", "table": other["id"], "value": "中文"},
        {"kind": "global", "value": ""},
        {"kind": "navigation", "value": "Overview"},
        {"kind": "local", "table": main["id"], "value": ""},
        {"kind": "global", "value": " income "},
        {"kind": "global", "value": "income"},
        {"kind": "local", "table": main["id"], "value": "debt"},
        {"kind": "local", "table": main["id"], "value": "_aux"},
        {"kind": "sort", "table": main["id"], "column": value_column},
        {"kind": "global", "value": ""},
        {"kind": "local", "table": main["id"], "value": ""},
        {"kind": "local", "table": other["id"], "value": ""},
        {"kind": "navigation", "value": "Overview"},
        {"kind": "global", "value": " debt", "checkpoint": "prefix_visible"},
        {"kind": "global", "value": "no_matching_row"},
        {"kind": "global", "value": " debt", "checkpoint": "prefix_after_hide"},
        {"kind": "global", "value": " debt", "checkpoint": "prefix_repeated"},
        {"kind": "navigation", "value": "Stats"},
        {"kind": "global", "value": " debt", "checkpoint": "prefix_hidden_page"},
        {"kind": "navigation", "value": "Overview", "checkpoint": "prefix_back_visible"},
        {"kind": "global", "value": "a  b", "checkpoint": "ascii_cell_spaces"},
        {"kind": "global", "value": literal_query, "checkpoint": "literal_cell_spaces"},
        {"kind": "global", "value": ""},
    ]
    completed = subprocess.run(
        [node, "-e", _DOM_FIXTURE],
        input=json.dumps({"script": script.group(1), "tables": parser.tables, "pages": pages, "actions": actions, "metricButtons": parser.metric_buttons, "metricPanels": parser.metric_panels}),
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    actual: list[dict[str, Any]] = json.loads(completed.stdout)
    global_query = ""
    local_queries = {table["id"]: "" for table in parser.tables}
    active = pages[0]
    for action, observation in zip(actions, actual):
        if action["kind"] == "global":
            global_query = action["value"].lower()
        elif action["kind"] == "local":
            local_queries[action["table"]] = action["value"].lower()
        elif action["kind"] == "navigation":
            active = action["value"]
        assert observation["active"] == [active]
        for table in parser.tables:
            expected = [
                row for row in table["rows"]
                if global_query in "\t".join(row).lower()
                and local_queries[table["id"]] in "\t".join(row).lower()
            ]
            assert sorted(observation["rows"][table["id"]]) == sorted(expected), action
    assert actual[2]["rows"][main["id"]] == []
    assert actual[7]["rows"] == actual[2]["rows"]
    assert actual[8]["rows"][main["id"]] == []
    assert actual[18]["rows"][main["id"]][0][0] == "income_aux"
    checkpoints: dict[str, dict[str, list[list[str]]]] = {
        action["checkpoint"]: observation["rows"]
        for action, observation in zip(actions, actual) if "checkpoint" in action
    }
    for name, rows in checkpoints.items():
        if name.startswith("prefix_"):
            assert rows == checkpoints["prefix_visible"]
            assert rows[main["id"]] == []
    assert bool(checkpoints["ascii_cell_spaces"][main["id"]]) == (literal_query == "a  b")
    assert checkpoints["literal_cell_spaces"][main["id"]][0][0] == "income_aux"
    assert len(actual[-1]["rows"][main["id"]]) == len(features)
    values = [int(row[value_column]) for row in actual[-1]["rows"][main["id"]]]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("value, expected", [(0, "0.00%"), (1, "100.00%"), (0.125, "12.50%")])
def test_shared_html_formatter_applies_percentage_to_integer_endpoints(value: float, expected: str) -> None:
    assert format_html_value(value, as_percent=True) == expected


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_profile_formats_raw_values_and_sorts_without_display_rounding(
    backend: str, snapshot: bool, tmp_path: Path,
) -> None:
    """百分比、同显示值、整数、类别、日期和缺失均按真实类型消费。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装，最小 DOM 事件回归未运行；真实浏览器另行验收。")
    frame = pd.DataFrame({
        "feature": ["one", "two", "three", "four"],
        "missing_rate": [1.0, 0.2, 0.09, None],
        "zeros_rate": [0, 1, 0.125, None],
        "mean": [0.10002, 0.10001, -2.0, float("inf")],
        "count": pd.Series([1000, 3, 1, None], dtype="Int64"),
        "weight_sum": [1.25, 0.5, 0.0, None],
        "mode_value": ["10", "2", "1", None],
        "bin_label": [1.234567, 0.125, 1000.0, None],
        "comparable": pd.Series([True, False, True, None], dtype="boolean"),
        "date": pd.to_datetime(["2026-01-02", "2025-01-01", "2026-01-01", None]),
        "auc": [0.75, 0.8, 0.5, None],
        "ks": [25.0, 0, 100, None],
        "psi": [0.25, 0.10001, 0.10002, None],
    })
    native = pl.from_pandas(frame) if backend == "polars" else frame
    trend = pd.DataFrame({"feature": frame["feature"], "总计 期.一": [1.0, 0.2, 0.09, None]})
    native_trend = pl.from_pandas(trend) if backend == "polars" else trend
    report = MarsProfileReport(
        native, dq_tables={"missing": native_trend, "zeros": native_trend},
        stats_tables={"mean": native_trend, "psi": native_trend, "指标.甲 <乙>": native_trend},
        feature_metadata={"one": {"display_name": "很长中文名" * 12 + "<script>", "data_source": "申请 & 外部"}},
    )
    summary = report._repr_html_()
    assert 'profile_report.write_html("profile_report.html")' in summary
    assert "指标.甲 &lt;乙&gt;" in summary and "<table" not in summary
    before = report.get_table("overview")
    exported = snapshot_report(report) if snapshot else report
    path = tmp_path / "formatted.html"
    exported.write_html(str(path))
    html = path.read_text(encoding="utf-8")
    parser = _ReportTables()
    parser.feed(html)
    main = next(table for table in parser.tables if table["page"] == "Overview")
    indexes = {name: main["headers"].index(name) for name in frame.columns}
    first = main["rows"][0]
    assert first[indexes["missing_rate"]] == "100.00%"
    assert first[indexes["zeros_rate"]] == "0.00%"
    assert main["rows"][2][indexes["zeros_rate"]] == "12.50%"
    assert first[indexes["count"]] == "1,000"
    assert first[indexes["weight_sum"]] == "1.25"
    assert first[indexes["psi"]] == "0.25"
    assert first[indexes["ks"]] == "25.00"
    assert first[indexes["auc"]] == "0.75"
    assert first[indexes["bin_label"]] == "1.234567"
    assert first[indexes["mean"]] == main["rows"][1][indexes["mean"]] == "0.10"
    assert main["rows"][3][indexes["mean"]] == "—"
    for column in ("mode_value", "bin_label", "comparable", "date", "display_name", "dtype", "data_source"):
        index = main["headers"].index(column)
        assert all("style" not in row[index] for row in main["attributes"])
    assert main["attributes"][0][indexes["mode_value"]]["data-sort-type"] == "text"
    assert main["attributes"][0][indexes["date"]]["data-sort-type"] == "date"
    assert main["attributes"][0][indexes["comparable"]]["data-sort-type"] == "boolean"
    assert "<script>" not in html.split("<script>", 1)[0] and "&lt;script&gt;" in html
    assert "KS uses 0–100 points" in html
    dq_table = next(table for table in parser.tables if table["page"] == "DQ")
    assert dq_table["rows"][0][-1] == "100.00%"
    all_ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(all_ids) == len(set(all_ids))
    assert all(button["aria-controls"] in all_ids for button in parser.metric_buttons)
    stats_buttons = [button for button in parser.metric_buttons if button["data-metric-group"] == "profile-page-3"]
    actions: list[dict[str, Any]] = [
        {"kind": "sort", "table": main["id"], "column": indexes["missing_rate"]},
        {"kind": "sort", "table": main["id"], "column": indexes["missing_rate"]},
        {"kind": "sort", "table": main["id"], "column": indexes["mean"]},
        {"kind": "sort", "table": main["id"], "column": indexes["mode_value"]},
        {"kind": "sort", "table": main["id"], "column": indexes["date"]},
        {"kind": "sort", "table": main["id"], "column": indexes["comparable"]},
        {"kind": "navigation", "value": "Stats"},
        {"kind": "metric", "value": stats_buttons[1]["id"]},
        {"kind": "navigation", "value": "DQ"},
        {"kind": "navigation", "value": "Stats"},
        {"kind": "key", "value": stats_buttons[1]["id"], "key": "End"},
        {"kind": "key", "value": stats_buttons[-1]["id"], "key": "ArrowRight"},
        {"kind": "global", "value": "申请 & 外部"},
        {"kind": "local", "table": main["id"], "value": "three"},
        {"kind": "local", "table": main["id"], "value": ""},
    ]
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert script is not None
    result = subprocess.run(
        [node, "-e", _DOM_FIXTURE], input=json.dumps({
            "script": script.group(1), "tables": parser.tables,
            "pages": list(dict.fromkeys(re.findall(r'data-page="([^"]+)"', html))),
            "actions": actions, "metricButtons": parser.metric_buttons, "metricPanels": parser.metric_panels,
        }), capture_output=True, text=True, encoding="utf-8", check=True,
    )
    actual: list[dict[str, Any]] = json.loads(result.stdout)
    orders = [[row[0] for row in observation["rows"][main["id"]]] for observation in actual]
    assert orders[0] == ["three", "two", "one", "four"]
    assert orders[1] == ["one", "two", "three", "four"]
    assert orders[2] == ["three", "two", "one", "four"]
    assert orders[3] == ["three", "one", "two", "four"]
    assert orders[4] == ["two", "three", "one", "four"]
    assert orders[5] == ["two", "three", "one", "four"]
    for index in (7, 8, 9):
        assert stats_buttons[1]["id"] in actual[index]["selected"]
    assert stats_buttons[-1]["id"] in actual[10]["selected"]
    assert stats_buttons[0]["id"] in actual[11]["selected"]
    assert orders[12] == orders[14] == ["one"] and orders[13] == []
    assert report.get_table("overview").equals(before)
