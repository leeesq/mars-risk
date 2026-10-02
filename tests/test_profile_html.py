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

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "div" and "page" in (attributes.get("class") or "").split():
            self.page = attributes["data-page"]
        elif tag == "table":
            self.table = {"id": attributes["id"], "page": self.page, "headers": [], "rows": []}
        elif tag == "tbody":
            self.in_body = True
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if self.in_body:
                self.table["rows"].append(self.row)
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
  fire(name) { this.listeners[name]({target: this}); }
}
const tables = input.tables.map(spec => {
  const table = {id: spec.id, page: spec.page};
  const body = {rows: spec.rows.map(cells => ({
    get innerText() {
      const visible = !this.hidden && pages.find(page => page.dataset.page === table.page).active;
      return visible ? cells.join('\t') : '\n      ' + cells.join('\n      ') + '\n    ';
    },
    hidden: false, cells: cells.map(text => ({innerText: text, textContent: text}))
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
const selectors = {'.table-search': locals, '.report-table': tables,
  '.nav-button': navigation, '.page': pages, '.report-table th': tables.flatMap(table => table.headers)};
const document = {
  getElementById: id => id === 'global-search' ? global : tables.find(table => table.id === id),
  querySelectorAll: selector => { if (!(selector in selectors)) throw Error(selector); return selectors[selector]; }
};
vm.runInNewContext(input.script, {document, Map, Number, String});
const observations = input.actions.map(action => {
  if (action.kind === 'global') { global.value = action.value; global.fire('input'); }
  if (action.kind === 'local') {
    const local = locals.find(input => input.dataset.table === action.table);
    local.value = action.value; local.fire('input');
  }
  if (action.kind === 'sort') tables.find(table => table.id === action.table).headers[action.column].fire('click');
  if (action.kind === 'navigation') navigation.find(button => button.dataset.page === action.value).fire('click');
  return {
    rows: Object.fromEntries(tables.map(table => [table.id,
      table.tBodies[0].rows.filter(row => !row.hidden).map(row => row.cells.map(cell => cell.innerText))])),
    active: pages.filter(page => page.active).map(page => page.dataset.page)
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
    main = next(table for table in parser.tables if table["headers"][:2] == ["feature", "value"])
    other = next(table for table in parser.tables if table["id"] != main["id"] and table["headers"][:2] == ["feature", "value"])
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
        {"kind": "sort", "table": main["id"], "column": 1},
        {"kind": "navigation", "value": "Stats"},
        {"kind": "local", "table": other["id"], "value": "中文"},
        {"kind": "global", "value": ""},
        {"kind": "navigation", "value": "Overview"},
        {"kind": "local", "table": main["id"], "value": ""},
        {"kind": "global", "value": " income "},
        {"kind": "global", "value": "income"},
        {"kind": "local", "table": main["id"], "value": "debt"},
        {"kind": "local", "table": main["id"], "value": "_aux"},
        {"kind": "sort", "table": main["id"], "column": 1},
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
        input=json.dumps({"script": script.group(1), "tables": parser.tables, "pages": pages, "actions": actions}),
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
    values = [int(row[1]) for row in actual[-1]["rows"][main["id"]]]
    assert values == sorted(values, reverse=True)
