"""真实 Chromium 的离线交互验收；期望值只来自公共 Python 报告 API。"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import Browser, BrowserContext, Page, Route, expect, sync_playwright

from mars.analysis import evaluate_score_policy, get_score_cell
from mars.reporting import load_report

_SCOPE = ("target", "group", "period")
_STATUS = {
    "empty": "空格",
    "unobserved": "未观测",
    "not_requested": "未请求目标",
    "invalid_denominator": "无有效分母",
    "low_sample": "低样本",
    "valid": "有效",
}


def _rows(report: Any, name: str, **query: Any) -> list[dict[str, Any]]:
    """保留公共查询的过滤口径，不读原始宽表。"""
    return report.get_table(name, **query).to_dicts()


def _check_number(text: str, expected: Any, scale: float = 1, digits: int = 2) -> None:
    """按照页面显示精度核对值，同时要求不可用值保留缺失。"""
    if expected is None or not isinstance(expected, (int, float)) or not math.isfinite(expected):
        assert "—" in text, (text, expected)
        return
    match = re.search(r"[+−-]?[\d,]+(?:\.\d+)?", text)
    assert match, (text, expected)
    actual = float(match.group().replace(",", "").replace("−", "-"))
    if scale == 1 and digits == 0:
        assert actual.is_integer() and actual == expected, (text, expected)
    else:
        assert abs(actual - expected * scale) <= 0.51 * 10 ** (-digits), (text, expected)


def _scope(page: Page) -> dict[str, Any]:
    """只读浏览器当前可重放证据；不读取或修改前端 state。"""
    reference: dict[str, Any] = json.loads(page.locator("#evidence-id").inner_text())
    return {key: reference["query"]["filters"][key] for key in _SCOPE}


def _select_scope(page: Page, scope: dict[str, Any]) -> None:
    """通过真实 select 和样本集按钮切换到保存的范围。"""
    target = "无标签 · 分布分析" if scope["target"] is None else str(scope["target"])
    page.get_by_label("目标", exact=True).select_option(label=target)
    group = "全部样本" if scope["group"] is None else str(scope["group"])
    page.get_by_role("group", name="样本集", exact=True).get_by_role(
        "button", name=group, exact=True
    ).click()
    assert page.evaluate("document.activeElement.matches('#dataset-seg button.active')")
    after_group = _scope(page)
    assert after_group["target"] == scope["target"] and after_group["group"] == scope["group"]
    if scope["period"] != after_group["period"]:
        page.locator("#period").select_option(
            label="未解析时间" if scope["period"] is None else str(scope["period"])
        )
    assert _scope(page) == scope


def _observe(context: BrowserContext) -> dict[str, Any]:
    """记录真实浏览器错误和资源请求，阻断任何非本地网络依赖。"""
    events: dict[str, Any] = {
        "pageerror": [], "console_error": [], "csp": [], "requests": [],
        "requestfailed": [], "external_requests": [],
    }
    context.add_init_script("""window.__acceptanceCsp=[];
      document.addEventListener('securitypolicyviolation', event =>
        window.__acceptanceCsp.push({directive:event.effectiveDirective,
          blockedURI:event.blockedURI,disposition:event.disposition}));""")

    def route_request(route: Route) -> None:
        parsed = urlparse(route.request.url)
        if parsed.scheme in {"file", "data", "about"} or parsed.hostname in {"127.0.0.1", "localhost"}:
            route.continue_()
        else:
            events["external_requests"].append(route.request.url)
            route.abort()

    def observe_page(page: Page) -> None:
        page.on("pageerror", lambda error: events["pageerror"].append(str(error)))
        page.on("console", lambda message: events["console_error"].append(message.text)
                if message.type == "error" else None)
        page.on("request", lambda request: events["requests"].append(request.url))
        page.on("requestfailed", lambda request: events["requestfailed"].append(
            {"url": request.url, "failure": request.failure}
        ))

    context.route("**/*", route_request)
    context.on("page", observe_page)
    return events


def _overflow(page: Page) -> dict[str, Any]:
    """断言主体适配视口，记录专用表格、矩阵和图表的内部滚动。"""
    sizes: dict[str, Any] = page.evaluate("""() => ({
      viewport:innerWidth,body:document.body.scrollWidth,
      document:document.documentElement.scrollWidth,
      containers:[...document.querySelectorAll('.matrix-scroll,.chart-scroll,.history-tables,.special-body')]
        .map(e=>({class:e.className,width:e.clientWidth,scroll:e.scrollWidth}))})""")
    assert sizes["body"] <= sizes["viewport"] + 1, sizes
    assert sizes["document"] <= sizes["viewport"] + 1, sizes
    return sizes


def _table_text(value: Any) -> str:
    """匹配通用表格的文本编码，布尔值遵循浏览器字符串形式。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _assert_policies(page: Page, payload: dict[str, Any], scope: dict[str, Any]) -> int:
    """真实选择已嵌入 policy 后逐表逐行核对保存证据。"""
    checked = 0
    page.locator(".history > summary").click()
    for index, policy in enumerate(payload["policies"]):
        page.locator("#policy").select_option(str(index))
        actual: dict[str, list[list[str]]] = page.locator("#policyTables").evaluate("""e =>
          Object.fromEntries([...e.querySelectorAll('h3')].map(h=>[h.textContent,
            [...h.nextElementSibling.querySelectorAll('tr')].map(row=>
              [...row.cells].map(cell=>cell.textContent))]))""")
        for name in ("changes", "summary", "regions", "axis_regions", "cell_decisions"):
            if name not in policy["tables"]:
                continue
            records = [row for row in policy["tables"][name]
                       if all(row[key] == scope[key] for key in _SCOPE)]
            expected = [] if not records else [list(records[0])] + [
                [_table_text(value) for value in row.values()] for row in records
            ]
            assert actual[name] == expected, (name, scope, actual[name], expected)
            checked += len(records)
    page.locator("#policy").select_option("-1")
    assert page.locator("#policyTables table").count() == 0
    page.locator(".history > summary").click()
    return checked


def _assert_cell(
    page: Page, report: Any, cell: dict[str, Any], scope: dict[str, Any],
    x_index: int, y_index: int, rows: list[dict[str, Any]], columns: list[dict[str, Any]],
    normal: list[dict[str, Any]],
) -> None:
    """点击真实单元格并核对详情、双向图、状态和公共证据重放。"""
    button = page.locator(f"#matrix button[data-r='{x_index}'][data-c='{y_index}']")
    button.click()
    expect(button).to_have_attribute("aria-pressed", "true")
    assert page.evaluate("document.activeElement.classList.contains('selected')")
    mode = page.locator("#metric-seg button.active").get_attribute("data-mode")
    _check_number(button.locator("strong").inner_text(),
                  cell["delta_vs_row"] if mode == "delta" else cell["bad_rate"], 100)
    assert (" pp" if mode == "delta" else "%") in button.locator("strong").inner_text() or cell["bad_rate"] is None
    volume, share = button.locator(".cell-volume").inner_text().split(" · ")
    _check_number(volume, cell["sample_count"], digits=0)
    _check_number(share, cell["sample_share"], 100, 1)
    for identifier, field, scale, digits in (
        ("detail-rate", "bad_rate", 100, 2),
        ("detail-delta", "delta_vs_row", 100, 2),
        ("detail-n", "sample_count", 1, 0),
        ("detail-obs", "observed_sample_count", 1, 0),
        ("detail-bad", "bad_sample_count", 1, 0),
        ("detail-coverage", "observed_coverage", 100, 1),
        ("detail-lift", "lift_vs_overall", 1, 2),
    ):
        _check_number(page.locator("#" + identifier).inner_text(), cell[field], scale, digits)
    expect(page.locator("#detail-state")).to_contain_text(_STATUS[cell["status"]])
    expect(page.locator("#live-region")).to_contain_text(f"选中 X{x_index + 1} Y{y_index + 1}")
    assert cell["status"] in button.get_attribute("class").split()
    if cell["status"] == "low_sample":
        expect(button.locator(".flag")).to_have_text("低 n")
    if cell["bad_rate"] == 0:
        expect(page.locator("#detail-state")).to_contain_text("有效零值")
        expect(page.locator("#detail-note")).to_contain_text("有效零值")
    row = next(row for row in rows if row["x_bin"] == cell["x_bin"])
    column = next(row for row in columns if row["y_bin"] == cell["y_bin"])
    _check_number(page.locator("#detail-base").inner_text(), row["bad_rate"], 100)
    assert "unweighted_ci_status=" + cell["unweighted_ci_status"] in page.locator("#detail-ci").get_attribute("title")
    assert "weighted_ci_status=" + cell["weighted_ci_status"] in page.locator("#detail-ci").get_attribute("title")
    ci = page.locator("#detail-ci").inner_text()
    if cell["unweighted_ci_lower"] is None:
        assert ci == "—"
    else:
        lower, upper = ci.split(" – ")
        _check_number(lower, cell["unweighted_ci_lower"], 100)
        _check_number(upper, cell["unweighted_ci_upper"], 100)
    assert "lift_status=" + cell["lift_status"] in page.locator("#detail-lift").get_attribute("title")

    # 图表有效点和断线按保存的风险率逐一检查；参考线来自完整边际。
    for axis, chart, labels, baseline in (
        ("y", "row-chart", "chart-labels", row["bad_rate"]),
        ("x", "column-chart", "column-labels", column["bad_rate"]),
    ):
        fixed = "x_bin" if axis == "y" else "y_bin"
        varying = [item for item in normal if item[fixed] == cell[fixed]]
        varying.sort(key=lambda item: item[axis + "_risk_rank"])
        titles = page.locator("#" + chart + " g title").all_text_contents()
        valid = [item for item in varying if item["bad_rate"] is not None]
        assert len(titles) == len(valid)
        for title, expected in zip(titles, valid):
            rate = title.split(" · ")[1]
            _check_number(rate, expected["bad_rate"], 100)
            assert _STATUS[expected["status"]] in title
        segments = sum(left["bad_rate"] is not None and right["bad_rate"] is not None
                       for left, right in zip(varying, varying[1:]))
        assert page.locator("#" + chart + " path[stroke-width='2']").count() == segments
        assert page.locator("#" + labels + " .chart-tick").count() == len(varying)
        baseline_text = page.locator("#" + chart + " text").all_text_contents()
        baseline_labels = [value for value in baseline_text if value.startswith("边际基线 ")]
        if baseline is None:
            assert not baseline_labels
        else:
            assert len(baseline_labels) == 1
            _check_number(baseline_labels[0], baseline, 100, 1)
    reference = json.loads(page.locator("#evidence-id").inner_text())
    assert reference["report_id"] == report.report_id and reference["table"] == "cells"
    assert reference["query"]["filters"] == {
        **scope, "x_bin": cell["x_bin"], "y_bin": cell["y_bin"]
    }
    replay = report.query_page(reference["table"], **reference["query"])
    assert replay["data"].to_dicts() == [cell]
    public = get_score_cell(report, cell["x_bin"], cell["y_bin"], filters=scope)
    assert public["page"]["data"].to_dicts() == [cell]


def _assert_rule(page: Page, report: Any, expression: str, scope: dict[str, Any]) -> None:
    """仅用 Python evaluate_score_policy 的 summary 和 decisions 核对规则。"""
    policy = evaluate_score_policy(report, {"type": "expression", "expression": expression})
    summary = _rows(policy, "summary", filters={**scope, "rule": "candidate", "retained": True})[0]
    result = page.locator("#custom-rule-result")
    expect(result).to_be_visible()
    assert result.get_attribute("data-rule") == expression.strip()
    assert result.get_attribute("data-status") == summary["status"]
    assert result.get_attribute("data-lift-status") == summary["lift_status"]
    values = result.locator(".rule-result-stats b").all_text_contents()
    for text, field, scale, digits in zip(values, (
        "sample_count", "sample_share", "observed_sample_count", "bad_sample_count",
        "bad_rate", "lift_vs_overall",
    ), (1, 100, 1, 1, 100, 1), (0, 1, 0, 0, 2, 2)):
        _check_number(text, summary[field], scale, digits)
    decisions = _rows(policy, "cell_decisions", filters=scope)
    bins = _rows(report, "bins")
    xs = {row["bin_id"]: row["risk_rank"] for row in bins if row["axis"] == "x" and row["kind"] == "normal"}
    ys = {row["bin_id"]: row["risk_rank"] for row in bins if row["axis"] == "y" and row["kind"] == "normal"}
    expected = {(xs[row["x_bin"]] - 1, ys[row["y_bin"]] - 1)
                for row in decisions if row["candidate_pass"] and row["x_bin"] in xs and row["y_bin"] in ys}
    actual = {(int(button.get_attribute("data-r")), int(button.get_attribute("data-c")))
              for button in page.locator("#matrix .rule-hit").all()}
    assert actual == expected, (expression, scope, actual, expected)


def _apply(page: Page, expression: str, *, enter: bool = False) -> None:
    """用输入、按钮或 Enter 触发表单，不调用任何前端函数。"""
    field = page.get_by_label("分箱规则", exact=True)
    field.fill(expression)
    if enter:
        field.press("Enter")
    else:
        page.get_by_role("button", name="应用规则", exact=True).click()


def _keyboard(page: Page, x_count: int, y_count: int) -> None:
    """通过 Tab 真正进入矩阵，确认方向键边界、焦点和 live region。"""
    page.get_by_role("button", name="Bad Rate", exact=True).focus()
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.matches('#matrix .cell.selected')")
    for key in ("ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown", "ArrowDown", "ArrowRight"):
        before = json.loads(page.locator("#evidence-id").inner_text())
        before_r = int(page.locator("#matrix .selected.cell").get_attribute("data-r"))
        before_c = int(page.locator("#matrix .selected.cell").get_attribute("data-c"))
        scroll = page.evaluate("({x:scrollX,y:scrollY})")
        page.keyboard.press(key)
        selected = page.locator("#matrix .cell.selected")
        r = max(0, min(x_count - 1, before_r + (1 if key == "ArrowDown" else -1 if key == "ArrowUp" else 0)))
        c = max(0, min(y_count - 1, before_c + (1 if key == "ArrowRight" else -1 if key == "ArrowLeft" else 0)))
        assert (int(selected.get_attribute("data-r")), int(selected.get_attribute("data-c"))) == (r, c)
        assert page.evaluate("document.activeElement.matches('#matrix .cell.selected')")
        expect(selected).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#live-region")).to_contain_text(f"选中 X{r + 1} Y{c + 1}")
        assert page.evaluate("({x:scrollX,y:scrollY})") == scroll
        assert _scope(page) == {key: before["query"]["filters"][key] for key in _SCOPE}
    for key, count in (("ArrowUp", x_count + 1), ("ArrowLeft", y_count + 1),
                       ("ArrowDown", x_count + 1), ("ArrowRight", y_count + 1)):
        for _ in range(count):
            page.keyboard.press(key)
        assert page.evaluate("document.activeElement.matches('#matrix .cell.selected')")
    selected = page.locator("#matrix .cell.selected")
    assert (int(selected.get_attribute("data-r")), int(selected.get_attribute("data-c"))) == (x_count - 1, y_count - 1)


def _empty(page: Page) -> None:
    """零 scope 页面必须准确空态，并禁用所有依赖真实格子的动作。"""
    expect(page.locator("#scope-caption")).to_have_text("空输入；没有真实 scope")
    expect(page.locator("#detail-title")).to_have_text("没有可选单元格")
    expect(page.locator("#matrix")).to_contain_text("没有样本范围")
    assert page.locator("#matrix button").count() == 0
    for identifier in ("target", "period", "rule-expression", "rule-clear", "copy-btn", "policy"):
        expect(page.locator("#" + identifier)).to_be_disabled()
    for control in page.locator("[data-mode],.rule-apply,.copy-example").all():
        expect(control).to_be_disabled()
    expect(page.locator("#period-filter")).to_be_hidden()
    assert page.locator("#rule-error").inner_text() == ""
    expect(page.locator("#custom-rule-result")).to_be_hidden()
    page.keyboard.press("Tab")
    page.keyboard.press("Enter")
    page.keyboard.press("ArrowRight")
    assert page.locator("#rule-error").inner_text() == ""
    assert page.locator("#evidence-id").inner_text() == ""


def _run_fixture(
    browser: Browser, fixture: dict[str, Any], output: Path, log: dict[str, Any],
) -> None:
    """完成一个已保存报告的范围、矩阵、规则、布局和状态验收。"""
    report = load_report(fixture["snapshot"])
    html = Path(fixture["html"])
    name = fixture["name"]
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN")
    events = _observe(context)
    page = context.new_page()
    page.set_default_timeout(10000)
    record: dict[str, Any] = {"name": name, "opening": html.as_uri(), "events": events,
                              "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
                              "cells": 0, "scopes": 0, "rules": 0, "policy_rows": 0,
                              "layouts": [], "screenshots": []}
    log["fixtures"].append(record)
    try:
        page.goto(html.as_uri(), wait_until="load")
        payload = json.loads(page.locator("#data").text_content())
        assert payload["description"] == report.describe()
        for policy in payload["policies"]:
            parameters = policy["description"]["parameters"]
            expected_policy = evaluate_score_policy(
                report, parameters["candidate"], baseline=parameters.get("baseline")
            )
            for table_name, expected_records in policy["tables"].items():
                # 固定 bins 可含非有限缺失定义；这里只对已保存回放的业务表。
                if table_name != "bins":
                    assert _rows(expected_policy, table_name) == expected_records
        assert page.locator("#report-id").inner_text() == report.report_id
        assert page.locator("h1").inner_text() == page.title()
        assert page.locator("script").count() == 2
        assert page.locator("script[src],iframe,object,embed").count() == 0
        scopes = [{key: row[key] for key in _SCOPE} for row in _rows(report, "overall")]
        bins = _rows(report, "bins")
        xs = sorted((row for row in bins if row["axis"] == "x" and row["kind"] == "normal"), key=lambda row: row["risk_rank"])
        ys = sorted((row for row in bins if row["axis"] == "y" and row["kind"] == "normal"), key=lambda row: row["risk_rank"])
        if not scopes:
            _empty(page)
        else:
            hint = page.locator("#rule-hint").inner_text()
            assert f"X <= X{min(2, len(xs))}" in hint
            assert f"Y >= Y{min(4, len(ys))}" in hint
            default_expression = page.get_by_label("分箱规则", exact=True).input_value()
            evaluate_score_policy(report, {"type": "expression", "expression": default_expression})
            for example in page.locator(".example-expression").all_text_contents():
                evaluate_score_policy(report, {"type": "expression", "expression": example})
            _check_number(page.locator("#ci-label").inner_text(),
                          report.describe()["parameters"]["confidence_level"], 100, 8)
            if all(scope["period"] is None for scope in scopes):
                expect(page.locator("#period-filter")).to_be_hidden()
            # 每个 scope 的当前格和完整边际都与报告公开表比较。
            for scope in scopes:
                _select_scope(page, scope)
                current = _rows(report, "overall", filters=scope)[0]
                cells = _rows(report, "cells", filters=scope)
                rows = _rows(report, "row_summary", filters=scope)
                columns = _rows(report, "column_summary", filters=scope)
                normal = [row for row in cells if row["x_risk_rank"] is not None and row["y_risk_rank"] is not None]
                for identifier, field, scale in (("total-n", "sample_count", 1),
                                                  ("total-obs", "observed_sample_count", 1),
                                                  ("total-rate", "bad_rate", 100)):
                    _check_number(page.locator("#" + identifier).inner_text(), current[field],
                                  scale, 0 if scale == 1 else 2)
                assert page.locator("#matrix .cell").count() == len(xs) * len(ys)
                labels = page.locator("#matrix .row-head b").all_text_contents()
                assert labels == [row["display_label"] for row in xs]
                assert page.locator("#matrix .col-head b").all_text_contents() == [row["display_label"] for row in ys]
                for index, x in enumerate(xs):
                    expected = next(row for row in rows if row["x_bin"] == x["bin_id"])
                    margin = page.locator("#matrix .row-margin").nth(index)
                    _check_number(margin.locator("b").inner_text(), expected["sample_count"], digits=0)
                    _check_number(margin.locator("span").inner_text(), expected["sample_share"], 100, 1)
                    _check_number(page.locator("#matrix .row-rate").nth(index).inner_text(), expected["bad_rate"], 100)
                for index, y in enumerate(ys):
                    expected = next(row for row in columns if row["y_bin"] == y["bin_id"])
                    margin = page.locator("#matrix .total-cell").nth(index + 1)
                    _check_number(margin.locator("b").inner_text(), expected["sample_count"], digits=0)
                    _check_number(margin.locator("span").inner_text(), expected["sample_share"], 100, 1)
                special = [row for row in cells if row["x_risk_rank"] is None or row["y_risk_rank"] is None]
                special_n = sum(row["sample_count"] for row in special)
                expect(page.locator("#special-summary")).to_contain_text(f"隐藏 {special_n:,} 人")
                expected_special = [row for row in special if row["sample_count"] > 0]
                page.locator(".specials > summary").click()
                special_text = page.locator("#special-body").evaluate("""e =>
                  [...e.querySelectorAll('tr')].slice(1).map(row=>[...row.cells].map(c=>c.textContent))""")
                assert len(special_text) == len(expected_special)
                for actual, expected_special_row in zip(special_text, expected_special):
                    classifications = []
                    for axis in ("x", "y"):
                        definition = next(row for row in bins if row["axis"] == axis and row["bin_id"] == expected_special_row[axis + "_bin"])
                        label = axis.upper() + str(definition["risk_rank"]) if definition["kind"] == "normal" else definition["kind"]
                        if definition["kind"] == "special":
                            label += " " + _table_text(definition["special_value"])
                        classifications.append(label)
                    assert actual[0] == " × ".join(classifications)
                    for text, field, scale, digits in zip(actual[1:6],
                        ("sample_count", "observed_sample_count", "bad_sample_count", "sample_share", "bad_rate"),
                        (1, 1, 1, 100, 100), (0, 0, 0, 1, 2)):
                        _check_number(text, expected_special_row[field], scale, digits)
                    assert actual[6] == expected_special_row["status"]
                page.locator(".specials > summary").click()
                for r, x in enumerate(xs):
                    for c, y in enumerate(ys):
                        if name == "medium" and (r, c) not in {
                            (0, 0), (len(xs) - 1, len(ys) - 1), (len(xs) // 2, len(ys) // 2)
                        }:
                            continue
                        cell = next(row for row in normal if row["x_bin"] == x["bin_id"] and row["y_bin"] == y["bin_id"])
                        _assert_cell(page, report, cell, scope, r, c, rows, columns, normal)
                        record["cells"] += 1
                # 默认规则和每个状态的格子都通过真实表单核对 Python policy。
                expressions = ["X >= 1 OR Y >= Y1", "X < 1", "x = x1 OR X = X1 AND y >= 1",
                               "(X = X1 OR X = X1) AND Y >= Y1",
                               "X != X1 AND Y > 1", "x == 1 OR y <= Y1"]
                if len(xs) >= 2 and len(ys) >= 2:
                    expressions.extend(["X = X1 OR X = X2 AND Y = Y2",
                                        "(X = X1 OR X = X2) AND Y = Y2"])
                status_cells: dict[str, dict[str, Any]] = {}
                for cell in normal:
                    status_cells.setdefault(cell["status"], cell)
                expressions.extend(f"X = X{cell['x_risk_rank']} AND Y = Y{cell['y_risk_rank']}" for cell in status_cells.values())
                for index, expression in enumerate(expressions):
                    _apply(page, expression, enter=index % 2 == 1)
                    _assert_rule(page, report, expression, scope)
                    record["rules"] += 1
                record["policy_rows"] += _assert_policies(page, payload, scope)
                record["scopes"] += 1

            _select_scope(page, scopes[0])
            expression = "X >= 1 OR Y >= Y1"
            _apply(page, expression)
            screenshot = output / (name + "-applied.png")
            page.locator("#custom-rule-result").scroll_into_view_if_needed()
            page.screenshot(path=str(screenshot), full_page=True)
            record["screenshots"].append(str(screenshot))
            # 保存各关键状态的真实选格视图，避免只从一张默认截图推断边界。
            if name in {"standard", "weighted"}:
                seen: set[str] = set()
                for state_scope in scopes:
                    _select_scope(page, state_scope)
                    for cell in _rows(report, "cells", filters=state_scope):
                        if cell["x_risk_rank"] is None or cell["y_risk_rank"] is None:
                            continue
                        kind = "valid_zero" if cell["bad_rate"] == 0 else cell["status"]
                        if kind in seen:
                            continue
                        seen.add(kind)
                        page.locator(f"#matrix button[data-r='{cell['x_risk_rank'] - 1}'][data-c='{cell['y_risk_rank'] - 1}']").click()
                        page.locator(".detail").scroll_into_view_if_needed()
                        screenshot = output / f"{name}-state-{kind}.png"
                        page.screenshot(path=str(screenshot), full_page=True)
                        record["screenshots"].append(str(screenshot))
                _select_scope(page, scopes[0])
            # 未提交输入、选格和示例复制不能偷偷改已应用规则。
            page.get_by_label("分箱规则", exact=True).fill("X < 1")
            page.locator("#matrix .cell").first.click()
            _assert_rule(page, report, expression, scopes[0])
            for scope in list(reversed(scopes)) + scopes:
                _select_scope(page, scope)
                _assert_rule(page, report, expression, scope)
                expect(page.get_by_label("分箱规则", exact=True)).to_have_value("X < 1")
            _select_scope(page, scopes[0])
            for boundary in (" " * 237 + "X=1", "(" * 12 + "X=1" + ")" * 12,
                             " OR ".join(["X=1"] * 24)):
                _apply(page, boundary, enter=True)
                expect(page.get_by_label("分箱规则", exact=True)).to_have_attribute("aria-invalid", "false")
                _assert_rule(page, report, boundary, scopes[0])
            _apply(page, expression)
            # 错误输入严格保留上一条有效规则，覆盖长度、token 和深度边界。
            invalid = [f"X = X{len(xs) + 1}", "X <= Y1", "X <= 0.5", "X <=",
                       "alert(1)", "</script><script>window.bad=true</script>",
                       "X=1 " * 61, "(" * 48 + "X=1" + ")" * 48,
                       "(" * 13 + "X=1" + ")" * 13]
            for value in invalid:
                _apply(page, value, enter=True)
                expect(page.get_by_label("分箱规则", exact=True)).to_have_attribute("aria-invalid", "true")
                expect(page.locator("#rule-error")).to_contain_text("仍显示上一条规则")
                _assert_rule(page, report, expression, scopes[0])
            assert not page.evaluate("Boolean(window.bad)")
            screenshot = output / (name + "-error.png")
            page.locator("#rule-form").scroll_into_view_if_needed()
            page.screenshot(path=str(screenshot), full_page=True)
            record["screenshots"].append(str(screenshot))
            before = json.loads(page.locator("#evidence-id").inner_text())
            page.get_by_role("button", name="恢复正常视图", exact=True).click()
            expect(page.locator("#custom-rule-result")).to_be_hidden()
            expect(page.locator("#rule-clear")).to_be_disabled()
            assert page.locator("#matrix .rule-hit").count() == 0
            assert json.loads(page.locator("#evidence-id").inner_text()) == before
            expect(page.get_by_label("分箱规则", exact=True)).to_have_attribute("aria-invalid", "false")
            _keyboard(page, len(xs), len(ys))

            # 指标切换保留选择；每个模式全报告色阶不因 scope 变化。
            for mode, label in (("rate", "Bad Rate"), ("delta", "相对 X 行基线 Δ")):
                selected = json.loads(page.locator("#evidence-id").inner_text())
                page.get_by_role("button", name=label, exact=True).click()
                expect(page.locator(f"[data-mode='{mode}']")).to_have_attribute("aria-pressed", "true")
                assert json.loads(page.locator("#evidence-id").inner_text()) == selected
                caption = page.locator("#scale-caption").inner_text()
                colors: dict[Any, str] = {}
                for scope in scopes:
                    _select_scope(page, scope)
                    assert page.locator("#scale-caption").inner_text() == caption
                    scope_cells = [cell for cell in _rows(report, "cells", filters=scope)
                                   if cell["x_risk_rank"] is not None and cell["y_risk_rank"] is not None]
                    scope_cells.sort(key=lambda cell: (cell["x_risk_rank"], cell["y_risk_rank"]))
                    actual_colors = page.locator("#matrix .cell").evaluate_all(
                        "elements=>elements.map(e=>getComputedStyle(e).backgroundColor)"
                    )
                    for cell, color in zip(scope_cells, actual_colors):
                        key = cell["delta_vs_row"] if mode == "delta" else cell["bad_rate"]
                        if key in colors:
                            assert colors[key] == color, (mode, key, colors[key], color)
                        colors[key] = color
                    # 应用规则和改选格不能重新标定热力的底色。
                    _apply(page, "X >= 1")
                    page.locator("#matrix .cell").first.click()
                    assert page.locator("#matrix .cell").evaluate_all(
                        "elements=>elements.map(e=>getComputedStyle(e).backgroundColor)"
                    ) == actual_colors
                    page.get_by_role("button", name="恢复正常视图", exact=True).click()
            _select_scope(page, scopes[0])
            for summary in ("特殊分箱", "三项阅读检查", "语法与范围", "已保存 policy 回放比较", "报告参数与证据目录"):
                locator = page.locator("summary").filter(has_text=summary)
                if locator.count():
                    locator.first.click()
                    assert locator.first.evaluate("e=>e.parentElement.open")
                    locator.first.click()
                    assert not locator.first.evaluate("e=>e.parentElement.open")
            page.get_by_role("button", name="指标口径", exact=True).click()
            expect(page.locator("#help")).to_be_visible()
            expect(page.locator("#help-btn")).to_have_attribute("aria-expanded", "true")
            page.get_by_role("button", name="指标口径", exact=True).click()
            expect(page.locator("#help")).to_be_hidden()

        # 相同固定主题在系统偏好变化时可用；不制造主题切换功能。
        for width in (1440, 1024, 768, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            for scheme in ("light", "dark"):
                page.emulate_media(color_scheme=scheme)
                record["layouts"].append({"width": width, "system_color_scheme": scheme,
                                          **_overflow(page)})
            if scopes:
                _select_scope(page, scopes[-1])
                page.locator("#matrix .cell").first.click()
                _apply(page, "X >= 1", enter=True)
                _assert_rule(page, report, "X >= 1", scopes[-1])
                expect(page.locator("#detail-title")).to_be_visible()
                if width == 390:
                    screenshot = output / (name + "-390-applied.png")
                    page.screenshot(path=str(screenshot), full_page=True)
                    record["screenshots"].append(str(screenshot))
                page.get_by_role("button", name="恢复正常视图", exact=True).click()
            else:
                _empty(page)
            screenshot = output / f"{name}-{width}.png"
            page.screenshot(path=str(screenshot), full_page=True)
            record["screenshots"].append(str(screenshot))
        events["csp"].extend(page.evaluate("window.__acceptanceCsp"))
        assert not events["pageerror"], events
        assert not events["console_error"], events
        assert not events["csp"], events
        assert not events["requestfailed"], events
        assert not events["external_requests"], events
        assert events["requests"] == [html.as_uri()], events
        record["status"] = "passed"
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        page.screenshot(path=str(output / (name + "-failed.png")), full_page=True)
        raise
    finally:
        context.close()


def _clipboard(browser: Browser, fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    """真实点击验证原生 clipboard、拒绝和 legacy fallback 的反馈与证据。"""
    record: dict[str, Any] = {}
    html = Path(fixture["html"])
    report = load_report(fixture["snapshot"])
    context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], locale="zh-CN")
    events = _observe(context)
    record["events"] = events
    page = context.new_page()
    try:
        page.goto(html.as_uri())
        reference = json.loads(page.locator("#evidence-id").inner_text())
        page.locator("#copy-btn").click()
        expect(page.locator("#toast")).to_have_text("证据已复制")
        copied = page.evaluate("navigator.clipboard.readText()")
        assert json.loads(copied) == reference
        replay = report.query_page(reference["table"], **reference["query"])
        assert replay["data"].height == 1
        record["file_native"] = {"status": "passed", "secure": page.evaluate("isSecureContext"),
                                  "clipboard_content_confirmed": True, "reference": reference}
        applied = page.locator("#custom-rule-result").get_attribute("data-rule")
        field = page.get_by_label("分箱规则", exact=True).input_value()
        page.locator(".copy-example").first.click()
        expect(page.locator("#toast")).to_have_text("已复制示例；自行粘贴后应用")
        example = page.locator(".example-expression").first.inner_text()
        assert page.evaluate("navigator.clipboard.readText()") == example
        assert page.get_by_label("分箱规则", exact=True).input_value() == field
        assert page.locator("#custom-rule-result").get_attribute("data-rule") == applied
        record["copy_example"] = "passed: copied only, input and applied result unchanged"
    finally:
        context.close()

    # 只注入浏览器环境故障，不调用或替换页面的复制、render、state 函数。
    for name, initialization, expected in (
        ("rejected", "Object.defineProperty(navigator,'clipboard',{value:{writeText:()=>Promise.reject(new DOMException('denied','NotAllowedError'))}})", "复制不可用，请手动选择并复制文本"),
        ("fallback", "Object.defineProperty(navigator,'clipboard',{value:undefined})", "证据已复制"),
        ("fallback_rejected", "Object.defineProperty(navigator,'clipboard',{value:undefined});document.execCommand=()=>false", "复制不可用，请手动选择并复制文本"),
    ):
        context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], locale="zh-CN")
        _observe(context)
        context.add_init_script(initialization)
        page = context.new_page()
        try:
            page.goto(html.as_uri())
            before = page.locator("#evidence-id").inner_text()
            page.locator("#copy-btn").click()
            expect(page.locator("#toast")).to_have_text(expected)
            assert page.locator("#evidence-id").inner_text() == before
            if name == "fallback":
                # 第二个未注入故障的安全上下文确认 execCommand 的实际系统内容。
                reader_context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
                reader = reader_context.new_page()
                reader.goto(html.as_uri())
                assert json.loads(reader.evaluate("navigator.clipboard.readText()")) == json.loads(before)
                record[name] = {"status": "passed", "content_confirmed": True,
                                "button_feedback": expected,
                                "method": "execCommand copy; independent native context read"}
                reader_context.close()
            else:
                record[name] = {"status": "passed", "button_feedback": expected}
            page.screenshot(path=str(output / ("clipboard-" + name + ".png")), full_page=True)
        finally:
            context.close()

    # 本地 HTTP 是额外的安全上下文复制检查，单独记录，不冒充 file 验收。
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(html.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], locale="zh-CN")
    _observe(context)
    page = context.new_page()
    try:
        page.goto(f"http://127.0.0.1:{server.server_port}/{html.name}")
        page.locator("#copy-btn").click()
        expect(page.locator("#toast")).to_have_text("证据已复制")
        copied = page.evaluate("navigator.clipboard.readText()")
        reference = json.loads(copied)
        assert reference == json.loads(page.locator("#evidence-id").inner_text())
        assert report.query_page(reference["table"], **reference["query"])["data"].height == 1
        record["localhost_native"] = {"status": "passed", "secure": page.evaluate("isSecureContext"),
                                       "clipboard_content_confirmed": True}
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join()
    return record


def _zoom(playwright: Any, fixture: dict[str, Any], output: Path, channel: str, headful: bool) -> dict[str, Any]:
    """使用 Chrome 设置的实际 200% 缩放，不以 CSS 或 pinch zoom 代替。"""
    profile = tempfile.mkdtemp(prefix="mars-browser-zoom-")
    context = playwright.chromium.launch_persistent_context(
        profile, channel=channel, headless=not headful, no_viewport=True,
        args=["--window-size=1440,1000"], locale="zh-CN",
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.pages[0]
    try:
        page.goto("chrome://settings/appearance")
        page.locator("#zoomLevel").select_option(label="200%")
        expect(page.locator("#zoomLevel")).to_have_value("2")
        page.goto(Path(fixture["html"]).as_uri())
        report = load_report(fixture["snapshot"])
        scopes = [{key: row[key] for key in _SCOPE} for row in _rows(report, "overall")]
        _select_scope(page, scopes[-1])
        page.locator("#matrix .cell").first.click()
        _apply(page, "X >= 1", enter=True)
        _assert_rule(page, report, "X >= 1", scopes[-1])
        expect(page.locator("#detail-title")).to_be_visible()
        page.locator("#copy-btn").click()
        expect(page.locator("#toast")).to_have_text("证据已复制")
        assert json.loads(page.evaluate("navigator.clipboard.readText()")) == json.loads(page.locator("#evidence-id").inner_text())
        sizes = page.evaluate("({outer:outerWidth,inner:innerWidth,dpr:devicePixelRatio,visual:visualViewport.scale})")
        assert sizes["dpr"] == 2 and sizes["visual"] == 1, sizes
        layout = _overflow(page)
        screenshot = output / "score-cross-zoom200.png"
        # 原生 200% 下以 CDP 抓完整物理视口，避免 Playwright 的 CSS clip 截半。
        clip = page.evaluate("({x:0,y:0,width:innerWidth*devicePixelRatio,height:innerHeight*devicePixelRatio,scale:1})")
        capture = context.new_cdp_session(page).send("Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": True, "fromSurface": True, "clip": clip,
        })
        screenshot.write_bytes(base64.b64decode(capture["data"]))
        return {"status": "passed", "mechanism": "Chrome settings page zoom 200%",
                "sizes": sizes, "layout": layout, "screenshot": str(screenshot)}
    finally:
        context.close()


def _snippet(browser: Browser, fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    """对公共 snippet 新进程导出的页面复查范围、规则和复制证据闭环。"""
    context = browser.new_context(locale="zh-CN", viewport={"width": 1440, "height": 1000},
                                  permissions=["clipboard-read", "clipboard-write"])
    events = _observe(context)
    page = context.new_page()
    checked = 0
    try:
        page.goto(Path(fixture["html"]).as_uri())
        report = load_report(fixture["snapshot"])
        scopes = [{key: row[key] for key in _SCOPE} for row in _rows(report, "overall")]
        for scope in (scopes[0], scopes[-1]):
            _select_scope(page, scope)
            page.locator("#matrix .cell").last.click()
            _apply(page, fixture["rule"], enter=True)
            _assert_rule(page, report, fixture["rule"], scope)
            page.locator("#copy-btn").click()
            expect(page.locator("#toast")).to_have_text("证据已复制")
            copied = json.loads(page.evaluate("navigator.clipboard.readText()"))
            assert copied == json.loads(page.locator("#evidence-id").inner_text())
            assert report.query_page(copied["table"], **copied["query"])["data"].height == 1
            checked += 1
        screenshot = output / "snippet-reexport.png"
        page.screenshot(path=str(screenshot), full_page=True)
        events["csp"].extend(page.evaluate("window.__acceptanceCsp"))
        assert not any(events[key] for key in ("pageerror", "console_error", "csp", "external_requests", "requestfailed"))
        return {"status": "passed", "scopes": checked, "expression": fixture["rule"],
                "opening": Path(fixture["html"]).as_uri(), "events": events,
                "screenshot": str(screenshot), "clipboard_content_confirmed": True}
    finally:
        context.close()


def _record_source(log: dict[str, Any]) -> None:
    """记录运行起点与最终生产提交，并以归一化模板字节核实源码对应关系。"""
    production = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = "src/mars/analysis/_score_cross_html.py"
    working = (Path(__file__).resolve().parents[2] / source).read_text(encoding="utf-8")
    committed = subprocess.check_output(["git", "show", f"{production}:{source}"]).decode("utf-8")
    working_hash = hashlib.sha256(working.replace("\r\n", "\n").encode()).hexdigest()
    committed_hash = hashlib.sha256(committed.replace("\r\n", "\n").encode()).hexdigest()
    log["initial_git_commit"] = log.pop("git_commit")
    log["production_commit"] = production
    log["production_source_verification"] = {
        "path": source, "normalization": "UTF-8 text, CRLF normalized to LF",
        "working_sha256": working_hash, "git_blob_sha256": committed_hash,
        "identical": working_hash == committed_hash,
        "unchanged_during_run": log["html_template_sha256"] == hashlib.sha256(
            (Path(__file__).resolve().parents[2] / source).read_bytes()
        ).hexdigest(),
    }


def main() -> int:
    """以独立可重复命令验收夹具，输出可审查的截图和结构化日志。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fixtures = [{"name": name, **fixture} for name, fixture in manifest["fixtures"].items()]
    log: dict[str, Any] = {"python": sys.version, "os": platform.platform(),
                          "engine": "Chromium", "headless": not args.headful,
                          "language": "zh-CN", "channel": args.channel,
                          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          "git_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
                          "html_template_sha256": hashlib.sha256(
                              (Path(__file__).resolve().parents[2] / "src/mars/analysis/_score_cross_html.py").read_bytes()
                          ).hexdigest(),
                          "manifest": str(args.manifest.resolve()), "fixtures": []}
    path = args.output / "score-cross-browser.json"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=args.channel, headless=not args.headful)
            log["browser_version"] = browser.version
            try:
                for fixture in fixtures:
                    _run_fixture(browser, fixture, args.output, log)
                    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
                usable = next(fixture for fixture, result in zip(fixtures, log["fixtures"])
                              if result["scopes"] > 0)
                log["clipboard"] = _clipboard(browser, usable, args.output)
                if "snippet_reexport" in manifest:
                    log["snippet_reexport"] = _snippet(browser, manifest["snippet_reexport"], args.output)
            finally:
                browser.close()
            log["zoom200"] = _zoom(playwright, usable, args.output, args.channel, args.headful)
        log["status"] = "passed"
        return 0
    except Exception:
        log["status"] = "failed"
        log["error"] = traceback.format_exc()
        print(log["error"], file=sys.stderr)
        return 1
    finally:
        _record_source(log)
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
