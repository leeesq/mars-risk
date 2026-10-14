"""以真实 Chrome 对既有 Profile/snapshot 离线页执行组合搜索与布局验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page


def _panel(page: Page, name: str) -> Any:
    """按实际公共表标题取得搜索控件所属的 panel。"""
    return page.locator("section.panel").filter(
        has=page.get_by_role("heading", name=name, exact=True, include_hidden=True)
    )


def _open_table(page: Page, name: str) -> Any:
    """通过原有导航页切换找到可交互的公共表。"""
    page.get_by_role("tab", name="Overview" if name == "overview" else "Stats", exact=True).click()
    if name != "overview":
        page.get_by_role("tab", name=name.split(".", 1)[1], exact=True).click()
    return _panel(page, name)


def _check_profile_rows(page: Page, name: str, expected: list[dict[str, Any]]) -> int:
    """以原值属性核对事实来源，展示百分比或两位小数不参与数值断言。"""
    table = _panel(page, name).locator("table")
    headers = table.locator("thead th").all_text_contents()
    rows = table.locator("tbody tr").evaluate_all(
        "rows=>rows.map(row=>Array.from(row.cells,cell=>({text:cell.textContent,type:cell.dataset.sortType,value:cell.dataset.sortValue})))"
    )
    assert len(rows) == len(expected), name
    if not expected:
        return 0
    assert headers == list(expected[0]), (name, headers)
    checked = 0
    for actual, saved in zip(rows, expected):
        for column, cell in zip(headers, actual):
            value = saved[column]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    assert cell["type"] == "number" and math.isclose(float(cell["value"]), value, rel_tol=1e-12)
                    checked += 1
                else:
                    assert cell["type"] == "missing"
            elif value is None:
                if column == "display_name":
                    assert cell["text"] == saved["feature"]
                else:
                    assert cell["type"] == "missing"
            else:
                assert cell["text"] == str(value)
    return checked


def _metric_cases(page: Page) -> list[dict[str, Any]]:
    """实际点击每个指标，验证独立导航、ARIA、键盘与返回后的选择状态。"""
    from playwright.sync_api import expect

    results: list[dict[str, Any]] = []
    for name in ("DQ", "Stats"):
        page.get_by_role("tab", name=name, exact=True).click()
        outer = page.locator(f'.page[data-page="{name}"]')
        tabs = outer.locator(".metric-tab")
        for index in range(tabs.count()):
            tab = tabs.nth(index)
            tab.click()
            selected = tab.get_attribute("aria-controls")
            expect(tab).to_have_attribute("aria-selected", "true")
            expect(outer.locator(".metric-panel:visible")).to_have_count(1)
            expect(page.locator(f'#{selected}')).to_be_visible()
            page.get_by_role("tab", name="Overview", exact=True).click()
            page.get_by_role("tab", name=name, exact=True).click()
            expect(tab).to_have_attribute("aria-selected", "true")
            results.append({"page": name, "metric": tab.text_content(), "panel": selected})
        if tabs.count() > 1:
            tabs.nth(tabs.count() - 1).focus()
            page.keyboard.press("ArrowRight")
            expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")
            page.keyboard.press("End")
            expect(tabs.nth(tabs.count() - 1)).to_have_attribute("aria-selected", "true")
            page.keyboard.press("Home")
            expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")
    return results


def _search_cases(page: Page, output: Path) -> dict[str, Any]:
    """逐次从当前两个条件核对原表行，并检查局部隔离与排序。"""
    from playwright.sync_api import expect

    names = ["overview", "stats.mean"]
    originals: dict[str, list[dict[str, str]]] = {
        name: _panel(page, name).locator("tbody tr").evaluate_all(
            "rows=>rows.map(row=>({feature:row.cells[0].textContent,text:Array.from(row.cells,cell=>cell.textContent).join('\t')}))"
        )
        for name in names
    }
    local_queries = {name: "" for name in names}
    global_query = ""
    checks: list[dict[str, Any]] = []
    literal_space_query: str = _panel(page, "overview").locator("table").evaluate(
        "table=>{const index=Array.from(table.querySelectorAll('thead th')).findIndex(cell=>cell.textContent==='display_name');return Array.from(table.tBodies[0].rows).find(row=>row.cells[0].textContent==='income_aux').cells[index].textContent}"
    )

    def check() -> None:
        """核对包括当前未显示导航页在内的每表 hidden 行集合。"""
        observed: dict[str, list[str]] = {}
        for name in names:
            actual = _panel(page, name).locator("tbody tr").evaluate_all(
                "rows=>rows.filter(row=>!row.hidden).map(row=>row.cells[0].innerText)"
            )
            expected = [
                row["feature"] for row in originals[name]
                if global_query.lower() in row["text"].lower()
                and local_queries[name].lower() in row["text"].lower()
            ]
            assert sorted(actual) == sorted(expected), (name, global_query, local_queries, actual)
            observed[name] = actual
        checks.append({"global": global_query, "local": dict(local_queries), "rows": observed})

    def global_fill(value: str) -> None:
        """操作页面中的全局输入并核对全部关联表。"""
        nonlocal global_query
        global_query = value
        page.get_by_role("textbox", name="Search all tables", exact=True).fill(value)
        check()

    def local_fill(name: str, value: str) -> None:
        """操作表自己的输入；导航换页仍保留当前组合条件。"""
        local_queries[name] = value
        _open_table(page, name).get_by_role("textbox", name="Search table").fill(value)
        check()

    _open_table(page, "overview")
    # 同一个字面条件必须与先前 hidden 状态、重复 input 和导航页无关。
    global_fill(" debt")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(0)
    global_fill("no_matching_row")
    global_fill(" debt")
    global_fill(" debt")
    _open_table(page, "stats.mean")
    global_fill(" debt")
    _open_table(page, "overview")
    check()
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(0)
    global_fill("辅助  收入")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(
        int(literal_space_query == "辅助  收入")
    )
    global_fill(literal_space_query)
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(1)
    global_fill("")
    global_fill("income")
    local_fill("overview", "debt")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(0)
    page.screenshot(path=str(output / "empty.png"), full_page=True)
    local_fill("overview", "")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(2)
    page.screenshot(path=str(output / "local-cleared.png"), full_page=True)
    global_fill("")
    local_fill("overview", "debt")
    global_fill("income")
    local_fill("overview", "income")
    global_fill("INCOME")
    local_fill("stats.mean", "debt")
    global_fill("")
    local_fill("overview", "_aux")
    global_fill("income")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(1)
    local_fill("overview", "")
    global_fill(" income ")
    global_fill("")
    local_fill("overview", "收入")
    global_fill("中文")
    expect(_panel(page, "overview").locator("tbody tr:visible")).to_have_count(1)
    global_fill("")
    local_fill("overview", "")
    local_fill("stats.mean", "")
    local_fill("overview", "income")
    global_fill("income")
    _panel(page, "overview").get_by_role("columnheader", name="mean", exact=True).click()
    check()
    _open_table(page, "stats.mean")
    check()
    _open_table(page, "overview")
    check()
    local_fill("overview", "")
    global_fill("")
    _panel(page, "overview").get_by_role("columnheader", name="mean", exact=True).click()
    values: list[float] = _panel(page, "overview").locator("tbody tr").evaluate_all(
        "rows=>rows.map(row=>Number(row.cells[Array.from(row.closest('table').querySelectorAll('thead th')).findIndex(th=>th.textContent==='mean')].dataset.sortValue))"
    )
    assert values == sorted(values, reverse=True)
    return {"cases": checks, "numeric_sort": values, "pagination": "navigation pages; no row pagination in renderer"}


def _full_profile_cases(page: Page, expected: dict[str, Any]) -> dict[str, Any]:
    """完整本地案例核对身份检索、两条件交集、未舍入排序和跨页状态。"""
    from playwright.sync_api import expect

    names = ["overview", "stats.psi"]
    originals: dict[str, list[dict[str, str]]] = {
        name: _panel(page, name).locator("tbody tr").evaluate_all(
            "rows=>rows.map(row=>({feature:row.cells[0].textContent,text:Array.from(row.cells,cell=>cell.textContent).join('\t')}))"
        ) for name in names
    }
    feature = next(row for row in expected["tables"]["overview"] if row["feature"] == "EXT_SOURCE_2")
    local_queries = {name: "" for name in names}
    global_query = ""
    checks: list[dict[str, Any]] = []

    def check() -> None:
        """在显示/隐藏页同时验证真实 hidden 集合，保留全部原统计行。"""
        observed: dict[str, list[str]] = {}
        for name in names:
            actual = _panel(page, name).locator("tbody tr").evaluate_all(
                "rows=>rows.filter(row=>!row.hidden).map(row=>row.cells[0].textContent)"
            )
            selected = [row["feature"] for row in originals[name] if global_query.lower() in row["text"].lower() and local_queries[name].lower() in row["text"].lower()]
            assert sorted(actual) == sorted(selected), (name, global_query, local_queries)
            observed[name] = actual
        checks.append({"global": global_query, "local": dict(local_queries), "rows": observed})

    _open_table(page, "overview")
    for query in (feature["feature"], feature["display_name"], feature["data_source"]):
        assert query, feature
        global_query = query
        page.get_by_role("textbox", name="Search all tables", exact=True).fill(query)
        check()
        expect(_panel(page, "overview").locator("tbody tr:visible")).not_to_have_count(0)
    local_queries["overview"] = feature["feature"]
    _panel(page, "overview").get_by_role("textbox", name="Search table").fill(feature["feature"])
    check()
    global_query = "no_matching_row"
    page.get_by_role("textbox", name="Search all tables", exact=True).fill(global_query)
    check()
    local_queries["overview"] = ""
    _panel(page, "overview").get_by_role("textbox", name="Search table").fill("")
    check()
    global_query = feature["feature"]
    page.get_by_role("textbox", name="Search all tables", exact=True).fill(global_query)
    _open_table(page, "stats.psi")
    check()
    local_queries["stats.psi"] = "no_matching_row"
    _panel(page, "stats.psi").get_by_role("textbox", name="Search table").fill(local_queries["stats.psi"])
    check()
    global_query = ""
    page.get_by_role("textbox", name="Search all tables", exact=True).fill("")
    check()
    local_queries["stats.psi"] = ""
    _panel(page, "stats.psi").get_by_role("textbox", name="Search table").fill("")
    check()
    sorted_values: dict[str, list[float | None]] = {}
    for name, column in (("overview", "missing_rate"), ("stats.psi", "4_oot_undue")):
        panel = _open_table(page, name)
        header = panel.get_by_role("columnheader", name=column, exact=True)
        for descending in (False, True):
            header.click()
            values: list[float | None] = panel.locator("table").evaluate(
                "(table,column)=>{const index=Array.from(table.querySelectorAll('thead th')).findIndex(th=>th.textContent===column);return Array.from(table.tBodies[0].rows,row=>row.cells[index].dataset.sortType==='missing'?null:Number(row.cells[index].dataset.sortValue));}", column,
            )
            finite = [value for value in values if value is not None]
            assert finite == sorted(finite, reverse=descending)
            assert values == finite + [None] * (len(values) - len(finite))
            sorted_values[f"{name}.{column}.{descending}"] = values
            check()
    return {"cases": checks, "raw_sort_values": sorted_values}


def _inspect(browser: Browser, entry: dict[str, Any], output: Path) -> dict[str, Any]:
    """同时验收原始 ProfileReport 和只读加载的 snapshot HTML。"""
    from representatives import _layout, _page

    expected = entry["expected"]
    results: dict[str, Any] = {}
    for kind, key in (("profile", "original_html"), ("snapshot", "html")):
        for width, scale in ((1440, 1), (768, 1), (390, 1), (1440, 2)):
            folder = output / f"{kind}-{width}-zoom{scale}"
            folder.mkdir(exist_ok=True)
            page, events = _page(browser, width)
            page.goto(Path(entry[key]).as_uri(), wait_until="load")
            if scale != 1:
                page.evaluate("scale=>document.documentElement.style.zoom=String(scale)", scale)
            numeric_count = sum(_check_profile_rows(page, name, rows) for name, rows in expected["tables"].items())
            searches = _full_profile_cases(page, expected) if entry.get("full_profile") else _search_cases(page, folder)
            metrics = _metric_cases(page)
            page.screenshot(path=str(folder / "final.png"), full_page=True)
            layout = _layout(page)
            assert layout["root"] <= width and layout["body"] <= width, layout
            assert not any(events[key] for key in ("pageerror", "console_error", "csp", "requestfailed")), events
            assert all(urlparse(url).scheme == "file" for url in events["requests"]), events
            results[f"{kind}-{width}-zoom{scale}"] = {
                "html": entry[key], "numeric_fields_checked": numeric_count,
                "html_sha256": hashlib.sha256(Path(entry[key]).read_bytes()).hexdigest(),
                "searches": searches, "metrics": metrics, "layout": layout, "events": events,
                "scale": {"factor": scale, "method": "Chromium CSS root zoom"},
            }
            page.context.close()
    return results


def main() -> None:
    """生成确定性的四特征报告，并输出真实截图、公共数值核对和日志。"""
    import polars as pl
    from playwright.sync_api import sync_playwright

    from mars.analysis import profile_stats
    from mars.reporting import load_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--original-html", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # 原报告和只加载持久化聚合表的 snapshot 共用同一 HTML renderer。
    frame = pl.DataFrame({
        "income": [1.0, 2.0, 3.0, 4.0], "debt": [4.0, 3.0, 2.0, 1.0],
        "income_aux": [2.0, 3.0, 4.0, 5.0], "中文收入": [10.0, 20.0, 30.0, 40.0],
    })
    report = profile_stats(
        frame, metrics=["missing", "zeros", "unique", "mode", "mean", "std", "psi"],
        benchmark_df=frame.head(2),
        feature_metadata={
            "income_aux": {"display_name": "辅助  收入", "data_source": "申请 & 外部"},
            "中文收入": {"display_name": "中文收入" * 20 + "<长名称>"},
        },
    )
    original, snapshot, archive = root / "profile.html", root / "snapshot.html", root / "profile.marsreport"
    report.write_html(str(original))
    report.save(archive, overwrite=True)
    restored = load_report(archive)
    restored.write_html(str(snapshot))
    entry = {
        "original_html": str(original), "html": str(snapshot),
        "expected": {"tables": {name: report.get_table(name).to_dicts() for name in report.describe()["tables"]}},
    }
    if args.archive:
        if not args.original_html:
            parser.error("--archive requires --original-html")
        actual = load_report(args.archive)
        actual_snapshot = root / "full-profile-snapshot.html"
        actual.write_html(str(actual_snapshot))
        entry = {
            "original_html": str(args.original_html.resolve()), "html": str(actual_snapshot),
            "full_profile": True,
            "expected": {"tables": {name: actual.get_table(name).to_dicts() for name in actual.describe()["tables"]}},
        }
    output = root / "profile-browser"
    output.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "src/mars/reporting/_profile_html.py"
    results: dict[str, Any] = {
        "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    }
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel=args.channel, headless=True)
        results["environment"] = {
            "browser": args.channel, "version": browser.version,
            "engine": "Chromium", "headless": True, "platform": platform.platform(),
            "locale": "zh-CN", "opening": "file://",
        }
        results["reports"] = _inspect(browser, entry, output)
        browser.close()
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": "passed", "log": str(output / "results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
