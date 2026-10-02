"""以真实 Chrome 对既有 Profile/snapshot 离线页执行组合搜索与布局验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    page.get_by_role("button", name="Overview" if name == "overview" else "Stats", exact=True).click()
    return _panel(page, name)


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
        "rows=>rows.map(row=>Number(row.cells[Array.from(row.closest('table').querySelectorAll('thead th')).findIndex(th=>th.innerText==='mean')].innerText))"
    )
    assert values == sorted(values, reverse=True)
    return {"cases": checks, "numeric_sort": values, "pagination": "navigation pages; no row pagination in renderer"}


def _inspect(browser: Browser, entry: dict[str, Any], output: Path) -> dict[str, Any]:
    """同时验收原始 ProfileReport 和只读加载的 snapshot HTML。"""
    from representatives import _check_rows, _layout, _page

    expected = entry["expected"]
    results: dict[str, Any] = {}
    for kind, key in (("profile", "original_html"), ("snapshot", "html")):
        for width in (1440, 390):
            folder = output / f"{kind}-{width}"
            folder.mkdir(exist_ok=True)
            page, events = _page(browser, width)
            page.goto(Path(entry[key]).as_uri(), wait_until="load")
            numeric_count = sum(_check_rows(page, name, rows) for name, rows in expected["tables"].items())
            searches = _search_cases(page, folder)
            page.screenshot(path=str(folder / "final.png"), full_page=True)
            layout = _layout(page)
            assert layout["root"] <= width and layout["body"] <= width, layout
            assert not any(events[key] for key in ("pageerror", "console_error", "csp", "requestfailed")), events
            assert all(urlparse(url).scheme == "file" for url in events["requests"]), events
            results[f"{kind}-{width}"] = {
                "html": entry[key], "numeric_fields_checked": numeric_count,
                "html_sha256": hashlib.sha256(Path(entry[key]).read_bytes()).hexdigest(),
                "searches": searches, "layout": layout, "events": events,
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
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # 原报告和只加载持久化聚合表的 snapshot 共用同一 HTML renderer。
    frame = pl.DataFrame({
        "income": [1.0, 2.0, 3.0, 4.0], "debt": [4.0, 3.0, 2.0, 1.0],
        "income_aux": [2.0, 3.0, 4.0, 5.0], "中文收入": [10.0, 20.0, 30.0, 40.0],
    })
    report = profile_stats(
        frame, metrics=["missing", "mean"],
        feature_metadata={"income_aux": {"display_name": "辅助  收入"}},
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
