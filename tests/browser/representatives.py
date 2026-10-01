"""在真实 Chromium 中抽查公共画像、风险、规则与相关性 HTML。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import Browser, Page, expect, sync_playwright


def _source_provenance() -> dict[str, Any]:
    """记录模板字节与生产提交关系，使先验收后提交的证据可核对。"""
    repository = Path(__file__).resolve().parents[2]
    files = ["src/mars/reporting/html_assets.py", "src/mars/reporting/_binning_html.py"]
    sources: dict[str, Any] = {}
    for relative in files:
        source = (repository / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", relative],
            cwd=repository,
            text=True,
            encoding="utf-8",
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository,
            text=True,
            encoding="utf-8",
        ).replace("\r\n", "\n")
        sources[relative] = {
            "commit": commit,
            "normalized_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "matches_committed_bytes": source == committed,
        }
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        encoding="utf-8",
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--short", "--", *files, "tests/browser/representatives.py"],
        cwd=repository,
        text=True,
        encoding="utf-8",
    ).strip()
    css = sources[files[0]]
    return {
        "head": head,
        "production_source_commit": css["commit"],
        "normalized_html_assets_sha256": css["normalized_sha256"],
        "source_matches_committed_bytes": all(
            value["matches_committed_bytes"] for value in sources.values()
        ),
        "sources": sources,
        "dirty": dirty,
    }


def _load(path: Path) -> dict[str, Any]:
    """读取 fixture 公共期望，不在浏览器重算指标。"""
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return value


def _record(page: Page) -> dict[str, list[Any]]:
    """记录真实页异常、CSP 和请求，并阻断意外外部资源。"""
    events: dict[str, list[Any]] = {
        "pageerror": [],
        "console_error": [],
        "csp": [],
        "requests": [],
        "requestfailed": [],
    }
    page.on("pageerror", lambda error: events["pageerror"].append(str(error)))
    page.on(
        "console",
        lambda message: (
            events["console_error"].append(message.text) if message.type == "error" else None
        ),
    )
    page.on("request", lambda request: events["requests"].append(request.url))
    page.on(
        "requestfailed",
        lambda request: events["requestfailed"].append(
            {"url": request.url, "failure": request.failure}
        ),
    )
    page.expose_function("recordCsp", lambda value: events["csp"].append(value))
    page.add_init_script(
        "document.addEventListener('securitypolicyviolation', e => window.recordCsp({directive:e.violatedDirective,blocked:e.blockedURI}));"
    )
    page.route(
        "**/*",
        lambda route: (
            route.continue_()
            if urlparse(route.request.url).scheme in {"file", "data", "about"}
            else route.abort()
        ),
    )
    return events


def _page(browser: Browser, width: int) -> tuple[Page, dict[str, Any]]:
    """建立固定语言与 viewport 的独立真实页面。"""
    context = browser.new_context(
        viewport={"width": width, "height": 1000}, locale="zh-CN", color_scheme="light"
    )
    page = context.new_page()
    return page, _record(page)


def _table(page: Page, name: str) -> Any:
    """按已有表标题定位快照导出表。"""
    return (
        page.locator("section.panel")
        .filter(has=page.get_by_role("heading", name=name, exact=True, include_hidden=True))
        .locator("table")
    )


def _check_rows(page: Page, name: str, expected: list[dict[str, Any]]) -> int:
    """核对完整小表的真实表头、行粒度和有限数字。"""
    table = _table(page, name)
    expect(table).to_have_count(1)
    rows: list[list[str]] = table.locator("tbody tr").evaluate_all(
        "rows=>rows.map(r=>Array.from(r.cells,c=>c.innerText))"
    )
    assert len(rows) == len(expected), (name, len(rows), len(expected))
    headers = table.locator("thead th").all_text_contents()
    if not expected:
        return 0
    assert headers[: len(expected[0])] == list(expected[0]), (name, headers)
    checked = 0
    for actual, saved in zip(rows, expected):
        for value, observed in zip(saved.values(), actual):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert math.isclose(float(observed), value, rel_tol=1e-5, abs_tol=1e-6), (
                    name,
                    value,
                    observed,
                )
                checked += 1
            elif isinstance(value, str):
                assert observed == value, (name, value, observed)
    return checked


def _layout(page: Page) -> dict[str, Any]:
    """只读取实际布局，判断专用表容器以外是否溢出。"""
    return page.evaluate(
        "({viewport:innerWidth,body:document.body.scrollWidth,root:document.documentElement.scrollWidth,containers:Array.from(document.querySelectorAll('.table-wrap,.mars-table-scroll'),e=>({client:e.clientWidth,scroll:e.scrollWidth}))})"
    )


def _snapshot_report(
    browser: Browser, entry: dict[str, Any], name: str, output: Path
) -> dict[str, Any]:
    """操作加载报告已有导航、搜索和排序，并比对公共事实。"""
    expected = _load(Path(entry["expected"]))
    results: dict[str, Any] = {}
    for width in (1440, 390):
        page, events = _page(browser, width)
        page.goto(Path(entry["html"]).as_uri(), wait_until="load")
        count = sum(
            _check_rows(page, table_name, rows) for table_name, rows in expected["tables"].items()
        )
        wanted = {
            "profile": "overview",
            "risk": "summary",
            "rule": "evaluation",
            "correlation": "pairs",
        }[name]
        page.get_by_role(
            "button", name="Overview" if wanted == "overview" else wanted, exact=True
        ).click()
        panel = page.locator("section.panel").filter(
            has=page.get_by_role("heading", name=wanted, exact=True)
        )
        local = panel.get_by_role("textbox", name="Search table")
        local.fill("no_such_feature_browser_probe")
        expect(panel.locator("tbody tr:visible")).to_have_count(0)
        local.fill("")
        expect(panel.locator("tbody tr:visible")).to_have_count(len(expected["tables"][wanted]))
        search_token = expected["tables"][wanted][0]["rule_id"] if name == "rule" else "income"
        local.fill(search_token)
        visible_text = panel.locator("tbody tr:visible").all_text_contents()
        assert visible_text and all(search_token in row for row in visible_text), name
        local.fill("")
        if name == "profile":
            panel.get_by_role("columnheader", name="mean", exact=True).click()
            values = panel.locator("tbody tr").evaluate_all(
                "rows=>rows.map(r=>Number(r.cells[3].innerText))"
            )
            assert values == sorted(values)
            page.get_by_role("textbox", name="Search all tables", exact=True).fill("income")
            expect(panel.locator("tbody tr:visible")).to_have_count(1)
            page.get_by_role("textbox", name="Search all tables", exact=True).fill("")
        page.screenshot(path=str(output / f"{name}-{width}.png"), full_page=True)
        layout = _layout(page)
        assert layout["root"] <= width and layout["body"] <= width, (name, width, layout)
        assert not any(
            events[key] for key in ("pageerror", "console_error", "csp", "requestfailed")
        ), (name, events)
        assert all(urlparse(url).scheme == "file" for url in events["requests"]), events
        results[str(width)] = {
            "numeric_fields_checked": count,
            "tables_checked": list(expected["tables"]),
            "navigation_search_sort": "passed",
            "layout": layout,
            "events": events,
        }
        page.context.close()
    return results


def _original_risk(browser: Browser, entry: dict[str, Any], output: Path) -> dict[str, Any]:
    """验收原风险报告已有目标选择、页面导航及全局搜索。"""
    results: dict[str, Any] = {}
    for width in (1440, 390):
        page, events = _page(browser, width)
        page.goto(Path(entry["original_html"]).as_uri(), wait_until="load")
        controls = page.locator(".mars-global-tools input,.mars-global-tools button").evaluate_all(
            "elements=>elements.map(e=>{const r=e.getBoundingClientRect();return{id:e.id,text:e.innerText,left:r.left,right:r.right}})"
        )
        if any(control["left"] < 0 or control["right"] > width for control in controls):
            page.screenshot(path=str(output / f"risk-toolbar-before-{width}.png"), full_page=True)
            (output / f"risk-toolbar-before-{width}.json").write_text(
                json.dumps({"controls": controls, "events": events}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise AssertionError(
                f"Risk report toolbar clips controls at viewport {width}: {controls}"
            )
        page.screenshot(path=str(output / f"risk-toolbar-after-{width}.png"), full_page=True)
        (output / f"risk-toolbar-after-{width}.json").write_text(
            json.dumps({"controls": controls, "events": events}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = page.locator("#semantics-section")
        metadata_tables = metadata.locator("table").evaluate_all(
            "tables=>tables.map(t=>({width:t.getBoundingClientRect().width,parent:t.parentElement.className,parentWidth:t.parentElement.clientWidth}))"
        )
        if metadata.locator(".mars-table-scroll").count() != len(metadata_tables):
            page.screenshot(path=str(output / f"risk-metadata-before-{width}.png"), full_page=True)
            (output / f"risk-metadata-before-{width}.json").write_text(
                json.dumps({"tables": metadata_tables}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise AssertionError("Business Metadata tables need a dedicated scrolling container.")
        containers = metadata.locator(".mars-table-scroll")
        for index in range(containers.count()):
            container = containers.nth(index)
            overflow = container.evaluate("e=>e.scrollWidth>e.clientWidth")
            if overflow:
                page.get_by_role("link", name="Business Metadata", exact=True).click()
                for _ in range(50):
                    page.keyboard.press("Tab")
                    if container.evaluate("e=>document.activeElement===e"):
                        break
                assert container.evaluate("e=>document.activeElement===e")
                page.keyboard.press("ArrowRight")
                page.wait_for_function(
                    "index=>document.querySelectorAll('#semantics-section .mars-table-scroll')[index].scrollLeft>0",
                    arg=index,
                )
                break
        metadata_containers = containers.evaluate_all(
            "elements=>elements.map(e=>({client:e.clientWidth,scroll:e.scrollWidth,scrollLeft:e.scrollLeft,left:e.getBoundingClientRect().left,right:e.getBoundingClientRect().right}))"
        )
        (output / f"risk-metadata-after-{width}.json").write_text(
            json.dumps(
                {"containers": metadata_containers, "events": events}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        page.screenshot(path=str(output / f"risk-metadata-after-{width}.png"), full_page=True)
        page.get_by_role("link", name="Summary", exact=True).click()
        regex = page.get_by_role("checkbox", name="Regex Mode", exact=True)
        regex.check()
        page.locator("#mars-global-search").fill("[")
        expect(page.locator("#mars-global-error")).not_to_have_text("")
        page.get_by_role("button", name="Clear Search", exact=True).click()
        regex.uncheck()
        page.locator("#mars-global-search").fill("income")
        expect(page.locator("#mars-summary-table tbody tr:visible")).not_to_have_count(0)
        page.get_by_role("button", name="Clear Search", exact=True).click()
        source_credit = page.get_by_role("checkbox", name="credit", exact=True)
        source_credit.uncheck()
        expect(
            page.locator("#mars-summary-table tbody tr[data-feature='debt']:visible")
        ).to_have_count(0)
        source_credit.check()
        expect(
            page.locator("#mars-summary-table tbody tr[data-feature='debt']:visible")
        ).not_to_have_count(0)
        page.get_by_role("button", name="Clear", exact=True).click()
        expect(source_credit).not_to_be_checked()
        expect(page.get_by_role("checkbox", name="application", exact=True)).not_to_be_checked()
        page.get_by_role("button", name="All", exact=True).click()
        expect(source_credit).to_be_checked()
        expect(page.get_by_role("checkbox", name="application", exact=True)).to_be_checked()
        page.locator("#mars-feature-jump-input").fill("income")
        page.get_by_role("button", name="Go", exact=True).click()
        expect(page.locator("#mars-feature-jump-error")).to_have_text("")
        expect(
            page.locator("#mars-summary-table tbody tr[data-feature='income']").first
        ).to_be_visible()
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Export Feature List", exact=True).click()
        download = download_info.value
        download_path = output / f"risk-feature-list-{width}.txt"
        download.save_as(download_path)
        feature_map = json.loads(download_path.read_text(encoding="utf-8"))
        assert feature_map == {"application": ["income"], "credit": ["debt"]}, feature_map
        page.get_by_role("link", name="Grouped Pivot", exact=True).click()
        if not page.locator("#pivot-section").evaluate("element => element.open"):
            page.locator("#pivot-section > summary").click()
        selector = page.locator("#mars-pivot-target")
        selector.select_option("later")
        expect(page.locator(".mars-pivot-view[data-y-value='later']")).to_be_visible()
        expect(page.locator(".mars-pivot-view[data-y-value='bad']")).to_be_hidden()
        selector.select_option("bad")
        expect(page.locator(".mars-pivot-view[data-y-value='bad']")).to_be_visible()
        page.screenshot(path=str(output / f"risk-original-target-{width}.png"), full_page=True)
        layout = _layout(page)
        assert not any(
            events[key] for key in ("pageerror", "console_error", "csp", "requestfailed")
        ), events
        assert all(urlparse(url).scheme == "file" for url in events["requests"]), events
        results[str(width)] = {
            "target_switch_global_search_navigation": "passed",
            "regex_jump_source_filters_existing_feature_export": "passed",
            "metadata_tables_contained_and_keyboard_scroll": "passed",
            "metadata_containers": metadata_containers,
            "toolbar_controls": controls,
            "layout": layout,
            "events": events,
        }
        page.context.close()
    return results


def _static_views(browser: Browser, entries: dict[str, Any], output: Path) -> dict[str, Any]:
    """核对规则原静态表与公开相关性 Styler，同一表示只做已有展示。"""
    results: dict[str, Any] = {}
    for name, key in (("rule", "original_html"), ("correlation", "matrix_html")):
        page, events = _page(browser, 1440)
        entry = entries[name]
        facts = _load(Path(entry["expected"]))
        page.goto(Path(entry[key]).as_uri(), wait_until="load")
        assert page.locator("table").count() > 0
        if name == "rule":
            expected = facts["tables"]["evaluation"]
            heading = page.get_by_role("heading", name="Evaluation", exact=True)
            table = heading.locator("xpath=following-sibling::table[1]")
            assert table.locator("tbody tr").count() == len(expected)
            headers = table.locator("thead th").all_text_contents()
            assert headers == list(expected[0]), headers
            assert "representative-browser-synthetic" in page.locator("body").inner_text()
            metadata = json.loads(page.locator("pre").inner_text())
            assert metadata["business_context"]["labels"]["bad"]["definition"] == '模拟违约<&>"'
        else:
            matrix = facts["matrix"]
            rows = page.locator("tbody tr").evaluate_all(
                "rows=>rows.map(r=>({name:r.querySelector('th').innerText,values:Array.from(r.querySelectorAll('td'),c=>c.innerText)}))"
            )
            for row in rows:
                expected = matrix[row["name"]]
                for actual, value in zip(row["values"], expected.values()):
                    assert math.isclose(float(actual), value, abs_tol=0.0005), (row, expected)
        assert not page.locator("script[src]").count()
        page.screenshot(path=str(output / f"{name}-original-1440.png"), full_page=True)
        assert not any(
            events[key] for key in ("pageerror", "console_error", "csp", "requestfailed")
        ), events
        results[name] = {"static_headers_rows_numbers_metadata": "passed", "events": events}
        page.context.close()
    return results


def main() -> None:
    """运行可重复的 file:// 真实 Chromium 代表报告验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = _load(output / "manifest.json")
    screenshots = output / "representative-browser"
    screenshots.mkdir(exist_ok=True)
    results: dict[str, Any] = {"source_provenance": _source_provenance()}
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel=args.channel, headless=True)
        results["environment"] = {
            "browser": args.channel,
            "version": browser.version,
            "engine": "Chromium",
            "headless": True,
            "platform": platform.platform(),
            "locale": "zh-CN",
            "opening": "file://",
        }
        for name, entry in manifest["representatives"].items():
            results[name] = _snapshot_report(browser, entry, name, screenshots)
        results["risk_original"] = _original_risk(
            browser, manifest["representatives"]["risk"], screenshots
        )
        results["static_originals"] = _static_views(
            browser, manifest["representatives"], screenshots
        )
        browser.close()
    (screenshots / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"result": "passed", "log": str(screenshots / "results.json")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
