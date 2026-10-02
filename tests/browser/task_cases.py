"""七任务页面、真实产品截图和同源交叉证据的 Chromium 验收。"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import Browser, Page, expect, sync_playwright

from mars.reporting import load_report
from score_cross import (
    _apply,
    _assert_cell,
    _assert_policies,
    _assert_rule,
    _keyboard,
    _observe,
    _overflow,
    _rows,
    _select_scope,
)

PAGES = [
    "", "demos/", "demos/data-quality/", "demos/binning-stability/",
    "demos/selection-correlation/", "demos/score-cross/", "demos/rule-evidence/",
    "demos/saved-reports/", "demos/report-delivery/",
]


class _QuietHandler(SimpleHTTPRequestHandler):
    """HTTP 行为由浏览器日志记录。"""

    def log_message(self, format: str, *args: Any) -> None:
        """关闭重复的访问日志。"""


def _json(path: Path, value: Any) -> None:
    """写入有限、可解析的验收记录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _layout(page: Page) -> dict[str, Any]:
    """验证主体不溢出，保留代码和表格自身滚动。"""
    values: dict[str, Any] = page.evaluate(
        """() => ({width: innerWidth, scroll: document.documentElement.scrollWidth,
        dpr:devicePixelRatio, containers:[...document.querySelectorAll('pre > code,.md-typeset__scrollwrap')]
        .filter(e=>e.scrollWidth>e.clientWidth+1)
        .map(e=>({width:e.clientWidth,scroll:e.scrollWidth,overflow:getComputedStyle(e).overflowX}))})"""
    )
    assert values["scroll"] <= values["width"] + 1, values
    return values


def _theme(page: Page, theme: str) -> None:
    """实际点击 Material 保留的主题切换控件。"""
    if page.locator("body").get_attribute("data-md-color-scheme") != theme:
        label = "切换到浅色模式" if theme == "default" else "切换到深色模式"
        page.locator(f'label[title="{label}"]:visible').click()
    expect(page.locator("body")).to_have_attribute("data-md-color-scheme", theme)


def _cross(browser: Browser, assets: Path, output: Path) -> dict[str, Any]:
    """范围、格子、梯度、规则、复制与离线请求逐项对公共 Python API。"""
    report = load_report(assets / "score-cross.marsreport")
    context = browser.new_context(
        viewport={"width": 1440, "height": 1000}, locale="zh-CN",
        permissions=["clipboard-read", "clipboard-write"],
    )
    events = _observe(context)
    page = context.new_page()
    page.goto((assets / "score-cross.html").as_uri())
    payload = json.loads(page.locator("#data").text_content() or "{}")
    assert payload["description"] == report.describe()
    scopes = [{key: row[key] for key in ("target", "group", "period")} for row in _rows(report, "overall")]
    bins = _rows(report, "bins")
    xs = sorted([b for b in bins if b["axis"] == "x" and b["kind"] == "normal"], key=lambda b: b["risk_rank"])
    ys = sorted([b for b in bins if b["axis"] == "y" and b["kind"] == "normal"], key=lambda b: b["risk_rank"])
    checked: list[dict[str, Any]] = []
    # 选择实际存在的发现、验证、无标签观察以及真实时间范围，不构造虚假 scope。
    picks = [next(s for s in scopes if s["group"] == group and s["target"] == "bad30")
             for group in ("discovery", "validation", "observation")]
    picks.append(next(s for s in scopes if s["target"] == "late60"))
    picks.append(next(s for s in scopes if s["target"] == "late60" and s["group"] == "observation"))
    for scope in picks:
        _select_scope(page, scope)
        cells = _rows(report, "cells", filters=scope)
        normal = [c for c in cells if c["x_risk_rank"] is not None and c["y_risk_rank"] is not None]
        rows = _rows(report, "row_summary", filters=scope)
        columns = _rows(report, "column_summary", filters=scope)
        for cell in (normal[0], normal[-1]):
            _assert_cell(page, report, cell, scope, cell["x_risk_rank"] - 1,
                         cell["y_risk_rank"] - 1, rows, columns, normal)
        _apply(page, "X >= 2 AND Y >= 3", enter=True)
        _assert_rule(page, report, "X >= 2 AND Y >= 3", scope)
        _apply(page, "X <= Y2")
        expect(page.locator("#rule-error")).not_to_have_text("")
        page.locator("#rule-clear").click()
        expect(page.locator("#custom-rule-result")).to_be_hidden()
        _assert_policies(page, payload, scope)
        page.locator("#copy-btn").click()
        expect(page.locator("#toast")).to_have_text("证据已复制")
        copied = json.loads(page.evaluate("navigator.clipboard.readText()"))
        assert copied["report_id"] == report.report_id
        assert report.query_page(copied["table"], **copied["query"])["data"].height == 1
        checked.append({"scope": scope, "states": sorted({c["status"] for c in cells})})
    _select_scope(page, picks[0])
    _keyboard(page, len(xs), len(ys))
    for width in (1440, 768, 390):
        page.set_viewport_size({"width": width, "height": 1000 if width > 390 else 844})
        _overflow(page)
        page.screenshot(path=str(output / f"cross-{width}.png"), full_page=width == 1440)
    _apply(page, "X >= 2 AND Y >= 3")
    page.screenshot(path=str(output / "cross-rule-mobile.png"), full_page=True)
    assert not any(events[k] for k in ("pageerror", "csp", "external_requests", "requestfailed")), events
    context.close()
    return {"scopes": checked, "events": events, "clipboard_replayed": True, "keyboard": True,
            "viewports": [1440, 768, 390], "offline": "file://; nonlocal requests blocked and none attempted"}


def _preview(browser: Browser, assets: Path, output: Path) -> dict[str, Any]:
    """验收原生分箱 PNG/SVG 和同源查询；不重绘或覆盖产品图。"""
    evidence = json.loads((assets / "binning-native-evidence.json").read_text(encoding="utf-8"))
    report = load_report(assets / "binning.marsreport")
    assert evidence["report_id"] == report.report_id
    for query in evidence["queries"]:
        reference = query["reference"]
        assert reference["report_id"] == report.report_id
        page_result = report.query_page(reference["table"], **reference["query"])
        assert page_result["data"].to_dicts() == query["rows"]
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN")
    events = _observe(context)
    page = context.new_page()
    rendered: list[dict[str, Any]] = []
    for key, element in (("chart_png", "img"), ("chart_svg", "svg")):
        path = assets / evidence[key]
        assert path.is_file()
        page.goto(path.as_uri())
        chart = page.locator(element).first
        expect(chart).to_be_visible()
        box = chart.bounding_box()
        assert box is not None and box["width"] > 0 and box["height"] > 0
        page.screenshot(path=str(output / f"native-binning-{key}.png"), full_page=True)
        rendered.append({"file": path.name, "box": box})
    assert not any(events[key] for key in ("pageerror", "csp", "external_requests", "requestfailed")), events
    context.close()
    return {"report_id": report.report_id, "feature": evidence["feature"],
            "source": evidence["source_api"], "rendered": rendered,
            "method": "native exported files; public-query replay; no composite Score Cross hero"}


def _docs(browser: Browser, site: Path, output: Path) -> dict[str, Any]:
    """本地站点真实点击页签、复制、折叠、下载并检查主题和各视口。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN",
                                  permissions=["clipboard-read", "clipboard-write"], accept_downloads=True)
    page = context.new_page()
    errors: list[str] = []
    missing: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("response", lambda response: missing.append(response.url) if response.status >= 400 and urlparse(response.url).hostname == "127.0.0.1" else None)
    layouts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    try:
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000 if width == 1440 else 844})
            for path in PAGES:
                page.goto(base + path, wait_until="networkidle")
                name = path.strip("/").replace("/", "-") or "home"
                for theme in ("default", "slate"):
                    _theme(page, theme)
                    layouts.append({"page": path, "theme": theme, **_layout(page)})
                    page.screenshot(path=str(output / f"{name}-{width}-{theme}.png"))
                if path and path != "demos/":
                    labels = page.locator("article .tabbed-labels label")
                    for label in labels.all():
                        label.click()
                        control = label.get_attribute("for")
                        expect(page.locator(f'[id="{control}"]')).to_be_checked()
                    # 实际复制代码，读取系统剪贴板核对完整内容。
                    copy = page.locator("article button.md-clipboard:visible").first
                    if copy.count():
                        expected = copy.evaluate("e => document.getElementById(e.dataset.clipboardTarget.slice(1)).textContent")
                        copy.click()
                        page.wait_for_function("expected => navigator.clipboard.readText().then(v=>v===expected)", arg=expected)
                    details = page.locator("article details").first
                    if details.count():
                        details.locator("summary").click()
                        expect(details).to_have_attribute("open", "")
                    page.keyboard.press("Tab")
                    assert page.evaluate("document.activeElement !== document.body")
                    actions.append({"page": path, "width": width, "tabs": labels.count(),
                                    "code_copy": copy.count() > 0, "expanded": details.count() > 0, "keyboard": True})
        # 下载从构建后的实际链接读取，不使用仓库路径代替 HTTP 验收。
        download_paths: list[dict[str, Any]] = []
        page.goto(base + "demos/report-delivery/", wait_until="networkidle")
        hrefs = page.locator('article a[href*="assets/cases/"]').evaluate_all("links=>[...new Set(links.map(a=>a.href))]")
        for href in hrefs:
            response = context.request.get(href)
            assert response.ok, (href, response.status)
            body = response.body()
            assert body
            download_paths.append({"url": urlparse(href).path, "bytes": len(body), "status": response.status})
        page.goto(base + "demos/binning-stability/", wait_until="networkidle")
        for filename in ("binning-native-main-score.png", "binning-native-main-score.svg", "binning.html"):
            link = page.locator(f'article a[href$="/{filename}"]').first
            assert link.count(), f"Native binning download is missing from the case page: {filename}"
            href = link.get_attribute("href")
            assert href is not None
            absolute = link.evaluate("element => element.href")
            response = context.request.get(absolute)
            assert response.ok, (absolute, response.status)
            assert response.body() == (site / "assets/cases" / filename).read_bytes()
            download_paths.append({"url": urlparse(absolute).path, "bytes": len(response.body()), "status": response.status})
        page.goto(base + "demos/report-delivery/", wait_until="networkidle")
        archive = page.locator('article a[href$="/cases.zip"]').first
        with page.expect_download() as downloading:
            archive.click()
        download = downloading.value
        assert download.suggested_filename == "cases.zip"
        destination = output / "browser-downloaded-cases.zip"
        download.save_as(destination)
        assert destination.read_bytes() == (site / "assets/cases/cases.zip").read_bytes()
        assert not errors and not missing, {"pageerror": errors, "local_http_errors": missing}
        return {"layouts": layouts, "actions": actions, "downloads": download_paths,
                "pageerror": errors, "local_http_errors": missing,
                "native_download": {"filename": download.suggested_filename, "bytes": destination.stat().st_size}}
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join()


def _zoom(playwright: Any, site: Path, output: Path, channel: str) -> dict[str, Any]:
    """在 Chrome 设置页选择真实 200%，抽查导航、页签、表格与按钮。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records: list[dict[str, Any]] = []
    try:
        with TemporaryDirectory(prefix="mars-cases-zoom-") as profile:
            context = playwright.chromium.launch_persistent_context(profile, channel=channel, headless=True,
                no_viewport=True, args=["--window-size=1440,1000"])
            page = context.pages[0]
            page.goto("chrome://settings/appearance")
            page.locator("#zoomLevel").select_option(label="200%")
            for path in ("", "demos/", "demos/score-cross/", "demos/saved-reports/"):
                page.goto(f"http://127.0.0.1:{server.server_port}/" + path, wait_until="networkidle")
                values = _layout(page)
                assert values["dpr"] == 2, values
                name = path.strip("/").replace("/", "-") or "home"
                page.screenshot(path=str(output / f"{name}-200-percent.png"), full_page=True)
                records.append({"page": path, **values})
            context.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return {"mechanism": "Chrome settings #zoomLevel 200%", "pages": records}


def main() -> None:
    """分阶段验收原生分箱图、交叉交互和最终严格构建。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--phase", choices=["preview", "verify"], default="verify")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"os": platform.platform(), "python": platform.python_version(),
                             "playwright": version("playwright"), "engine": "Chromium", "channel": args.channel,
                             "access": "native local file HTML / strict-build local HTTP", "headless": True}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        record["browser_version"] = browser.version
        try:
            if args.phase == "preview":
                record["preview"] = _preview(browser, args.assets.resolve(), args.output.resolve())
            else:
                record["cross"] = _cross(browser, args.assets.resolve(), args.output.resolve())
                if args.site_dir is None:
                    parser.error("verify 阶段需要 --site-dir 指向严格构建结果")
                record["docs"] = _docs(browser, args.site_dir.resolve(), args.output.resolve())
        finally:
            browser.close()
        if args.phase == "verify":
            record["zoom"] = _zoom(playwright, args.site_dir.resolve(), args.output.resolve(), args.channel)
    record["status"] = "passed"
    _json(args.output / f"{args.phase}-results.json", record)
    print(json.dumps({"status": "passed", "phase": args.phase, "browser_version": record["browser_version"]}))


if __name__ == "__main__":
    main()
