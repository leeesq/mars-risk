"""通过真实 Chromium 检查本地 MkDocs 站点；先严格构建再传入 --site-dir。"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import platform
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlparse

from playwright.sync_api import Page, expect, sync_playwright


class _QuietHandler(SimpleHTTPRequestHandler):
    """保持服务器输出安静，请求证据由浏览器统一记录。"""

    def log_message(self, format: str, *args: Any) -> None:
        """忽略服务器终端日志，保留 HTTP 行为。"""


def _capture(page: Page, output: Path, name: str, *, full_page: bool = False) -> None:
    """保存对应真实交互状态的截图。"""
    page.screenshot(path=str(output / f"{name}.png"), full_page=full_page)


def _capture_zoom(page: Page, output: Path, name: str) -> None:
    """按真实缩放后的设备像素捕获窗口，避开普通截图的半宽裁切。"""
    dimensions: dict[str, float] = page.evaluate(
        "({width:innerWidth, height:innerHeight, dpr:devicePixelRatio})"
    )
    session = page.context.new_cdp_session(page)
    try:
        screenshot: dict[str, Any] = session.send("Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": True, "fromSurface": True,
            "clip": {
                "x": 0, "y": 0, "width": dimensions["width"] * dimensions["dpr"],
                "height": dimensions["height"] * dimensions["dpr"], "scale": 1,
            },
        })
        (output / f"{name}.png").write_bytes(base64.b64decode(screenshot["data"]))
        assert page.evaluate("innerWidth") == dimensions["width"]
    finally:
        session.detach()


def _layout(page: Page) -> dict[str, Any]:
    """读取主体、内容以及专用滚动容器的实际尺寸。"""
    dimensions: dict[str, Any] = page.evaluate(
        """() => ({
          width: innerWidth,
          scrollWidth: document.documentElement.scrollWidth,
          devicePixelRatio,
          visualScale: visualViewport.scale,
          overflow: [...document.querySelectorAll('.md-typeset__scrollwrap, pre > code')]
            .filter(e => e.scrollWidth > e.clientWidth + 1)
            .map(e => ({tag: e.tagName, width: e.clientWidth, scrollWidth:e.scrollWidth,
                       overflow:getComputedStyle(e).overflowX}))
        })"""
    )
    assert dimensions["scrollWidth"] <= dimensions["width"], dimensions
    return dimensions


def _watch(page: Page, logs: dict[str, list[Any]]) -> None:
    """在普通及真实缩放页面记录加载、错误和 CSP 事件。"""
    page.on("pageerror", lambda error: logs["pageerror"].append(str(error)))
    page.on("console", lambda msg: logs["console_error"].append(
        {"text": msg.text, "location": msg.location}
    ) if msg.type == "error" else None)
    page.on("request", lambda request: logs["requests"].append(request.url))
    page.on("requestfailed", lambda request: logs["requestfailed"].append(
        {"url": request.url, "failure": request.failure}
    ))
    page.on("response", lambda response: logs["http_error"].append(
        {"url": response.url, "status": response.status}
    ) if response.status >= 400 else None)
    page.expose_function("recordCspViolation", lambda event: logs["csp"].append(event))
    page.add_init_script("""document.addEventListener('securitypolicyviolation', e =>
      window.recordCspViolation({directive:e.violatedDirective, blocked:e.blockedURI}));""")


def _theme(page: Page, scheme: str) -> None:
    """通过 Material 已有切换按钮选择主题。"""
    if page.locator("body").get_attribute("data-md-color-scheme") != scheme:
        title = "切换到浅色模式" if scheme == "default" else "切换到深色模式"
        page.locator(f'label[title="{title}"]:visible').click()
    expect(page.locator("body")).to_have_attribute("data-md-color-scheme", scheme)


def _scroll_table(page: Page) -> dict[str, Any]:
    """对实际超宽的表格发出横向滚轮事件；自然换行的表格只核对尺寸。"""
    tables = page.locator("article .md-typeset__scrollwrap")
    dimensions: list[dict[str, int]] = tables.evaluate_all(
        "elements => elements.map(e => ({width:e.clientWidth, scrollWidth:e.scrollWidth}))"
    )
    index = max(range(len(dimensions)), key=lambda i: dimensions[i]["scrollWidth"] - dimensions[i]["width"])
    table = tables.nth(index)
    table.scroll_into_view_if_needed()
    if dimensions[index]["scrollWidth"] > dimensions[index]["width"] + 1:
        table.hover()
        page.mouse.wheel(500, 0)
        page.wait_for_function(
            "index => document.querySelectorAll('article .md-typeset__scrollwrap')[index].scrollLeft > 0",
            arg=index,
        )
    return table.evaluate(
        "e => ({clientWidth:e.clientWidth, scrollWidth:e.scrollWidth, scrollLeft:e.scrollLeft})"
    )


def _search_api(page: Page) -> None:
    """实际输入搜索并点击 API 命中结果。"""
    page.wait_for_load_state("networkidle")
    search = page.get_by_role("textbox", name="搜索", exact=True)
    opener = page.locator('label[for="__search"].md-header__button:visible')
    if opener.count():
        opener.click()
    search.click()
    search.press("Control+A")
    search.press_sequentially("write_score_cross_html")
    match = page.locator('.md-search-result__link[href*="#mars.analysis.write_score_cross_html"]')
    expect(match).to_be_visible(timeout=15000)
    match.click()
    page.wait_for_url("**/reference/analysis/**")
    expect(page.locator('[id="mars.analysis.write_score_cross_html"]')).to_be_visible()


def _run(site: Path, output: Path, channel: str) -> dict[str, Any]:
    """运行桌面、窄屏、主题及真实导航搜索并记录结果。"""
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    logs: dict[str, list[Any]] = {
        "pageerror": [], "console_error": [], "csp": [], "requests": [],
        "requestfailed": [], "http_error": [],
    }
    results: dict[str, Any] = {
        "os": platform.platform(), "python": platform.python_version(),
        "engine": "Chromium", "channel": channel, "headless": True,
        "playwright": version("playwright"), "mkdocs": version("mkdocs"),
        "mkdocs_material": version("mkdocs-material"),
        "locale": "zh-CN", "open_method": "local HTTP, strict MkDocs build",
        "viewports": [], "logs": logs,
    }
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=channel, headless=True)
            results["browser_version"] = browser.version
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN")
            page = context.new_page()
            _watch(page, logs)
            page.goto(base, wait_until="networkidle")

            # 首页保留区按现有身份和动态来源核对，不下载或替换动态徽章。
            expect(page.locator(".mars-home-logo")).to_have_attribute("src", "assets/mars-logo.svg")
            expect(page.locator(".mars-home-wordmark")).to_have_attribute("alt", "MODELING ANALYSIS RISK SCORE")
            assert page.locator(".mars-home-badges a").count() == 6
            assert page.locator(".mars-task-card").count() == 6
            assert "0.0.28" in page.locator(".mars-home").inner_text()
            results["badges"] = page.locator(".mars-home-badges a").evaluate_all(
                "elements => elements.map(e => ({href:e.href, image:e.firstElementChild.src}))"
            )
            for scheme in ("slate", "default"):
                _theme(page, scheme)
                results["viewports"].append({"page": "home", "theme": scheme, **_layout(page)})
                _capture(page, output, f"home-desktop-{scheme}", full_page=True)
            page.locator(".tabbed-labels").get_by_text("AI Agent：查询与复用", exact=True).click()
            expect(page.locator(".tabbed-block:visible")).to_contain_text("load_report")
            _capture(page, output, "home-agent-tab")

            # 任务卡的中文锚点由既有显式 span 保存，实际点击必须滚动到目标章节。
            page.locator(".mars-task-card").filter(has_text="SCORE CROSS").click()
            page.wait_for_url("**/user-guide/correlation-and-score-cross/**")
            target = page.locator('[id="固定分段交叉"]')
            expect(target).to_be_attached()
            page.wait_for_function("scrollY > 100")
            results["task_anchor"] = {
                "hash": unquote(urlparse(page.url).fragment),
                "top": target.bounding_box()["y"], "scroll_y": page.evaluate("scrollY"),
            }
            assert 0 <= results["task_anchor"]["top"] < 300
            _capture(page, output, "guide-task-anchor")
            assert page.locator(".mars-home").count() == 0
            assert page.locator("article table").count() >= 3
            for scheme in ("default", "slate"):
                _theme(page, scheme)
                results["viewports"].append({"page": "guide", "theme": scheme, **_layout(page)})
                _capture(page, output, f"guide-desktop-{scheme}")

            _search_api(page)
            assert page.locator(".mars-home").count() == 0
            assert page.locator("article table").count() > 0
            for scheme in ("slate", "default"):
                _theme(page, scheme)
                results["viewports"].append({"page": "api", "theme": scheme, **_layout(page)})
                _capture(page, output, f"api-desktop-{scheme}")
            page.locator("article").get_by_role("link", name="相关性与模型分交叉", exact=True).click()
            page.wait_for_url("**/user-guide/correlation-and-score-cross/")

            # 窄屏保留汉堡导航和搜索；表格/代码只在各自容器中滚动。
            page.set_viewport_size({"width": 390, "height": 844})
            for path, name in (("", "home"), ("user-guide/correlation-and-score-cross/", "guide"), ("reference/analysis/", "api")):
                page.goto(base + path, wait_until="networkidle")
                for scheme in ("default", "slate"):
                    _theme(page, scheme)
                    results["viewports"].append({"page": name, "theme": scheme, **_layout(page)})
                    _capture(page, output, f"{name}-mobile-{scheme}")
                if name == "home":
                    page.locator('label[for="__drawer"].md-header__button:visible').click()
                    expect(page.locator("#__drawer")).to_be_checked()
                    expect(page.locator(".md-nav--primary")).to_be_visible()
                    _capture(page, output, "home-mobile-navigation")
                    page.locator('label[for="__drawer"].md-overlay').click(position={"x": 380, "y": 100})
                    expect(page.locator("#__drawer")).not_to_be_checked()
                else:
                    results[f"{name}_mobile_table_scroll"] = _scroll_table(page)
                    _capture(page, output, f"{name}-mobile-table")
            _search_api(page)
            _capture(page, output, "api-mobile-search-result")
            results["mobile_search"] = "passed"

            browser.close()

            # Chrome 设置页修改真实默认缩放，不把 viewport 或 CSS 缩放当作 200%。
            with TemporaryDirectory(prefix="mars-docs-zoom-") as profile:
                zoom_context = playwright.chromium.launch_persistent_context(
                    profile, channel=channel, headless=True, no_viewport=True,
                    args=["--window-size=1440,1000"],
                )
                try:
                    zoom_page = zoom_context.pages[0]
                    _watch(zoom_page, logs)
                    zoom_page.goto("chrome://settings/appearance")
                    zoom_page.locator("#zoomLevel").select_option(label="200%")
                    expect(zoom_page.locator("#zoomLevel")).to_have_value("2")
                    zoom_results: list[dict[str, Any]] = []
                    for path, name in (
                        ("", "home"),
                        ("user-guide/correlation-and-score-cross/#固定分段交叉", "guide"),
                        ("reference/analysis/#mars.analysis.write_score_cross_html", "api"),
                    ):
                        zoom_page.goto(base + path, wait_until="networkidle")
                        dimensions = _layout(zoom_page)
                        assert dimensions["devicePixelRatio"] == 2
                        zoom_results.append({"page": name, **dimensions})
                        _capture_zoom(zoom_page, output, f"{name}-zoom-200")
                    results["browser_zoom_200"] = {
                        "method": "Chrome appearance settings #zoomLevel selected 200%",
                        "window_width": zoom_page.evaluate("outerWidth"), "pages": zoom_results,
                    }
                finally:
                    zoom_context.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not logs["pageerror"], logs["pageerror"]
    assert not logs["csp"], logs["csp"]
    assert not [event for event in logs["http_error"] if urlparse(event["url"]).hostname == "127.0.0.1"]
    results["status"] = "passed"
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    """解析构建目录与产物路径并执行真实浏览器验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    if not (args.site_dir / "index.html").is_file():
        parser.error("--site-dir 需要先由 python -m mkdocs build --strict 生成")
    results = _run(args.site_dir.resolve(), args.output.resolve(), args.channel)
    print(json.dumps({"status": results["status"], "browser_version": results["browser_version"], "viewports": len(results["viewports"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
