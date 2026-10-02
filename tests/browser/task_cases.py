"""七任务页面、真实产品截图和同源交叉证据的 Chromium 验收。"""

from __future__ import annotations

import argparse
import functools
import html
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
    _check_number,
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
    """先截真实矩阵，再在可维护排版中展示同格公共查询，不改报告数字。"""
    report = load_report(assets / "score-cross.marsreport")
    first = report.get_table("overall", filters={"group": "discovery", "target": "bad30"}, limit=1).to_dicts()[0]
    scope = {key: first[key] for key in ("group", "target", "period")}
    context = browser.new_context(viewport={"width": 1150, "height": 1000}, locale="zh-CN")
    page = context.new_page()
    page.goto((assets / "score-cross.html").as_uri())
    _select_scope(page, scope)
    cell = report.get_table("cells", filters={**scope, "x_bin": "b1", "y_bin": "b3"}).to_dicts()[0]
    page.locator(f'#matrix button[data-r="{cell["x_risk_rank"]-1}"][data-c="{cell["y_risk_rank"]-1}"]').click()
    page.locator(".maincol > section.card").first.screenshot(path=str(assets / "score-cross-matrix.png"))
    reference = json.loads(page.locator("#evidence-id").inner_text())
    assert reference["report_id"] == report.report_id
    row = report.query_page(reference["table"], **reference["query"])["data"].to_dicts()[0]
    for identifier, field, scale, digits in (
        ("detail-rate", "bad_rate", 100, 2), ("detail-delta", "delta_vs_row", 100, 2),
        ("detail-lift", "lift_vs_overall", 1, 2), ("detail-n", "sample_count", 1, 0),
        ("detail-obs", "observed_sample_count", 1, 0),
    ):
        _check_number(page.locator("#" + identifier).inner_text(), row[field], scale, digits)
    excerpt = {"report_id": report.report_id, "table": "cells", "scope": scope,
               "x_bin": row["x_bin"], "y_bin": row["y_bin"],
               **{field: row[field] for field in ("status", "sample_count", "observed_sample_count", "bad_sample_count", "observed_weight_sum", "bad_weight_sum", "bad_rate", "delta_vs_row", "lift_vs_overall")}}
    _json(assets / "preview-evidence.json", {"reference": reference, "row": row, "excerpt": excerpt,
                                            "method": "real browser matrix screenshot + public query; layout only"})
    # 排版仅组合真实截图与真实有限 JSON，手机只裁切标签之外的空白，不绘制新矩阵。
    escaped = html.escape(json.dumps(excerpt, ensure_ascii=False, indent=2))
    viewer = assets / "preview.html"
    viewer.write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MARS — 同一报告的人工与 Agent 证据</title><style>
*{{box-sizing:border-box}}body{{margin:0;color:#24203b;background:#f4f0fc;font:16px/1.5 "Segoe UI","Microsoft YaHei",sans-serif}}
main{{padding:22px}}header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}}
h1{{font-size:23px;margin:0}}header span{{color:#69538c;font-size:13px}}.content{{display:grid;grid-template-columns:minmax(0,1fr) 355px;gap:14px}}
.panel{{background:white;border:1px solid #ded5ef;border-radius:12px;overflow:hidden}}.panel h2{{font-size:15px;margin:0;padding:10px 15px;background:#faf8ff}}
.matrix img{{width:100%;display:block}}pre{{font:13px/1.6 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:14px;margin:0}}
footer{{font-size:12px;color:#69538c;padding:12px 2px 0}}@media(max-width:600px){{main{{padding:12px}}header{{display:block}}h1{{font-size:20px}}.content{{grid-template-columns:1fr}}.matrix{{overflow:auto}}.matrix img{{width:680px;max-width:none}}pre{{font-size:13px}}}}
</style><main><header><h1>一个 Report · 人工矩阵与 Agent 证据</h1><span>MARS / SYNTHETIC · discovery / bad30 / {scope['period']}</span></header>
<div class="content"><section class="panel matrix"><h2>真实 Score Cross 矩阵 · Δ 单位 pp</h2><img src="score-cross-matrix.png" alt="真实交叉矩阵"></section>
<section class="panel"><h2>同一选格 · 公共 query_page 摘录</h2><pre>{escaped}</pre></section></div>
<footer>截图和 JSON 来自同一 .marsreport · 加权率分母为有效标签权重；完整行及引用见 preview-evidence.json · 合成数据</footer></main></html>''', encoding="utf-8")
    page.set_viewport_size({"width": 1120, "height": 1000})
    page.goto(viewer.as_uri())
    page.locator("main").screenshot(path=str(assets / "readme-preview.png"))
    # 窄屏预览保留同一格的身份、范围、单位与状态，使用原生详情而不裁掉矩阵轴。
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto((assets / "score-cross.html").as_uri())
    _select_scope(page, scope)
    page.locator(f'#matrix button[data-r="{cell["x_risk_rank"]-1}"][data-c="{cell["y_risk_rank"]-1}"]').click()
    detail = page.locator("aside.detail")
    detail.scroll_into_view_if_needed()
    box = detail.bounding_box()
    assert box is not None
    page.screenshot(path=str(assets / "score-cross-cell.png"), clip={**box, "height": 425})
    mobile_excerpt = {"table": "cells", "scope": scope, "x_bin": row["x_bin"], "y_bin": row["y_bin"],
                      "status": row["status"], "bad_rate": row["bad_rate"],
                      "delta_vs_row": row["delta_vs_row"], "full_evidence": "preview-evidence.json"}
    mobile_viewer = assets / "preview-mobile.html"
    mobile_viewer.write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f0fc;color:#24203b;font:15px/1.45 "Microsoft YaHei",sans-serif}}
main{{padding:12px}}h1{{font-size:19px;margin:0 0 8px}}p{{font-size:12px;margin:6px 0}}
img{{width:100%;display:block;border-radius:8px}}pre{{font:12px/1.5 Consolas,monospace;background:#fff;border:1px solid #ded5ef;border-radius:8px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}}
</style><main><h1>同一报告 · 真实选格与 Agent 证据</h1><p>discovery / bad30 / {scope['period']} · 移动预览为选格详情；完整矩阵见桌面图</p>
<img src="score-cross-cell.png" alt="真实选格详情"><pre>{html.escape(json.dumps(mobile_excerpt, ensure_ascii=False, indent=2))}</pre>
<p>公共 query_page 真实摘录 · 完整身份与引用见 preview-evidence.json · 合成数据</p></main></html>''', encoding="utf-8")
    page.goto(mobile_viewer.as_uri())
    page.locator("main").screenshot(path=str(assets / "readme-preview-mobile.png"))
    context.close()
    return {"reference": reference, "key_fields": excerpt, "matrix_source": "native HTML report screenshot",
            "layout": "preview.html; JSON excerpt is labeled, complete original linked"}


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
    """按阶段生成真实预览并对最终严格构建做浏览器验收。"""
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
