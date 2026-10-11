"""实际检查公众号 HTML 在手机与桌面上的图片、表格、代码和正文。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


def main() -> None:
    """读取本地文章并保存各视口截图和机器布局检查，不发布内容。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        try:
            for width in (375, 390, 1440):
                page = browser.new_page(viewport={"width": width, "height": 900})
                errors: list[str] = []
                page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                page.goto(args.html.resolve().as_uri(), wait_until="load")
                page.wait_for_function("() => [...document.images].every(i=>i.complete && i.naturalWidth>0)")
                record: dict[str, Any] = page.evaluate("""() => ({
                    width:innerWidth, scroll:document.documentElement.scrollWidth,
                    images:[...document.images].map(e=>({source:e.getAttribute('src'),
                        width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height,
                        nativeWidth:e.naturalWidth,nativeHeight:e.naturalHeight})),
                    tables:[...document.querySelectorAll('table')].map(e=>({width:e.getBoundingClientRect().width,
                        scroll:e.scrollWidth,font:getComputedStyle(e).fontSize})),
                    code:[...document.querySelectorAll('pre')].map(e=>({width:e.getBoundingClientRect().width,
                        scroll:e.scrollWidth,font:getComputedStyle(e).fontSize})),
                    h1:[...document.querySelectorAll('h1')].map(e=>getComputedStyle(e).fontSize)
                })""")
                assert record["scroll"] <= width + 1, record
                assert record["images"] and len(record["images"]) <= 3, record
                for image in record["images"]:
                    assert image["width"] <= width + 1
                    assert abs(image["width"] / image["height"] - image["nativeWidth"] / image["nativeHeight"]) < 0.01
                for table in record["tables"]:
                    assert table["width"] <= width + 1 and table["scroll"] <= table["width"] + 1, table
                assert not errors, errors
                page.screenshot(path=str(args.output / f"wechat-{width}.png"), full_page=True)
                page.screenshot(path=str(args.output / f"wechat-{width}-first-screen.png"))
                page.locator("table").first.screenshot(path=str(args.output / f"wechat-{width}-rules.png"))
                records.append(record)
                page.close()
        finally:
            browser.close()
    (args.output / "wechat-layout.json").write_text(
        json.dumps({"status": "passed", "engine": "Chrome", "layouts": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("WeChat layout passed: 375 / 390 / 1440px")


if __name__ == "__main__":
    main()
