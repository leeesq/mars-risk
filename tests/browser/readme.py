"""在真实 GitHub 页面检查两种 README 的品牌、图片、语言链接及主题视口。"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from playwright.sync_api import expect, sync_playwright


def _render_local(root: Path, directory: Path) -> None:
    """渲染未提交的 README，并将网站资产链接指向当前本地构建。"""
    import markdown

    for name in ("README.md", "README.en.md"):
        source = (root / name).read_text(encoding="utf-8")
        source = source.replace("https://leeesq.github.io/mars-risk/", (root / "site").as_uri() + "/")
        rendered = markdown.markdown(source, extensions=["fenced_code", "tables"])
        html = (
            '<html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{margin:0;background:#fff;color:#24292f}article{max-width:1040px;margin:auto;padding:24px;'
            'font:16px/1.65 system-ui}img{max-width:100%;height:auto}pre,table{max-width:100%;overflow:auto;'
            'display:block}pre{padding:12px;background:#f6f8fa}a{color:#0969da}h1{font-size:28px}'
            '@media(prefers-color-scheme:dark){body{background:#0d1117;color:#c9d1d9}pre{background:#161b22}}</style>'
            '<article class="markdown-body">' + rendered + '</article></html>'
        )
        (directory / (name + ".html")).write_text(html, encoding="utf-8")


def main() -> None:
    """读取指定公开提交；不写入 GitHub 或登录账户。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--local", action="store_true", help="验收未提交的本地两语 README，不访问 GitHub 旧版")
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="mars-local-readme-")
    local_directory = Path(temporary.name)
    if args.local:
        _render_local(Path(__file__).resolve().parents[2], local_directory)
    args.output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                             "playwright": version("playwright"), "engine": "Chromium", "headless": True,
                             "method": "local current Markdown rendering" if args.local else "actual public GitHub branch file rendering", "ref": args.ref,
                             "http_retries": [], "pages": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        record["browser_version"] = browser.version
        try:
            for scheme in ("light", "dark"):
                for width in (1440, 390):
                    context = browser.new_context(color_scheme=scheme,
                        viewport={"width": width, "height": 1000 if width == 1440 else 844})
                    page = context.new_page()
                    page.set_default_timeout(30000)
                    for filename, language in (("README.md", "zh"), ("README.en.md", "en")):
                        url = (local_directory / (filename + ".html")).as_uri() if args.local else f"https://github.com/leeesq/mars-risk/blob/{args.ref}/{filename}"
                        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        for attempt in range(2):
                            if response is None or response.status not in {429, 502, 503, 504}:
                                break
                            record["http_retries"].append({"url": url, "status": response.status})
                            page.screenshot(path=str(args.output / f"http-{response.status}-{language}-{attempt}.png"))
                            page.wait_for_timeout(2000)
                            response = page.goto(url + "?plain=0", wait_until="domcontentloaded", timeout=60000)
                        if not args.local and (response is None or not response.ok):
                            (args.output / "readme-github-results.json").write_text(
                                json.dumps({**record, "status": "unverified", "failed_url": url,
                                            "http_status": response.status if response else None},
                                           ensure_ascii=False, indent=2), encoding="utf-8",
                            )
                        assert args.local or (response is not None and response.ok), (url, response.status if response else None)
                        article = page.locator('article.markdown-body').first
                        expect(article).to_be_visible()
                        expect(article.get_by_role("img", name="MARS", exact=True)).to_be_visible()
                        expect(article.get_by_role("img", name="MODELING ANALYSIS RISK SCORE", exact=True)).to_be_visible()
                        opposite = "README.md" if language == "en" else "README.en.md"
                        switch = article.locator(f'a[href$="/{opposite}"]').first
                        assert switch.count() == 1
                        assert f"/blob/{args.ref}/{opposite}" in (switch.get_attribute("href") or "")
                        preview = article.locator('img[src*="binning-native-home-credit"]')
                        expect(preview).to_be_visible(timeout=30000)
                        page.wait_for_function("() => [...document.querySelectorAll('article.markdown-body img')].filter(i=>/mars-logo|mars-wordmark|binning-native-home-credit/.test(i.src)).every(i=>i.complete && i.naturalWidth>0)")
                        page.wait_for_function("() => [...document.querySelectorAll('article.markdown-body img')].every(i=>i.complete)")
                        images = article.locator("img").evaluate_all(
                            "images=>images.map(i=>({alt:i.alt,source:i.currentSrc,width:i.naturalWidth}))"
                        )
                        for label in ("PyPI", "Docs", "Python", "Downloads", "CI", "License"):
                            assert any(i["alt"] == label and i["width"] > 0 for i in images), (label, images)
                        sizes = article.evaluate("e=>({width:e.clientWidth,scroll:e.scrollWidth})")
                        assert sizes["scroll"] <= sizes["width"] + 1, sizes
                        article.screenshot(path=str(args.output / f"readme-{language}-{width}-{scheme}.png"), timeout=60000)
                        page.evaluate("window.scrollTo(0, 0)")
                        page.screenshot(path=str(args.output / f"readme-{language}-{width}-{scheme}-first-screen.png"))
                        record["pages"].append({"file": filename, "url": page.url, "viewport": width,
                                                "scheme": scheme, "sizes": sizes, "images": images,
                                                "color_mode": page.locator("html").get_attribute("data-color-mode"),
                                                "background": article.evaluate("e=>getComputedStyle(e).backgroundColor"),
                                                "body_background": page.locator("body").evaluate("e=>getComputedStyle(e).backgroundColor"),
                                                "preview_source": preview.evaluate("e=>e.currentSrc")})
                    context.close()
        except Exception as error:
            record.update(status="unverified", error_type=type(error).__name__)
            (args.output / "readme-github-results.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            raise
        finally:
            browser.close()
    record["status"] = "passed"
    (args.output / "readme-github-results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "pages": len(record["pages"]), "browser": record["browser_version"]}))


if __name__ == "__main__":
    main()
