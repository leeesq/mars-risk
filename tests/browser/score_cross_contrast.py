"""通过真实 Chromium、computed styles 和渲染底色检查离线报告文字对比度。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import score_cross
from PIL import Image
from PIL import __version__ as pillow_version
from playwright.sync_api import Browser, Page, expect, sync_playwright

from mars.analysis import evaluate_score_policy, write_score_cross_html
from mars.reporting import load_report

_TEXT_RECORDS = r"""() => {
 const records=[];
 for(const element of document.querySelectorAll('body *')) {
  if(element.closest('script,style,title,option,.visually-hidden'))continue;
  const style=getComputedStyle(element), box=element.getBoundingClientRect();
  if(style.visibility!=='visible'||style.display==='none'||!box.width||!box.height)continue;
  let opacity=1,ancestor=element,clipped=false;
  const ancestors=[];
  while(ancestor){const s=getComputedStyle(ancestor);opacity*=Number(s.opacity);
   ancestors.push({tag:ancestor.tagName,class:ancestor.className.baseVal??ancestor.className,
    opacity:Number(s.opacity),background:s.backgroundColor,image:s.backgroundImage});
   if(ancestor!==element&&/(auto|hidden|scroll|clip)/.test(s.overflow+s.overflowX+s.overflowY)){
    const b=ancestor.getBoundingClientRect();
    if(box.right<=b.left||box.left>=b.right||box.bottom<=b.top||box.top>=b.bottom)clipped=true;
   }ancestor=ancestor.parentElement;
  }
  if(!opacity||clipped)continue;
  if(element.matches('input[type="text"],select,textarea')){
   const value=element.matches('select')?element.selectedOptions[0]?.textContent:element.value;
   if(value){const horizontal=Number.parseFloat(style.paddingLeft)+Number.parseFloat(style.paddingRight),
    vertical=Number.parseFloat(style.paddingTop)+Number.parseFloat(style.paddingBottom);
    records.push({text:String(value).slice(0,200),tag:element.tagName,
     selector:element.id?'#'+element.id:element.tagName.toLowerCase(),probe:null,
     color:style.color,opacity,font_size:Number.parseFloat(style.fontSize),
     font_weight:style.fontWeight,disabled:element.matches(':disabled'),ancestors,
     rect:{x:box.x+scrollX+Number.parseFloat(style.paddingLeft),
      y:box.y+scrollY+Number.parseFloat(style.paddingTop),width:box.width-horizontal,height:box.height-vertical}});
   }
  }
  const nodes=[...element.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE&&n.textContent.trim());
  for(const node of nodes){
   const range=document.createRange();range.selectNodeContents(node);
   for(const r of range.getClientRects()) {
    if(!r.width||!r.height)continue;
    records.push({text:node.textContent.trim().slice(0,200),tag:element.tagName,
     probe:element.closest('[data-probe]')?.dataset.probe??null,
     selector:element.id?'#'+element.id:element.tagName.toLowerCase()+'.'+
      String(element.className.baseVal??element.className).trim().replace(/\s+/g,'.'),
     color:element instanceof SVGElement?style.fill:style.color,opacity,
     font_size:Number.parseFloat(style.fontSize),font_weight:style.fontWeight,
     disabled:element.matches(':disabled'),ancestors,
     rect:{x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height}});
   }
  }
 }
 return records;
}"""


def _rgba(value: str) -> tuple[float, float, float, float]:
    """解析 Chromium 序列化的 rgb/rgba 颜色，保留透明度。"""
    values = [float(value) for value in re.findall(r"[\d.]+", value)]
    if len(values) not in (3, 4):
        raise ValueError(f"不支持的 computed color: {value!r}")
    return values[0], values[1], values[2], values[3] if len(values) == 4 else 1.0


def _luminance(rgb: tuple[float, float, float]) -> float:
    """按 WCAG sRGB 转换计算相对亮度。"""
    linear = [
        channel / 255 / 12.92
        if channel / 255 <= 0.04045
        else ((channel / 255 + 0.055) / 1.055) ** 2.4
        for channel in rgb
    ]
    return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def _contrast(foreground: tuple[float, float, float], background: tuple[int, ...]) -> float:
    """返回两个实际合成颜色的亮度对比。"""
    left = _luminance(foreground)
    right = _luminance((background[0], background[1], background[2]))
    return (max(left, right) + 0.05) / (min(left, right) + 0.05)


def _capture(page: Page, path: Path) -> None:
    """原生缩放时以 CDP 物理范围截图，避开 Playwright 的 CSS clip 截半问题。"""
    sizes = page.evaluate(
        "({width:Math.max(innerWidth,document.documentElement.scrollWidth),"
        "height:Math.max(innerHeight,document.documentElement.scrollHeight),dpr:devicePixelRatio})"
    )
    if sizes["dpr"] == 1:
        page.screenshot(path=str(path), full_page=True)
        return
    session = page.context.new_cdp_session(page)
    try:
        screenshot = session.send("Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": True, "fromSurface": True,
            "clip": {"x": 0, "y": 0, "width": sizes["width"] * sizes["dpr"],
                     "height": sizes["height"] * sizes["dpr"], "scale": 1},
        })
        path.write_bytes(base64.b64decode(screenshot["data"]))
    finally:
        session.detach()


def _pixels(image: Image.Image) -> Iterable[tuple[int, int, int]]:
    """只读 RGB 像素，兼容旧 Pillow 的 getdata 与新 flattened API。"""
    getter = getattr(image, "get_flattened_data", image.getdata)
    return cast(Iterable[tuple[int, int, int]], getter())


def _measure(
    page: Page, output: Path, name: str, *, reset_scroll: bool = True,
) -> dict[str, Any]:
    """捕获真实底色并逐可见文字计算最差值，包含纹理和祖先透明度。"""
    if reset_scroll:
        page.evaluate("scrollTo(0,0)")
    records: list[dict[str, Any]] = page.evaluate(_TEXT_RECORDS)
    screenshot = output / f"{name}.png"
    _capture(page, screenshot)
    # 只改变验收页面的文字绘制，保留背景、纹理、选框、布局与 opacity。
    style = page.add_style_tag(content=(
        "body *{color:transparent!important;-webkit-text-fill-color:transparent!important;"
        "text-shadow:none!important}svg text{fill:transparent!important}"
    ))
    background_path = output / f"{name}-background.png"
    _capture(page, background_path)
    style.evaluate("e=>e.remove()")
    original = Image.open(screenshot).convert("RGB")
    background = Image.open(background_path).convert("RGB")
    css_width = page.evaluate("Math.max(innerWidth,document.documentElement.scrollWidth)")
    scale = original.width / css_width
    measured: list[dict[str, Any]] = []
    for record in records:
        rect = record["rect"]
        left = max(0, math.ceil(rect["x"] * scale))
        top = max(0, math.ceil(rect["y"] * scale))
        right = min(background.width, math.floor((rect["x"] + rect["width"]) * scale))
        bottom = min(background.height, math.floor((rect["y"] + rect["height"]) * scale))
        if right <= left or bottom <= top:
            continue
        red, green, blue, alpha = _rgba(record["color"])
        alpha *= record["opacity"]
        foreground_pixels = _pixels(original.crop((left, top, right, bottom)))
        background_pixels = _pixels(background.crop((left, top, right, bottom)))
        # 只采样真实字形覆盖的位置，避免把边框或邻近图表线误认为字后背景。
        pixels = {
            rgb for actual, rgb in zip(foreground_pixels, background_pixels)
            if max(abs(one - two) for one, two in zip(actual, rgb)) >= 3
        }
        if not pixels:
            continue
        # 有背景的透明祖先需要额外群组合成；当前模板须不触发未支持的前提。
        unsupported_opacity = [
            ancestor for ancestor in record["ancestors"]
            if ancestor["opacity"] < 1 and _rgba(ancestor["background"])[3] > 0
            and any(tuple(_rgba(ancestor["background"])[:3]) != rgb for rgb in pixels)
        ]
        worst = min(
            (
                _contrast(
                    tuple(alpha * fg + (1 - alpha) * bg for fg, bg in zip((red, green, blue), rgb)),
                    rgb,
                ),
                rgb,
            )
            for rgb in pixels
        )
        weight = int(record["font_weight"])
        large = record["font_size"] >= 24 or (record["font_size"] >= 18.6667 and weight >= 700)
        threshold = 3.0 if large else 4.5
        measured.append({**record, "contrast": worst[0], "background": worst[1],
                         "threshold": threshold, "large_text": large,
                         "unsupported_background_group_opacity": unsupported_opacity})
    failures = [record for record in measured if record["contrast"] < record["threshold"]
                or record["unsupported_background_group_opacity"]]
    return {"state": name, "screenshot": str(screenshot), "background": str(background_path),
            "text_count": len(measured), "worst": min(measured, key=lambda item: item["contrast"])
            if measured else None, "failures": failures, "measurements": measured}


def _continuous_scale(page: Page, output: Path) -> dict[str, Any]:
    """用生产 color() 遍历完整 RGB 舍入区间，真实渲染普通与纹理组合探针。"""
    metadata: dict[str, Any] = page.evaluate(r"""() => {
      const oldMode=state.mode,holder=document.createElement('section');
      holder.className='card';holder.id='contrast-probes';
      holder.style.cssText='margin:15px;padding:12px;display:grid;grid-template-columns:repeat(8,102px);gap:5px';
      const metadata={scales:{...scales},modes:{}};
      for(const mode of ['delta','rate'])for(const sign of mode==='delta'?[-1,1]:[1]){
        state.mode=mode;
        const c=value=>({delta_vs_row:value,bad_rate:value});
        const start=color(c(sign*Number.EPSILON*scales[mode]))[0],end=color(c(sign*scales[mode]))[0];
        const rgb=value=>value.match(/\d+/g).map(Number);
        const a=rgb(start),b=rgb(end),boundaries=new Set([0,1]);
        a.forEach((v,i)=>{const n=Math.abs(b[i]-v);for(let j=0;j<n;j++)boundaries.add((j+.5)/n)});
        const sorted=[...boundaries].sort((a,b)=>a-b),samples=[...sorted];
        sorted.slice(1).forEach((v,i)=>samples.push((v+sorted[i])/2));
        const colors=new Map();samples.forEach(f=>{
          const value=sign*f*scales[mode],pair=color(c(value));colors.set(pair[0],{value,pair});
        });
        const extreme=color(c(sign));colors.set(extreme[0],{value:sign,pair:extreme});
        metadata.modes[mode+sign]={start,end,rounding_intervals:sorted.length-1,distinct_rgb:colors.size};
        colors.forEach(({value,pair},index)=>{
          for(const texture of [false,true]){
            const button=document.createElement('button');
            button.className='cell num '+(texture?'low_sample selected rule-hit':'valid');
            button.dataset.probe=mode+':'+sign+':'+index+':'+texture;
            button.style.setProperty('--cell-bg',pair[0]);button.style.setProperty('--cell-ink',pair[1]);
            if(texture){const flag=document.createElement('span');flag.className='flag';flag.textContent='低 n';button.append(flag)}
            const strong=document.createElement('strong');strong.textContent=mode==='delta'?pp(value):pct(value);
            const metrics=document.createElement('span');metrics.className='cell-metrics';metrics.textContent='坏账 0% · Lift 0.00';
            const volume=document.createElement('span');volume.className='cell-volume';volume.textContent='n 1 · 0.1%';
            button.append(strong,metrics,volume);holder.append(button);
          }
        });
      }
      state.mode=oldMode;document.body.append(holder);
      return metadata;
    }""")
    page.add_style_tag(content="body>.topbar,body>.shell{display:none!important}")
    result = _measure(page, output, "continuous-rgb-scale-probes")
    result["probe_metadata"] = metadata
    result["provenance"] = "synthetic rendered DOM probe; production color() and computed CSS"
    return result


def _fixture(
    browser: Browser, html: Path, output: Path, name: str, *, all_scopes: bool,
    probes: bool, width: int = 1440,
) -> list[dict[str, Any]]:
    """检查真实 fixture 的普通、展开、规则选中、hover、错误和禁用状态。"""
    context = browser.new_context(locale="zh-CN", viewport={"width": width, "height": 1000})
    score_cross._observe(context)
    page = context.new_page()
    states: list[dict[str, Any]] = []
    try:
        page.goto(html.as_uri())
        states.append(_measure(page, output, name + "-initial-disabled"))
        if page.locator("#matrix .cell").count():
            page.locator("#help-btn").click()
            for details in page.locator("details").all():
                details.locator("summary").click()
            if page.locator("#policy option").count() > 1:
                page.locator("#policy").select_option("0")
            page.locator("#rule-expression").fill("X >= 1")
            page.locator(".rule-apply").click()
            page.locator("#matrix .cell").first.click()
            page.locator("#matrix .cell").first.hover()
            states.append(_measure(page, output, name + "-delta-hit-selected-hover-expanded"))
            page.locator("#metric-seg button[data-mode='rate']").click()
            page.locator("#matrix .cell").last.click()
            states.append(_measure(page, output, name + "-rate-last-selected"))
            page.locator("#rule-expression").fill("X <= Y2")
            page.locator(".rule-apply").click()
            states.append(_measure(page, output, name + "-invalid-rule"))
            if all_scopes:
                payload = json.loads(page.locator("#data").text_content() or "{}")
                scopes = [{key: row[key] for key in score_cross._SCOPE}
                          for row in payload["tables"]["overall"]]
                for index, scope in enumerate(scopes):
                    score_cross._select_scope(page, scope)
                    for mode in ("delta", "rate"):
                        page.locator(f"#metric-seg button[data-mode='{mode}']").click()
                        cells = page.locator("#matrix .cell")
                        for cell in (cells.first, cells.last):
                            cell.click()
                            cell.press("ArrowRight")
                        states.append(_measure(page, output, f"{name}-scope{index}-{mode}"))
            if probes:
                states.append(_continuous_scale(page, output))
    finally:
        context.close()
    return states


def _native_zoom(playwright: Any, html: Path, output: Path, channel: str) -> dict[str, Any]:
    """通过 Chrome 设置真实缩放到 200%，采样实际显示的文字和底色。"""
    context = playwright.chromium.launch_persistent_context(
        tempfile.mkdtemp(prefix="mars-contrast-zoom-"), channel=channel, headless=True,
        no_viewport=True, args=["--window-size=1440,1000"], locale="zh-CN",
    )
    page = context.pages[0]
    try:
        page.goto("chrome://settings/appearance")
        page.locator("#zoomLevel").select_option(label="200%")
        expect(page.locator("#zoomLevel")).to_have_value("2")
        page.goto(html.as_uri())
        sizes = page.evaluate(
            "({outer:outerWidth,inner:innerWidth,dpr:devicePixelRatio,visual:visualViewport.scale})"
        )
        assert sizes["dpr"] == 2 and sizes["visual"] == 1, sizes
        page.locator("#help-btn").click()
        page.locator(".reading-help > summary").click()
        page.locator("#rule-expression").fill("X >= 1")
        page.locator(".rule-apply").click()
        page.locator("#matrix .cell").first.click()
        result = _measure(page, output, "standard-native-zoom200")
        result["native_zoom"] = {"mechanism": "chrome://settings/appearance 200%", "sizes": sizes}
        result["layout"] = score_cross._overflow(page)
        return result
    finally:
        context.close()


def _copy_feedback(browser: Browser, html: Path, output: Path) -> list[dict[str, Any]]:
    """真实点击复制，分别捕获成功及环境拒绝时的可见 toast。"""
    states: list[dict[str, Any]] = []
    for rejected in (False, True):
        context = browser.new_context(
            locale="zh-CN", viewport={"width": 1440, "height": 1000},
            permissions=["clipboard-read", "clipboard-write"],
        )
        if rejected:
            context.add_init_script(
                "Object.defineProperty(navigator,'clipboard',{value:{writeText:()=>"
                "Promise.reject(new DOMException('denied','NotAllowedError'))}})"
            )
        page = context.new_page()
        try:
            page.goto(html.as_uri())
            page.locator("#copy-btn").click()
            expected = "复制不可用，请手动选择并复制文本" if rejected else "证据已复制"
            expect(page.locator("#toast")).to_have_text(expected)
            states.append(_measure(page, output, "copy-feedback-" + ("rejected" if rejected else "success")))
            assert any(item["selector"] == "#toast" for item in states[-1]["measurements"])
        finally:
            context.close()
    return states


def _hover_controls(browser: Browser, html: Path, output: Path) -> list[dict[str, Any]]:
    """逐个真实控件保持鼠标 hover，检查初始禁用和规则选中组合的文字背景。"""
    context = browser.new_context(locale="zh-CN", viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    states: list[dict[str, Any]] = []
    try:
        page.goto(html.as_uri())
        # 滚动保持在真实 hovered 控件的位置，不把顶部重置造成的 hover 丢失算通过。
        for index, button in enumerate(page.locator(".btn,.seg button").all()):
            button.hover()
            assert button.evaluate("e=>e.matches(':hover')")
            result = _measure(page, output, f"control-hover-{index}", reset_scroll=False)
            result["hover_control"] = button.evaluate(
                "e=>({text:e.textContent,classes:e.className,disabled:e.disabled,hover:e.matches(':hover')})"
            )
            assert result["hover_control"]["hover"]
            states.append(result)
        page.locator("#rule-expression").fill("X >= 1")
        page.locator(".rule-apply").click()
        for status in score_cross._STATUS:
            cell = page.locator(f"#matrix .cell.{status}").first
            if not cell.count():
                continue
            cell.click()
            cell.hover()
            assert cell.evaluate("e=>e.matches(':hover')&&e.classList.contains('selected')")
            result = _measure(page, output, f"cell-{status}-hit-selected-hover", reset_scroll=False)
            result["hover_control"] = cell.evaluate(
                "e=>({text:e.textContent,classes:e.className,hover:e.matches(':hover')})"
            )
            assert result["hover_control"]["hover"]
            states.append(result)
    finally:
        context.close()
    return states


def _reexport(fixture: dict[str, Any], path: Path) -> None:
    """加载现有快照并按原 HTML 声明的公开 policy 重导出，保留回放文字覆盖。"""
    report = load_report(fixture["snapshot"])
    old_html = Path(fixture["html"]).read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', old_html, re.S)
    if match is None:
        raise ValueError(f"找不到现有 fixture 的聚合 payload: {fixture['html']}")
    payload = json.loads(match.group(1))
    policies = []
    for policy in payload.get("policies", []):
        parameters = policy["description"]["parameters"]
        policies.append(evaluate_score_policy(
            report, parameters["candidate"], baseline=parameters["baseline"]
        ))
    write_score_cross_html(report, path, report_name=fixture["title"], policy_reports=policies)


def main() -> int:
    """运行手工浏览器门禁并保存真实测量记录。

    Returns
    -------
    int
        全部可见文字通过时返回 0，否则返回 1。

    Examples
    --------
    在仓库根目录运行 ``python tests/browser/score_cross_contrast.py --help`` 查看参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", nargs="+", default=None)
    parser.add_argument("--reexport", action="store_true")
    parser.add_argument("--all-scopes", action="store_true")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--narrow", action="store_true")
    parser.add_argument("--zoom200", action="store_true")
    parser.add_argument("--copy-feedback", action="store_true")
    parser.add_argument("--hover-controls", action="store_true")
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    log: dict[str, Any] = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": str(args.manifest.resolve()), "reexport": args.reexport,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pillow_version": pillow_version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "src/mars/analysis/_score_cross_html.py").read_bytes()
        ).hexdigest(),
        "html_sha256": {},
        "method": "computed color/fill/font/ancestor opacity; text-transparent Chromium "
                  "PNG samples in actual text rectangles, including heat/texture/background",
        "ordinary_threshold": 4.5,
        "large_threshold": "3 only at >=24px, or >=18.6667px with weight>=700",
        "states": [],
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        log["browser_version"] = browser.version
        try:
            for name, fixture in manifest["fixtures"].items():
                if args.fixtures and name not in args.fixtures:
                    continue
                html = Path(fixture["html"])
                if args.reexport:
                    html = args.output / f"{name}.html"
                    _reexport(fixture, html)
                log["html_sha256"][name] = hashlib.sha256(html.read_bytes()).hexdigest()
                log["states"].extend(_fixture(
                    browser, html, args.output, name,
                    all_scopes=args.all_scopes, probes=args.probes and name == "standard",
                ))
                if args.narrow:
                    log["states"].extend(_fixture(
                        browser, html, args.output, name + "-390",
                        all_scopes=False, probes=False, width=390,
                    ))
                if args.zoom200 and name == "standard":
                    log["states"].append(_native_zoom(playwright, html, args.output, args.channel))
                if args.copy_feedback and name == "standard":
                    log["states"].extend(_copy_feedback(browser, html, args.output))
                if args.hover_controls and name == "standard":
                    log["states"].extend(_hover_controls(browser, html, args.output))
        finally:
            browser.close()
    measurements = [item for state in log["states"] for item in state["measurements"]]
    log["worst"] = min(measurements, key=lambda item: item["contrast"]) if measurements else None
    log["failure_count"] = sum(len(state["failures"]) for state in log["states"])
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["source_unchanged"] = log["source_sha256"] == hashlib.sha256(
        (Path(__file__).resolve().parents[2] / "src/mars/analysis/_score_cross_html.py").read_bytes()
    ).hexdigest()
    log["status"] = "passed" if not log["failure_count"] and log["source_unchanged"] else "failed"
    path = args.output / "contrast.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": log["status"], "failure_count": log["failure_count"],
                      "worst": log["worst"], "artifact": str(path)}))
    return int(log["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
