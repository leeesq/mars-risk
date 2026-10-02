"""真实 Chromium 周期语义回归，复用已有离线、证据与键盘验收工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import polars as pl
from playwright.sync_api import Browser, Page, sync_playwright

from fixtures import _small_report
from mars.analysis import cross_scores, write_score_cross_html
from mars.reporting import Report, ReportSnapshot, load_report
from score_cross import (
    _apply,
    _assert_rule,
    _check_number,
    _empty,
    _keyboard,
    _observe,
    _overflow,
    _scope,
)

_SCOPE = ("target", "group", "period")


def _time_report(
    *,
    multiple: bool,
    group: bool = True,
    timed: bool = True,
    mixed_group: bool = False,
    unresolved: bool = False,
    mixed_null: bool = False,
) -> Report:
    """以公共入口生成两个目标、真实样本集和一个或两个日期周期。"""
    dates: list[str | None] = (
        (["2026-01-01"] * 4 + ["2026-02-01"] * 4) * 2 if multiple else ["2026-01-01"] * 16
    )
    if mixed_group:
        dates[8:] = ["2026-01-01"] * 8
    if unresolved:
        dates = [None] * 16
    if mixed_null:
        dates[:4] = [None] * 4
        dates[4:] = ["2026-01-01"] * 12
    frame: pl.DataFrame = pl.DataFrame(
        {
            "x": [0.1, 0.2, 0.8, 0.9] * 4,
            "y": [0.1, 0.8, 0.2, 0.9] * 4,
            "bad": [0, 1, 1, 0] * 4,
            "later": [1, 0, None, 0] * 4,
            "cohort": ["Total"] * 8 + ["OOT"] * 8,
            "date": pl.Series(dates, dtype=pl.String),
        }
    )
    return cross_scores(
        frame,
        score_x="x",
        score_y="y",
        score_directions={"x": "lower_risk", "y": "higher_risk"},
        targets=["bad", "later"],
        cutpoints={"x": [0.5], "y": [0.5]},
        group_col="cohort" if group else None,
        time_col="date" if timed else None,
        time_grain="month" if timed else None,
        min_observed=2,
    )


def _compatible_snapshot(
    report: Report,
    *,
    period: str | None = None,
    replace_period: bool = False,
    remove_metadata: bool = False,
    drop_fields: tuple[str, ...] = (),
) -> ReportSnapshot:
    """构造旧契约或任意合法标签快照；所有统计、分母和证据行原样保留。"""
    description: dict[str, Any] = report.describe()
    if remove_metadata:
        for key in ("scope_dimensions", "time_col", "time_grain"):
            description["parameters"].pop(key, None)
    for key in drop_fields:
        description["parameters"].pop(key, None)
    tables: dict[str, Any] = {name: report.get_table(name) for name in description["tables"]}
    if replace_period:
        for name, table in tables.items():
            if "period" in table.columns:
                tables[name] = table.with_columns(pl.lit(period, dtype=pl.String).alias("period"))
    return ReportSnapshot(tables, description)


def _fixtures(output: Path) -> list[dict[str, Any]]:
    """保存并重载公共报告，确保实际浏览器只消费持久化聚合结果。"""
    base = _small_report()
    single = _time_report(multiple=False)
    multiple = _time_report(multiple=True)
    cases: list[tuple[str, Report, str]] = [
        ("no-time-total", base, "none"),
        ("no-time-none", _compatible_snapshot(base, replace_period=True), "none"),
        ("single-time", single, "single"),
        ("multi-time", multiple, "multi"),
        ("time-without-group", _time_report(multiple=True, group=False), "multi"),
        ("single-group-in-multi-report", _time_report(multiple=True, mixed_group=True), "mixed"),
        ("real-time-null", _time_report(multiple=False, unresolved=True), "unparsed"),
        ("one-real-period-plus-null", _time_report(multiple=False, mixed_null=True), "mixed_null"),
        (
            "valid-total-time-label",
            _compatible_snapshot(single, period="Total", replace_period=True),
            "single",
        ),
        ("legacy-unknown-multi", _compatible_snapshot(multiple, remove_metadata=True), "unknown"),
        ("legacy-unknown-total", _compatible_snapshot(base, remove_metadata=True), "unknown"),
        (
            "legacy-explicit-no-time",
            _compatible_snapshot(base, drop_fields=("scope_dimensions",)),
            "none",
        ),
        (
            "legacy-explicit-time",
            _compatible_snapshot(single, drop_fields=("scope_dimensions",)),
            "single",
        ),
        (
            "legacy-partial-time",
            _compatible_snapshot(multiple, drop_fields=("scope_dimensions", "time_grain")),
            "unknown",
        ),
        ("empty", _small_report(empty=True), "empty"),
    ]
    # group_col 和 period 来源独立：真实样本集 Total 不应使无时间报告产生周期。
    cases.append(("valid-total-group-no-time", _time_report(multiple=False, timed=False), "none"))
    fixtures: list[dict[str, Any]] = []
    for name, report, mode in cases:
        snapshot = output / (name + ".marsreport")
        report.save(snapshot, overwrite=True)
        restored = load_report(snapshot)
        assert restored.describe() == report.describe(), name
        for table in report.describe()["tables"]:
            assert restored.get_table(table).equals(report.get_table(table)), (name, table)
        html = output / (name + ".html")
        write_score_cross_html(restored, html, report_name=name)
        fixtures.append({"name": name, "mode": mode, "html": str(html), "snapshot": str(snapshot)})
    return fixtures


def _select(page: Page, scope: dict[str, Any]) -> None:
    """通过真实控件访问原始 scope；既不改表值，也不触及前端 state。"""
    target = "无标签 · 分布分析" if scope["target"] is None else str(scope["target"])
    page.locator("#target").select_option(label=target)
    group = "全部样本" if scope["group"] is None else str(scope["group"])
    page.get_by_role("group", name="样本集", exact=True).get_by_role(
        "button", name=group, exact=True
    ).click()
    assert page.evaluate("document.activeElement.matches('#dataset-seg button.active')")
    if _scope(page)["period"] != scope["period"]:
        label = "未解析时间" if scope["period"] is None else str(scope["period"])
        page.locator("#period").select_option(label=label)
    assert _scope(page) == scope, (scope, _scope(page))


def _assert_semantics(page: Page, mode: str, scope: dict[str, Any]) -> dict[str, Any]:
    """核对当前真实 DOM 的筛选、范围、详情、阅读提示和规则范围一致性。"""
    reading = page.locator("#reading-period").text_content() or ""
    texts: dict[str, str] = {
        identifier: page.locator("#" + identifier).inner_text()
        for identifier in ("scope-caption", "detail-interval", "custom-rule-result")
    }
    period_visible = page.locator("#period-filter").is_visible()
    period_disabled = page.locator("#period").is_disabled()
    label = page.locator("label[for=period]").text_content()
    if mode == "none":
        assert not period_visible, (mode, label, texts, reading)
        assert "跨期" in reading and "未" in reading and "可切换" not in reading, reading
        for text in texts.values():
            assert "未设置时间维度" in text, text
            assert "period=Total" not in text and "period=未提供" not in text, text
    elif mode == "single":
        assert not period_visible or period_disabled, (mode, label, reading)
        assert "一个周期" in reading or "单周期" in reading, reading
        assert "跨期" in reading and "未" in reading and "可切换" not in reading, reading
        for text in texts.values():
            assert str(scope["period"]) in text, (scope, text)
    elif mode == "multi":
        assert period_visible and not period_disabled, (mode, label, reading)
        assert label == "周期", label
        assert "可切换" in reading, reading
        for text in texts.values():
            assert str(scope["period"]) in text, (scope, text)
    elif mode == "unknown":
        assert "来源" in reading and "未" in reading and "跨期" in reading, reading
        assert "周期可切换" not in reading, reading
        if page.locator("#period option").count() > 1:
            assert period_visible and not period_disabled and label == "分组范围", label
        for text in texts.values():
            assert str(scope["period"]) in text and "来源" in text, text
    elif mode == "unparsed":
        assert not period_visible or period_disabled, (mode, label, reading)
        assert "没有可解析" in reading and "跨期" in reading and "未" in reading, reading
        for text in texts.values():
            assert "未解析时间" in text, text
    elif mode == "single_unparsed":
        assert period_visible and not period_disabled and label == "时间范围", label
        assert "一个已解析周期" in reading and "未解析时间范围" in reading, reading
        assert "未做跨期持续性验证" in reading, reading
        visible_period = "未解析时间" if scope["period"] is None else str(scope["period"])
        for text in texts.values():
            assert visible_period in text, (scope, text)
    return {
        "scope": scope,
        "reading": reading,
        "texts": texts,
        "period_visible": period_visible,
        "period_disabled": period_disabled,
        "label": label,
    }


def _run_fixture(browser: Browser, fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    """逐目标、样本集和保存范围检查语义，保留失败截图并继续其余夹具。"""
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN")
    events = _observe(context)
    page = context.new_page()
    page.set_default_timeout(3000)
    report = load_report(fixture["snapshot"])
    scopes: list[dict[str, Any]] = [
        {key: row[key] for key in _SCOPE} for row in report.get_table("overall").to_dicts()
    ]
    record: dict[str, Any] = {
        **fixture,
        "events": events,
        "scopes": [],
        "layouts": [],
        "snapshot_roundtrip": True,
        "screenshots": [],
    }
    try:
        page.goto(Path(fixture["html"]).as_uri())
        if not scopes:
            _empty(page)
            reading = page.locator("#reading-period").text_content() or ""
            assert "跨期" in reading and "未" in reading, reading
        else:
            for scope in scopes + list(reversed(scopes)):
                _select(page, scope)
                page.locator("#matrix .cell").first.click()
                assert page.evaluate("document.activeElement.matches('#matrix .cell.selected')")
                _apply(page, "X >= 1", enter=True)
                mode = fixture["mode"]
                if mode == "mixed":
                    period_count = len(
                        {
                            row["period"]
                            for row in scopes
                            if row["target"] == scope["target"] and row["group"] == scope["group"]
                        }
                    )
                    mode = "multi" if period_count > 1 else "single"
                elif mode == "mixed_null":
                    available_periods = {
                        row["period"]
                        for row in scopes
                        if row["target"] == scope["target"] and row["group"] == scope["group"]
                    }
                    mode = "single_unparsed" if None in available_periods else "single"
                record["scopes"].append(_assert_semantics(page, mode, scope))
                _assert_rule(page, report, "X >= 1", scope)
                overall = report.get_table("overall", filters=scope).to_dicts()[0]
                _check_number(
                    page.locator("#total-n").inner_text(), overall["sample_count"], digits=0
                )
                cell_reference: dict[str, Any] = json.loads(
                    page.locator("#evidence-id").inner_text()
                )
                assert (
                    report.query_page(cell_reference["table"], **cell_reference["query"])[
                        "data"
                    ].height
                    == 1
                )
                selected = page.locator("#matrix .cell.selected")
                assert "rule-hit" in (selected.get_attribute("class") or "")
                borders = selected.evaluate(
                    "e=>({outer:getComputedStyle(e).outlineStyle,inner:getComputedStyle(e,'::before').borderStyle})"
                )
                assert borders == {"outer": "solid", "inner": "dashed"}, borders
            bins: list[dict[str, Any]] = report.get_table("bins").to_dicts()
            x_count = sum(row["axis"] == "x" and row["kind"] == "normal" for row in bins)
            y_count = sum(row["axis"] == "y" and row["kind"] == "normal" for row in bins)
            _keyboard(page, x_count, y_count)
        page.locator(".reading-help > summary").click()
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            record["layouts"].append(_overflow(page))
            screenshot = output / (fixture["name"] + f"-{width}.png")
            page.screenshot(path=str(screenshot), full_page=True)
            record["screenshots"].append(str(screenshot))
        events["csp"].extend(page.evaluate("window.__acceptanceCsp"))
        assert not any(
            events[key]
            for key in ("pageerror", "console_error", "csp", "external_requests", "requestfailed")
        ), events
        record["status"] = "passed"
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        screenshot = output / (fixture["name"] + "-failure.png")
        page.screenshot(path=str(screenshot), full_page=True)
        record["screenshots"].append(str(screenshot))
        record["failure_dom"] = page.locator(
            "#period-filter,#scope-caption,#detail-interval,#reading-period,#custom-rule-result"
        ).all_text_contents()
    finally:
        context.close()
    return record


def main() -> int:
    """运行离线真浏览器回归，保存逐例实测结果和版本、源码哈希。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--only", action="append", help="仅运行指定夹具名称，可重复传入。")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    template_path = Path(__file__).resolve().parents[2] / "src/mars/analysis/_score_cross_html.py"
    template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    fixtures = _fixtures(args.output)
    assert hashlib.sha256(template_path.read_bytes()).hexdigest() == template_hash, (
        "HTML 模板在生成夹具时被修改，须固定最终源码后重跑。"
    )
    if args.only:
        fixtures = [fixture for fixture in fixtures if fixture["name"] in args.only]
        if not fixtures:
            parser.error("--only 未匹配到任何夹具")
    log: dict[str, Any] = {
        "python": sys.version,
        "os": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "html_template_sha256": template_hash,
        "viewport": {"width": 1440, "height": 1000},
        "narrow": {"width": 390, "height": 1000},
        "channel": args.channel,
        "headless": True,
        "fixtures": [],
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        log["browser_version"] = browser.version
        try:
            for fixture in fixtures:
                record = _run_fixture(browser, fixture, args.output)
                log["fixtures"].append(record)
                print(f"{fixture['name']}: {record['status']}")
        finally:
            browser.close()
    log["unchanged_during_run"] = (
        hashlib.sha256(template_path.read_bytes()).hexdigest() == template_hash
    )
    log["status"] = (
        "passed"
        if (
            all(record["status"] == "passed" for record in log["fixtures"])
            and log["unchanged_during_run"]
        )
        else "failed"
    )
    (args.output / "score-cross-time-browser.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if log["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
