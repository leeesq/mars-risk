"""从公共统计构造单一人口的原生展示视图，不恢复 estimator。"""

from __future__ import annotations

from copy import deepcopy
from html import escape
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl

from mars.feature._scopes import resolve_scopes
from mars.reporting._query import query_table


def scope_views(
    report: Any, segments: list[Any] | None = None, include_all: bool = True,
) -> list[tuple[str, Any]]:
    """按目录解析人口，投影各事实表并隔离日期元数据，沿用原生 adapter。"""
    tables = report._query_tables()
    if "scopes" not in tables:
        if segments is not None or not include_all:
            raise ValueError("This report has no segment scopes.")
        return [("all", report)]
    catalog = tables["scopes"]
    if not isinstance(catalog, pl.DataFrame):
        catalog = pl.from_pandas(catalog)
    results: list[tuple[str, Any]] = []
    selected = resolve_scopes(catalog, segments, include_all)
    if segments is None and include_all:
        selected = [report._query_metadata().get("default_scope_id", "all")]
    for scope_id in selected:
        scoped = {
            name: query_table(frame, filters={"scope_id": scope_id})
            if "scope_id" in frame.columns else frame
            for name, frame in tables.items() if name != "scopes"
        }
        # 原生模板不增加客群列；过滤已在公共事实范围中完成。
        scoped = {
            name: frame.drop("scope_id") if isinstance(frame, pl.DataFrame)
            and "scope_id" in frame.columns else frame.drop(columns=["scope_id"])
            if "scope_id" in frame.columns else frame
            for name, frame in scoped.items()
        }
        meta = deepcopy(report._query_metadata())
        meta.update(meta.pop("scope_metadata", {}).get(scope_id, {}))
        row = catalog.filter(pl.col("scope_id") == scope_id).to_dicts()[0]
        meta["selected_scope"] = row
        meta["scope_caption"] = row["display_name"]
        meta["scope_table_references"] = {
            name: {"report_id": report.report_id, "table": name,
                   "query": {"filters": {"scope_id": scope_id}}}
            for name, frame in tables.items() if name != "scopes" and "scope_id" in frame.columns
        }
        if report.report_type == "MarsBinningReport":
            from mars.reporting.binning_report import MarsBinningReport

            view: Any = MarsBinningReport(
                scoped["summary"],
                {name[6:]: frame for name, frame in scoped.items() if name.startswith("trend.")},
                scoped["detail"], group_col=meta.get("group_col", getattr(report, "group_col", None)),
                detail_group_col=meta.get("detail_group_col", getattr(report, "detail_group_col", None)),
                dt_col=meta.get("dt_col", getattr(report, "dt_col", None)),
                missing_by_day_table=scoped.get("missing_by_day"),
                risk_corr_reference_table=scoped.get("risk_corr_reference"),
                report_meta=meta, feature_metadata=report.feature_metadata,
                business_context=report.business_context,
            )
        elif "overview" in scoped:
            from mars.reporting.profile_report import MarsProfileReport

            view = MarsProfileReport(
                scoped["overview"],
                {n[3:]: f for n, f in scoped.items() if n.startswith("dq.")},
                {n[6:]: f for n, f in scoped.items() if n.startswith("stats.")},
                comparison_tables={n[11:]: f for n, f in scoped.items() if n.startswith("comparison.")},
                report_meta=meta, feature_metadata=report.feature_metadata,
                business_context=report.business_context,
            )
        else:
            raise ValueError("Native scope views require binning or profile statistics.")
        view._calculation_status = scoped.get("calculation_status")
        view.report_id = report.report_id
        view.source = deepcopy(report.source)
        results.append((scope_id, view))
    return results


def render_scopes(report: Any, method: str, options: dict[str, Any]) -> Any:
    """让各人口沿用原生绘图，实际图数共用上限，文件名包含稳定身份。"""
    from mars.reporting._types import MarsHtmlRenderResult

    segments = options.pop("segments", None)
    include_all = options.pop("include_all", True)
    maximum = options.get("max_plots", 20)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_plots must be a positive integer.")
    results: list[Any] = []
    blocks: list[str] = []
    assets: list[Any] = []
    figures: list[Any] = []
    count = 0
    for identity, view in scope_views(report, segments, include_all):
        if count >= maximum:
            break
        current = {**options, "max_plots": maximum - count}
        if "filename_prefix" in current:
            current["filename_prefix"] = f"{current['filename_prefix']}_{identity}"
        result = getattr(view, method)(**current)
        if isinstance(result, MarsHtmlRenderResult):
            blocks.append(result.html)
            assets.extend(result.assets)
            figures.extend(result.figures or [])
            count += result.html.count('class="mars-risk-trend"')
        else:
            results.extend(result)
            count += len(result)
    if method == "render_risk_trends_html":
        return MarsHtmlRenderResult("".join(blocks), assets, figures if options.get("return_figures") else None)
    return results


def write_scoped_html(
    report: Any, path: str, segments: list[Any] | None, include_all: bool,
    options: dict[str, Any],
) -> None:
    """各人口复用现有离线页面，父级选择器将来源、特征和分组限制在同一人口。"""
    views = scope_views(report, segments, include_all)
    destination = Path(path)
    blocks: list[str] = []
    choices: list[str] = []
    remaining = options.get("max_plots", 500)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        for index, (identity, view) in enumerate(views):
            label = view._query_metadata().get("selected_scope", {}).get("display_name", identity)
            current = dict(options)
            if "max_plots" in current:
                current["max_plots"] = max(remaining, 1)
                if remaining == 0:
                    current["include_charts"] = False
            file = Path(temporary) / f"scope-{index}.html"
            view.write_html(str(file), **current)
            content = file.read_text(encoding="utf-8")
            source_assets = file.with_name(f"{file.stem}_assets")
            if source_assets.is_dir():
                target_assets = destination.parent / f"{destination.stem}_assets" / identity
                target_assets.resolve().relative_to(destination.parent.resolve())
                copytree(source_assets, target_assets, dirs_exist_ok=True)
                content = content.replace(
                    source_assets.name + "/", f"{destination.stem}_assets/{identity}/",
                )
            remaining = max(0, remaining - content.count('class="mars-chart-card"'))
            choices.append(f'<option value="{index}">{escape(label)} [{escape(identity[:12])}]</option>')
            hidden = " hidden" if index else ""
            blocks.append(
                f'<iframe data-scope-id="{identity}" title="{escape(label, quote=True)}"'
                f' srcdoc="{escape(content, quote=True)}"{hidden}></iframe>'
            )
    title = escape(options.get("report_name", "MARS Report"))
    destination.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title}</title><style>body{{margin:0;font-family:Arial}}'
        'header{padding:12px}iframe{border:0;width:100%;height:calc(100vh - 60px)}'
        'iframe[hidden]{display:none}</style></head><body>'
        f'<header><label>客群 <select id="scope-select">{"".join(choices)}</select></label>'
        ' All 为完整人口；Total 为当前客群跨分组汇总。图数受 max_plots 限制，完整统计保留。</header>'
        + ("".join(blocks) or '<p>没有选中的人口。</p>')
        + '<script>const frames=[...document.querySelectorAll("iframe")],'
        'selector=document.getElementById("scope-select");'
        'window.marsSetScopePage=(frame,view,replace)=>{'
        'const hash="#"+new URLSearchParams({scope:frame.dataset.scopeId,view});'
        'if(location.hash!==hash)history[replace?"replaceState":"pushState"](null,"",hash);};'
        'function selectScope(index){frames.forEach((f,i)=>f.hidden=i!==index);selector.value=String(index);}'
        'function applyHash(){const query=new URLSearchParams(location.hash.slice(1));'
        'const index=frames.findIndex(f=>f.dataset.scopeId===query.get("scope"));'
        'if(index>=0)selectScope(index);'
        'const view=query.get("view");const frame=frames[Number(selector.value)];'
        'if(view&&frame&&typeof frame.contentWindow.marsNavigateTo==="function")'
        'frame.contentWindow.marsNavigateTo(view,true);}'
        'selector.addEventListener("change",e=>{selectScope(Number(e.target.value));'
        'window.marsSetScopePage(frames[Number(e.target.value)],"overview",false);});'
        'addEventListener("popstate",applyHash);applyHash();'
        '</script></body></html>', encoding="utf-8",
    )
