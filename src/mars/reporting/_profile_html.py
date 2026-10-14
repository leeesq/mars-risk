"""画像与通用快照共用的离线 HTML 表格展示。"""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mars.compute import to_pandas_frame
from mars.reporting._binning_html_helpers import interpolate_rgb
from mars.reporting._metadata import export_semantics, is_metadata_column
from mars.utils.html import format_html_value, is_missing_html_value


def _cell_type(value: Any) -> tuple[str, str]:
    """保留原始排序键；字符串类别按文本排序，不解析其展示文本。"""
    if is_missing_html_value(value):
        return "missing", ""
    if isinstance(value, (bool, np.bool_)):
        return "boolean", str(int(value))
    if isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        return "date", pd.Timestamp(value).isoformat()
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return "missing", ""
        return "number", repr(value.item() if isinstance(value, np.generic) else value)
    return "text", str(value)


def _numeric_columns(dataframe: pd.DataFrame, fields: dict[str, Any]) -> list[str]:
    """排除身份、类别、状态、日期与 Boolean，色阶仅消费真实数值。"""
    excluded_units = {
        "identifier", "type", "category", "dimension", "state", "state_reason",
        "original_value", "display", "index", "ordinal",
    }
    return [
        column for column in dataframe.select_dtypes(include="number").columns
        if not is_metadata_column(column)
        and fields.get(column, {}).get("unit") not in excluded_units
    ]


def _table_html(
    name: str, frame: object, table_index: int, fields: dict[str, Any] | None = None,
) -> str:
    """从同一原表渲染格式、原值排序键和轻量色阶，不复制全量 JSON。"""
    dataframe: pd.DataFrame = to_pandas_frame(frame)
    definitions: dict[str, Any] = fields or {}
    numeric = _numeric_columns(dataframe, definitions)
    limits: dict[str, tuple[float, float]] = {}
    for column in numeric:
        values = dataframe[column].dropna()
        values = values[np.isfinite(values)]
        if not values.empty:
            limits[column] = float(values.min()), float(values.max())
    headers = "".join(
        f'<th scope="col" tabindex="0" aria-sort="none" '
        f'title="{escape(str(definitions.get(column, {}).get("meaning", column)), quote=True)}">'
        f'{escape(str(column))}</th>'
        for column in dataframe.columns
    )
    rows: list[str] = []
    metric = name.rsplit(".", 1)[-1]
    for row in dataframe.itertuples(index=False, name=None):
        cells: list[str] = []
        for column, value in zip(dataframe.columns, row):
            kind, raw = _cell_type(value)
            unit = definitions.get(column, {}).get("unit")
            # AUC 的 ratio 单位沿用其数值格式，不能和质量率混同。
            percent = unit in {"ratio", "ratio_difference"} and metric not in {"auc", "auc_bin"} and column not in {"auc", "auc_bin"}
            if column not in numeric:
                percent = False
            if kind == "missing":
                text = "—"
            elif kind == "number" and column not in numeric:
                text = str(value)
            elif kind == "number" and not percent and (
                isinstance(value, (int, np.integer))
                or str(definitions.get(column, {}).get("dtype", "")).lower().startswith(("int", "uint"))
                or unit == "count" and float(value).is_integer()
            ):
                text = f"{int(value):,}"
            else:
                text = format_html_value(value, as_percent=percent, missing_text="—")
            style = ""
            if column in limits and kind == "number":
                low, high = limits[column]
                if high > low:
                    fraction = (float(value) - low) / (high - low)
                    risk = metric in {"missing", "psi"} or column in {"missing", "missing_rate", "psi", "psi_max"}
                    end = (247, 184, 174) if risk else (167, 208, 232)
                    red, green, blue = interpolate_rgb((249, 251, 253), end, fraction)
                    style = f' style="background-color:rgb({red},{green},{blue})"'
            cell_class = ' class="missing"' if kind == "missing" else (
                ' class="metadata-cell"' if is_metadata_column(column) else (
                    ' class="sparkline"' if column == "distribution" else ""
                )
            )
            display = escape(text)
            if is_metadata_column(column) and kind != "missing":
                display = f'<span class="metadata-text">{display}</span>'
            # title 可查未舍入原值；用户字典、类别及描述始终进行属性/文本转义。
            title = escape(str(value) if kind == "missing" else raw, quote=True)
            cells.append(
                f'<td data-sort-type="{kind}" data-sort-value="{escape(raw, quote=True)}"'
                f' title="{title}"{cell_class}{style}>{display}</td>'
            )
        rows.append(f'<tr>{"".join(cells)}</tr>')
    unit_note = '<p class="unit-note">KS uses 0–100 points; rates use percentages. Hover cells for original values.</p>' if any(field.get("unit") == "points_0_100" for field in definitions.values()) else ""
    return (
        f'<section class="panel"><h2>{escape(name)}</h2>{unit_note}'
        f'<input class="table-search" data-table="table-{table_index}" '
        'placeholder="Search this table" aria-label="Search table">'
        f'<div class="table-wrap"><table class="report-table" id="table-{table_index}">'
        f'<thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
    )


_SCRIPT = r"""
const globalSearch=document.getElementById('global-search');
const tableSearches=new Map([...document.querySelectorAll('.table-search')].map(input=>[input.dataset.table,input]));
function filterTable(table){
  const globalQuery=globalSearch.value.toLowerCase(),localQuery=tableSearches.get(table.id)?.value.toLowerCase()||'';
  for(const row of table.tBodies[0]?.rows||[]){
    const text=[...row.cells].map(cell=>cell.textContent).join('\t').toLowerCase();
    row.hidden=!(text.includes(globalQuery)&&text.includes(localQuery));
  }
}
function wireTabs(buttons, panels, key){
  function select(button){
    for(const peer of buttons){
      const active=peer===button;
      peer.setAttribute('aria-selected',String(active));peer.tabIndex=active?0:-1;
    }
    for(const panel of panels) panel.classList.toggle('active',panel.dataset[key]===button.dataset[key]);
  }
  buttons.forEach((button,index)=>{
    button.addEventListener('click',()=>select(button));
    button.addEventListener('keydown',event=>{
      let next=index;
      if(event.key==='ArrowRight'||event.key==='ArrowDown') next=(index+1)%buttons.length;
      else if(event.key==='ArrowLeft'||event.key==='ArrowUp') next=(index+buttons.length-1)%buttons.length;
      else if(event.key==='Home') next=0;
      else if(event.key==='End') next=buttons.length-1;
      else return;
      event.preventDefault();select(buttons[next]);buttons[next].focus();
    });
  });
}
wireTabs([...document.querySelectorAll('.nav-button')],[...document.querySelectorAll('.page')],'page');
const metricButtons=[...document.querySelectorAll('.metric-tab')],metricPanels=[...document.querySelectorAll('.metric-panel')];
for(const group of new Set(metricButtons.map(button=>button.dataset.metricGroup))){
  wireTabs(metricButtons.filter(button=>button.dataset.metricGroup===group),metricPanels.filter(panel=>panel.dataset.metricGroup===group),'metricPanel');
}
tableSearches.forEach(input=>input.addEventListener('input',()=>filterTable(document.getElementById(input.dataset.table))));
globalSearch.addEventListener('input',()=>document.querySelectorAll('.report-table').forEach(filterTable));
function compareCells(x,y,asc){
  if(x.dataset.sortType==='missing'||y.dataset.sortType==='missing'){
    return x.dataset.sortType===y.dataset.sortType?0:x.dataset.sortType==='missing'?1:-1;
  }
  const numeric=['number','boolean'];
  const sameNumbers=numeric.includes(x.dataset.sortType)&&numeric.includes(y.dataset.sortType);
  const difference=sameNumbers?Number(x.dataset.sortValue)-Number(y.dataset.sortValue):x.dataset.sortValue.localeCompare(y.dataset.sortValue);
  return asc?difference:-difference;
}
document.querySelectorAll('.report-table th').forEach(th=>{
  function sort(){
    const table=th.closest('table'),body=table.tBodies[0],index=[...th.parentNode.children].indexOf(th),asc=th.dataset.asc!=='true';
    [...body.rows].sort((a,b)=>compareCells(a.cells[index],b.cells[index],asc)).forEach(row=>body.appendChild(row));
    for(const header of th.parentNode.children) header.setAttribute('aria-sort',header===th?(asc?'ascending':'descending'):'none');
    th.dataset.asc=String(asc);
  }
  th.addEventListener('click',sort);
  th.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();sort();}});
});
"""


def render_profile_html(report: Any, *, report_name: str) -> str:
    """渲染原始报告及快照共用的格式、组合搜索和两级页签。"""
    description: dict[str, Any] = report.describe()
    pages: list[tuple[str, list[tuple[str, object]]]] = [
        ("Metadata", list(export_semantics(description).items()))
    ]
    grouped: dict[str, list[tuple[str, object]]] = {}
    for name in description["tables"]:
        frame = report._display_frame(report.get_table(name))
        prefix = name.split(".", 1)[0]
        page_name = {"overview": "Overview", "dq": "DQ", "stats": "Stats", "comparison": "Comparisons"}.get(prefix, name)
        grouped.setdefault(page_name, []).append((name, frame))
    if description.get("report_type") == "MarsProfileReport":
        for page_name in ("Overview", "DQ", "Stats", "Comparisons"):
            pages.append((page_name, grouped.pop(page_name, [])))
    pages.extend(grouped.items())
    table_index = 0
    page_html: list[str] = []
    for page_index, (page_name, tables) in enumerate(pages):
        panels: list[str] = []
        metric_navigation: list[str] = []
        page_id = f"profile-page-{page_index}"
        for metric_index, (table_name, frame) in enumerate(tables):
            panel = _table_html(table_name, frame, table_index, description["tables"].get(table_name, {}).get("fields"))
            table_index += 1
            if page_name in {"DQ", "Stats"}:
                tab_id, panel_id = f"metric-tab-{page_index}-{metric_index}", f"metric-panel-{page_index}-{metric_index}"
                selected = metric_index == 0
                metric_navigation.append(
                    f'<button class="metric-tab" role="tab" id="{tab_id}" '
                    f'data-metric-group="{page_id}" data-metric-panel="{panel_id}" '
                    f'aria-controls="{panel_id}" aria-selected="{str(selected).lower()}" '
                    f'tabindex="{0 if selected else -1}">{escape(table_name.split(".", 1)[-1])}</button>'
                )
                panel = (
                    f'<div id="{panel_id}" class="metric-panel{" active" if selected else ""}" '
                    f'role="tabpanel" aria-labelledby="{tab_id}" data-metric-group="{page_id}" '
                    f'data-metric-panel="{panel_id}">{panel}</div>'
                )
            panels.append(panel)
        tabs = f'<div class="metric-tabs" role="tablist" aria-label="{escape(page_name)} metrics">{"".join(metric_navigation)}</div>' if metric_navigation else ""
        content = "".join(panels) or '<section class="panel"><p>No calculated tables available.</p></section>'
        page_html.append(
            f'<div id="{page_id}" class="page{" active" if page_index == 0 else ""}" '
            f'role="tabpanel" aria-labelledby="profile-nav-{page_index}" data-page="{escape(page_name, quote=True)}">'
            f'{tabs}{content}</div>'
        )
    navigation = "".join(
        f'<button class="nav-button" role="tab" id="profile-nav-{index}" '
        f'data-page="{escape(name, quote=True)}" aria-controls="profile-page-{index}" '
        f'aria-selected="{str(index == 0).lower()}" tabindex="{0 if index == 0 else -1}">{escape(name)}</button>'
        for index, (name, _) in enumerate(pages)
    )
    title = escape(report_name)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}}body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f7fa;color:#223}}header{{position:sticky;top:0;background:#173b57;color:white;padding:18px 24px;z-index:2}}h1{{margin:0;font-size:1.6em;overflow-wrap:anywhere}}main{{padding:20px;max-width:1600px;margin:auto;min-width:0}}nav,.metric-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.nav-button,.metric-tab{{border:1px solid #ffffff66;background:#ffffff16;color:white;padding:7px 12px;border-radius:5px;cursor:pointer}}.nav-button[aria-selected=true]{{background:#ffffff33}}.metric-tabs{{margin:0 0 12px}}.metric-tab{{background:white;border-color:#cbd9e4;color:#173b57;overflow-wrap:anywhere;max-width:100%}}.metric-tab[aria-selected=true]{{background:#173b57;color:white}}button:focus-visible,th:focus-visible{{outline:3px solid #6aafd1;outline-offset:2px}}.global-search,.table-search{{padding:9px 12px;border:1px solid #ccd6df;border-radius:5px}}.global-search{{width:min(560px,100%);margin-top:12px}}.table-search{{margin:0 0 10px;width:min(380px,100%)}}.page,.metric-panel{{display:none}}.page.active,.metric-panel.active{{display:block}}.panel{{background:white;padding:18px;margin-bottom:18px;border-radius:8px;box-shadow:0 1px 4px #0002;min-width:0}}.table-wrap{{overflow:auto;max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e4e8ec;text-align:left;white-space:nowrap}}th{{background:#edf3f7;cursor:pointer;position:sticky;top:0}}tr:hover{{outline:1px solid #dce8ef}}h2{{margin-top:0;overflow-wrap:anywhere}}.metadata-cell{{white-space:normal;min-width:110px;overflow-wrap:anywhere}}.metadata-text{{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;overflow:hidden;max-width:320px;line-height:1.5;max-height:4.5em}}.missing{{color:#83919a;font-style:italic}}.unit-note{{font-size:13px;color:#647681}}.sparkline{{font-family:Consolas,Menlo,monospace;color:#1f77b4;white-space:pre}}@media(max-width:600px){{header{{padding:14px}}main{{padding:12px}}.panel{{padding:12px}}}}
</style></head><body><header><h1>{title}</h1><input id="global-search" class="global-search" placeholder="Search all tables" aria-label="Search all tables"><nav role="tablist" aria-label="Report pages">{navigation}</nav></header><main>{"".join(page_html)}</main>
<script>{_SCRIPT}</script></body></html>"""


def write_profile_html(report: Any, *, path: str, report_name: str) -> None:
    """写入完整画像文档，父目录或渲染失败时明确报错。"""
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"HTML report parent directory does not exist: {destination.parent}")
    try:
        destination.write_text(render_profile_html(report, report_name=report_name), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to export profile HTML to '{destination}'.") from exc
