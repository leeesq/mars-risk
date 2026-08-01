"""Self-contained HTML export for data profile reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from mars.compute import to_pandas_frame


def _table_html(name: str, frame: object, table_index: int) -> str:
    """Render one escaped report table."""
    dataframe = to_pandas_frame(frame).copy()
    table = dataframe.to_html(
        index=False,
        escape=True,
        border=0,
        classes=["report-table"],
        table_id=f"table-{table_index}",
    )
    return (
        f'<section class="panel"><h2>{escape(name)}</h2>'
        f'<input class="table-search" data-table="table-{table_index}" '
        'placeholder="Search this table" aria-label="Search table">'
        f'<div class="table-wrap">{table}</div></section>'
    )


def _metadata_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    """Convert metadata into a deterministic two-column table."""
    rows = []
    for key in sorted(metadata):
        value = metadata[key]
        rows.append({"key": key, "value": repr(value) if not isinstance(value, str) else value})
    return pd.DataFrame(rows, columns=["key", "value"])


def render_profile_html(report: Any, *, report_name: str) -> str:
    """Render a complete interactive profile document."""
    pages: list[tuple[str, list[tuple[str, object]]]] = []
    if report.report_meta:
        pages.append(("Metadata", [("Metadata", _metadata_frame(report.report_meta))]))
    pages.append(("Overview", [("Overview", report.overview_table)]))
    if report.dq_tables:
        pages.append(("DQ", list(report.dq_tables.items())))
    if report.stats_tables:
        pages.append(("Stats", list(report.stats_tables.items())))
    if report.comparison_tables:
        pages.append(("Comparisons", list(report.comparison_tables.items())))
    table_index = 0
    page_html: list[str] = []
    for page_index, (page_name, tables) in enumerate(pages):
        panels: list[str] = []
        for table_name, frame in tables:
            panels.append(_table_html(table_name, frame, table_index))
            table_index += 1
        active = " active" if page_index == 0 else ""
        page_html.append(
            f'<div class="page{active}" data-page="{escape(page_name)}">'
            f'{"".join(panels)}</div>'
        )
    navigation = "".join(
        f'<button class="nav-button" data-page="{escape(name)}">{escape(name)}</button>'
        for name, _ in pages
    )
    title = escape(report_name)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f7fa;color:#223}}header{{position:sticky;top:0;background:#173b57;color:white;padding:18px 24px;z-index:2}}main{{padding:20px;max-width:1600px;margin:auto}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.nav-button{{border:1px solid #ffffff66;background:#ffffff16;color:white;padding:7px 12px;border-radius:5px;cursor:pointer}}.global-search,.table-search{{box-sizing:border-box;padding:9px 12px;border:1px solid #ccd6df;border-radius:5px}}.global-search{{width:min(560px,100%);margin-top:12px}}.table-search{{margin:0 0 10px;width:min(380px,100%)}}.page{{display:none}}.page.active{{display:block}}.panel{{background:white;padding:18px;margin-bottom:18px;border-radius:8px;box-shadow:0 1px 4px #0002}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e4e8ec;text-align:left;white-space:nowrap}}th{{background:#edf3f7;cursor:pointer;position:sticky;top:0}}tr:hover{{background:#f8fbfd}}h2{{margin-top:0}}
</style></head><body><header><h1>{title}</h1><input id="global-search" class="global-search" placeholder="Search all tables" aria-label="Search all tables"><nav>{navigation}</nav></header><main>{"".join(page_html)}</main>
<script>
function filterTable(table,query){{for(const row of table.tBodies[0]?.rows||[]){{row.hidden=!row.innerText.toLowerCase().includes(query)}}}}
document.querySelectorAll('.nav-button').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('.page').forEach(page=>page.classList.toggle('active',page.dataset.page===button.dataset.page))}}));
document.querySelectorAll('.table-search').forEach(input=>input.addEventListener('input',()=>filterTable(document.getElementById(input.dataset.table),input.value.toLowerCase())));
document.getElementById('global-search').addEventListener('input',event=>document.querySelectorAll('.report-table').forEach(table=>filterTable(table,event.target.value.toLowerCase())));
document.querySelectorAll('.report-table th').forEach(th=>th.addEventListener('click',()=>{{const table=th.closest('table'),body=table.tBodies[0],index=[...th.parentNode.children].indexOf(th),asc=th.dataset.asc!=='true';[...body.rows].sort((a,b)=>{{const x=a.cells[index].innerText,y=b.cells[index].innerText,nx=Number(x),ny=Number(y),cmp=Number.isNaN(nx)||Number.isNaN(ny)?x.localeCompare(y):nx-ny;return asc?cmp:-cmp}}).forEach(row=>body.appendChild(row));th.dataset.asc=String(asc)}}));
</script></body></html>"""


def write_profile_html(report: Any, *, path: str, report_name: str) -> None:
    """Write a profile report and fail explicitly on any export error."""
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError(f"HTML report parent directory does not exist: {destination.parent}")
    try:
        html = render_profile_html(report, report_name=report_name)
        destination.write_text(html, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to export profile HTML to '{destination}'.") from exc
