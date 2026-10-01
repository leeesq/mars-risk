"""聚合双分数报告的 Notebook 与自包含离线 HTML 视图。"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from mars.reporting._artifact import Report
from mars.reporting._serialization import encode

from .score_cross import get_score_bin_definitions


def get_score_cell(
    report: Report,
    x_bin: str,
    y_bin: str,
    *,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """查询完整格子证据和固定区间，不接触明细或重新分箱。

    Parameters
    ----------
    report : Report
        原始 score_cross 报告或加载后的快照。
    x_bin : str
        保存的 X bin_id。
    y_bin : str
        保存的 Y bin_id。
    filters : dict[str, Any] | None
        target/group/period 等原生过滤；可同时返回同一格子多个样本范围。

    Returns
    -------
    dict[str, Any]
        page（含 report_id/table/query）、两个区间原生表及完整格子统计。

    Raises
    ------
    ValueError
        报告或 bin_id 无效。

    Examples
    --------
    >>> get_score_cell(restored, "b0", "b1", filters={"group": "OOT"})  # doctest: +SKIP
    """
    get_score_bin_definitions(report)
    intervals = {
        axis: report.get_table("bins", filters={"axis": axis, "bin_id": bin_id})
        for axis, bin_id in (("x", x_bin), ("y", y_bin))
    }
    if any(not len(frame) for frame in intervals.values()):
        raise ValueError("Unknown bin_id.")
    return {
        "page": report.query_page(
            "cells", filters={**(filters or {}), "x_bin": x_bin, "y_bin": y_bin}, limit=None
        ),
        "intervals": intervals,
    }


def show_score_matrix(
    report: Report,
    *,
    filters: dict[str, Any] | None = None,
    metric: str = "bad_rate",
    include_special: bool = False,
    color_range: tuple[float, float] | None = None,
) -> Any:
    """先限定单一样本范围再显示低到高风险矩阵，附真实边际和总体。

    Parameters
    ----------
    report : Report
        score_cross 报告或加载后的快照。
    filters : dict[str, Any] | None
        target/group/period 原生筛选；匹配多组时使用首组并在 caption 明示。
    metric : str
        bad_rate、delta_vs_row 或 sample_share，展示单位为百分比/百分点。
    include_special : bool
        是否展开特殊箱；默认折叠但注明未展示样本数。
    color_range : tuple[float, float] | None
        小数单位固定色标；delta 必须以 0 为中心；默认坏率/占比 [0,1]。

    Returns
    -------
    Any
        可 Notebook 显示及 to_html/to_excel 的 Pandas Styler；格子标注有表现人数。

    Raises
    ------
    ValueError
        报告、指标、色标或样本范围无效。

    Examples
    --------
    >>> show_score_matrix(restored, filters={"target": "bad", "group": "OOT"})  # doctest: +SKIP
    """
    definitions = get_score_bin_definitions(report)
    if metric not in {"bad_rate", "delta_vs_row", "sample_share"}:
        raise ValueError("metric must be bad_rate, delta_vs_row or sample_share.")
    scopes = report.get_table("overall", filters=filters, limit=1)
    if not len(scopes):
        raise ValueError("No matching sample scope.")
    scope = {c: scopes[c][0] for c in ("target", "group", "period")}
    all_cells = report.get_table("cells", filters=scope)
    omitted = 0
    if not include_special:
        normal = pl.col("x_risk_rank").is_not_null() & pl.col("y_risk_rank").is_not_null()
        omitted = all_cells.filter(~normal)["sample_count"].sum()
        all_cells = all_cells.filter(normal)
    columns = ["x_bin", "y_bin", metric, "observed_sample_count", "status"]
    # 只有选定范围和展示列转换 Pandas；颜色与文本共享同一页证据。
    frame = all_cells.select(columns).to_pandas()
    x_bins = report.get_table("bins", filters={"axis": "x"}).to_dicts()
    y_bins = report.get_table("bins", filters={"axis": "y"}).to_dicts()
    xs = [r["bin_id"] for r in x_bins if include_special or r["kind"] == "normal"]
    ys = [r["bin_id"] for r in y_bins if include_special or r["kind"] == "normal"]
    values = frame.pivot(index="x_bin", columns="y_bin", values=metric).reindex(
        index=xs, columns=ys
    )
    text = pd.DataFrame("", index=xs, columns=ys)
    statuses = pd.DataFrame("", index=xs, columns=ys)
    for row in frame.to_dict("records"):
        value = row[metric]
        risk = (
            "—"
            if pd.isna(value)
            else f"{value * 100:.2f}" + ("pp" if metric == "delta_vs_row" else "%")
        )
        count = row["observed_sample_count"]
        text.loc[row["x_bin"], row["y_bin"]] = (
            f"{risk} | n={'—' if pd.isna(count) else int(count)} | {row['status']}"
        )
        statuses.loc[row["x_bin"], row["y_bin"]] = row["status"]
    # TOTAL 从边际整数统计取值；delta 的边际不定义为格子差值的均值。
    margin_metric = "sample_share" if metric == "sample_share" else "bad_rate"
    for axis, ids, table in (("x", xs, "row_summary"), ("y", ys, "column_summary")):
        marginal = report.get_table(table, filters=scope)
        for row in marginal.iter_rows(named=True):
            if row[f"{axis}_bin"] not in ids:
                continue
            v = row[margin_metric]
            label = "—" if v is None else f"{100 * v:.2f}%"
            if axis == "x":
                text.loc[row["x_bin"], "TOTAL"] = label
            else:
                text.loc["TOTAL", row["y_bin"]] = label
    v = scopes[margin_metric][0]
    text.loc["TOTAL", "TOTAL"] = "—" if v is None else f"{100 * v:.2f}%"
    bounds = color_range or ((-1.0, 1.0) if metric == "delta_vs_row" else (0.0, 1.0))
    if not bounds[0] < bounds[1] or metric == "delta_vs_row" and bounds[0] != -bounds[1]:
        raise ValueError("Color range must increase; delta range must be centered on zero.")
    styles = pd.DataFrame("", index=text.index, columns=text.columns)
    for x in xs:
        for y in ys:
            value = values.loc[x, y]
            if pd.isna(value):
                color = "#eee"
            else:
                strength = min(
                    1.0,
                    max(
                        0.0,
                        abs(value) / bounds[1]
                        if metric == "delta_vs_row"
                        else (value - bounds[0]) / (bounds[1] - bounds[0]),
                    ),
                )
                color = f"rgba({'45,100,200' if metric == 'delta_vs_row' and value < 0 else '210,60,45'},{0.08 + 0.65 * strength})"
            styles.loc[x, y] = f"background-color:{color};" + (
                "border:2px dashed #777;" if statuses.loc[x, y] == "low_sample" else ""
            )
    metadata = report.describe()["feature_metadata"]
    names = [
        f"{metadata.get(d['score'], {}).get('display_name') or d['score']} [{d['score']}]"
        for d in definitions.values()
    ]
    caption = f"X={names[0]}; Y={names[1]}; {scope}; omitted special samples={omitted}; TOTAL={margin_metric}; low_sample dashed border"
    return (
        text.style.apply(lambda _: styles, axis=None)
        .format(escape="html")
        .format_index(escape="html", axis=0)
        .format_index(escape="html", axis=1)
        .set_caption(escape(caption))
    )


def write_score_cross_html(
    report: Report,
    path: str | Path,
    *,
    report_name: str = "MARS Score Cross",
    policy_reports: list[Report] | None = None,
) -> None:
    """导出只含聚合数据的离线可交互 HTML，支持查询、指标切换、详情和已回放规则比较。

    Parameters
    ----------
    report : Report
        score_cross 报告，支持加载后的快照。
    path : str | Path
        目标路径；父目录须存在。
    report_name : str
        转义后的报告标题。
    policy_reports : list[Report] | None
        显式回放的派生报告；离线切换并查看留存、四区域、覆盖差及格子差异。

    Returns
    -------
    None
        写入自包含 HTML；无 CDN、远程请求或个体明细。

    Raises
    ------
    ValueError
        报告或规则父 ID 不匹配。

    Examples
    --------
    >>> write_score_cross_html(restored, "cross.html", policy_reports=[replay])  # doctest: +SKIP
    """
    get_score_bin_definitions(report)
    policies: list[dict[str, Any]] = []
    for policy in policy_reports or []:
        if (
            policy.report_type != "score_policy"
            or policy.describe()["parameters"]["parent_report_id"] != report.report_id
        ):
            raise ValueError("Policy report must derive from this report_id.")
        policies.append(
            {
                "description": policy.describe(),
                "tables": {n: policy.get_table(n).to_dicts() for n in policy.describe()["tables"]},
            }
        )
    payload = {
        "description": report.describe(),
        "tables": {n: report.get_table(n).to_dicts() for n in report.describe()["tables"]},
        "policies": policies,
    }
    serialized = encode(payload).replace("<", "\\u003c").replace("&", "\\u0026")
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font-family:system-ui;margin:24px;color:#222}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #aaa;padding:10px}button,select,input{margin:6px;padding:6px}td.cell{cursor:pointer}.low{outline:2px dashed #777}pre{white-space:pre-wrap;max-height:500px;overflow:auto}.changed{box-shadow:inset 0 0 0 4px #9b32a8}</style>
<h1>__TITLE__</h1><p id="names"></p><label>样本范围<select id="scope"></select></label>
<label>着色<select id="metric"><option>bad_rate</option><option>delta_vs_row</option><option>sample_share</option></select></label>
<label><input type="checkbox" id="special">展开特殊箱</label><label>固定色标绝对上限（小数）<input id="scale" type="number" min="0.000001" step="0.01" value="1"></label>
<p id="note"></p><div id="matrix"></div><h2>格子详情与同 X 分段梯度</h2><pre id="details">点击格子查看区间、人数、覆盖、Wilson 区间、状态和证据。</pre><div id="gradient"></div>
<h2>历史规则回放</h2><p>紫框表示基准与候选决策不同的格子。实际样本留存差异见 changes；扩量区域不代表放量安全。</p><select id="policy"></select><div id="policyTables"></div>
<details><summary>口径、状态与固定分段</summary><pre id="semantics"></pre></details>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),t=data.tables,d=data.description;
const $=id=>document.getElementById(id), same=(r,s)=>['target','group','period'].every(k=>r[k]===s[k]);
const put=(el,v)=>el.textContent=String(v), json=v=>JSON.stringify(v,null,2);
const fmt=v=>v===null?'—':(100*v).toFixed(2)+'%';
function table(rows,el){el.replaceChildren();if(!rows.length)return;const tab=document.createElement('table'),head=tab.insertRow();Object.keys(rows[0]).forEach(k=>put(head.appendChild(document.createElement('th')),k));rows.forEach(r=>{const tr=tab.insertRow();Object.values(r).forEach(v=>put(tr.insertCell(),typeof v==='object'?json(v):v===null?'—':v));});el.appendChild(tab);}
const scopeRows=t.overall;scopeRows.forEach((s,i)=>{const o=document.createElement('option');o.value=i;put(o,json({target:s.target,group:s.group,period:s.period}));$('scope').appendChild(o);});
put($('names'),['x','y'].map(a=>{const id=d.parameters['score_'+a];return a.toUpperCase()+'='+((d.feature_metadata[id]||{}).display_name||id)+' ['+id+']';}).join('; '));
put($('semantics'),json(d));
const none=document.createElement('option');none.value=-1;put(none,'无规则比较');$('policy').appendChild(none);
data.policies.forEach((p,i)=>{const o=document.createElement('option');o.value=i;put(o,json(p.description.parameters.candidate));$('policy').appendChild(o);});
function render(){if(!scopeRows.length){put($('note'),'empty input; no actual scopes');return;}
const s=scopeRows[Number($('scope').value)],metric=$('metric').value,expand=$('special').checked,scale=Number($('scale').value)||1;
const xs=t.bins.filter(b=>b.axis==='x'&&(expand||b.kind==='normal')),ys=t.bins.filter(b=>b.axis==='y'&&(expand||b.kind==='normal'));
const all=t.cells.filter(r=>same(r,s)),cells=all.filter(r=>xs.some(b=>b.bin_id===r.x_bin)&&ys.some(b=>b.bin_id===r.y_bin));
const omitted=all.filter(r=>!cells.includes(r)).reduce((n,r)=>n+r.sample_count,0);
put($('note'),'两轴低风险→高风险；未展示特殊样本='+omitted+'；总体 n='+s.sample_count+'；有表现='+s.observed_sample_count+'；风险='+fmt(s.bad_rate)+'；空/未表现为 —，低样本为虚线框；delta 单位 pp，其他比例 %；固定色标 '+(metric==='delta_vs_row'?'±':'0..')+scale);
const p=data.policies[Number($('policy').value)],dec=p?p.tables.cell_decisions.filter(r=>same(r,s)):[];
const tab=document.createElement('table'),head=tab.insertRow();put(head.appendChild(document.createElement('th')),'X / Y');ys.forEach(b=>put(head.appendChild(document.createElement('th')),b.bin_id+' rank='+b.risk_rank));put(head.appendChild(document.createElement('th')),'TOTAL bad_rate');
xs.forEach(x=>{const tr=tab.insertRow();put(tr.appendChild(document.createElement('th')),x.bin_id+' rank='+x.risk_rank);ys.forEach(y=>{const r=cells.find(c=>c.x_bin===x.bin_id&&c.y_bin===y.bin_id),td=tr.insertCell();td.className='cell'+(r.status==='low_sample'?' low':'');const v=r[metric];put(td,(v===null?'—':(v*100).toFixed(2)+(metric==='delta_vs_row'?'pp':'%'))+' | n='+r.observed_sample_count+' | '+r.status);td.style.backgroundColor=v===null?'#eee':'rgba('+(metric==='delta_vs_row'&&v<0?'45,100,200':'210,60,45')+','+(0.08+0.65*Math.min(1,Math.max(0,Math.abs(v)/scale)))+')';const decision=dec.find(c=>c.x_bin===x.bin_id&&c.y_bin===y.bin_id);if(decision&&decision.baseline_pass!==decision.candidate_pass)td.classList.add('changed');td.onclick=()=>{put($('details'),json({reference:{report_id:d.report_id,table:'cells',dimensions:{target:s.target,group:s.group,period:s.period,x_bin:x.bin_id,y_bin:y.bin_id}},x_interval:x,y_interval:y,statistics:r}));table(all.filter(c=>c.x_bin===x.bin_id).map(c=>({y_bin:c.y_bin,y_risk_rank:c.y_risk_rank,bad_rate:c.bad_rate,observed_sample_count:c.observed_sample_count,delta_vs_row:c.delta_vs_row,status:c.status})),$('gradient'));};});const margin=t.row_summary.find(r=>same(r,s)&&r.x_bin===x.bin_id);put(tr.insertCell(),fmt(margin.bad_rate));});
const last=tab.insertRow();put(last.appendChild(document.createElement('th')),'TOTAL bad_rate');ys.forEach(y=>put(last.insertCell(),fmt(t.column_summary.find(r=>same(r,s)&&r.y_bin===y.bin_id).bad_rate)));put(last.insertCell(),fmt(s.bad_rate));$('matrix').replaceChildren(tab);
$('policyTables').replaceChildren();if(p){['changes','summary','regions','axis_regions'].forEach(name=>{if(!p.tables[name])return;const h=document.createElement('h3');put(h,name);$('policyTables').appendChild(h);const el=document.createElement('div');$('policyTables').appendChild(el);table(p.tables[name].filter(r=>same(r,s)),el);});}}
['scope','metric','special','scale','policy'].forEach(id=>$(id).addEventListener('change',render));render();
</script></html>"""
    Path(path).write_text(
        html.replace("__TITLE__", escape(report_name)).replace("__DATA__", serialized),
        encoding="utf-8",
    )
