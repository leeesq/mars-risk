"""MARS 评估报告 HTML 样式与运行脚本构建工具。"""

from __future__ import annotations

import json
from collections.abc import Sequence


def build_html_styles() -> str:
    """
    构建评估报告使用的内联 CSS 样式。

    Returns
    -------
    str
        可直接嵌入 HTML 文档的 ``<style>`` 内容。

    Examples
    --------
    >>> styles = build_html_styles()
    >>> ".mars-page" in styles
    True
    """
    return """
            :root { --bg:#f5f7fb; --panel:#fff; --panel-soft:#f9fbfd; --ink:#203040; --muted:#607080; --line:#d9e3eb; --line-soft:#ebf1f6; --accent:#3b87ad; --danger:#c44f4f; --shadow:0 16px 36px rgba(51,82,108,.08); }
            body { margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:radial-gradient(circle at top right,#edf6fb 0%,#f5f7fb 40%,#f8fbfd 100%); color:var(--ink); }
            .mars-page { max-width:1640px; margin:0 auto; padding:22px; }
            .mars-hero,.mars-section { background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:var(--shadow); }
            .mars-hero { padding:22px 24px; margin-bottom:16px; position:relative; overflow:hidden; }
            .mars-hero::after { content:""; position:absolute; inset:auto -80px -90px auto; width:240px; height:240px; background:radial-gradient(circle, rgba(59,135,173,.14) 0%, rgba(59,135,173,0) 72%); pointer-events:none; }
            .mars-hero h1 { margin:0 0 8px 0; font-size:30px; }
            #mars-page-top { position:relative; top:0; }
            .mars-hero p,.mars-footnote,.mars-section-subtitle,.mars-search-error,.mars-view-label,.mars-pivot-source-title,.mars-result-status,.mars-export-helper { color:var(--muted); position:relative; z-index:1; }
            .mars-meta,.mars-nav,.mars-inline-controls { display:flex; flex-wrap:wrap; gap:10px; }
            .mars-meta { margin-top:12px; position:relative; z-index:1; }
            .mars-pill,.mars-nav a { border:1px solid var(--line); background:#f7fbff; border-radius:999px; padding:6px 12px; font-size:13px; color:#36546d; text-decoration:none; }
            .mars-global-tools { margin-top:16px; display:grid; grid-template-columns:minmax(280px,420px) auto minmax(240px,340px) minmax(280px,1fr) minmax(180px,240px); gap:10px; align-items:start; position:relative; z-index:1; }
            .mars-filter-input,.mars-select-group select,.mars-clear-button,.mars-mini-button { border:1px solid var(--line); border-radius:12px; background:#fff; font-size:14px; }
            .mars-filter-input { padding:10px 12px; width:100%; box-sizing:border-box; }
            .mars-search-cluster { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:center; }
            .mars-select-group { display:inline-flex; gap:8px; align-items:center; font-size:13px; }
            .mars-select-group select { padding:8px 10px; }
            .mars-source-panel { border:1px solid var(--line); border-radius:14px; background:#fff; padding:10px 12px; min-width:280px; }
            .mars-source-header,.mars-source-options { display:flex; flex-wrap:wrap; gap:8px; }
            .mars-source-header { align-items:center; justify-content:space-between; margin-bottom:10px; }
            .mars-source-header strong { font-size:13px; color:#355b74; }
            .mars-source-link { border:0; background:transparent; color:var(--accent); cursor:pointer; font-size:12px; padding:0; }
            .mars-source-option { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line-soft); border-radius:999px; padding:5px 10px; background:#f9fbfe; font-size:13px; }
            .mars-clear-button,.mars-mini-button { padding:9px 12px; cursor:pointer; }
            .mars-toggle { display:inline-flex; align-items:center; gap:8px; font-size:13px; }
            .mars-export-block { display:grid; gap:6px; align-content:start; }
            .mars-export-helper { font-size:12px; line-height:1.35; }
            .mars-nav { margin:14px 0 18px 0; }
            .mars-overview-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
            .mars-kpi-card { border:1px solid var(--line-soft); border-radius:14px; background:linear-gradient(180deg,#fbfdff 0%,#f7fbff 100%); padding:14px; }
            .mars-kpi-label { font-size:12px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:.04em; }
            .mars-kpi-value { font-size:16px; font-weight:700; color:#244258; line-height:1.35; word-break:break-word; }
            .mars-legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
            .mars-legend-chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line-soft); border-radius:999px; padding:6px 10px; background:#fff; font-size:12px; color:#436179; }
            .mars-section { margin-bottom:16px; overflow:hidden; }
            .mars-section>summary,.mars-metric-block>summary { cursor:pointer; list-style:none; font-weight:700; }
            .mars-section>summary { padding:16px 18px; background:#f7fbff; border-bottom:1px solid var(--line-soft); }
            .mars-section>summary::-webkit-details-marker,.mars-metric-block>summary::-webkit-details-marker { display:none; }
            .mars-section-body { padding:14px 18px 18px 18px; }
            .mars-section-subtitle { padding:12px 18px 0 18px; font-size:13px; }
            .mars-metric-block { border:1px solid var(--line-soft); border-radius:14px; background:var(--panel-soft); margin-bottom:12px; padding:12px; }
            .mars-metric-block>summary { margin-bottom:10px; color:#355b74; }
            .mars-table-wrap { min-width:0; }
            .mars-table-toolbar { display:grid; grid-template-columns:minmax(240px,360px); gap:6px; margin-bottom:10px; }
            .mars-chart-controls { display:grid; grid-template-columns:minmax(240px,360px) auto; gap:10px; align-items:start; }
            .mars-chart-search { min-width:240px; }
            .mars-summary-filter { border:1px solid var(--line-soft); border-radius:14px; background:#fbfdff; padding:12px; margin-bottom:10px; }
            .mars-summary-filter-label { display:block; margin-bottom:8px; font-size:13px; font-weight:600; color:#355b74; }
            .mars-result-status { min-height:16px; font-size:12px; margin:6px 0 10px 0; }
            .mars-table-scroll { position:relative; overflow:auto; border:1px solid var(--line-soft); border-radius:14px; background:#fff; }
            .mars-data-table { width:max-content; min-width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
            .mars-th,.mars-td { border-bottom:1px solid var(--line-soft); padding:8px 10px; white-space:nowrap; text-align:left; vertical-align:top; }
            .mars-th { position:sticky; top:0; background:#eef6fb; z-index:1; }
            .mars-td { position:relative; z-index:0; }
            .mars-sticky-col { position:sticky; background-clip:padding-box; overflow:hidden; }
            .mars-feature-col { min-width:var(--mars-feature-col-width, 220px); width:var(--mars-feature-col-width, 220px); max-width:var(--mars-feature-col-width, 220px); box-sizing:border-box; }
            .mars-secondary-col { min-width:var(--mars-secondary-col-width, 110px); width:var(--mars-secondary-col-width, 110px); max-width:var(--mars-secondary-col-width, 110px); box-sizing:border-box; }
            .mars-bin-col { min-width:var(--mars-bin-col-width, 140px); width:var(--mars-bin-col-width, 140px); max-width:var(--mars-bin-col-width, 140px); box-sizing:border-box; }
            .mars-data-table .mars-td.mars-feature-col,
            .mars-data-table .mars-td.mars-secondary-col,
            .mars-pivot-table .mars-td.mars-bin-col { background:#fff; }
            .mars-data-table .mars-th.mars-feature-col,
            .mars-data-table .mars-th.mars-secondary-col,
            .mars-pivot-table .mars-th.mars-bin-col { background:#eef6fb; }
            .mars-data-table .mars-th.mars-feature-col { left:0; z-index:6; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-data-table .mars-td.mars-feature-col { left:0; z-index:4; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-data-table .mars-th.mars-secondary-col { left:var(--mars-feature-col-width, 220px); z-index:5; box-shadow:2px 0 0 rgba(217,227,235,.72); }
            .mars-data-table .mars-td.mars-secondary-col { left:var(--mars-feature-col-width, 220px); z-index:3; box-shadow:2px 0 0 rgba(217,227,235,.72); }
            .mars-pivot-table .mars-th.mars-feature-col { left:0; z-index:7; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-pivot-table .mars-td.mars-feature-col { left:0; z-index:5; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-pivot-table .mars-th.mars-bin-col { left:var(--mars-feature-col-width, 220px); z-index:6; padding-right:18px; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-pivot-table .mars-td.mars-bin-col { left:var(--mars-feature-col-width, 220px); z-index:4; box-shadow:2px 0 0 rgba(217,227,235,.85); }
            .mars-th.is-numeric,.mars-td.is-numeric { text-align:right; }
            .mars-sort-button { width:100%; min-width:0; overflow:hidden; border:0; background:transparent; padding:0; margin:0; color:inherit; font:inherit; display:inline-flex; align-items:center; justify-content:space-between; gap:8px; cursor:pointer; }
            .mars-sort-label { display:block; min-width:0; overflow:hidden; text-overflow:ellipsis; }
            .mars-cell-text { display:block; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
            .mars-sticky-cell-inner { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
            .mars-th.mars-feature-col { padding-right:18px; }
            .mars-resize-handle { position:absolute; top:0; right:0; width:10px; height:100%; cursor:col-resize; user-select:none; touch-action:none; }
            .mars-resize-handle::after { content:""; position:absolute; top:20%; bottom:20%; left:4px; width:2px; border-radius:2px; background:rgba(53,91,116,.22); }
            .mars-feature-jump { min-width:240px; }
            .mars-pivot-table .mars-th, .mars-pivot-table .mars-td { background-clip:padding-box; }
            .mars-jump-highlight { animation:mars-jump-pulse 1.2s ease-out 1; }
            .mars-jump-highlight-cell { animation:mars-jump-pulse 1.2s ease-out 1; }
            .mars-table-ownership-sentinel { height:0; margin:0; padding:0; pointer-events:none; }
            .mars-floating-header-host { position:fixed; top:0; left:0; width:0; display:none; border:0; border-radius:14px; background:#fff; box-shadow:0 14px 32px rgba(32,48,64,.16), inset 0 0 0 1px var(--line-soft); overflow:hidden; z-index:60; }
            .mars-floating-header-host.is-visible { display:block; }
            .mars-floating-header-scroll { overflow:hidden; background:#fff; }
            .mars-floating-header-table { width:max-content; min-width:100%; margin:0; table-layout:fixed; }
            .mars-floating-header-table tbody { display:none; }
            .mars-floating-header-table .mars-th { top:0; z-index:8; }
            .mars-floating-header-table .mars-th.mars-feature-col { z-index:10; }
            .mars-floating-header-table .mars-th.mars-secondary-col,
            .mars-floating-header-table .mars-th.mars-bin-col { z-index:9; }
            .mars-back-to-top { position:fixed; right:24px; bottom:24px; border:1px solid rgba(53,91,116,.18); border-radius:999px; background:rgba(255,255,255,.96); color:#355b74; box-shadow:0 14px 28px rgba(32,48,64,.14); padding:11px 16px; font-size:13px; font-weight:600; cursor:pointer; opacity:0; transform:translateY(12px); pointer-events:none; transition:opacity .18s ease, transform .18s ease, box-shadow .18s ease; z-index:70; }
            .mars-back-to-top.is-visible { opacity:1; transform:translateY(0); pointer-events:auto; }
            .mars-back-to-top:hover,.mars-back-to-top:focus-visible { box-shadow:0 18px 34px rgba(32,48,64,.2); outline:none; }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td { position:relative; z-index:2; filter:saturate(1.08) brightness(.98); box-shadow:inset 0 0 0 9999px rgba(255,237,177,.34), inset 0 2px 0 rgba(233,153,49,.86), inset 0 -2px 0 rgba(233,153,49,.86) !important; transition:box-shadow .32s ease, filter .32s ease, outline-color .32s ease; }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td:first-child { border-left:3px solid rgba(233,153,49,.86); }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td:last-child { border-right:3px solid rgba(233,153,49,.86); }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td .mars-cell-text { font-weight:600; }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td.mars-feature-col { color:#122636; }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td.mars-jump-highlight-cell { outline:2px solid rgba(245,158,11,.48); outline-offset:-2px; box-shadow:inset 0 0 0 9999px rgba(255,247,213,.82), inset 6px 0 0 #f59e0b, inset 0 2px 0 rgba(233,153,49,.9), inset 0 -2px 0 rgba(233,153,49,.9) !important; filter:saturate(1.12) brightness(1); }
            .mars-data-table tbody tr.mars-jump-highlight > .mars-td.mars-jump-highlight-cell .mars-cell-text { font-weight:700; color:#0f2131; text-shadow:0 1px 0 rgba(255,255,255,.55); }
            .mars-sort-indicator::before { content:"\\2195"; color:#8aa1b3; font-size:11px; }
            th[data-sort-dir="asc"] .mars-sort-indicator::before { content:"\\2191"; color:var(--accent); }
            th[data-sort-dir="desc"] .mars-sort-indicator::before { content:"\\2193"; color:var(--accent); }
            .mars-empty { border:1px dashed var(--line); border-radius:14px; padding:16px; background:#fbfdff; font-size:13px; }
            .mars-scope-empty { margin-top:10px; }
            .mars-scope-empty[hidden] { display:none !important; }
            .mars-chart-card { border:1px solid var(--line-soft); border-radius:14px; background:#fff; padding:12px; margin-bottom:12px; box-shadow:0 8px 20px rgba(51,82,108,.05); }
            .mars-pivot-source-title-cell { background:#edf6fb; color:#355b74; font-weight:700; letter-spacing:.02em; }
            .mars-pivot-feature { font-weight:600; color:#2f495e; }
            .mars-pivot-feature-blank .mars-cell-text { visibility:hidden; }
            .mars-pivot-spacer-row td { border-bottom:0; padding:5px 0; background:linear-gradient(180deg,transparent 0%,rgba(233,239,245,.65) 100%); }
            .mars-chart-card h4 { margin:0 0 10px 0; font-size:16px; }
            .mars-footnote { font-size:12px; margin-top:12px; }
    """.strip()



def build_html_runtime_script(summary_filter_columns: Sequence[str]) -> str:
    """
    构建评估报告表格筛选、排序和跳转交互脚本。

    Parameters
    ----------
    summary_filter_columns : Sequence[str]
        汇总表允许参与全局搜索与筛选的列名集合。

    Returns
    -------
    str
        可直接嵌入 HTML 文档的 ``<script>`` 内容。

    Examples
    --------
    >>> script = build_html_runtime_script(["feature", "group"])
    >>> "marsSummaryFilterColumns" in script
    True
    """
    template = """
            const marsSummaryFilterColumns = new Set(__SUMMARY_FILTER_COLUMNS__);
            const marsState = {
                globalQuery:"",
                regexMode:false,
                localQueries:{},
                selectedSources:[],
                appliedSummaryExpression:"",
                summaryAllowedFeatures:null,
                refreshScheduled:false,
                refreshFrameId:null,
                postPaintFrameId:null,
                refreshTimerId:null,
                pendingRefreshTokens:[],
                pendingLayoutToken:null,
                layoutFrameId:null,
                resizeState:null,
                resizeFrameScheduled:false,
                floatingHeaderTableId:"",
                floatingHeaderScrollBox:null,
                floatingHeaderFrameId:null,
                jumpHighlightTimerId:null,
                jumpHighlightArmTimerId:null,
                jumpHighlightNode:null,
                jumpHighlightCell:null
            };
            function marsBuildMatcher(query) { const q=(query||"").trim(); if(!q) return {ok:true,match:()=>true}; if(marsState.regexMode) { try { const regex=new RegExp(q,"i"); return {ok:true,match:(text)=>regex.test(text||"")}; } catch(err) { return {ok:false,error:err.message}; } } const terms=q.toLowerCase().split(/\\s+/).filter(Boolean); return {ok:true,match:(text)=>terms.every((term)=>(text||"").toLowerCase().includes(term))}; }
            function marsSetError(id, message) { const node=document.getElementById(id); if(node) node.textContent=message||""; }
            function marsNormalizeFeatureValue(value) { return (value||"").trim().toLowerCase(); }
            function marsResolveLocalScope(scopeId) { return scopeId==="mars-chart-cards" ? "charts" : `table:${scopeId}`; }
            function marsMergeRefreshToken(scopeToken) {
                const token=(scopeToken||"all").trim() || "all";
                if(token==="all") { marsState.pendingRefreshTokens=["all"]; return; }
                if(marsState.pendingRefreshTokens.includes("all")) return;
                if(!marsState.pendingRefreshTokens.includes(token)) marsState.pendingRefreshTokens.push(token);
            }
            function marsMergeLayoutToken(scopeToken) {
                const token=(scopeToken||"all").trim() || "all";
                if(token==="all" || marsState.pendingLayoutToken==="all" || !marsState.pendingLayoutToken) {
                    marsState.pendingLayoutToken = token==="all" ? "all" : marsState.pendingLayoutToken || token;
                    return;
                }
                if(marsState.pendingLayoutToken!==token) marsState.pendingLayoutToken="all";
            }
            function marsQueueRefresh(scopeToken="all", delayMs=0) {
                marsMergeRefreshToken(scopeToken);
                if(marsState.refreshFrameId) window.cancelAnimationFrame(marsState.refreshFrameId);
                if(marsState.postPaintFrameId) window.cancelAnimationFrame(marsState.postPaintFrameId);
                if(marsState.refreshTimerId) window.clearTimeout(marsState.refreshTimerId);
                marsState.refreshScheduled = true;
                marsState.refreshFrameId = window.requestAnimationFrame(() => {
                    marsState.refreshFrameId = null;
                    marsState.postPaintFrameId = window.requestAnimationFrame(() => {
                        marsState.postPaintFrameId = null;
                        marsState.refreshTimerId = window.setTimeout(() => {
                            marsState.refreshTimerId = null;
                            marsState.refreshScheduled = false;
                            marsFlushRefreshQueue();
                        }, Math.max(0, Number(delayMs) || 0));
                    });
                });
            }
            function marsQueueTextRefresh(scopeToken="all") { marsQueueRefresh(scopeToken, 80); }
            function marsQueueLayoutSync(scopeToken="all") {
                marsMergeLayoutToken(scopeToken);
                if(marsState.layoutFrameId) return;
                marsState.layoutFrameId = window.requestAnimationFrame(() => {
                    marsState.layoutFrameId = null;
                    const token = marsState.pendingLayoutToken || "all";
                    marsState.pendingLayoutToken = null;
                    marsSyncScopeLayouts(token);
                });
            }
            function marsSetGlobalQuery(value) { marsState.globalQuery=value||""; marsQueueTextRefresh("all"); }
            function marsSetLocalQuery(scopeId, value) { marsState.localQueries[scopeId]=value||""; marsQueueTextRefresh(marsResolveLocalScope(scopeId)); }
            function marsSetRegexMode(enabled) { marsState.regexMode=!!enabled; marsQueueRefresh("all"); }
            function marsSetDataSources() { const boxes=Array.from(document.querySelectorAll(".mars-source-checkbox")); marsState.selectedSources=boxes.filter((box)=>box.checked).map((box)=>box.value); marsQueueRefresh("all"); }
            function marsHandleDataSourceToggle() { marsSetDataSources(); }
            function marsHandlePivotTargetChange() { marsQueueRefresh("pivot"); marsQueueLayoutSync("pivot"); }
            function marsHandleChartTargetChange() { marsQueueRefresh("charts"); }
            function marsSelectAllSources() { document.querySelectorAll(".mars-source-checkbox").forEach((box)=>{ box.checked=true; }); marsSetDataSources(); }
            function marsClearSources() { document.querySelectorAll(".mars-source-checkbox").forEach((box)=>{ box.checked=false; }); marsSetDataSources(); }
            function marsClearGlobalSearch() { const input=document.getElementById("mars-global-search"); if(input) input.value=""; marsState.globalQuery=""; marsQueueTextRefresh("all"); }
            function marsTokenizeSummaryExpression(expr) {
                const text=(expr||"").trim();
                if(!text) return {ok:true,tokens:[]};
                const tokenPattern=/\\s*(>=|<=|==|!=|>|<|\\&|\\||\\(|\\)|-?(?:\\d+\\.\\d*|\\d*\\.\\d+|\\d+)(?:[eE][+-]?\\d+)?|[A-Za-z_][A-Za-z0-9_]*)\\s*/gy;
                const tokens=[];
                let cursor=0;
                while(cursor < text.length) {
                    tokenPattern.lastIndex = cursor;
                    const match=tokenPattern.exec(text);
                    if(!match) return {ok:false,error:"Invalid expression syntax."};
                    tokens.push(match[1]);
                    if(tokenPattern.lastIndex<=cursor) return {ok:false,error:"Invalid expression syntax."};
                    cursor=tokenPattern.lastIndex;
                }
                return {ok:true,tokens};
            }
            function marsParseSummaryExpression(expr) {
                const tokenResult=marsTokenizeSummaryExpression(expr);
                if(!tokenResult.ok) return tokenResult;
                const tokens=tokenResult.tokens;
                if(!tokens.length) return {ok:true,ast:null};
                let idx=0;
                function peek() { return tokens[idx]; }
                function consume(expected) {
                    const token=tokens[idx];
                    if(expected && token!==expected) throw new Error(`Expected '${expected}'`);
                    idx+=1;
                    return token;
                }
                function parsePrimary() {
                    const token=peek();
                    if(token===undefined) throw new Error("Unexpected end of expression.");
                    if(token==="(") { consume("("); const node=parseOr(); if(peek()!==")") throw new Error("Missing closing parenthesis."); consume(")"); return node; }
                    if(/^-?(?:\\d+\\.\\d*|\\d*\\.\\d+|\\d+)(?:[eE][+-]?\\d+)?$/.test(token)) { consume(); return {type:"number", value:Number(token)}; }
                    if(/^[A-Za-z_][A-Za-z0-9_]*$/.test(token)) {
                        if(!marsSummaryFilterColumns.has(token)) throw new Error(`Unknown metric: ${token}`);
                        consume();
                        return {type:"identifier", value:token};
                    }
                    throw new Error(`Unexpected token: ${token}`);
                }
                function parseComparison() {
                    const left=parsePrimary();
                    const token=peek();
                    if(["<", "<=", ">", ">=", "==", "!="].includes(token)) {
                        consume();
                        const right=parsePrimary();
                        return {type:"compare", op:token, left, right};
                    }
                    if(!["identifier", "compare", "and", "or"].includes(left.type)) throw new Error("Standalone values must be metric names.");
                    return left;
                }
                function parseAnd() {
                    let node=parseComparison();
                    while(peek()==="&") { consume("&"); node={type:"and", left:node, right:parseComparison()}; }
                    return node;
                }
                function parseOr() {
                    let node=parseAnd();
                    while(peek()==="|") { consume("|"); node={type:"or", left:node, right:parseAnd()}; }
                    return node;
                }
                try {
                    const ast=parseOr();
                    if(idx!==tokens.length) throw new Error(`Unexpected token: ${peek()}`);
                    return {ok:true,ast};
                } catch(err) {
                    return {ok:false,error:err.message};
                }
            }
            function marsEvaluateSummaryNode(node, metrics) {
                if(!node) return true;
                if(node.type==="number") return node.value;
                if(node.type==="identifier") return Number(metrics?.[node.value]);
                if(node.type==="compare") {
                    const left=Number(marsEvaluateSummaryNode(node.left, metrics));
                    const right=Number(marsEvaluateSummaryNode(node.right, metrics));
                    if(!Number.isFinite(left) || !Number.isFinite(right)) return false;
                    return node.op===">" ? left>right : node.op===">=" ? left>=right : node.op==="<" ? left<right : node.op==="<=" ? left<=right : node.op==="==" ? left===right : left!==right;
                }
                if(node.type==="and") return Boolean(marsEvaluateSummaryNode(node.left, metrics)) && Boolean(marsEvaluateSummaryNode(node.right, metrics));
                if(node.type==="or") return Boolean(marsEvaluateSummaryNode(node.left, metrics)) || Boolean(marsEvaluateSummaryNode(node.right, metrics));
                return false;
            }
            function marsSetSummaryExpression(value) {
                const expr=(value||"").trim();
                if(!expr) {
                    marsState.appliedSummaryExpression="";
                    marsSetError("mars-summary-expression-error", "");
                    marsQueueTextRefresh("all");
                    return;
                }
                const parsed=marsParseSummaryExpression(expr);
                if(!parsed.ok) {
                    marsSetError("mars-summary-expression-error", parsed.error || "Invalid expression.");
                    marsQueueTextRefresh("all");
                    return;
                }
                marsState.appliedSummaryExpression=expr;
                marsSetError("mars-summary-expression-error", "");
                marsQueueTextRefresh("all");
            }
            function marsUpdateTableSpecialRows(table) { const rows=Array.from(table.querySelectorAll("tbody tr")); const visibleBySource=new Set(); const visibleByFeatureSource=new Set(); rows.forEach((row)=>{ const role=row.dataset.role||"data"; if(role==="data"&&row.style.display!=="none") { const source=row.dataset.dataSource||""; const feature=row.dataset.feature||""; visibleBySource.add(source); visibleByFeatureSource.add(`${source}||${feature}`); } }); rows.forEach((row)=>{ const role=row.dataset.role||"data"; if(role==="source") { row.style.display=visibleBySource.has(row.dataset.dataSource||"")?"":"none"; } else if(role==="spacer") { const key=`${row.dataset.dataSource||""}||${row.dataset.feature||""}`; row.style.display=visibleByFeatureSource.has(key)?"":"none"; } }); }
            function marsSourceSelected(source) { if(source==="__aggregate__") return true; const hasBoxes=document.querySelectorAll(".mars-source-checkbox").length>0; if(!hasBoxes) return true; return marsState.selectedSources.includes(source||"UNMAPPED"); }
            function marsReadRowMetrics(row) { let metrics={}; try { metrics=JSON.parse(row.dataset.metrics||"{}"); } catch(err) { metrics={}; } return metrics; }
            function marsSummaryRowAllowedWithoutLocal(row, globalMatcher=null, summaryParsed=null) {
                if(!row) return false;
                const matcher=globalMatcher||marsBuildMatcher(marsState.globalQuery);
                if(!matcher.ok) return false;
                const parsed=summaryParsed||marsParseSummaryExpression(marsState.appliedSummaryExpression);
                if(!parsed.ok) return false;
                const source=row.dataset.dataSource||"UNMAPPED";
                const text=row.dataset.searchText||row.textContent||"";
                return marsSourceSelected(source) && matcher.match(text) && marsEvaluateSummaryNode(parsed.ast, marsReadRowMetrics(row));
            }
            function marsGetSummaryFeatureAllowSet() {
                const table=document.getElementById("mars-summary-table");
                if(!table) return null;
                const globalMatcher=marsBuildMatcher(marsState.globalQuery);
                if(!globalMatcher.ok) return null;
                const parsed=marsParseSummaryExpression(marsState.appliedSummaryExpression);
                if(!parsed.ok) return marsState.summaryAllowedFeatures;
                const features=new Set();
                table.querySelectorAll("tbody tr[data-feature]").forEach((row)=>{
                    const feature=row.dataset.feature||"";
                    if(feature && marsSummaryRowAllowedWithoutLocal(row, globalMatcher, parsed)) features.add(feature);
                });
                return features;
            }
            function marsFeatureAllowed(feature) { if(!(marsState.summaryAllowedFeatures instanceof Set)) return true; return marsState.summaryAllowedFeatures.has(feature||""); }
            function marsSetScopeStatus(scopeId, visibleCount, totalCount, noun) {
                const node=document.getElementById(`${scopeId}-status`);
                if(!node) return;
                const visible=Math.max(0, Number(visibleCount) || 0);
                const total=Math.max(0, Number(totalCount) || 0);
                if(total===0 || visible===0) { node.textContent=`0 ${noun} matched current filters.`; return; }
                if(visible===total) { node.textContent=`${visible} ${noun} shown.`; return; }
                node.textContent=`${visible} of ${total} ${noun} shown.`;
            }
            function marsToggleScopeEmpty(scopeId, visible) {
                const node=document.getElementById(`${scopeId}-empty`);
                if(node) node.hidden=!visible;
            }
            function marsUpdateTableFeedback(tableId, totalCount, visibleCount) {
                marsSetScopeStatus(tableId, visibleCount, totalCount, "rows");
                marsToggleScopeEmpty(tableId, visibleCount===0);
            }
            function marsApplyTableFilter(tableId) {
                const table=document.getElementById(tableId);
                if(!table) return;
                const globalMatcher=marsBuildMatcher(marsState.globalQuery);
                if(!globalMatcher.ok) { marsSetError("mars-global-error", `Invalid regex: ${globalMatcher.error}`); return; }
                marsSetError("mars-global-error", "");
                const localMatcher=marsBuildMatcher(marsState.localQueries[tableId]||"");
                if(!localMatcher.ok) { marsSetError(`${tableId}-error`, `Invalid regex: ${localMatcher.error}`); return; }
                marsSetError(`${tableId}-error`, "");
                const summaryParsed=marsParseSummaryExpression(marsState.appliedSummaryExpression);
                const isSummary=table.dataset.tableKind==="summary";
                const dataRows = Array.from(table.querySelectorAll("tbody tr")).filter((row)=>(row.dataset.role||"data")==="data");
                dataRows.forEach((row)=>{
                    const source=row.dataset.dataSource||"UNMAPPED";
                    const feature=row.dataset.feature||"";
                    const text=row.dataset.searchText||row.textContent||"";
                    const globalVisible=marsSourceSelected(source)&&globalMatcher.match(text);
                    if(!globalVisible) { row.style.display="none"; return; }
                    const summaryVisible=isSummary
                        ? (summaryParsed.ok ? marsSummaryRowAllowedWithoutLocal(row, globalMatcher, summaryParsed) : true)
                        : marsFeatureAllowed(feature);
                    const visible=summaryVisible&&localMatcher.match(text);
                    row.style.display=visible?"":"none";
                });
                marsUpdateTableSpecialRows(table);
                const visibleCount = dataRows.filter((row)=>row.style.display!=="none").length;
                marsUpdateTableFeedback(tableId, dataRows.length, visibleCount);
            }
            function marsSortTable(tableId, trigger) { const table=document.getElementById(tableId); if(!table) return; const th=trigger.closest("th"); const colIndex=Number(th.dataset.colIndex||Array.from(th.parentNode.children).indexOf(th)); if(colIndex<0) return; const sourceHeader=table.querySelector(`thead th[data-col-index="${colIndex}"]`) || th; const tbody=table.querySelector("tbody"); const rows=Array.from(tbody.querySelectorAll("tr")).filter((row)=>(row.dataset.role||"data")==="data"); let nextDir="asc"; if(table.dataset.sortCol===String(colIndex)) nextDir=table.dataset.sortDir==="asc"?"desc":"asc"; const sortType=sourceHeader.dataset.sortType||th.dataset.sortType||"text"; rows.sort((a,b)=>{ const va=a.children[colIndex]?.dataset.sortValue||""; const vb=b.children[colIndex]?.dataset.sortValue||""; if(sortType==="number") { const na=Number(va), nb=Number(vb); const sa=Number.isFinite(na)?na:(nextDir==="asc"?Infinity:-Infinity); const sb=Number.isFinite(nb)?nb:(nextDir==="asc"?Infinity:-Infinity); return nextDir==="asc"?sa-sb:sb-sa; } return nextDir==="asc"?va.localeCompare(vb,undefined,{numeric:true,sensitivity:"base"}):vb.localeCompare(va,undefined,{numeric:true,sensitivity:"base"}); }); rows.forEach((row)=>tbody.appendChild(row)); table.dataset.sortCol=String(colIndex); table.dataset.sortDir=nextDir; table.querySelectorAll("thead th[data-sort-dir]").forEach((cell)=>cell.removeAttribute("data-sort-dir")); sourceHeader.dataset.sortDir=nextDir; marsApplyTableFilter(tableId); marsQueueLayoutSync(`table:${tableId}`); marsScheduleViewportRefresh(); }
            function marsUpdatePivotViews() {
                const targetValue=document.getElementById("mars-pivot-target")?.value||null;
                document.querySelectorAll(".mars-pivot-view").forEach((view)=>{
                    const sameTarget=!targetValue||view.dataset.yValue===targetValue;
                    view.style.display=sameTarget?"":"none";
                });
            }
            function marsUpdateChartViews() {
                const targetValue=document.getElementById("mars-chart-target")?.value||null;
                const globalMatcher=marsBuildMatcher(marsState.globalQuery);
                const localMatcher=marsBuildMatcher(marsState.localQueries["mars-chart-cards"]||"");
                if(!globalMatcher.ok) { marsSetError("mars-global-error", `Invalid regex: ${globalMatcher.error}`); return; }
                marsSetError("mars-global-error", "");
                if(!localMatcher.ok) { marsSetError("mars-chart-cards-error", `Invalid regex: ${localMatcher.error}`); return; }
                marsSetError("mars-chart-cards-error", "");
                let totalCards=0;
                let visibleCards=0;
                document.querySelectorAll(".mars-chart-view").forEach((view)=>{
                    const visibleTarget=!targetValue||view.dataset.yValue===targetValue;
                    view.style.display=visibleTarget?"":"none";
                    if(!visibleTarget) return;
                    view.querySelectorAll(".mars-chart-card").forEach((card)=>{
                        totalCards += 1;
                        const source=card.dataset.dataSource||"UNMAPPED";
                        const feature=card.dataset.feature||"";
                        const text=card.dataset.searchText||card.textContent||"";
                        const globalVisible=marsSourceSelected(source)&&globalMatcher.match(text)&&marsFeatureAllowed(feature);
                        const visible=globalVisible&&localMatcher.match(text);
                        card.style.display=visible?"":"none";
                        if(visible) visibleCards += 1;
                    });
                });
                marsSetScopeStatus("mars-chart-cards", visibleCards, totalCards, "charts");
                marsToggleScopeEmpty("mars-chart-cards", visibleCards===0);
            }
            function marsBuildExportFeatureMap() {
                const table=document.getElementById("mars-summary-table");
                if(!table) return {};
                const summaryParsed=marsParseSummaryExpression(marsState.appliedSummaryExpression);
                const featureMap=new Map();
                const sourceOrder=Array.from(document.querySelectorAll(".mars-source-checkbox")).map((box)=>box.value);
                table.querySelectorAll("tbody tr[data-feature]").forEach((row)=>{
                    const source=row.dataset.dataSource||"UNMAPPED";
                    const feature=row.dataset.feature||"";
                    if(!feature || !marsSourceSelected(source)) return;
                    if(summaryParsed.ok && !marsEvaluateSummaryNode(summaryParsed.ast, marsReadRowMetrics(row))) return;
                    if(!featureMap.has(source)) featureMap.set(source, new Set());
                    featureMap.get(source).add(feature);
                });
                const payload={};
                const assignedSources=new Set();
                sourceOrder.forEach((source)=>{
                    const values=featureMap.has(source) ? Array.from(featureMap.get(source)).sort((a,b)=>a.localeCompare(b, undefined, {numeric:true, sensitivity:"base"})) : [];
                    if(values.length) {
                        payload[source]=values;
                        assignedSources.add(source);
                    }
                });
                featureMap.forEach((features, source)=>{
                    if(assignedSources.has(source)) return;
                    const values=Array.from(features).sort((a,b)=>a.localeCompare(b, undefined, {numeric:true, sensitivity:"base"}));
                    if(values.length) payload[source]=values;
                });
                return payload;
            }
            function marsDownloadTextFile(text, fileName) { const blob=new Blob([text], {type:"text/plain;charset=utf-8"}); const link=document.createElement("a"); link.href=URL.createObjectURL(blob); link.download=fileName; link.click(); URL.revokeObjectURL(link.href); }
            function marsExportFeatures() { const featureMap=marsBuildExportFeatureMap(); marsDownloadTextFile(JSON.stringify(featureMap, null, 2), "mars_features.txt"); }
            function marsGetFloatingHeaderHost() { return document.getElementById("mars-floating-header-host"); }
            function marsGetFloatingHeaderScroll() { return document.getElementById("mars-floating-header-scroll"); }
            function marsGetTableScrollBox(table) { return table?.closest(".mars-table-scroll") || null; }
            function marsAncestorsDetailsOpen(node) {
                let parent=node?.closest("details");
                while(parent) {
                    if(!parent.open) return false;
                    parent=parent.parentElement?.closest("details");
                }
                return true;
            }
            function marsHasClientRects(node) {
                return Boolean(node?.getClientRects && node.getClientRects().length);
            }
            function marsIntersectsViewport(rect) {
                return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
            }
            function marsTableIsActuallyVisible(table, scrollBox, thead) {
                if(!table || !scrollBox || !thead) return false;
                if(!marsAncestorsDetailsOpen(scrollBox)) return false;
                if(!marsHasClientRects(scrollBox) || !marsHasClientRects(table) || !marsHasClientRects(thead)) return false;
                const scrollRect=scrollBox.getBoundingClientRect();
                const tableRect=table.getBoundingClientRect();
                const theadRect=thead.getBoundingClientRect();
                if(scrollRect.width <= 0 || scrollRect.height <= 0 || tableRect.width <= 0 || tableRect.height <= 0 || theadRect.width <= 0 || theadRect.height <= 0) return false;
                return marsIntersectsViewport(scrollRect) && marsIntersectsViewport(tableRect);
            }
            function marsHideFloatingHeader() {
                const host=marsGetFloatingHeaderHost();
                const scrollHost=marsGetFloatingHeaderScroll();
                if(scrollHost) scrollHost.innerHTML="";
                if(host) {
                    host.hidden=true;
                    host.classList.remove("is-visible");
                    host.style.left="0px";
                    host.style.width="0px";
                    host.removeAttribute("data-table-id");
                }
                marsState.floatingHeaderTableId="";
                marsState.floatingHeaderScrollBox=null;
            }
            function marsGetFirstVisibleDataRowTop(table) {
                const rows=Array.from(table?.querySelectorAll("tbody tr") || []).filter((row) => {
                    if(row.offsetParent===null || row.style.display==="none") return false;
                    return (row.dataset.role || "data")==="data";
                });
                for(const row of rows) {
                    const rect=row.getBoundingClientRect();
                    if(rect.bottom > 0) return rect.top;
                }
                if(rows.length) return rows[0].getBoundingClientRect().top;
                const tbody=table?.querySelector("tbody");
                if(tbody) return tbody.getBoundingClientRect().top;
                return table?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY;
            }
            function marsCollectLeafColumnWidths(table) {
                const thead=table?.querySelector("thead");
                const rows=Array.from(thead?.rows || []);
                if(!rows.length) return [];
                const occupancy=[];
                const leafColumns=[];
                const totalRows=rows.length;
                rows.forEach((row, rowIndex) => {
                    occupancy[rowIndex] = occupancy[rowIndex] || [];
                    let colIndex=0;
                    Array.from(row.cells).forEach((cell) => {
                        while(occupancy[rowIndex][colIndex]) colIndex += 1;
                        const colSpan=Math.max(1, Number(cell.colSpan) || 1);
                        const rowSpan=Math.max(1, Number(cell.rowSpan) || 1);
                        for(let r=rowIndex; r<Math.min(totalRows, rowIndex + rowSpan); r += 1) {
                            occupancy[r] = occupancy[r] || [];
                            for(let c=colIndex; c<colIndex + colSpan; c += 1) occupancy[r][c]=true;
                        }
                        if(rowIndex + rowSpan >= totalRows) {
                            const baseWidth=Math.max(1, Number(cell.getBoundingClientRect().width || cell.offsetWidth || 0));
                            const sharedWidth=baseWidth / colSpan;
                            for(let c=0; c<colSpan; c += 1) {
                                leafColumns[colIndex + c] = Math.max(1, Math.ceil(sharedWidth));
                            }
                        }
                        colIndex += colSpan;
                    });
                });
                return leafColumns.filter((width)=>Number.isFinite(width) && width > 0);
            }
            function marsBuildFloatingHeaderColGroup(table) {
                const widths=marsCollectLeafColumnWidths(table);
                if(!widths.length) return null;
                const colgroup=document.createElement("colgroup");
                widths.forEach((width) => {
                    const col=document.createElement("col");
                    col.style.width=`${width}px`;
                    col.style.minWidth=`${width}px`;
                    col.style.maxWidth=`${width}px`;
                    colgroup.appendChild(col);
                });
                return colgroup;
            }
            function marsCloneFloatingHeader(table) {
                const host=marsGetFloatingHeaderHost();
                const scrollHost=marsGetFloatingHeaderScroll();
                const sourceScrollBox=marsGetTableScrollBox(table);
                const thead=table?.querySelector("thead");
                const colgroup=marsBuildFloatingHeaderColGroup(table);
                if(!host || !scrollHost || !sourceScrollBox || !thead || !colgroup) {
                    marsHideFloatingHeader();
                    return;
                }
                const cloneTable=document.createElement("table");
                cloneTable.className=`${table.className} mars-floating-header-table`;
                cloneTable.setAttribute("aria-hidden", "true");
                const inlineStyle=table.getAttribute("style");
                if(inlineStyle) cloneTable.setAttribute("style", inlineStyle);
                cloneTable.appendChild(colgroup);
                cloneTable.appendChild(thead.cloneNode(true));
                scrollHost.innerHTML="";
                scrollHost.appendChild(cloneTable);
                host.hidden=false;
                host.classList.add("is-visible");
                host.dataset.tableId=table.id;
                marsState.floatingHeaderTableId=table.id;
                marsState.floatingHeaderScrollBox=sourceScrollBox;
                marsSyncFloatingHeaderMetrics(table);
            }
            function marsSyncFloatingHeaderMetrics(table) {
                const host=marsGetFloatingHeaderHost();
                const scrollHost=marsGetFloatingHeaderScroll();
                const sourceScrollBox=marsGetTableScrollBox(table);
                const cloneTable=scrollHost?.querySelector("table");
                const thead=table?.querySelector("thead");
                if(!host || !scrollHost || !sourceScrollBox || !cloneTable || !thead) {
                    marsHideFloatingHeader();
                    return;
                }
                const scrollRect=sourceScrollBox.getBoundingClientRect();
                const headerRect=thead.getBoundingClientRect();
                if(scrollRect.width <= 0 || headerRect.height <= 0) {
                    marsHideFloatingHeader();
                    return;
                }
                const colgroup=marsBuildFloatingHeaderColGroup(table);
                if(!colgroup) {
                    marsHideFloatingHeader();
                    return;
                }
                const existingColgroup=cloneTable.querySelector("colgroup");
                if(existingColgroup) cloneTable.replaceChild(colgroup, existingColgroup);
                else cloneTable.insertBefore(colgroup, cloneTable.firstChild);
                const contentLeft=scrollRect.left + (sourceScrollBox.clientLeft || 0);
                const visibleWidth=Math.max(0, Math.ceil(sourceScrollBox.clientWidth || scrollRect.width || 0));
                host.style.left=`${Math.max(0, contentLeft)}px`;
                host.style.width=`${visibleWidth}px`;
                host.style.top="0px";
                scrollHost.style.height=`${Math.ceil(headerRect.height)}px`;
                const inlineStyle=table.getAttribute("style");
                if(inlineStyle) cloneTable.setAttribute("style", inlineStyle);
                const tableWidth=Math.max(
                    colgroup.childElementCount
                        ? Array.from(colgroup.children).reduce((sum, col) => sum + (parseFloat(col.style.width) || 0), 0)
                        : 0,
                    Math.ceil(table.scrollWidth || 0),
                    Math.ceil(table.getBoundingClientRect().width || 0),
                );
                cloneTable.style.width=`${tableWidth}px`;
                cloneTable.style.minWidth=`${tableWidth}px`;
                cloneTable.style.maxWidth=`${tableWidth}px`;
                scrollHost.scrollLeft=sourceScrollBox.scrollLeft;
            }
            function marsResolveFloatingHeaderOwner() {
                const visibleTables=[];
                document.querySelectorAll(".mars-table-scroll[data-table-id]").forEach((scrollBox) => {
                    const table=scrollBox.querySelector("table.mars-data-table[id]");
                    const thead=table?.querySelector("thead");
                    if(!marsTableIsActuallyVisible(table, scrollBox, thead)) return;
                    const theadRect=thead.getBoundingClientRect();
                    const tableRect=table.getBoundingClientRect();
                    visibleTables.push({
                        table,
                        scrollBox,
                        theadTop:theadRect.top,
                        headerHeight:Math.max(1, Math.ceil(theadRect.height || 0)),
                        tableBottom:tableRect.bottom,
                        firstDataRowTop:marsGetFirstVisibleDataRowTop(table),
                    });
                });
                if(!visibleTables.length) return null;
                const hostHeight=Math.ceil(marsGetFloatingHeaderHost()?.getBoundingClientRect().height || 0);
                const hasVisibleReadingTable=visibleTables.some(({ theadTop, firstDataRowTop, tableBottom, headerHeight }) => {
                    const readingBandBottom=Math.max(1, hostHeight || headerHeight);
                    return tableBottom > 0 && (theadTop <= readingBandBottom || firstDataRowTop <= readingBandBottom);
                });
                if(!hasVisibleReadingTable) return null;
                const ownerCandidates=visibleTables.filter(({ theadTop, tableBottom, headerHeight }) => {
                    const floatingHeaderHeight=Math.max(1, hostHeight || headerHeight);
                    return theadTop <= 0 && tableBottom > floatingHeaderHeight;
                });
                if(!ownerCandidates.length) return null;
                ownerCandidates.sort((a,b)=>b.theadTop-a.theadTop);
                const owner=ownerCandidates[0];
                const readingLine=Math.max(1, hostHeight || owner.headerHeight) + 1;
                const shouldReleaseOwner=visibleTables.some((item) => {
                    if(item.table.id===owner.table.id) return false;
                    return item.theadTop > 0 && item.firstDataRowTop <= readingLine;
                });
                if(shouldReleaseOwner) return null;
                return owner;
            }
            function marsRefreshFloatingHeader() {
                const candidate=marsResolveFloatingHeaderOwner();
                if(!candidate) {
                    marsHideFloatingHeader();
                    return;
                }
                if(marsState.floatingHeaderTableId!==candidate.table.id) {
                    marsCloneFloatingHeader(candidate.table);
                    return;
                }
                marsState.floatingHeaderScrollBox=candidate.scrollBox;
                marsSyncFloatingHeaderMetrics(candidate.table);
            }
            function marsScheduleViewportRefresh() {
                if(marsState.floatingHeaderFrameId) return;
                marsState.floatingHeaderFrameId=window.requestAnimationFrame(() => {
                    marsState.floatingHeaderFrameId=null;
                    marsRefreshFloatingHeader();
                    marsUpdateBackToTopVisibility();
                });
            }
            function marsHandleTableHorizontalScroll(event) {
                const scrollBox=event.currentTarget;
                if(scrollBox!==marsState.floatingHeaderScrollBox) return;
                const scrollHost=marsGetFloatingHeaderScroll();
                if(scrollHost) scrollHost.scrollLeft=scrollBox.scrollLeft;
            }
            function marsRegisterTableScrollListeners() {
                document.querySelectorAll(".mars-table-scroll[data-table-id]").forEach((scrollBox) => {
                    if(scrollBox.dataset.headerScrollBound==="1") return;
                    scrollBox.dataset.headerScrollBound="1";
                    scrollBox.addEventListener("scroll", marsHandleTableHorizontalScroll, {passive:true});
                });
            }
            function marsBackToTop() {
                const anchor=document.getElementById("mars-page-top");
                if(anchor) {
                    anchor.scrollIntoView({behavior:"smooth", block:"start"});
                    return;
                }
                window.scrollTo({top:0, behavior:"smooth"});
            }
            function marsUpdateBackToTopVisibility() {
                const button=document.getElementById("mars-back-to-top");
                if(!button) return;
                button.classList.toggle("is-visible", window.scrollY > 600);
            }
            function marsColumnWidthProperty(columnKey) { return columnKey==="feature" ? "--mars-feature-col-width" : columnKey==="secondary" ? "--mars-secondary-col-width" : "--mars-bin-col-width"; }
            function marsColumnDefaultWidth(columnKey) { return columnKey==="feature" ? 220 : columnKey==="secondary" ? 110 : 140; }
            function marsColumnMinWidth(columnKey) { return columnKey==="feature" ? 140 : 90; }
            function marsApplyColumnWidth(table, columnKey, width) {
                if(!table) return;
                const safeWidth=Math.max(marsColumnMinWidth(columnKey), Number(width)||marsColumnDefaultWidth(columnKey));
                table.style.setProperty(marsColumnWidthProperty(columnKey), `${safeWidth}px`);
            }
            function marsSyncStickyLayout(table) {
                if(!table) return;
                const featureHeader=table.querySelector("thead .mars-feature-col");
                if(featureHeader) {
                    const featureWidth=Math.max(140, Math.ceil(featureHeader.getBoundingClientRect().width || marsColumnDefaultWidth("feature")));
                    marsApplyColumnWidth(table, "feature", featureWidth);
                }
                const secondaryHeader=table.querySelector("thead .mars-secondary-col");
                if(secondaryHeader) {
                    const secondaryWidth=Math.max(90, Math.ceil(secondaryHeader.getBoundingClientRect().width || marsColumnDefaultWidth("secondary")));
                    marsApplyColumnWidth(table, "secondary", secondaryWidth);
                }
                const binHeader=table.querySelector("thead .mars-bin-col");
                if(binHeader) {
                    const binWidth=Math.max(90, Math.ceil(binHeader.getBoundingClientRect().width || marsColumnDefaultWidth("bin")));
                    marsApplyColumnWidth(table, "bin", binWidth);
                }
            }
            function marsTablesForScope(scopeToken) {
                if(scopeToken==="all") return Array.from(document.querySelectorAll("table.mars-data-table[id]"));
                if(scopeToken==="pivot") return Array.from(document.querySelectorAll("table.mars-pivot-table[id]"));
                if(scopeToken.startsWith("table:")) {
                    const table=document.getElementById(scopeToken.slice(6));
                    return table ? [table] : [];
                }
                return [];
            }
            function marsSyncScopeLayouts(scopeToken="all") {
                marsTablesForScope(scopeToken).forEach((table)=>marsSyncStickyLayout(table));
                marsRegisterTableScrollListeners();
                marsRefreshFloatingHeader();
            }
            function marsOpenAncestorSections(node) {
                let parent=node?.closest("details");
                while(parent) {
                    parent.open=true;
                    parent=parent.parentElement?.closest("details");
                }
            }
            function marsFindSummaryFeatureNode(feature, visibleOnly=false) {
                const target=marsNormalizeFeatureValue(feature);
                const nodes=Array.from(document.querySelectorAll("#mars-summary-table tbody tr[data-feature]"));
                const candidateNodes=visibleOnly ? nodes.filter((node)=>node.style.display!=="none" && node.offsetParent!==null) : nodes;
                for(const node of candidateNodes) {
                    if(marsNormalizeFeatureValue(node.dataset.feature)===target) return node;
                }
                for(const node of candidateNodes) {
                    if(marsNormalizeFeatureValue(node.dataset.feature).includes(target)) return node;
                }
                return null;
            }
            function marsClearSummaryLocalQuery() {
                marsState.localQueries["mars-summary-table"]="";
                const input=document.getElementById("mars-summary-table-query");
                if(input) input.value="";
            }
            function marsClearJumpHighlight() {
                if(marsState.jumpHighlightArmTimerId) {
                    window.clearTimeout(marsState.jumpHighlightArmTimerId);
                    marsState.jumpHighlightArmTimerId=null;
                }
                if(marsState.jumpHighlightTimerId) {
                    window.clearTimeout(marsState.jumpHighlightTimerId);
                    marsState.jumpHighlightTimerId=null;
                }
                if(marsState.jumpHighlightNode) marsState.jumpHighlightNode.classList.remove("mars-jump-highlight");
                if(marsState.jumpHighlightCell) marsState.jumpHighlightCell.classList.remove("mars-jump-highlight-cell");
                marsState.jumpHighlightNode=null;
                marsState.jumpHighlightCell=null;
            }
            function marsActivateJumpHighlight(node, featureCell) {
                marsClearJumpHighlight();
                if(!node) return;
                marsState.jumpHighlightNode=node;
                marsState.jumpHighlightCell=featureCell||null;
                node.classList.add("mars-jump-highlight");
                if(featureCell) featureCell.classList.add("mars-jump-highlight-cell");
                marsState.jumpHighlightTimerId=window.setTimeout(() => {
                    if(marsState.jumpHighlightNode===node) {
                        node.classList.remove("mars-jump-highlight");
                        if(featureCell) featureCell.classList.remove("mars-jump-highlight-cell");
                        marsState.jumpHighlightTimerId=null;
                        marsState.jumpHighlightNode=null;
                        marsState.jumpHighlightCell=null;
                    }
                }, 3000);
            }
            function marsFocusSummaryFeature(node) {
                if(!node) return;
                const featureCell=node.querySelector(".mars-feature-col");
                const scrollBox=node.closest(".mars-table-scroll");
                marsOpenAncestorSections(node);
                marsClearJumpHighlight();
                window.requestAnimationFrame(() => {
                    node.scrollIntoView({behavior:"smooth", block:"center", inline:"nearest"});
                    if(featureCell) featureCell.scrollIntoView({behavior:"smooth", block:"nearest", inline:"start"});
                    if(scrollBox) scrollBox.scrollTo({left:0, behavior:"smooth"});
                    marsState.jumpHighlightArmTimerId=window.setTimeout(() => {
                        marsState.jumpHighlightArmTimerId=null;
                        marsActivateJumpHighlight(node, featureCell);
                    }, 140);
                });
            }
            function marsJumpToFeature() {
                const input=document.getElementById("mars-feature-jump-input");
                const value=(input?.value||"").trim();
                if(!value) {
                    marsSetError("mars-feature-jump-error", "Enter a feature name to jump.");
                    return;
                }
                let node=marsFindSummaryFeatureNode(value, true);
                if(node) {
                    marsSetError("mars-feature-jump-error", "");
                    marsFocusSummaryFeature(node);
                    return;
                }
                node=marsFindSummaryFeatureNode(value, false);
                if(!node) {
                    marsSetError("mars-feature-jump-error", `Feature "${value}" does not exist in Summary.`);
                    return;
                }
                const globalMatcher=marsBuildMatcher(marsState.globalQuery);
                const summaryParsed=marsParseSummaryExpression(marsState.appliedSummaryExpression);
                if(marsSummaryRowAllowedWithoutLocal(node, globalMatcher, summaryParsed)) {
                    marsClearSummaryLocalQuery();
                    marsQueueRefresh("table:mars-summary-table");
                    window.requestAnimationFrame(() => {
                        window.requestAnimationFrame(() => {
                            const refreshedNode=marsFindSummaryFeatureNode(value, true) || marsFindSummaryFeatureNode(value, false);
                            marsSetError("mars-feature-jump-error", "");
                            marsFocusSummaryFeature(refreshedNode);
                        });
                    });
                    return;
                }
                marsSetError("mars-feature-jump-error", `Feature "${value}" is hidden by data source, global search, or summary filter.`);
            }
            function marsStartColumnResize(event, tableId, columnKey) {
                event.preventDefault();
                event.stopPropagation();
                const table=document.getElementById(tableId);
                if(!table) return;
                const property=marsColumnWidthProperty(columnKey);
                const computed=getComputedStyle(table);
                const startWidth=parseFloat(computed.getPropertyValue(property)) || marsColumnDefaultWidth(columnKey);
                marsState.resizeState={ tableId, columnKey, property, startX:event.clientX, startWidth, pendingWidth:startWidth };
                document.body.style.cursor="col-resize";
                document.body.style.userSelect="none";
            }
            function marsHandleColumnResize(event) {
                if(!marsState.resizeState) return;
                const { startX, startWidth, columnKey } = marsState.resizeState;
                const table=document.getElementById(marsState.resizeState.tableId);
                if(!table) return;
                const minWidth=marsColumnMinWidth(columnKey);
                const nextWidth=Math.max(minWidth, startWidth + (event.clientX - startX));
                marsState.resizeState.pendingWidth=nextWidth;
                if(marsState.resizeFrameScheduled) return;
                marsState.resizeFrameScheduled=true;
                window.requestAnimationFrame(() => {
                    marsState.resizeFrameScheduled=false;
                    if(!marsState.resizeState) return;
                    const activeTable=document.getElementById(marsState.resizeState.tableId);
                    marsApplyColumnWidth(activeTable, marsState.resizeState.columnKey, marsState.resizeState.pendingWidth);
                    marsRefreshFloatingHeader();
                });
            }
            function marsStopColumnResize() {
                if(!marsState.resizeState) return;
                const table=document.getElementById(marsState.resizeState.tableId);
                if(table) marsSyncStickyLayout(table);
                marsState.resizeState=null;
                document.body.style.cursor="";
                document.body.style.userSelect="";
                marsScheduleViewportRefresh();
            }
            function marsRefreshSummaryContext() {
                const summaryFeatures=marsGetSummaryFeatureAllowSet();
                marsState.summaryAllowedFeatures=summaryFeatures instanceof Set ? summaryFeatures : null;
            }
            function marsRefreshSummaryTable() { marsApplyTableFilter("mars-summary-table"); }
            function marsRefreshGenericTables() {
                document.querySelectorAll("table.mars-data-table[id]").forEach((table)=>{
                    if(table.id==="mars-summary-table" || table.classList.contains("mars-pivot-table")) return;
                    marsApplyTableFilter(table.id);
                });
            }
            function marsRefreshPivotScope() {
                marsUpdatePivotViews();
                document.querySelectorAll("table.mars-pivot-table[id]").forEach((table)=>marsApplyTableFilter(table.id));
                marsQueueLayoutSync("pivot");
            }
            function marsRefreshScopeToken(scopeToken) {
                if(scopeToken==="all") {
                    marsRefreshSummaryContext();
                    marsRefreshSummaryTable();
                    marsRefreshGenericTables();
                    marsRefreshPivotScope();
                    marsUpdateChartViews();
                    return;
                }
                if(scopeToken==="pivot") { marsRefreshPivotScope(); return; }
                if(scopeToken==="charts") { marsUpdateChartViews(); return; }
                if(scopeToken==="summary") { marsRefreshSummaryTable(); return; }
                if(scopeToken.startsWith("table:")) { marsApplyTableFilter(scopeToken.slice(6)); }
            }
            function marsFlushRefreshQueue() {
                const tokens = marsState.pendingRefreshTokens.length ? marsState.pendingRefreshTokens.slice() : ["all"];
                marsState.pendingRefreshTokens = [];
                if(tokens.includes("all")) {
                    marsRefreshScopeToken("all");
                    marsScheduleViewportRefresh();
                    return;
                }
                tokens.forEach((token)=>marsRefreshScopeToken(token));
                marsScheduleViewportRefresh();
            }
            window.addEventListener("mousemove", marsHandleColumnResize);
            window.addEventListener("mouseup", marsStopColumnResize);
            window.addEventListener("resize", () => { marsQueueLayoutSync("all"); marsScheduleViewportRefresh(); });
            window.addEventListener("scroll", marsScheduleViewportRefresh, {passive:true});
            document.addEventListener("toggle", () => { marsHideFloatingHeader(); marsQueueLayoutSync("all"); marsScheduleViewportRefresh(); }, true);
            window.addEventListener("DOMContentLoaded", () => {
                marsRegisterTableScrollListeners();
                marsSetDataSources();
                marsQueueLayoutSync("all");
                marsQueueRefresh("all");
                marsUpdateBackToTopVisibility();
                marsRefreshFloatingHeader();
            });
    """
    return template.replace("__SUMMARY_FILTER_COLUMNS__", json.dumps(list(summary_filter_columns), ensure_ascii=False))
