"""分箱评估 HTML 导出实现。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from mars.compute import to_pandas_frame
from mars.core.constants import DIVISION_EPSILON, FLOAT_TOLERANCE
from mars.reporting._binning_html_helpers import (
    build_html_document,
    build_scope_feedback_html,
    column_colspan,
    escape_attr,
    format_sort_value,
    interpolate_rgb,
    is_percent_column,
    normalize_search_text,
    reorder_group_columns,
    resolve_chart_sort_column,
    semantic_for_metric,
    slugify,
    sort_metric_display_df,
    sticky_class_for_role,
    sticky_inner_class_for_role,
    summary_style_rule,
    table_sticky_role,
    three_color_rgb,
    trend_style_rule,
    wrap_html_section,
)
from mars.reporting._matplotlib import require_pyplot
from mars.reporting._time_range import TimeRange, resolve_report_time_range
from mars.reporting.html_assets import build_html_runtime_script, build_html_styles
from mars.utils.html import format_html_value, is_missing_html_value
from mars.utils.logger import logger

if TYPE_CHECKING:
    pass


_CHART_ASSET_THRESHOLD = 50


class _BinningHtmlRenderer:
    """分箱报告 HTML 导出能力。"""

    def __init__(self, report: Any) -> None:
        self._report = report

    def __getattr__(self, name: str) -> Any:
        """将只读数据访问委托给 report 容器。"""
        return getattr(self._report, name)

    def _repr_html_(self: Any) -> str:
        """返回 Jupyter 环境下的评估摘要面板。"""
        # 内部展示逻辑统一转为 Pandas 处理
        df_summary_pd = to_pandas_frame(self.summary_table)
        n_feats = len(df_summary_pd)

        # 简单统计报警数 (修正为新的小写列名 psi_max)
        high_risk_psi = 0
        if "psi_max" in df_summary_pd.columns:
            high_risk_psi = sum(df_summary_pd["psi_max"] > 0.25)

        # 样式定义
        pill_style = (
            "background-color: #e8f4f8; color: #2980b9; border: 1px solid #bce0eb; "
            "padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin-right: 4px;"
        )

        # 动态生成 Trend 链接
        trend_keys = list(self.trend_tables.keys())
        trend_pills = "".join([f"<span style='{pill_style}'>'{k}'</span>" for k in trend_keys])

        lines = []
        # 查看类操作
        lines.append('👉 <code>.show_summary()</code> &nbsp;<span style="color:#7f8c8d">View Feature Ranking</span>')
        lines.append(f'👉 <code>.show_trend(metric)</code> <span style="color:#7f8c8d">metric: {trend_pills}</span>')
        lines.append('👉 <code>.plot_risk_trends()</code> &nbsp;<span style="color:#7f8c8d">Show Binning Charts</span>')

        # 数据访问与导出入口
        lines.append('<hr style="margin: 8px 0; border: 0; border-top: 1px dashed #ccc;">')
        lines.append('📥 <code>.get_evaluation_data()</code> &nbsp;<span style="color:#7f8c8d">Get Raw Data (summary, trends, detail)</span>')
        lines.append('💾 <code>.write_excel()</code> &nbsp;<span style="color:#7f8c8d">Export to Excel</span>')

        return f"""
        <div style="border-left: 5px solid #8e44ad; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0; font-family: 'Segoe UI', sans-serif;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📉 Mars Binning Report</h3>

            <div style="display: flex; gap: 30px; margin-bottom: 12px; font-size: 0.95em;">
                <div><strong>🏷️ Features:</strong> {n_feats}</div>
                <div><strong>🚨 High PSI (>0.25):</strong> <span style="color: {'red' if high_risk_psi > 0 else 'green'}; font-weight:bold;">{high_risk_psi}</span></div>
                <div><strong>📅 Group By:</strong> {self.group_col if self.group_col else 'None (Total Only)'}</div>
            </div>

            <div style="font-size:0.9em; line-height:1.8; color:#2c3e50; background: white; padding: 10px; border: 1px solid #e0e0e0; border-radius: 4px;">
                { "<br>".join(lines) }
            </div>
        </div>
        """

    @staticmethod
    def _slugify(value: str) -> str:
        """将任意标题转换为可作为 HTML id 的稳定片段。"""
        return slugify(value)

    @staticmethod
    def _wrap_html_section(title: str, body: str, section_id: str, subtitle: str | None = None, open_by_default: bool = True) -> str:
        """将一段 HTML 内容包装成可折叠的报告 section。"""
        return wrap_html_section(title, body, section_id, subtitle, open_by_default)

    @staticmethod
    def _normalize_search_text(*parts: Any) -> str:
        """将多个文本片段合并为前端搜索使用的标准小写串。"""
        return normalize_search_text(*parts)

    @classmethod
    def _is_percent_column(
        cls: type[_BinningHtmlRenderer],
        col_name: Any,
        *,
        metric_name: str | None = None,
    ) -> bool:
        """
        判断列在 HTML 表格中是否应按百分比格式展示。

        该判断结合列名和指标名处理 ``missing``、``bad_rate``、``pct`` 等
        风控报告常见字段，避免普通标识列被误格式化。
        """
        return is_percent_column(col_name, metric_name=metric_name)

    @staticmethod
    def _interpolate_rgb(start: Tuple[int, int, int], end: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        """在两个 RGB 颜色之间按比例插值。"""
        return interpolate_rgb(start, end, ratio)

    @classmethod
    def _three_color_rgb(
        cls: type[_BinningHtmlRenderer],
        ratio: float,
        *,
        reverse: bool = False,
    ) -> Tuple[int, int, int]:
        """生成红黄绿三段式色阶中的 RGB 颜色。"""
        return three_color_rgb(ratio, reverse=reverse)

    @classmethod
    def _column_colspan(cls: type[_BinningHtmlRenderer], col_name: Any) -> int:
        """根据扁平化列名中的分隔符估算表头 colspan。"""
        return column_colspan(col_name)

    @classmethod
    def _format_sort_value(cls: type[_BinningHtmlRenderer], value: Any, sort_type: str) -> str:
        """为前端排序属性生成稳定的字符串化值。"""
        return format_sort_value(value, sort_type)

    @staticmethod
    def _reorder_group_columns(df: pd.DataFrame, leading_cols: List[str]) -> pd.DataFrame:
        """
        按报告展示习惯重排分组列。

        指定的前置列会保持在最左侧，``Total`` 固定放在最右侧，其余分组列
        使用稳定排序，保证 Notebook、Excel 与 HTML 报告列序一致。
        """
        return reorder_group_columns(df, leading_cols)

    @staticmethod
    def _resolve_chart_sort_column(summary_df: pd.DataFrame, requested: str) -> str | None:
        """解析图表排序列，缺失请求列时回退到风险摘要或首个数值列。"""
        return resolve_chart_sort_column(summary_df, requested)

    @staticmethod
    def _semantic_for_metric(metric: str) -> str:
        """返回指标在热力图中的业务方向语义。"""
        return semantic_for_metric(metric)

    @staticmethod
    def _escape_attr(value: Any) -> str:
        """按 HTML 属性上下文转义任意值。"""
        return escape_attr(value)

    @staticmethod
    def _trend_style_rule(metric: str | None) -> Dict[str, Any] | None:
        """
        返回趋势指标对应的阈值色阶规则。

        不同指标的好坏方向不同，规则中会显式编码锚点、颜色和部分高亮阈值，
        供 HTML 表格和图例复用。
        """
        return trend_style_rule(metric)

    @classmethod
    def _summary_style_rule(
        cls: type[_BinningHtmlRenderer],
        metric: str | None,
    ) -> Dict[str, Any] | None:
        """解析汇总表指标对应的阈值色阶规则。"""
        return summary_style_rule(metric)

    @staticmethod
    def _sort_metric_display_df(df: pd.DataFrame) -> pd.DataFrame:
        """按 Total 或 feature 列稳定排序趋势指标展示表。"""
        return sort_metric_display_df(df)

    @classmethod
    def _build_threshold_legend_html(
        cls: type[_BinningHtmlRenderer],
        items: List[Tuple[str, str]],
        *,
        legend_id: str,
    ) -> str:
        """构建阈值图例的 HTML chip 列表。"""
        if not items:
            return ""
        chips = "".join(
            f'<span class="mars-legend-chip"><strong>{html.escape(label)}</strong> {html.escape(desc)}</span>'
            for label, desc in items
        )
        return f'<div id="{legend_id}" class="mars-legend">{chips}</div>'

    @classmethod
    def _build_dataset_overview_html(
        cls: type[_BinningHtmlRenderer],
        report_meta: Dict[str, Any],
    ) -> str:
        """
        构建报告首页的数据集概览卡片。

        输入来自 ``report_meta``，输出为自包含 HTML 字符串；缺少元信息时返回
        空字符串，让调用方自然跳过该 section。
        """
        if not report_meta:
            return ""

        def fmt_value(value: Any) -> str:
            """将缺失元信息统一渲染为首页卡片可展示文本。"""
            if value is None or value == "":
                return "N/A"
            return html.escape(str(value))

        cards = [
            ("Rows", fmt_value(report_meta.get("row_count"))),
            ("Features", fmt_value(report_meta.get("feature_count"))),
            ("Profile By", fmt_value(report_meta.get("profile_by_input"))),
            ("Groups", fmt_value(report_meta.get("group_count"))),
        ]
        if report_meta.get("dt_col"):
            cards.append(
                (
                    "Time Range",
                    fmt_value(report_meta.get("start_dt")) + " ~ " + fmt_value(report_meta.get("end_dt")),
                )
            )
        targets = report_meta.get("targets") or []
        cards.append(("Targets", fmt_value(", ".join(str(v) for v in targets) if targets else None)))

        event_rate_map = report_meta.get("event_rate_by_target") or {}
        if event_rate_map:
            rate_text = " | ".join(
                f"{target}: {format_html_value(value, as_percent=True)}" if value is not None else f"{target}: N/A"
                for target, value in event_rate_map.items()
            )
        else:
            rate_text = "N/A"
        cards.append(("Event Rate", html.escape(rate_text)))
        if report_meta.get("feature_start_aware_reference"):
            active_features = report_meta.get("feature_start_reference_features") or []
            cards.append(
                (
                    "Start-Aware Baseline",
                    fmt_value(f"Enabled ({len(active_features)} features)"),
                )
            )

        card_html = "".join(
            f'<div class="mars-kpi-card"><div class="mars-kpi-label">{label}</div><div class="mars-kpi-value">{value}</div></div>'
            for label, value in cards
        )
        return f'<section id="dataset-overview" class="mars-overview-grid">{card_html}</section>'

    @classmethod
    def _build_feature_jump_html(
        cls: type[_BinningHtmlRenderer],
        features: List[str],
    ) -> str:
        """构建 Summary 表格的特征跳转控件。"""
        feature_values = sorted({str(feature) for feature in features if str(feature).strip()})
        if not feature_values:
            return ""
        option_html = "".join(
            f'<option value="{html.escape(feature)}"></option>'
            for feature in feature_values
        )
        return (
            '<div class="mars-feature-jump">'
            '<label class="mars-summary-filter-label" for="mars-feature-jump-input">Jump to Feature</label>'
            '<div class="mars-search-cluster">'
            '<input id="mars-feature-jump-input" class="mars-filter-input" type="search" '
            'list="mars-feature-jump-list" placeholder="Jump to Summary feature..." '
            'onkeydown="if(event.key===\'Enter\'){event.preventDefault();marsJumpToFeature();}" />'
            '<datalist id="mars-feature-jump-list">'
            f'{option_html}'
            '</datalist>'
            '<button type="button" class="mars-mini-button" onclick="marsJumpToFeature()">Go</button>'
            '</div>'
            '<div class="mars-footnote">Jumps to the matching row in Summary. Charts are also searchable.</div>'
            '<div id="mars-feature-jump-results" class="mars-jump-results"></div>'
            '<div id="mars-feature-jump-error" class="mars-search-error"></div>'
            '</div>'
        )

    @staticmethod
    def _table_sticky_role(column_name: Any) -> str | None:
        """识别表格列是否需要固定在横向滚动区域左侧。"""
        return table_sticky_role(column_name)

    @staticmethod
    def _sticky_class_for_role(role: str | None) -> str:
        """将粘性列角色映射为外层单元格 CSS class。"""
        return sticky_class_for_role(role)

    @staticmethod
    def _sticky_inner_class_for_role(role: str | None) -> str:
        """将粘性列角色映射为内层单元格 CSS class。"""
        return sticky_inner_class_for_role(role)

    @staticmethod
    def _build_scope_feedback_html(scope_id: str, *, empty_text: str) -> str:
        """构建局部表格筛选状态和空结果提示区域。"""
        return build_scope_feedback_html(scope_id, empty_text=empty_text)

    @classmethod
    def _build_html_document(
        cls: type[_BinningHtmlRenderer],
        *,
        report_name: str,
        styles: str,
        body_html: str,
        runtime_script: str,
    ) -> str:
        """组装自包含 HTML 文档外壳。"""
        return build_html_document(
            report_name=report_name,
            styles=styles,
            body_html=body_html,
            runtime_script=runtime_script,
        )

    @classmethod
    def _build_global_tools_html(
        cls: type[_BinningHtmlRenderer],
        *,
        feature_jump_html: str,
        source_options: str,
    ) -> str:
        """构建全局搜索、数据源过滤和导出工具条。"""
        export_block_html = (
            '<div class="mars-export-block">'
            '<button type="button" class="mars-clear-button" onclick="marsExportFeatures()">Export Feature List</button>'
            '<div class="mars-export-helper">Export uses Summary expression + Data Source only. Global and local searches only affect display.</div>'
            '</div>'
        )
        return (
            '<div class="mars-global-tools">'
            '<div class="mars-search-cluster">'
            '<input id="mars-global-search" class="mars-filter-input" type="search" placeholder="Global search across tables and charts..." '
            'oninput="marsSetGlobalQuery(this.value)" '
            'onkeydown="if(event.key===\'Enter\'){event.preventDefault();marsJumpToGlobalSearch();}" />'
            '<button type="button" class="mars-clear-button" onclick="marsClearGlobalSearch()">Clear Search</button>'
            '</div>'
            '<label class="mars-toggle"><input id="mars-regex-mode" type="checkbox" onchange="marsSetRegexMode(this.checked)" /> Regex Mode</label>'
            f'{feature_jump_html}'
            '<div class="mars-source-panel">'
            '<div class="mars-source-header">'
            '<strong>Data Source</strong>'
            '<div>'
            '<button type="button" class="mars-source-link" onclick="marsSelectAllSources()">All</button>'
            '<button type="button" class="mars-source-link" onclick="marsClearSources()">Clear</button>'
            '</div>'
            '</div>'
            f'<div id="mars-data-source-group" class="mars-source-options">{source_options}</div>'
            '</div>'
            f'{export_block_html}'
            '</div>'
        )

    def _build_summary_section_html(
        self: Any,
        *,
        summary_pd: pd.DataFrame,
        feature_sources: Dict[str, str],
        sort_by: str,
        ascending: bool,
    ) -> str | None:
        """
        构建特征汇总评估 section。

        该 section 负责排序、字段显隐、阈值图例、表达式筛选框和增强表格
        组装；汇总表为空时返回 ``None``。
        """
        if summary_pd.empty:
            return None

        summary_df = summary_pd.copy()
        if sort_by in summary_df.columns:
            summary_df = summary_df.sort_values(sort_by, ascending=ascending)
        if "mono" in summary_df.columns:
            summary_df = summary_df.drop(columns=["mono"])

        summary_semantics = {col: self._semantic_for_metric(col) for col in summary_df.columns}
        summary_percent_cols = [col for col in summary_df.columns if self._is_percent_column(col)]
        summary_style_rules = {
            col: self._summary_style_rule(col)
            for col in summary_df.columns
            if self._summary_style_rule(col) is not None
        }
        summary_legend_html = self._build_threshold_legend_html(
            [
                ("IV", "0.01 / 0.05 / 0.10, purple above 0.20"),
                ("KS", "4 / 8 / 12, purple above 16"),
                ("AUC", "0.525 / 0.55 / 0.575, purple above 0.625"),
                ("PSI", "0 / 0.1 / 0.25"),
                ("Missing", "0 / 0.5 / 1"),
                ("Lift Min", "0.5 / 0.6 / 0.7 / 0.8, purple at or below 0.5"),
                ("Lift Max", "1.2 / 1.3 / 1.4, purple above 1.5"),
            ],
            legend_id="mars-summary-legend",
        )
        summary_filter_html = (
            '<div class="mars-summary-filter">'
            '<label class="mars-summary-filter-label" for="mars-summary-expression">Feature Filter Expression</label>'
            '<input id="mars-summary-expression" class="mars-filter-input" type="search" '
            'placeholder="e.g. iv > 0.05 & (psi_max < 0.1 | missing < 0.2)" '
            'oninput="marsSetSummaryExpression(this.value)" />'
            '<div id="mars-summary-expression-error" class="mars-search-error"></div>'
            '<div class="mars-footnote">Available metrics: iv, ks, auc, psi_max, rc_min, lift_min, lift_max, missing, missing_min, missing_max.</div>'
            '<div class="mars-footnote">Supported operators: &gt;, &gt;=, &lt;, &lt;=, ==, !=, &amp;, |, ( ). Use &amp; for AND and | for OR.</div>'
            f'{summary_legend_html}'
            '</div>'
        )
        summary_table_html = self._build_enhanced_table_html(
            summary_df,
            "mars-summary-table",
            search_placeholder="Search summary table...",
            feature_sources=feature_sources,
            semantic_map=summary_semantics,
            percent_cols=summary_percent_cols,
            style_rule_map=summary_style_rules,
            extra_toolbar_html=summary_filter_html,
            table_kind="summary",
        )
        summary_section_html: str = self._wrap_html_section(
            "Summary",
            summary_table_html,
            "summary-section",
            subtitle="Feature-level ranking and monitoring summary.",
        )
        return summary_section_html

    def _build_trend_sections_html(
        self: Any,
        *,
        trend_pd_map: Dict[str, pd.DataFrame],
        missing_by_day_pd: pd.DataFrame | None,
        feature_sources: Dict[str, str],
    ) -> List[Tuple[str, str, str]]:
        """
        构建缺失率日趋势和核心指标趋势 section 列表。

        返回值中的元组依次为 section id、导航标题和 HTML 内容，供主页面
        统一拼接导航和主体。
        """
        sections: List[Tuple[str, str, str]] = []

        if missing_by_day_pd is not None and not missing_by_day_pd.empty:
            missing_day_df = self._reorder_group_columns(
                missing_by_day_pd.copy(),
                ["feature", "dtype"],
            )
            missing_day_semantics = {
                col: "risk_high"
                for col in missing_day_df.columns
                if col not in {"feature", "dtype"}
            }
            missing_day_percent_cols = [
                col
                for col in missing_day_df.columns
                if self._is_percent_column(col, metric_name="missing")
            ]
            missing_day_style_rules = {
                col: self._trend_style_rule("missing")
                for col in missing_day_df.columns
                if col not in {"feature", "dtype"}
            }
            missing_day_html = self._build_enhanced_table_html(
                missing_day_df,
                "mars-missing-by-day",
                search_placeholder="Search daily missing trend...",
                feature_sources=feature_sources,
                semantic_map=missing_day_semantics,
                percent_cols=missing_day_percent_cols,
                style_rule_map=missing_day_style_rules,
            )
            missing_day_subtitle = (
                f"Daily missing-rate trend derived from dt_col={self.dt_col}."
            )
        else:
            missing_day_html = (
                '<div class="mars-empty">No daily missing-rate table is available. '
                "Evaluate with a valid time_col to generate Missing By Day data.</div>"
            )
            missing_day_subtitle = (
                "No daily missing-rate data was generated for this report."
            )
        sections.append(
            (
                "missing-day-section",
                "Missing By Day",
                self._wrap_html_section(
                    "Missing Trend By Day",
                    missing_day_html,
                    "missing-day-section",
                    subtitle=missing_day_subtitle,
                ),
            )
        )

        trend_blocks: List[str] = []
        trend_legend_html = self._build_threshold_legend_html(
            [
                ("Missing", "0 / 0.5 / 1"),
                ("PSI", "0 / 0.1 / 0.25"),
                ("IV", "0.01 / 0.05 / 0.10, purple above 0.20"),
                ("KS", "4 / 8 / 12, purple above 16"),
                ("Lift", "1.2 / 1.3 / 1.4, purple above 1.5"),
                ("AUC", "0.525 / 0.55 / 0.575, purple above 0.625"),
                ("Risk Corr", "0.2 / 0.5 / 0.8"),
            ],
            legend_id="mars-trend-legend",
        )
        for metric in ["missing", "psi", "iv", "ks", "lift", "auc", "risk_corr"]:
            if metric not in trend_pd_map:
                continue
            trend_df = self._sort_metric_display_df(
                self._reorder_group_columns(trend_pd_map[metric].copy(), ["feature", "dtype"])
            )
            trend_semantics = {col: self._semantic_for_metric(metric) for col in trend_df.columns if col not in {"feature", "dtype"}}
            trend_percent_cols = [
                col for col in trend_df.columns
                if self._is_percent_column(col, metric_name=metric)
            ]
            trend_style_rules = {
                col: self._trend_style_rule(metric)
                for col in trend_df.columns
                if col not in {"feature", "dtype"}
            }
            table_html = self._build_enhanced_table_html(
                trend_df,
                f"mars-trend-{self._slugify(metric)}",
                search_placeholder=f"Search {metric} trend...",
                feature_sources=feature_sources,
                semantic_map=trend_semantics,
                percent_cols=trend_percent_cols,
                style_rule_map=trend_style_rules,
            )
            trend_blocks.append(f'<details class="mars-metric-block" open><summary>{html.escape(metric.upper())}</summary>{table_html}</details>')
        if trend_blocks:
            sections.append((
                "trend-section",
                "Trend Tables",
                self._wrap_html_section(
                    "Trend Tables",
                    trend_legend_html + "".join(trend_blocks),
                    "trend-section",
                    subtitle="Trend tables with local search, default Total ranking, and Excel-like color scales.",
                ),
            ))

        return sections

    @staticmethod
    def _resolve_chart_embed_mode(
        chart_embed_mode: str,
        chart_count: int,
    ) -> Literal["inline", "asset"]:
        """根据图表数量解析 HTML 图像嵌入模式。"""
        if chart_embed_mode not in {"auto", "inline", "asset"}:
            raise ValueError(
                "chart_embed_mode must be one of 'auto', 'inline', or 'asset'."
            )
        if chart_embed_mode == "asset":
            return "asset"
        if chart_embed_mode == "inline":
            return "inline"
        return "asset" if chart_count > _CHART_ASSET_THRESHOLD else "inline"

    @staticmethod
    def _build_chart_asset_filename(
        *,
        index: int,
        target_name: str,
        feature: str,
    ) -> str:
        """为风险趋势图生成稳定且可读的图片资产文件名。"""
        return (
            f"risk_trend_{index:04d}_{slugify(target_name)}_"
            f"{slugify(feature)}.png"
        )

    @staticmethod
    def _write_chart_asset(figure: Any, asset_path: Path) -> None:
        """将单个风险趋势图写为 PNG 资产并释放 Matplotlib 对象。"""
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot = require_pyplot(feature_name="HTML risk trend asset rendering")
        try:
            figure.savefig(asset_path, format="png", dpi=150, bbox_inches="tight")
        finally:
            pyplot.close(figure)

    @staticmethod
    def _mark_page_view(section_html: str, page_key: str) -> str:
        """为已有 HTML section 添加 SPA 页面视图标记。"""
        return section_html.replace(
            'class="mars-section"',
            f'class="mars-section mars-page-view" data-mars-view="{page_key}"',
            1,
        )

    @staticmethod
    def _page_key_for_section(section_id: str) -> str:
        """将报告 section id 映射为 SPA 页面 key。"""
        return {
            "trend-section": "trends",
            "chart-section": "charts",
        }.get(section_id, section_id.removesuffix("-section"))

    def _build_chart_section_html(
        self: Any,
        *,
        detail_pd: pd.DataFrame,
        summary_pd: pd.DataFrame,
        feature_sources: Dict[str, str],
        max_plots: int,
        sort_by: str,
        ascending: bool,
        chart_embed_mode: str,
        chart_asset_dir: Path | None,
    ) -> str | None:
        """
        构建特征分箱风险趋势图 section。

        方法会复用 ``MarsPlotter`` 的绘图路径，并按目标变量和排序配置生成
        可筛选的图表卡片；无明细数据时返回 ``None``。
        """
        time_range: TimeRange = resolve_report_time_range(
            report_meta=self.report_meta,
            dt_col=self.dt_col,
        )
        if detail_pd.empty:
            return None

        if "y" in detail_pd.columns:
            chart_y_values = [
                str(value)
                for value in detail_pd["y"].dropna().astype(str).drop_duplicates().tolist()
            ]
        else:
            chart_y_values = ["Target"]
        chart_controls = (
            '<div class="mars-inline-controls mars-chart-controls">'
            '<input class="mars-filter-input mars-chart-search" type="search" '
            'placeholder="Search chart features..." '
            'oninput="marsSetLocalQuery(\'mars-chart-cards\', this.value)" />'
            '<div id="mars-chart-cards-error" class="mars-search-error"></div>'
        )
        if len(chart_y_values) > 1:
            chart_options = "".join(
                f'<option value="{html.escape(y_val)}">{html.escape(y_val)}</option>'
                for y_val in chart_y_values
            )
            chart_controls += (
                '<label class="mars-select-group">Chart Target'
                '<select id="mars-chart-target" onchange="marsHandleChartTargetChange()">'
                f'{chart_options}</select></label>'
            )
        chart_controls += "</div>"

        chart_candidates: list[tuple[str, pd.DataFrame, str, str]] = []
        for y_val in chart_y_values:
            if "y" in detail_pd.columns:
                chart_detail_pd = detail_pd[detail_pd["y"].astype(str) == y_val].copy()
            else:
                chart_detail_pd = detail_pd.copy()

            if "target" in summary_pd.columns:
                chart_summary_pd = summary_pd[
                    summary_pd["target"].astype(str) == y_val
                ].copy()
            else:
                chart_summary_pd = summary_pd.copy()
            chart_sort_col = self._resolve_chart_sort_column(chart_summary_pd, sort_by)
            if not chart_summary_pd.empty and chart_sort_col:
                chart_summary_pd = chart_summary_pd.sort_values(
                    chart_sort_col,
                    ascending=ascending,
                )
            if not chart_summary_pd.empty and "feature" in chart_summary_pd.columns:
                chart_features = chart_summary_pd["feature"].drop_duplicates().tolist()[:max_plots]
            else:
                chart_features = chart_detail_pd["feature"].drop_duplicates().tolist()[:max_plots]
            for feature in chart_features:
                data_source = feature_sources.get(str(feature), "UNMAPPED")
                chart_candidates.append(
                    (y_val, chart_detail_pd, str(feature), str(data_source))
                )

        effective_embed_mode = self._resolve_chart_embed_mode(
            chart_embed_mode,
            len(chart_candidates),
        )
        if effective_embed_mode == "asset" and chart_asset_dir is None:
            raise ValueError("chart_asset_dir is required for asset chart embedding.")

        chart_views: List[str] = []
        chart_cards_by_target: Dict[str, List[str]] = {y_val: [] for y_val in chart_y_values}
        try:
            from mars.reporting.plotter import MarsPlotter

            for index, (y_val, chart_detail_pd, feature, data_source) in enumerate(
                chart_candidates,
                start=1,
            ):
                if effective_embed_mode == "inline":
                    block_html = MarsPlotter.render_feature_binning_risk_trend_html(
                        df_detail=chart_detail_pd,
                        feature=feature,
                        group_col=self.detail_group_col or "mars_group",
                        target_name=y_val,
                        show_risk="both",
                        dpi=150,
                        time_range=time_range,
                    )
                    if not block_html:
                        continue
                    image_html = block_html
                else:
                    assert chart_asset_dir is not None
                    figure = MarsPlotter._build_feature_binning_risk_figure(
                        df_detail=chart_detail_pd,
                        feature=feature,
                        group_col=self.detail_group_col or "mars_group",
                        target_name=y_val,
                        show_risk="both",
                        time_range=time_range,
                    )
                    if figure is None:
                        continue
                    filename = self._build_chart_asset_filename(
                        index=index,
                        target_name=y_val,
                        feature=feature,
                    )
                    asset_path = chart_asset_dir / filename
                    self._write_chart_asset(figure, asset_path)
                    asset_src = f"{chart_asset_dir.name}/{filename}"
                    image_html = (
                        f'<img class="mars-risk-trend-image" loading="lazy" '
                        f'data-src="{self._escape_attr(asset_src)}" '
                        f'alt="{html.escape(feature)} risk trend" />'
                    )
                chart_cards_by_target[y_val].append(
                    f'<article class="mars-chart-card" data-feature="{self._escape_attr(feature)}" '
                    f'data-target="{self._escape_attr(y_val)}" '
                    f'data-data-source="{self._escape_attr(data_source)}" '
                    f'data-search-text="{self._escape_attr(self._normalize_search_text(feature, y_val, data_source))}">'
                    f'<h4>{html.escape(feature)}</h4>{image_html}</article>'
                )
        except Exception as exc:
            logger.warning("HTML chart rendering skipped due to error: %s", exc)

        for y_val in chart_y_values:
            chart_cards = chart_cards_by_target[y_val]
            if not chart_cards:
                chart_cards.append('<div class="mars-empty">No chart data available for this target.</div>')
            chart_views.append(
                f'<div class="mars-chart-view" data-y-value="{self._escape_attr(y_val)}">'
                f'{"".join(chart_cards)}</div>'
            )

        if not chart_views:
            for y_val in chart_y_values:
                chart_views.append(
                    f'<div class="mars-chart-view" data-y-value="{self._escape_attr(y_val)}">'
                    f'<div class="mars-empty">Chart rendering is unavailable in the current environment.</div>'
                    f"</div>"
                )

        chart_feedback_html = self._build_scope_feedback_html("mars-chart-cards", empty_text="No charts match current filters.")
        chart_section_html: str = self._wrap_html_section(
            "Charts",
            chart_controls + chart_feedback_html + f'<div id="mars-chart-cards">{"".join(chart_views)}</div>',
            "chart-section",
            subtitle="Risk trend charts rendered from the shared plotting path.",
            open_by_default=False,
        )
        return chart_section_html

    @classmethod
    def _build_threshold_style(
        cls: type[_BinningHtmlRenderer],
        value: float,
        rule: Dict[str, Any],
    ) -> str:
        """
        将数值和阈值规则转换为单元格内联样式。

        规则支持多锚点颜色插值和高值紫色强调，用于复刻 Excel 风格的
        条件格式。
        """
        anchors = tuple(float(v) for v in rule["anchors"])
        colors = tuple(rule["colors"])
        purple_above = rule.get("purple_above")
        purple_rgb = tuple(rule.get("purple_rgb", (160, 98, 196)))

        if len(anchors) != len(colors):
            raise ValueError("Threshold style rules require the same number of anchors and colors.")

        red, green, blue = colors[-1]
        if value <= anchors[0]:
            red, green, blue = colors[0]
        else:
            segment_found = False
            for idx in range(len(anchors) - 1):
                start_anchor = anchors[idx]
                end_anchor = anchors[idx + 1]
                if value <= end_anchor:
                    ratio = (
                        0.5
                        if abs(end_anchor - start_anchor) < FLOAT_TOLERANCE
                        else (value - start_anchor) / (end_anchor - start_anchor)
                    )
                    red, green, blue = cls._interpolate_rgb(colors[idx], colors[idx + 1], ratio)
                    segment_found = True
                    break
            if not segment_found and purple_above is not None:
                high = anchors[-1]
                upper = float(purple_above)
                ratio = (
                    1.0
                    if abs(upper - high) < FLOAT_TOLERANCE
                    else min(max((value - high) / (upper - high), 0.0), 1.0)
                )
                red, green, blue = cls._interpolate_rgb(colors[-1], purple_rgb, ratio)

        alpha = 0.84 if purple_above is not None and value >= float(purple_above) else 0.72
        font_style = " color: #fff; font-weight: 600;" if purple_above is not None and value >= float(purple_above) else ""
        return f"background-color: rgba({red}, {green}, {blue}, {alpha});{font_style}"

    @classmethod
    def _cell_style(
        cls: type[_BinningHtmlRenderer],
        value: Any,
        *,
        semantic: str,
        vmin: float | None,
        vmax: float | None,
        style_rule: Dict[str, Any] | None = None,
        data_bar: bool = False,
    ) -> str:
        """
        为 HTML 表格单元格生成条件格式样式。

        该方法统一处理风险高为坏、指标高为好、发散指标和 data bar 四类
        展示语义；无法解析为有限数值时返回空样式。
        """
        if is_missing_html_value(value):
            return ""

        try:
            num = float(value)
        except (TypeError, ValueError):
            return ""

        if not np.isfinite(num):
            return ""

        styles: List[str] = []
        if style_rule is not None:
            styles.append(cls._build_threshold_style(num, style_rule))

        if vmin is not None and vmax is not None and np.isfinite(vmin) and np.isfinite(vmax):
            span = vmax - vmin
            ratio = 0.5 if abs(span) < FLOAT_TOLERANCE else (num - vmin) / span
            ratio = max(0.0, min(1.0, ratio))
            if style_rule is None:
                if semantic == "risk_high":
                    red, green, blue = cls._three_color_rgb(ratio, reverse=True)
                    styles.append(f"background-color: rgba({red}, {green}, {blue}, 0.72);")
                elif semantic == "good_high":
                    red, green, blue = cls._three_color_rgb(ratio, reverse=False)
                    styles.append(f"background-color: rgba({red}, {green}, {blue}, 0.72);")
                elif semantic == "diverging":
                    max_abs = max(abs(vmin), abs(vmax), FLOAT_TOLERANCE)
                    diverging_ratio = min(abs(num) / max_abs, 1.0)
                    if num >= 0:
                        red, green, blue = cls._interpolate_rgb((255, 235, 132), (99, 190, 123), diverging_ratio)
                    else:
                        red, green, blue = cls._interpolate_rgb((255, 235, 132), (248, 105, 107), diverging_ratio)
                    styles.append(f"background-color: rgba({red}, {green}, {blue}, 0.72);")

            if data_bar and style_rule is None:
                bar_ratio = (
                    ratio
                    if semantic != "diverging"
                    else min(abs(num) / max(abs(vmin), abs(vmax), FLOAT_TOLERANCE), 1.0)
                )
                bar_color = "#8bbf9d" if semantic not in {"risk_high", "diverging"} else "#ea8f8f"
                if semantic == "good_high":
                    bar_color = "#7fc68d"
                if semantic == "diverging" and num >= 0:
                    bar_color = "#7fc68d"
                styles.append(
                    f"background-image: linear-gradient(90deg, {bar_color} 0%, {bar_color} {bar_ratio * 100:.2f}%, transparent {bar_ratio * 100:.2f}%);"
                    "background-repeat: no-repeat;"
                )
        return "".join(styles)

    @classmethod
    def _build_enhanced_table_html(
        cls: type[_BinningHtmlRenderer],
        df: pd.DataFrame,
        table_id: str,
        *,
        search_placeholder: str,
        feature_sources: Dict[str, str] | None = None,
        semantic_map: Dict[str, str] | None = None,
        data_bar_cols: List[str] | None = None,
        percent_cols: List[str] | None = None,
        style_rule_map: Dict[str, Dict[str, Any]] | None = None,
        extra_toolbar_html: str = "",
        table_kind: str = "generic",
        empty_text: str = "No data available.",
    ) -> str:
        """
        构建带搜索、排序、粘性列和条件格式的 HTML 表格。

        该方法是评估报告 v2 表格渲染的统一入口，负责把数据源、指标语义、
        百分比格式和阈值样式编码进前端可识别的 ``data-*`` 属性。
        """
        if df.empty:
            return f'<div class="mars-empty">{html.escape(empty_text)}</div>'

        semantic_map = semantic_map or {}
        data_bar_cols = set(data_bar_cols or [])
        percent_cols = set(percent_cols or [])
        style_rule_map = style_rule_map or {}
        feature_sources = feature_sources or {}

        sort_types = {
            col: "number" if pd.api.types.is_numeric_dtype(df[col]) else "text"
            for col in df.columns
        }
        numeric_bounds: Dict[str, tuple[float | None, float | None]] = {}
        for col in df.columns:
            if sort_types[col] == "number":
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                if numeric_series.notna().any():
                    numeric_bounds[col] = (float(numeric_series.min()), float(numeric_series.max()))
                else:
                    numeric_bounds[col] = (None, None)

        header_cells: List[str] = []
        for col in df.columns:
            sort_type = sort_types[col]
            numeric_class = " is-numeric" if sort_type == "number" else ""
            sticky_role = cls._table_sticky_role(col)
            sticky_class = cls._sticky_class_for_role(sticky_role)
            sticky_inner_class = cls._sticky_inner_class_for_role(sticky_role)
            resize_handle = (
                f'<span class="mars-resize-handle" onmousedown="marsStartColumnResize(event, \'{table_id}\', \'feature\')"></span>'
                if sticky_role == "feature" else ""
            )
            header_cells.append(
                f'<th class="mars-th{numeric_class}{sticky_class}" data-sort-type="{sort_type}" data-col-index="{len(header_cells)}">'
                f'<button type="button" class="mars-sort-button{sticky_inner_class}" onclick="marsSortTable(\'{table_id}\', this)">'
                f'<span class="mars-sort-label">{html.escape(str(col))}</span><span class="mars-sort-indicator"></span>'
                f'</button>{resize_handle}</th>'
            )

        body_rows = []
        for _, row in df.iterrows():
            feature = str(row["feature"]) if "feature" in df.columns else ""
            data_source = str(row.get("data_source", feature_sources.get(feature, "UNMAPPED")))
            metric_payload = {
                str(col): float(row[col])
                for col in df.columns
                if sort_types.get(col) == "number" and not is_missing_html_value(row[col])
            }
            search_text = cls._normalize_search_text(
                feature,
                data_source,
                *[
                    format_html_value(row[col], as_percent=(col in percent_cols))
                    for col in df.columns
                ],
            )

            row_cells = []
            for col in df.columns:
                sort_type = sort_types[col]
                display_val = format_html_value(row[col], as_percent=(col in percent_cols))
                sort_val = cls._format_sort_value(row[col], sort_type)
                numeric_class = " is-numeric" if sort_type == "number" else ""
                semantic = semantic_map.get(col, "neutral")
                style_rule = style_rule_map.get(col)
                vmin, vmax = numeric_bounds.get(col, (None, None))
                style = cls._cell_style(
                    row[col],
                    semantic=semantic,
                    vmin=vmin,
                    vmax=vmax,
                    style_rule=style_rule,
                    data_bar=col in data_bar_cols,
                )
                sticky_role = cls._table_sticky_role(col)
                sticky_class = cls._sticky_class_for_role(sticky_role)
                sticky_inner_class = cls._sticky_inner_class_for_role(sticky_role)
                row_cells.append(
                    f'<td class="mars-td{numeric_class}{sticky_class}" data-col="{cls._escape_attr(col)}" '
                    f'data-sort-value="{cls._escape_attr(sort_val)}" style="{cls._escape_attr(style)}">'
                    f'<span class="mars-cell-text{sticky_inner_class}">{html.escape(display_val)}</span></td>'
                )

            body_rows.append(
                f'<tr data-feature="{cls._escape_attr(feature)}" data-data-source="{cls._escape_attr(data_source)}" '
                f'data-search-text="{cls._escape_attr(search_text)}" '
                f'data-metrics="{cls._escape_attr(json.dumps(metric_payload, ensure_ascii=False, separators=(",", ":")))}">'
                f'{"".join(row_cells)}</tr>'
            )

        return f"""
        <div id="{table_id}-wrap" class="mars-table-wrap">
            <div class="mars-table-toolbar">
                <input
                    id="{table_id}-query"
                    class="mars-filter-input"
                    type="search"
                    placeholder="{html.escape(search_placeholder)}"
                    oninput="marsSetLocalQuery('{table_id}', this.value)"
                />
                <div id="{table_id}-error" class="mars-search-error"></div>
            </div>
            {extra_toolbar_html}
            {cls._build_scope_feedback_html(table_id, empty_text="No rows match current filters.")}
            <div class="mars-table-ownership-sentinel" data-table-id="{table_id}" data-sentinel-role="start" aria-hidden="true"></div>
            <div class="mars-table-scroll" data-table-id="{table_id}">
                <table id="{table_id}" class="mars-data-table" data-sort-col="" data-sort-dir="" data-table-kind="{cls._escape_attr(table_kind)}" style="--mars-feature-col-width: 220px; --mars-secondary-col-width: 110px;">
                    <thead><tr>{''.join(header_cells)}</tr></thead>
                    <tbody>{''.join(body_rows)}</tbody>
                </table>
            </div>
            <div class="mars-table-ownership-sentinel" data-table-id="{table_id}" data-sentinel-role="end" aria-hidden="true"></div>
        </div>
        """

    @classmethod
    def _build_grouped_pivot_section_html(
        cls: type[_BinningHtmlRenderer],
        detail_pd: pd.DataFrame,
        *,
        group_col: str,
        feature_sources: Dict[str, str],
    ) -> str:
        """
        构建按特征分箱和时间分组展开的透视 section。

        该 section 将明细表聚合成首尾组、正常组和空值组三类视图，便于在
        HTML 报告中横向比较各分组的风险、占比和 Lift。
        """
        if detail_pd.empty or group_col not in detail_pd.columns:
            return '<div class="mars-empty">No grouped pivot data available.</div>'

        work_df = detail_pd.copy()
        if "bin_label" in work_df.columns:
            work_df = work_df[work_df["bin_label"].astype(str) != "Total"]
        work_df = work_df[work_df[group_col].astype(str) != "Total"]
        if "bin_type" in work_df.columns:
            work_df = work_df[work_df["bin_type"].astype(str).isin(["首尾组", "正常组", "空值组"])]
        if work_df.empty:
            return '<div class="mars-empty">No grouped pivot data available.</div>'

        missing_idx = -1
        try:
            from mars.feature.binning.base import MarsBinnerBase

            missing_idx = int(MarsBinnerBase.IDX_MISSING)
        except Exception:
            missing_idx = -1

        work_df["bin_index_num"] = pd.to_numeric(work_df.get("bin_index"), errors="coerce")
        missing_mask = work_df["bin_index_num"].eq(missing_idx)
        if "bin_label" in work_df.columns:
            missing_mask = missing_mask | work_df["bin_label"].astype(str).str.lower().eq("missing")

        max_bin_index = (
            work_df.loc[~missing_mask, ["feature", group_col, "bin_index_num"]]
            .groupby(["feature", group_col], dropna=False)["bin_index_num"]
            .transform("max")
        )
        work_df["pivot_bin_type"] = np.where(
            missing_mask,
            "空值组",
            np.where(
                work_df["bin_index_num"].eq(0) | work_df["bin_index_num"].eq(max_bin_index),
                "首尾组",
                "正常组",
            ),
        )

        if "data_source" not in work_df.columns:
            work_df["data_source"] = work_df["feature"].map(feature_sources).fillna("UNMAPPED")
        else:
            work_df["data_source"] = work_df["data_source"].fillna("UNMAPPED")

        y_values = (
            [str(v) for v in work_df["y"].dropna().astype(str).drop_duplicates().tolist()]
            if "y" in work_df.columns
            else ["Target"]
        )
        control_parts = []
        if len(y_values) > 1:
            options = "".join(
                f'<option value="{html.escape(y_value)}">{html.escape(y_value)}</option>'
                for y_value in y_values
            )
            control_parts.append(
                f'<label class="mars-select-group">Pivot Target'
                f'<select id="mars-pivot-target" onchange="marsHandlePivotTargetChange()">{options}</select>'
                f'</label>'
            )

        metrics = [
            ("pct", "pct", "good_high", True, True),
            ("risk", "risk", "risk_high", True, True),
            ("lift", "lift", "good_high", False, True),
            ("bad_count", "bad count", "risk_high", False, False),
            ("total_count", "total count", "good_high", False, False),
            ("iv", "iv", "good_high", False, False),
        ]
        view_blocks: List[str] = []

        for y_val in y_values:
            view_df = work_df.copy()
            if "y" in view_df.columns:
                view_df = view_df[view_df["y"].astype(str) == y_val]
            if view_df.empty:
                continue

            for col in ["pct", "bad", "count", "lift", "iv_bin"]:
                if col not in view_df.columns:
                    view_df[col] = np.nan
            view_df["bad"] = pd.to_numeric(view_df["bad"], errors="coerce").fillna(0.0)
            view_df["count"] = pd.to_numeric(view_df["count"], errors="coerce").fillna(0.0)
            view_df["pct"] = pd.to_numeric(view_df["pct"], errors="coerce").fillna(0.0)
            view_df["lift"] = pd.to_numeric(view_df["lift"], errors="coerce")
            view_df["iv_bin"] = pd.to_numeric(view_df["iv_bin"], errors="coerce").fillna(0.0)

            grouped = (
                view_df.groupby(
                    ["feature", "bin_label", "bin_index", "pivot_bin_type", group_col],
                    dropna=False,
                )
                .agg(
                    bad_count=("bad", "sum"),
                    total_count=("count", "sum"),
                    lift=("lift", "max"),
                    iv=("iv_bin", "sum"),
                )
                .reset_index()
            )
            group_denominator = grouped.groupby(
                ["feature", group_col],
                dropna=False,
            )["total_count"].transform("sum")
            grouped["pct"] = grouped["total_count"] / group_denominator.replace(0, np.nan)
            grouped["pct"] = grouped["pct"].fillna(0.0)
            grouped["risk"] = grouped["bad_count"] / (grouped["total_count"] + DIVISION_EPSILON)

            totals = (
                grouped.groupby(["feature", "bin_label", "bin_index", "pivot_bin_type"], dropna=False)
                .agg(
                    bad_count=("bad_count", "sum"),
                    total_count=("total_count", "sum"),
                    lift=("lift", "max"),
                    iv=("iv", "sum"),
                )
                .reset_index()
            )
            feature_denominator = totals.groupby(["feature"], dropna=False)["total_count"].transform("sum")
            totals["pct"] = totals["total_count"] / feature_denominator.replace(0, np.nan)
            totals["pct"] = totals["pct"].fillna(0.0)
            totals["risk"] = totals["bad_count"] / (totals["total_count"] + DIVISION_EPSILON)
            totals[group_col] = "Total"
            feature_rank = (
                totals.groupby(["feature"], dropna=False)["iv"]
                .sum()
                .reset_index()
                .sort_values(["iv", "feature"], ascending=[False, True])
            )
            grouped = pd.concat([grouped, totals], ignore_index=True, sort=False)
            ordered_groups = ["Total"] + sorted(
                [g for g in grouped[group_col].astype(str).unique().tolist() if g != "Total"]
            )

            metric_bounds = {}
            for metric_key, _, _, _, _ in metrics:
                vals = pd.to_numeric(grouped[metric_key], errors="coerce")
                metric_bounds[metric_key] = (
                    float(vals.min()) if vals.notna().any() else None,
                    float(vals.max()) if vals.notna().any() else None,
                )

            header_top = [
                '<th rowspan="2" class="mars-th mars-sticky-col mars-feature-col">Feature'
                f'<span class="mars-resize-handle" onmousedown="marsStartColumnResize(event, \'mars-pivot-{cls._slugify(y_val)}\', \'feature\')"></span>'
                '</th>',
                '<th rowspan="2" class="mars-th mars-sticky-col mars-bin-col">Bin'
                f'<span class="mars-resize-handle" onmousedown="marsStartColumnResize(event, \'mars-pivot-{cls._slugify(y_val)}\', \'bin\')"></span>'
                '</th>',
            ]
            header_bottom: List[str] = []
            for _, metric_label, _, _, _ in metrics:
                header_top.append(
                    f'<th class="mars-th is-numeric" colspan="{len(ordered_groups)}">{html.escape(metric_label)}</th>'
                )
                for group_value in ordered_groups:
                    header_bottom.append(f'<th class="mars-th is-numeric">{html.escape(str(group_value))}</th>')

            row_chunks: List[str] = []
            total_columns = 2 + len(metrics) * len(ordered_groups)
            grouped["bin_index"] = pd.to_numeric(grouped["bin_index"], errors="coerce")
            grouped = grouped.sort_values(["feature", "bin_index", "bin_label", group_col], na_position="last")
            feature_order = feature_rank["feature"].astype(str).tolist()
            for feature in feature_order:
                feature_df = grouped[grouped["feature"].astype(str) == feature].copy()
                if feature_df.empty:
                    continue
                bin_frames = []
                for (bin_label, bin_index, pivot_bin_type), bin_df in feature_df.groupby(
                    ["bin_label", "bin_index", "pivot_bin_type"],
                    dropna=False,
                ):
                    bin_frames.append((bin_index, str(bin_label), str(pivot_bin_type), bin_df.set_index(group_col)))

                bin_frames.sort(key=lambda item: (float(item[0]) if pd.notna(item[0]) else np.inf, item[1]))
                first_row = True
                for _, bin_label, pivot_bin_type, row_df in bin_frames:
                    row_cells = [
                        f'<td class="mars-td mars-pivot-feature mars-sticky-col mars-feature-col{" mars-pivot-feature-blank" if not first_row else ""}"><span class="mars-cell-text mars-sticky-cell-inner">{html.escape(feature if first_row else "")}</span></td>',
                        f'<td class="mars-td mars-pivot-bin mars-sticky-col mars-bin-col"><span class="mars-cell-text mars-sticky-cell-inner">{html.escape(bin_label)}</span></td>',
                    ]
                    for metric_key, _, semantic, is_percent, show_bar in metrics:
                        vmin, vmax = metric_bounds[metric_key]
                        for group_value in ordered_groups:
                            value = row_df[metric_key].get(group_value, np.nan)
                            display_val = format_html_value(value, as_percent=is_percent)
                            sort_val = cls._format_sort_value(value, "number")
                            style = cls._cell_style(value, semantic=semantic, vmin=vmin, vmax=vmax, data_bar=show_bar)
                            row_cells.append(
                                f'<td class="mars-td is-numeric" data-col="{cls._escape_attr(metric_key)}|{cls._escape_attr(group_value)}" '
                                f'data-sort-value="{cls._escape_attr(sort_val)}" style="{cls._escape_attr(style)}">{html.escape(display_val)}</td>'
                            )

                    search_text = cls._normalize_search_text(feature, bin_label, pivot_bin_type)
                    row_chunks.append(
                        f'<tr data-feature="{cls._escape_attr(feature)}" data-data-source="__aggregate__" '
                        f'data-search-text="{cls._escape_attr(search_text)}">{"".join(row_cells)}</tr>'
                    )
                    first_row = False

                row_chunks.append(
                    f'<tr class="mars-pivot-spacer-row" data-role="spacer" data-feature="{cls._escape_attr(feature)}" '
                    f'data-data-source="__aggregate__"><td colspan="{total_columns}"></td></tr>'
                )

            if row_chunks:
                table_id = f"mars-pivot-{cls._slugify(y_val)}"
                table_html = (
                    f'<div class="mars-table-toolbar">'
                    f'<input id="{table_id}-query" class="mars-filter-input" type="search" placeholder="Search grouped pivot..." '
                    f'oninput="marsSetLocalQuery(\'{table_id}\', this.value)" />'
                    f'<div id="{table_id}-error" class="mars-search-error"></div>'
                    f'</div>'
                    f'{cls._build_scope_feedback_html(table_id, empty_text="No grouped pivot rows match current filters.")}'
                    f'<div class="mars-table-ownership-sentinel" data-table-id="{cls._escape_attr(table_id)}" data-sentinel-role="start" aria-hidden="true"></div>'
                    f'<div class="mars-table-scroll" data-table-id="{cls._escape_attr(table_id)}">'
                    f'<table id="{table_id}" class="mars-data-table mars-pivot-table" data-sort-col="" data-sort-dir="" data-table-kind="pivot" style="--mars-feature-col-width: 220px; --mars-bin-col-width: 140px;">'
                    f'<thead><tr>{"".join(header_top)}</tr><tr>{"".join(header_bottom)}</tr></thead>'
                    f'<tbody>{"".join(row_chunks)}</tbody></table></div>'
                    f'<div class="mars-table-ownership-sentinel" data-table-id="{cls._escape_attr(table_id)}" data-sentinel-role="end" aria-hidden="true"></div>'
                )
                view_blocks.append(
                    f'<div class="mars-pivot-view" data-y-value="{cls._escape_attr(y_val)}">'
                    f'<div class="mars-view-label">{html.escape(y_val)}</div>{table_html}</div>'
                )

        if not view_blocks:
            return '<div class="mars-empty">No grouped pivot data available.</div>'
        controls_html = f'<div class="mars-inline-controls">{"".join(control_parts)}</div>' if control_parts else ""
        return controls_html + "".join(view_blocks)

    def write_html(
        self: Any,
        path: str = "mars_bin_report.html",
        *,
        report_name: str = "MARS Evaluation Report",
        max_plots: int = 500,
        chart_embed_mode: Literal["auto", "inline", "asset"] = "auto",
        sort_by: str = "iv",
        ascending: bool = False,
        include_summary: bool = True,
        include_trends: bool = True,
        include_detail: bool = True,
        include_charts: bool = True,
    ) -> None:
        """
        导出支持页面切换和大规模图表懒加载的交互式 HTML 报告。

        Parameters
        ----------
        path : str
            输出文件路径。
        report_name : str
            HTML 页面标题与报告名称。
        max_plots : int
            每个 target 的图表区域最多展示的特征数量，默认 500。
        chart_embed_mode : Literal["auto", "inline", "asset"]
            图表图片的嵌入模式。``auto`` 在图表数量超过 50 张时写入旁路资产并懒加载；
            ``inline`` 强制内嵌；``asset`` 强制写入与 HTML 同级的资产目录。
        sort_by : str
            图表和汇总视图默认使用的排序指标。
        ascending : bool
            是否按 ``sort_by`` 升序排列。
        include_summary : bool
            是否包含汇总表区域。
        include_trends : bool
            是否包含趋势分析区域。
        include_detail : bool
            是否包含分箱明细区域。
        include_charts : bool
            是否包含图表区域。

        Notes
        -----
        ``inline`` 模式为单文件报告；大规模图表的 ``auto`` / ``asset`` 模式会在
        HTML 同级生成图片资产目录，适合脱离 Notebook 独立分享或归档。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12], "ks": [18.0]})
        >>> detail = pl.DataFrame({"feature": ["age"], "bin_index": [0], "count": [100]})
        >>> report = MarsBinningReport(summary, {}, detail)
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "report.html"
        ...     report.write_html(str(path), include_charts=False, include_detail=False)
        ...     path.exists()
        True
        """
        self._write_html_v2(
            path=path,
            report_name=report_name,
            max_plots=max_plots,
            chart_embed_mode=chart_embed_mode,
            sort_by=sort_by,
            ascending=ascending,
            include_summary=include_summary,
            include_trends=include_trends,
            include_detail=include_detail,
            include_charts=include_charts,
        )

    def _write_html_v2(
        self: Any,
        *,
        path: str,
        report_name: str,
        max_plots: int,
        chart_embed_mode: Literal["auto", "inline", "asset"],
        sort_by: str,
        ascending: bool,
        include_summary: bool,
        include_trends: bool,
        include_detail: bool,
        include_charts: bool,
    ) -> None:
        """
        写入新版页面化 HTML 评估报告及可选图表资产。

        方法负责收集汇总表、明细表、趋势表、图表和数据源筛选配置，随后
        组装导航、概览、各业务 section 与运行脚本并写入目标路径。
        """
        summary_pd = to_pandas_frame(self.summary_table).copy()
        detail_pd = to_pandas_frame(self.detail_table).copy()
        trend_pd_map = {metric: to_pandas_frame(df).copy() for metric, df in self.trend_tables.items()}
        missing_by_day_pd = (
            to_pandas_frame(self.missing_by_day_table).copy()
            if self.missing_by_day_table is not None
            else None
        )

        _ = include_detail
        feature_sources = dict(self.feature_data_source or {})
        if not feature_sources and not summary_pd.empty and {"feature", "data_source"}.issubset(summary_pd.columns):
            feature_sources = dict(
                zip(
                    summary_pd["feature"].astype(str),
                    summary_pd["data_source"].astype(str),
                    strict=False,
                )
            )
        if not feature_sources and not detail_pd.empty and {"feature", "data_source"}.issubset(detail_pd.columns):
            source_df = detail_pd[["feature", "data_source"]].dropna().drop_duplicates()
            feature_sources = dict(
                zip(
                    source_df["feature"].astype(str),
                    source_df["data_source"].astype(str),
                    strict=False,
                )
            )

        n_features = len(summary_pd) if not summary_pd.empty else detail_pd["feature"].nunique() if "feature" in detail_pd.columns else 0
        group_label = self.group_col if self.group_col else "None (Total Only)"
        all_sources = sorted(
            set(feature_sources.values())
            | set(summary_pd["data_source"].astype(str).tolist() if "data_source" in summary_pd.columns else [])
            | set(detail_pd["data_source"].astype(str).tolist() if "data_source" in detail_pd.columns else [])
        )
        safe_report_name = report_name or "MARS Evaluation Report"
        summary_filter_columns = [
            "iv", "ks", "auc", "psi_max", "rc_min",
            "lift_min", "lift_max",
            "missing", "missing_min", "missing_max",
        ]
        feature_jump_html = self._build_feature_jump_html(
            summary_pd["feature"].astype(str).tolist() if "feature" in summary_pd.columns else detail_pd["feature"].astype(str).tolist() if "feature" in detail_pd.columns else []
        )

        html_parts: List[str] = []
        nav_items: List[Tuple[str, str]] = []

        overview_html = self._build_dataset_overview_html(self.report_meta)
        if overview_html:
            html_parts.append(
                self._mark_page_view(
                    self._wrap_html_section(
                        "Dataset Overview",
                        overview_html,
                        "overview-section",
                        subtitle="Dataset context, grouping setup, and target-level baseline stats.",
                    ),
                    "overview",
                )
            )
            nav_items.append(("overview-section", "Overview"))

        if include_summary:
            summary_html = self._build_summary_section_html(
                summary_pd=summary_pd,
                feature_sources=feature_sources,
                sort_by=sort_by,
                ascending=ascending,
            )
            if summary_html:
                html_parts.append(self._mark_page_view(summary_html, "summary"))
                nav_items.append(("summary-section", "Summary"))

        if include_trends:
            for section_id, label, section_html in self._build_trend_sections_html(
                trend_pd_map=trend_pd_map,
                missing_by_day_pd=missing_by_day_pd,
                feature_sources=feature_sources,
            ):
                page_key = self._page_key_for_section(section_id)
                html_parts.append(self._mark_page_view(section_html, page_key))
                nav_items.append((section_id, label))

        if not detail_pd.empty:
            pivot_body = self._build_grouped_pivot_section_html(
                detail_pd,
                group_col=self.detail_group_col or "mars_group",
                feature_sources=feature_sources,
            )
            html_parts.append(
                self._mark_page_view(
                    self._wrap_html_section(
                        "Grouped Pivot",
                        pivot_body,
                        "pivot-section",
                        subtitle="Binned distribution and risk comparison across groups.",
                        open_by_default=False,
                    ),
                    "pivot",
                )
            )
            nav_items.append(("pivot-section", "Grouped Pivot"))

        if include_charts:
            chart_asset_dir = Path(path).with_suffix("").with_name(
                f"{Path(path).stem}_assets"
            )
            chart_html = self._build_chart_section_html(
                detail_pd=detail_pd,
                summary_pd=summary_pd,
                feature_sources=feature_sources,
                max_plots=max_plots,
                sort_by=sort_by,
                ascending=ascending,
                chart_embed_mode=chart_embed_mode,
                chart_asset_dir=chart_asset_dir,
            )
            if chart_html:
                html_parts.append(self._mark_page_view(chart_html, "charts"))
                nav_items.append(("chart-section", "Charts"))

        nav_html = "".join(
            f'<a class="mars-page-nav" data-page="{html.escape(self._page_key_for_section(section_id))}" '
            f'href="#{html.escape(self._page_key_for_section(section_id))}" '
            f'onclick="marsNavigateTo(this.dataset.page); return false;">{html.escape(label)}</a>'
            for section_id, label in nav_items
        )
        source_options = "".join(
            f'<label class="mars-source-option"><input type="checkbox" class="mars-source-checkbox" '
            f'value="{html.escape(source)}" checked onchange="marsHandleDataSourceToggle()" />'
            f'<span>{html.escape(source)}</span></label>'
            for source in all_sources
        )


        sections_html = "".join(html_parts)
        global_tools_html = self._build_global_tools_html(
            feature_jump_html=feature_jump_html,
            source_options=source_options,
        )
        body_html = f"""
            <div id="mars-page-top" aria-hidden="true"></div>
            <div id="mars-floating-header-host" class="mars-floating-header-host" hidden>
                <div id="mars-floating-header-scroll" class="mars-floating-header-scroll"></div>
            </div>
            <div class="mars-page">
                <div class="mars-hero">
                    <h1>{html.escape(safe_report_name)}</h1>
                    <p>Interactive monitoring report with source-aware tables, Excel-like color scales, grouped pivot views, and shared trend charts.</p>
                    <div class="mars-meta">
                        <div class="mars-pill">Features: {n_features}</div>
                        <div class="mars-pill">Trend Metrics: {len(trend_pd_map)}</div>
                        <div class="mars-pill">Group By: {html.escape(str(group_label))}</div>
                    </div>
                    {global_tools_html}
                    <div id="mars-global-error" class="mars-search-error"></div>
                </div>
                <div class="mars-nav">{nav_html}</div>
                {sections_html}
                <div class="mars-footnote">HTML export is self-contained. detail_table remains available in Python and Excel workflows.</div>
            </div>
            <button id="mars-back-to-top" class="mars-back-to-top" type="button" aria-label="Back to top" onclick="marsBackToTop()">Top</button>
        """
        page_html = self._build_html_document(
            report_name=safe_report_name,
            styles=build_html_styles(),
            body_html=body_html,
            runtime_script=build_html_runtime_script(summary_filter_columns=summary_filter_columns),
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(page_html)

        logger.info("Exported binning report to HTML: %s", path)
