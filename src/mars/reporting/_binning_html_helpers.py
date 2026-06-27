"""分箱 HTML 报告的无状态内部 helper。"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd

from mars.utils.html import is_missing_html_value


def slugify(value: str) -> str:
    """将任意标题转换为可作 HTML id 的稳定片段。"""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "section"


def wrap_html_section(
    title: str,
    body: str,
    section_id: str,
    subtitle: str | None = None,
    open_by_default: bool = True,
) -> str:
    """将一段 HTML 内容包装成可折叠的报告 section。"""
    open_attr = " open" if open_by_default else ""
    subtitle_html = f'<div class="mars-section-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return f"""
        <details id="{section_id}" class="mars-section"{open_attr}>
            <summary>{html.escape(title)}</summary>
            {subtitle_html}
            <div class="mars-section-body">
                {body}
            </div>
        </details>
        """


def normalize_search_text(*parts: Any) -> str:
    """将多个文本片段合并为前端搜索使用的标准小写串。"""
    joined = " ".join("" if part is None else str(part) for part in parts)
    return " ".join(joined.split()).strip().lower()


def is_percent_column(col_name: Any, *, metric_name: str | None = None) -> bool:
    """判断 HTML 表格列是否应该按百分比格式展示。"""
    col_lower = str(col_name).strip().lower()
    metric_lower = str(metric_name or "").strip().lower()
    non_percent_labels = {"feature", "dtype", "data_source", "target", "y", "bin"}
    if col_lower in non_percent_labels:
        return False
    if metric_lower in {"missing", "bad_rate"} and col_lower not in non_percent_labels:
        return True
    if col_lower in {"pct", "risk", "missing", "missing_rate", "bad_rate", "cum_bad_rate"}:
        return True
    if col_lower.startswith("missing"):
        return True
    return (
        col_lower.endswith(("_rate", "_ratio", "_pct"))
        or col_lower.startswith(("pct_", "risk_"))
        or col_lower.endswith("%")
    )


def interpolate_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    """在两个 RGB 颜色之间按比例插值。"""
    return (
        int(round(start[0] + (end[0] - start[0]) * ratio)),
        int(round(start[1] + (end[1] - start[1]) * ratio)),
        int(round(start[2] + (end[2] - start[2]) * ratio)),
    )


def three_color_rgb(ratio: float, *, reverse: bool = False) -> tuple[int, int, int]:
    """生成红黄绿三段式色阶中的 RGB 颜色。"""
    ratio = max(0.0, min(1.0, ratio))
    low = (248, 105, 107) if not reverse else (99, 190, 123)
    mid = (255, 235, 132)
    high = (99, 190, 123) if not reverse else (248, 105, 107)
    if ratio <= 0.5:
        return interpolate_rgb(low, mid, ratio * 2.0)
    return interpolate_rgb(mid, high, (ratio - 0.5) * 2.0)


def column_colspan(col_name: Any) -> int:
    """根据扁平化列名中的分隔符估算表头 colspan。"""
    return max(1, str(col_name).count("|") + 1)


def format_sort_value(value: Any, sort_type: str) -> str:
    """为前端排序属性生成稳定的字符串化值。"""
    if is_missing_html_value(value):
        return ""
    if sort_type == "number":
        try:
            return f"{float(value):.12g}"
        except (TypeError, ValueError):
            return ""
    return str(value)


def reorder_group_columns(df: pd.DataFrame, leading_cols: list[str]) -> pd.DataFrame:
    """按报告展示习惯重排分组列。"""
    if df.empty:
        return df
    head_cols = [col for col in leading_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in head_cols]
    non_total = sorted([col for col in other_cols if col != "Total"])
    tail_cols = ["Total"] if "Total" in other_cols else []
    return df[head_cols + non_total + tail_cols]


def resolve_chart_sort_column(summary_df: pd.DataFrame, requested: str) -> str | None:
    """解析图表排序列，缺失时回退到风险摘要或首个数值列。"""
    if requested in summary_df.columns:
        return requested
    if "psi_max" in summary_df.columns:
        return "psi_max"
    numeric_cols = [
        col
        for col in summary_df.select_dtypes(include=[np.number]).columns
        if col not in {"bin_index"}
    ]
    return numeric_cols[0] if numeric_cols else None


def semantic_for_metric(metric: str) -> str:
    """返回指标在热力图中的业务方向语义。"""
    metric = str(metric).lower()
    if metric.startswith("missing") or metric in {"psi", "psi_max", "missing_rate"}:
        return "risk_high"
    if metric.startswith("lift") or metric in {"iv", "auc", "ks", "risk_corr", "rc_min"}:
        return "good_high"
    if metric == "mono":
        return "diverging"
    return "neutral"


def escape_attr(value: Any) -> str:
    """按 HTML 属性上下文转义任意值。"""
    return html.escape("" if value is None else str(value), quote=True)


def trend_style_rule(metric: str | None) -> dict[str, Any] | None:
    """返回趋势指标对应的阈值色阶规则。"""
    metric_key = str(metric or "").lower()
    purple_rgb = (160, 98, 196)
    green = (99, 190, 123)
    yellow = (255, 235, 132)
    red = (248, 105, 107)
    rules: dict[str, dict[str, Any]] = {
        "missing": {"anchors": (0.0, 0.5, 1.0), "colors": (green, yellow, red)},
        "psi": {"anchors": (0.0, 0.1, 0.25), "colors": (green, yellow, red)},
        "iv": {"anchors": (0.01, 0.05, 0.1), "colors": (red, yellow, green), "purple_above": 0.2, "purple_rgb": purple_rgb},
        "ks": {"anchors": (4.0, 8.0, 12.0), "colors": (red, yellow, green), "purple_above": 16.0, "purple_rgb": purple_rgb},
        "auc": {"anchors": (0.525, 0.55, 0.575), "colors": (red, yellow, green), "purple_above": 0.625, "purple_rgb": purple_rgb},
        "lift": {"anchors": (1.2, 1.3, 1.4), "colors": (red, yellow, green), "purple_above": 1.5, "purple_rgb": purple_rgb},
        "risk_corr": {"anchors": (0.2, 0.5, 0.8), "colors": (red, yellow, green)},
    }
    return rules.get(metric_key)


def summary_style_rule(metric: str | None) -> dict[str, Any] | None:
    """解析汇总表指标对应的阈值色阶规则。"""
    metric_key = str(metric or "").lower()
    if metric_key in {"iv", "ks", "auc", "psi_max", "rc_min", "lift_max", "missing", "missing_min", "missing_max"}:
        mapped = {
            "psi_max": "psi",
            "rc_min": "risk_corr",
            "lift_max": "lift",
            "missing_min": "missing",
            "missing_max": "missing",
        }.get(metric_key, metric_key)
        return trend_style_rule(mapped)
    if metric_key == "lift_min":
        return {
            "anchors": (0.5, 0.6, 0.7, 0.8),
            "colors": ((160, 98, 196), (99, 190, 123), (255, 235, 132), (248, 105, 107)),
        }
    return None


def sort_metric_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """按 Total 或 feature 列稳定排序趋势指标展示表。"""
    if df.empty:
        return df
    if "Total" in df.columns:
        sort_values = pd.to_numeric(df["Total"], errors="coerce")
        return (
            df.assign(__mars_total_sort=sort_values)
            .sort_values(["__mars_total_sort", "feature"], ascending=[False, True], na_position="last")
            .drop(columns="__mars_total_sort")
        )
    if "feature" in df.columns:
        return df.sort_values("feature", ascending=True)
    return df


def table_sticky_role(column_name: Any) -> str | None:
    """识别表格列是否需要固定在横向滚动区域左侧。"""
    column_lower = str(column_name).strip().lower()
    if column_lower == "feature":
        return "feature"
    if column_lower == "dtype":
        return "secondary"
    return None


def sticky_class_for_role(role: str | None) -> str:
    """将粘性列角色映射为外层单元格 CSS class。"""
    if not role:
        return ""
    return f" mars-sticky-col mars-{role}-col"


def sticky_inner_class_for_role(role: str | None) -> str:
    """将粘性列角色映射为内层单元格 CSS class。"""
    if not role:
        return ""
    return " mars-sticky-cell-inner"


def build_scope_feedback_html(scope_id: str, *, empty_text: str) -> str:
    """构建局部表格筛选状态和空结果提示区域。"""
    return (
        f'<div id="{scope_id}-status" class="mars-result-status" aria-live="polite"></div>'
        f'<div id="{scope_id}-empty" class="mars-empty mars-scope-empty" hidden>{html.escape(empty_text)}</div>'
    )


def build_html_document(
    *,
    report_name: str,
    styles: str,
    body_html: str,
    runtime_script: str,
) -> str:
    """组装自包含 HTML 文档外壳。"""
    template = """
        <!DOCTYPE html>
        <html lang="zh">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>__TITLE__</title>
            <style>
__STYLES__
            </style>
        </head>
        <body>
__BODY_HTML__
            <script>
__RUNTIME_SCRIPT__
            </script>
        </body>
        </html>
        """
    return (
        template
        .replace("__TITLE__", html.escape(report_name))
        .replace("__STYLES__", styles)
        .replace("__BODY_HTML__", body_html)
        .replace("__RUNTIME_SCRIPT__", runtime_script)
    )
