"""风险趋势图绘制的内部支撑函数。"""

from __future__ import annotations

import base64
import uuid
from io import BytesIO
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import HTML, display

from mars.core.constants import DIVISION_EPSILON


def as_pandas_detail_frame(df_detail: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """将绘图明细表统一转换为 Pandas DataFrame。"""
    if isinstance(df_detail, pl.DataFrame):
        return df_detail.to_pandas()
    return df_detail


def figure_to_base64(fig: plt.Figure, dpi: int = 150, close: bool = True) -> str:
    """将 Matplotlib 图表序列化为 Base64 PNG 字符串。"""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
    buffer.seek(0)
    image_text = base64.b64encode(buffer.read()).decode("utf-8")
    if close:
        plt.close(fig)
    return image_text


def build_image_html(image_text: str) -> str:
    """为已序列化的 PNG 图片构建可缩放 HTML 片段。"""
    unique_id = str(uuid.uuid4())
    container_id = f"cont_{unique_id}"
    image_id = f"img_{unique_id}"
    function_id = unique_id.replace("-", "_")
    return f"""
        <style>
            #{container_id} {{
                width: 100%;
                overflow-x: hidden;
                border: 1px solid #e0e0e0;
                padding: 5px;
                cursor: zoom-in;
                transition: all 0.2s ease;
                margin-bottom: 25px;
            }}
            #{image_id} {{
                width: 100%;
                height: auto;
                display: block;
            }}
        </style>

        <div id="{container_id}" ondblclick="toggleZoom_{function_id}(this)">
            <img id="{image_id}" src="data:image/png;base64,{image_text}" title="双击图片：放大查看细节 / 缩小查看全貌" />
        </div>

        <script>
        function toggleZoom_{function_id}(container) {{
            var img = container.querySelector('img');
            if (img.style.width === '100%' || img.style.width === '') {{
                img.style.width = 'auto';
                img.style.maxWidth = 'none';
                container.style.overflowX = 'auto';
                container.style.cursor = 'zoom-out';
            }} else {{
                img.style.width = '100%';
                img.style.maxWidth = '100%';
                container.style.overflowX = 'hidden';
                container.style.cursor = 'zoom-in';
            }}
        }}
        </script>
        """


def show_scrollable(fig: plt.Figure, dpi: int = 150) -> None:
    """将 Matplotlib 图表包装为 Notebook 可滚动 HTML。"""
    display(HTML(build_image_html(figure_to_base64(fig, dpi=dpi, close=True))))


def build_rollup_row_mask(df_detail: pd.DataFrame) -> pd.Series:
    """标记分箱明细中的汇总行。"""
    rollup_mask = pd.Series(False, index=df_detail.index, dtype=bool)
    if "bin_label" in df_detail.columns:
        rollup_mask = rollup_mask | (df_detail["bin_label"].astype(str) == "Total")
    if "bin_type" in df_detail.columns:
        rollup_mask = rollup_mask | (df_detail["bin_type"].astype(str) == "汇总组")
    return rollup_mask


def summarize_binning_metrics(df_detail: pd.DataFrame) -> tuple[float, float, float]:
    """按单个面板口径汇总 IV、KS 和方向修正后的 AUC。"""
    iv_value = float(df_detail["iv_bin"].sum()) if "iv_bin" in df_detail.columns else 0.0
    ks_value = float(df_detail["ks_bin"].max()) if "ks_bin" in df_detail.columns else 0.0
    auc_value = float(df_detail["auc_bin"].sum()) if "auc_bin" in df_detail.columns else 0.0
    if auc_value < 0.5:
        auc_value = 1 - auc_value
    return iv_value, ks_value, auc_value


def calculate_panel_risk_corr(
    panel_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
) -> float:
    """按参考坏率表计算单个面板的 RC。"""
    if reference_df is None or reference_df.empty:
        return 1.0

    current_df = panel_df[panel_df["bin_index"] >= 0].copy()
    if current_df.empty:
        return 1.0

    merged_df = current_df.merge(
        reference_df[["feature", "bin_index", "base_br"]],
        on=["feature", "bin_index"],
        how="left",
    )
    merged_df = merged_df.dropna(subset=["bad_rate", "base_br"])
    if merged_df.empty:
        return 1.0

    current_values = merged_df["bad_rate"].to_numpy(dtype=float)
    baseline_values = merged_df["base_br"].to_numpy(dtype=float)
    if len(current_values) <= 1:
        return 1.0
    if np.std(current_values) <= DIVISION_EPSILON or np.std(baseline_values) <= DIVISION_EPSILON:
        return 1.0

    corr_value = merged_df["bad_rate"].corr(merged_df["base_br"], method="spearman")
    if np.isnan(corr_value):
        return 1.0
    return float(corr_value)


def normalize_show_risk(show_risk: str) -> Literal["count", "amt", "both"]:
    """规范化公开的风险线展示模式。"""
    normalized = str(show_risk).strip().lower()
    if normalized not in {"count", "amt", "both"}:
        raise ValueError("`show_risk` only supports 'count', 'amt', or 'both'.")
    return cast(Literal["count", "amt", "both"], normalized)


def has_amount_risk_columns(df_detail: pd.DataFrame) -> bool:
    """判断明细表是否具备可绘制的金额风险列。"""
    required_cols = {"amt_bad_rate", "lift_amt"}
    if not required_cols.issubset(df_detail.columns):
        return False
    return bool(df_detail["amt_bad_rate"].notna().any())


def split_feature_plot_frames(
    df_detail: pd.DataFrame,
    *,
    feature: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """拆分单特征绘图所需的普通明细、Total 面板和汇总行。"""
    df_feat: pd.DataFrame = df_detail[df_detail["feature"] == feature].copy()
    if df_feat.empty or group_col not in df_feat.columns:
        return df_feat, pd.DataFrame(), pd.DataFrame(), False

    group_values = df_feat[group_col].astype(str)
    has_total_panel = "Total" in group_values.values
    rollup_mask = build_rollup_row_mask(df_feat)
    plot_df = df_feat.loc[~rollup_mask].copy()
    if has_total_panel:
        total_metric_df = plot_df[plot_df[group_col].astype(str) == "Total"].copy()
        total_rollup_df = df_feat[
            (df_feat[group_col].astype(str) == "Total") & rollup_mask
        ].copy()
    else:
        total_metric_df = plot_df.copy()
        total_rollup_df = df_feat.loc[rollup_mask].copy()
    return plot_df, total_metric_df, total_rollup_df, has_total_panel


def resolve_total_count(total_metric_df: pd.DataFrame, total_rollup_df: pd.DataFrame) -> int:
    """按汇总行、Total 面板、普通明细的优先级解析全局样本数。"""
    if not total_rollup_df.empty and "total_count" in total_rollup_df.columns:
        return int(total_rollup_df["total_count"].iloc[0])
    if "total_count" in total_metric_df.columns and not total_metric_df.empty:
        return int(total_metric_df["total_count"].iloc[0])
    if not total_metric_df.empty:
        return int(total_metric_df["count"].sum())
    return 0


def resolve_feature_trend(
    total_metric_df: pd.DataFrame,
    total_rollup_df: pd.DataFrame,
) -> str:
    """解析图表标题中的单调趋势描述。"""
    trend_source_df = total_rollup_df if not total_rollup_df.empty else total_metric_df
    if "trend" in trend_source_df.columns:
        raw_trend = trend_source_df["trend"].iloc[0]
        if pd.notna(raw_trend) and str(raw_trend).lower() != "undefined":
            return str(raw_trend)

    df_trend_calc = total_metric_df[
        total_metric_df["bin_index"] >= 0
    ].sort_values("bin_index")
    if len(df_trend_calc) <= 1:
        return "n.a."

    x_arr = df_trend_calc["bin_index"].values
    y_arr = df_trend_calc["bad_rate"].values
    if np.std(y_arr) <= DIVISION_EPSILON:
        return "flat"

    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    if corr >= 0.5:
        return f"asc({corr:.2f})"
    if corr <= -0.5:
        return f"desc({corr:.2f})"
    return f"n.a.({corr:.2f})"


def resolve_missing_summary(total_metric_df: pd.DataFrame, total_count: int) -> str:
    """解析图表标题中的全局缺失率文本。"""
    missing_row = total_metric_df[total_metric_df["bin_index"] == -1]
    if missing_row.empty or total_count <= 0:
        return "nan%"
    miss_count = missing_row["count"].sum()
    return f"{miss_count / total_count:.2%}"


def resolve_panel_groups(
    plot_df: pd.DataFrame,
    *,
    group_col: str,
    has_total_panel: bool,
) -> tuple[list[str], list[str], str]:
    """解析图表面板顺序和标题中的时间范围。"""
    groups = [group for group in plot_df[group_col].astype(str).unique() if group != "Total"]
    groups = sorted(groups)
    all_groups = groups + (["Total"] if has_total_panel else [])
    time_range = f"[{groups[0]} ~ {groups[-1]}]" if groups else ""
    return groups, all_groups, time_range


def resolve_feature_reference(
    reference_df: pd.DataFrame | None,
    *,
    feature: str,
) -> pd.DataFrame | None:
    """按当前特征筛选 RC 参考表。"""
    if reference_df is None or reference_df.empty:
        return None
    return reference_df[reference_df["feature"].astype(str) == feature].copy()


def resolve_global_axis_limits(
    plot_df: pd.DataFrame,
    *,
    group_col: str,
    all_groups: list[str],
    has_target_global: bool,
    amount_risk_available: bool,
) -> tuple[float, float, float]:
    """解析所有面板共享的样本占比和风险线轴上限。"""
    global_max_count = 0.0
    global_max_bad = 0.0
    global_max_amt_bad = 0.0
    for group in all_groups:
        tmp_df = plot_df[plot_df[group_col] == group]
        if tmp_df.empty:
            continue

        tmp_counts = (
            tmp_df["count"] / tmp_df["count"].sum()
            if "count_dist" not in tmp_df.columns
            else tmp_df["count_dist"]
        )
        if len(tmp_counts) > 0:
            global_max_count = max(global_max_count, tmp_counts.max())

        if has_target_global and "bad_rate" in tmp_df.columns:
            tmp_bads = tmp_df["bad_rate"]
            if len(tmp_bads) > 0:
                global_max_bad = max(global_max_bad, tmp_bads.max())
        if amount_risk_available and "amt_bad_rate" in tmp_df.columns:
            tmp_amt_bads = tmp_df["amt_bad_rate"].dropna()
            if len(tmp_amt_bads) > 0:
                global_max_amt_bad = max(global_max_amt_bad, tmp_amt_bads.max())
    return global_max_count, global_max_bad, global_max_amt_bad


def sort_batch_features(
    df_detail: pd.DataFrame,
    *,
    features: list[str],
    group_col: str,
    sort_by: str,
    ascending: bool,
) -> list[str]:
    """按指定指标确定批量绘图的特征顺序。"""
    if not sort_by or sort_by.lower() not in {"iv", "ks", "auc"}:
        return features

    feature_stats: list[dict[str, float | str]] = []
    sort_metric = sort_by.lower()
    for feat in features:
        df_feat = df_detail[df_detail["feature"] == feat]
        if df_feat.empty:
            continue
        df_calc = (
            df_feat[df_feat[group_col] == "Total"]
            if "Total" in df_feat[group_col].values
            else df_feat
        )
        val = 0.0
        if sort_metric == "iv":
            val = float(df_calc["iv_bin"].sum())
        elif sort_metric == "ks":
            val = float(df_calc["ks_bin"].max() * 100)
        elif sort_metric == "auc":
            val = float(df_calc["auc_bin"].sum())
            if val < 0.5:
                val = 1 - val
        feature_stats.append({"feature": feat, "score": val})

    df_stats = pd.DataFrame(feature_stats)
    if df_stats.empty:
        return features
    sorted_values = df_stats.sort_values(by="score", ascending=ascending)["feature"].tolist()
    return [str(feature) for feature in sorted_values]
