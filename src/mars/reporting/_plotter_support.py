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
