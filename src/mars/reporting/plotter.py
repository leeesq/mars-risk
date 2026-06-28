"""MARS 风控特征趋势图绘制与 HTML 渲染工具。"""

from typing import Literal, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import HTML, display
from matplotlib.ticker import FuncFormatter

from mars.compute import RiskCorrBaseline
from mars.reporting._plotting.support import (
    as_pandas_detail_frame,
    build_image_html,
    build_rollup_row_mask,
    calculate_panel_risk_corr,
    figure_to_base64,
    has_amount_risk_columns,
    normalize_show_risk,
    resolve_feature_reference,
    resolve_feature_trend,
    resolve_global_axis_limits,
    resolve_missing_summary,
    resolve_panel_groups,
    resolve_total_count,
    show_scrollable,
    sort_batch_features,
    split_feature_plot_frames,
    summarize_binning_metrics,
)


class MarsPlotter:
    """
    专注于风控特征效能与稳定性分析的可视化工具。

    该工具统一处理分箱明细表到 Matplotlib 图形和嵌入式 HTML 的转换，
    适用于 Notebook 交互展示和 HTML 报告生成。

    Attributes
    ----------
    UNIT_WIDTH : int
        单个子图的基准宽度。
    UNIT_HEIGHT : float
        单个子图的基准高度。

    Examples
    --------
    >>> plotter = MarsPlotter()
    >>> isinstance(plotter.UNIT_WIDTH, (int, float))
    True
    """

    UNIT_WIDTH = 3  # 单个子图的基准宽度
    UNIT_HEIGHT = 2.75 # 单个子图的基准高度

    @staticmethod
    def _as_pandas_detail_frame(df_detail: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """
        将绘图输入统一转换为 Pandas DataFrame。

        Parameters
        ----------
        df_detail : Union[pd.DataFrame, pl.DataFrame]
            绘图明细表。

        Returns
        -------
        pd.DataFrame
            可直接用于 Matplotlib 和 Pandas 切片的表对象。
        """
        return as_pandas_detail_frame(df_detail)

    @staticmethod
    def _show_scrollable(fig: plt.Figure, dpi: int = 150) -> None:
        """
        将 Matplotlib 图表包装进可滚动、可点击放大的交互式 HTML 容器。

        Parameters
        ----------
        fig : plt.Figure
            待显示的图表对象。
        dpi : int
            图像分辨率。
        """
        show_scrollable(fig, dpi=dpi)

    @staticmethod
    def _build_rollup_row_mask(df_detail: pd.DataFrame) -> pd.Series:
        """标记分箱明细中的汇总行。"""
        return build_rollup_row_mask(df_detail)

    @staticmethod
    def _summarize_binning_metrics(df_detail: pd.DataFrame) -> tuple[float, float, float]:
        """按单个面板口径汇总 IV、KS 和 AUC。"""
        return summarize_binning_metrics(df_detail)

    @staticmethod
    def _calculate_panel_risk_corr(
        panel_df: pd.DataFrame,
        reference_df: pd.DataFrame | None,
    ) -> float:
        """按参考坏率表计算单个面板的 RC。"""
        return calculate_panel_risk_corr(panel_df, reference_df)

    @staticmethod
    def _normalize_show_risk(
        show_risk: str,
    ) -> Literal["count", "amt", "both"]:
        """规范化公开的风险线展示模式。"""
        return normalize_show_risk(show_risk)

    @staticmethod
    def _has_amount_risk_columns(df_detail: pd.DataFrame) -> bool:
        """判断明细表是否具备可绘制的金额风险列。"""
        return has_amount_risk_columns(df_detail)

    @staticmethod
    def _figure_to_base64(fig: plt.Figure, dpi: int = 150, close: bool = True) -> str:
        """将 Matplotlib 图表序列化为 Base64 PNG 字符串。"""
        return figure_to_base64(fig, dpi=dpi, close=close)

    @staticmethod
    def _build_image_html(img_str: str) -> str:
        """为已序列化的 PNG 图片构建可缩放 HTML 片段。"""
        return build_image_html(img_str)

    @staticmethod
    def render_feature_binning_risk_trend_html(
        df_detail: Union[pd.DataFrame, pl.DataFrame],
        feature: str,
        group_col: str = "month",
        target_name: str = "Target",
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        dpi: int | None = 150,
    ) -> str:
        """
        将单个特征的分箱风险趋势图渲染为可嵌入 HTML 片段。

        Parameters
        ----------
        df_detail : Union[pd.DataFrame, pl.DataFrame]
            分箱评估明细表。
        feature : str
            需要绘制的特征名。
        group_col : str
            趋势分组列名。
        target_name : str
            目标变量展示名称。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            当前特征绘图使用的 RC 参考坏率表；传入后图中 RC 与报告口径保持一致。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        dpi : int | None
            输出 PNG 的渲染分辨率。为 ``None`` 时使用默认值 ``150``。

        Returns
        -------
        str
            包含图像与缩放脚本的 HTML 片段。若特征或分组无法绘制，则返回空字符串。

        Examples
        --------
        >>> import pandas as pd
        >>> df_detail = pd.DataFrame({
        ...     "feature": ["age", "age", "age", "age"],
        ...     "month": ["202601", "202601", "202602", "202602"],
        ...     "bin_index": [0, 1, 0, 1],
        ...     "bin_label": ["young", "mature", "young", "mature"],
        ...     "count": [80, 20, 70, 30],
        ...     "bad": [4, 3, 5, 6],
        ...     "bad_rate": [0.05, 0.15, 0.071, 0.20],
        ...     "lift": [0.8, 1.5, 0.9, 1.7],
        ...     "iv_bin": [0.01, 0.03, 0.02, 0.04],
        ...     "ks_bin": [5.0, 12.0, 6.0, 14.0],
        ...     "auc_bin": [0.31, 0.30, 0.32, 0.31],
        ...     "psi_bin": [0.0, 0.0, 0.01, 0.02],
        ... })
        >>> html = MarsPlotter.render_feature_binning_risk_trend_html(df_detail, "age", dpi=80)
        >>> isinstance(html, str)
        True
        """
        fig = MarsPlotter._build_feature_binning_risk_figure(
            df_detail=df_detail,
            feature=feature,
            group_col=group_col,
            target_name=target_name,
            risk_corr_reference_df=risk_corr_reference_df,
            show_risk=show_risk,
        )
        if fig is None:
            return ""

        img_str = MarsPlotter._figure_to_base64(fig, dpi=dpi or 150, close=True)
        return MarsPlotter._build_image_html(img_str)

    @staticmethod
    def _build_feature_binning_risk_figure(
        df_detail: Union[pd.DataFrame, pl.DataFrame],
        feature: str,
        group_col: str = "month",
        target_name: str = "Target",
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
    ) -> plt.Figure | None:
        """构建风险趋势图对象但不直接展示。"""
        df_detail = MarsPlotter._as_pandas_detail_frame(df_detail)
        show_risk = MarsPlotter._normalize_show_risk(show_risk)
        amount_risk_available = MarsPlotter._has_amount_risk_columns(df_detail)
        if show_risk == "amt" and not amount_risk_available:
            raise ValueError("Amount risk columns are unavailable in detail_table.")
        reference_pd = (
            MarsPlotter._as_pandas_detail_frame(risk_corr_reference_df)
            if risk_corr_reference_df is not None
            else None
        )

        plot_df, total_metric_df, total_rollup_df, has_total_panel = (
            split_feature_plot_frames(
                df_detail,
                feature=feature,
                group_col=group_col,
            )
        )
        if plot_df.empty and total_metric_df.empty and total_rollup_df.empty:
            return None

        total_count = resolve_total_count(total_metric_df, total_rollup_df)

        has_target_global = (
            "bad_rate" in total_metric_df.columns
            and total_metric_df["bad_rate"].notna().any()
        )

        if has_target_global:
            global_iv, global_ks, global_auc = MarsPlotter._summarize_binning_metrics(
                total_metric_df,
            )

        trend_str = (
            resolve_feature_trend(total_metric_df, total_rollup_df)
            if has_target_global
            else "n.a."
        )
        miss_str = resolve_missing_summary(total_metric_df, total_count)
        _, all_groups, time_range = resolve_panel_groups(
            plot_df,
            group_col=group_col,
            has_total_panel=has_total_panel,
        )
        feature_reference_pd = resolve_feature_reference(
            reference_pd,
            feature=feature,
        )

        n_panels = len(all_groups)
        if n_panels == 0:
            return None

        total_width = MarsPlotter.UNIT_WIDTH * n_panels
        total_height = MarsPlotter.UNIT_HEIGHT + 0.7
        fig = plt.figure(figsize=(total_width, total_height))

        base_h = 2.5
        fs_title = base_h * 1.8 + 2
        fs_label = base_h * 1.5 + 1.5
        fs_text = base_h * 1.5 + 1

        gs = gridspec.GridSpec(
            1, n_panels,
            figure=fig,
            wspace=0.09,
            left=0.05, right=0.95, top=0.75, bottom=0.15
        )

        if has_target_global:
            summary_str_1 = f"{feature},  {target_name},  Total: {int(total_count)},  {time_range}"
            summary_str_2 = f"IV: {global_iv:.3f},  KS: {global_ks:.1f},  AUC: {global_auc:.2f},  Missing: {miss_str},  Trend: {trend_str}"
        else:
            summary_str_1 = f"{feature}  (Label-Free Mode),  Total: {int(total_count)},  {time_range}"
            summary_str_2 = f"Missing: {miss_str},  PSI Check Only"

        fig.text(
            0.04, 0.94, summary_str_1 + "\n" + summary_str_2,
            fontsize=fs_title + 0.85, va="top", ha="left", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc", alpha=0.8),
        )

        global_max_count, global_max_bad, global_max_amt_bad = (
            resolve_global_axis_limits(
                plot_df,
                group_col=group_col,
                all_groups=all_groups,
                has_target_global=has_target_global,
                amount_risk_available=amount_risk_available,
            )
        )

        to_percent = FuncFormatter(lambda y, _: f"{y:.0%}")

        for i, group in enumerate(all_groups):
            ax = plt.subplot(gs[i])

            for spine in ax.spines.values():
                spine.set_linewidth(0.2)

            df_g = plot_df[plot_df[group_col] == group].sort_values("bin_index")
            if df_g.empty:
                continue

            rc_val = MarsPlotter._calculate_panel_risk_corr(
                df_g,
                feature_reference_pd,
            )

            has_target = "bad_rate" in df_g.columns and df_g["bad_rate"].notna().any()
            x = range(len(df_g))
            labels = df_g["bin_label"].tolist()
            indices = df_g["bin_index"].tolist()
            counts = df_g["count"] / df_g["count"].sum() if "count_dist" not in df_g.columns else df_g["count_dist"]

            label_bar = "Count Dist" if i == 0 else None
            ax.bar(x, counts, color="grey", label=label_bar, alpha=0.4)
            ax.set_ylim(0, global_max_count * 1.3)

            if i == 0:
                ax.yaxis.set_major_formatter(to_percent)
                ax.tick_params(axis="y", labelsize=fs_label + 1.5, colors="grey", length=0)
            else:
                ax.set_yticks([])

            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=90, fontsize=fs_label + 1.5)
            ax.tick_params(axis="x", length=0)

            if has_target:
                bads = df_g["bad_rate"]
                ax2 = ax.twinx()
                for spine in ax2.spines.values():
                    spine.set_linewidth(0.2)

                mask_normal = np.array(indices) >= 0
                mask_special = ~mask_normal
                x_arr = np.array(list(x))
                bads_arr = np.array(bads)
                draw_count_risk = show_risk in {"count", "both"}
                draw_amount_risk = show_risk in {"amt", "both"} and amount_risk_available

                color_red = "#fc5853"
                color_amount_risk = "#d4a017"
                color_blue = "#210fe8"
                color_grey = "#555555"
                color_amt_lift_text = "#6a0dad"
                color_amt_bad_rate_text = "#b57edc"
                color_amt_lift_special_text = "#355cde"
                color_amt_bad_rate_special_text = "#7f8cff"

                if draw_count_risk and mask_normal.any():
                    ax2.plot(x_arr[mask_normal], bads_arr[mask_normal], color=color_red, linewidth=1.2, zorder=1)
                    ax2.scatter(x_arr[mask_normal], bads_arr[mask_normal], color=color_red, s=6.5, zorder=2)

                if draw_count_risk and mask_special.any():
                    ax2.scatter(x_arr[mask_special], bads_arr[mask_special], color=color_blue, s=6.5, zorder=2)

                amount_bads_arr = None
                if draw_amount_risk:
                    amount_bads_arr = df_g["amt_bad_rate"].to_numpy(dtype=float)
                    if mask_normal.any():
                        ax2.plot(
                            x_arr[mask_normal],
                            amount_bads_arr[mask_normal],
                            color=color_amount_risk,
                            linewidth=1.2,
                            zorder=1,
                        )
                        ax2.scatter(
                            x_arr[mask_normal],
                            amount_bads_arr[mask_normal],
                            color=color_amount_risk,
                            s=6.5,
                            zorder=2,
                        )
                    if mask_special.any():
                        ax2.scatter(
                            x_arr[mask_special],
                            amount_bads_arr[mask_special],
                            color=color_amount_risk,
                            s=6.5,
                            zorder=2,
                        )

                risk_axis_max = 0.0
                if draw_count_risk:
                    risk_axis_max = max(risk_axis_max, global_max_bad)
                if draw_amount_risk:
                    risk_axis_max = max(risk_axis_max, global_max_amt_bad)
                y_max_limit = risk_axis_max * 1.25 if risk_axis_max > 0 else 1.0
                ax2.set_ylim(0, y_max_limit)

                if i == len(all_groups) - 1:
                    ax2.yaxis.set_major_formatter(to_percent)
                    ax2.tick_params(axis="y", labelsize=fs_label + 1.5, colors="#a23633", length=0)
                else:
                    ax2.set_yticks([])

                if draw_count_risk:
                    for j, val in enumerate(bads):
                        is_special = indices[j] < 0
                        color_lift_text = color_blue if is_special else "black"

                        if "lift" in df_g.columns:
                            lift_val = df_g["lift"].iloc[j]
                            offset_up = y_max_limit * 0.02
                            ax2.text(j, val + offset_up, f"{lift_val:.1f}", color=color_lift_text, ha="center", va="bottom", fontweight="bold", fontsize=fs_text + 2.6)

                        offset_down = y_max_limit * 0.03
                        ax2.text(j, val - offset_down, f"{val:.1%}", color=color_grey, ha="center", va="top", fontweight="bold", fontsize=fs_text + 0.8)

                if draw_amount_risk and amount_bads_arr is not None:
                    for j, amt_val in enumerate(amount_bads_arr):
                        if np.isnan(amt_val):
                            continue
                        is_special = indices[j] < 0
                        lift_amt_val = df_g["lift_amt"].iloc[j]
                        offset_up = y_max_limit * 0.02
                        offset_down = y_max_limit * 0.03
                        lift_amt_text_color = (
                            color_amt_lift_special_text
                            if is_special
                            else color_amt_lift_text
                        )
                        amt_bad_rate_text_color = (
                            color_amt_bad_rate_special_text
                            if is_special
                            else color_amt_bad_rate_text
                        )
                        ax2.text(
                            j,
                            amt_val + offset_up,
                            f"{lift_amt_val:.1f}",
                            color=lift_amt_text_color,
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            fontsize=fs_text + 2.4,
                        )
                        ax2.text(
                            j,
                            amt_val - offset_down,
                            f"{amt_val:.1%}",
                            color=amt_bad_rate_text_color,
                            ha="center",
                            va="top",
                            fontweight="bold",
                            fontsize=fs_text + 0.6,
                        )

            for j, val in enumerate(counts):
                ax.text(j, max(counts) * 0.05, f"{val:.1%}", color="#333333", ha="center", va="bottom", fontsize=fs_text + 0.5)

            total_count_g = df_g["count"].sum() if "count" in df_g.columns else df_g["total_count"].iloc[0]
            psi_val = df_g["psi_bin"].sum() if "psi_bin" in df_g.columns else 0.0

            g_miss_row = df_g[df_g["bin_index"] == -1]
            g_miss_str = f"{g_miss_row['count'].sum() / total_count_g:.0%}" if not g_miss_row.empty and total_count_g > 0 else "0%"

            if has_target:
                total_bad = df_g["bad"].sum()
                avg_bad_rate = total_bad / total_count_g if total_count_g > 0 else 0
                ax.set_title(f"{group}   ({int(total_bad)}/{int(total_count_g)}, {avg_bad_rate:.1%})", fontsize=fs_title + 0.85, y=1.05, ha="center")

                iv_val, ks_val, auc_val = MarsPlotter._summarize_binning_metrics(df_g)

                rc_str = f"RC:{rc_val:.2f}" if not np.isnan(rc_val) else "RC:n.a."
                rc_color = "red" if (not np.isnan(rc_val) and rc_val < 0.7) else "#555555"

                perf_text = f"IV: {iv_val:.2f},  KS: {ks_val:.1f},  AUC: {auc_val:.2f},"
                ax.text(0.602, 1.015, perf_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_title + 0.85, color="black")
                ax.text(0.607, 1.015, f"  PSI: {psi_val:.2f},", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="red" if psi_val > 0.1 else "black")
                ax.text(0.837, 0.945, f" {rc_str}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.36, color=rc_color)
                ax.text(0.837, 1.015, f" Miss:{g_miss_str}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="#555555")

                if draw_count_risk:
                    ax2.axhline(avg_bad_rate, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
                if draw_amount_risk and "amt_bad_rate" in df_g.columns:
                    total_good_amt = df_g["good_amt"].sum() if "good_amt" in df_g.columns else 0.0
                    total_bad_amt = df_g["bad_amt"].sum() if "bad_amt" in df_g.columns else 0.0
                    total_observed_amt = total_good_amt + total_bad_amt
                    avg_amt_bad_rate = (
                        total_bad_amt / total_observed_amt
                        if total_observed_amt > 0
                        else np.nan
                    )
                    if not np.isnan(avg_amt_bad_rate):
                        ax2.axhline(
                            avg_amt_bad_rate,
                            color=color_amount_risk,
                            linestyle="--",
                            alpha=0.5,
                            linewidth=0.8,
                        )
                df_normal = df_g[df_g["bin_index"] >= 0].sort_values("bin_index")
                if not df_normal.empty:
                    for suffix, idx in [("L", 0), ("R", -1)]:
                        row = df_normal.iloc[idx]
                        lft, bd = row.get("lift", 0), int(row.get("bad", 0))
                        rt = bd / total_bad if total_bad > 0 else 0
                        text = f"{suffix}: {lft:.2f}, {bd}, {rt:.1%}"
                        ax.text(0.02, 0.987 if suffix == "L" else 0.935, text, transform=ax.transAxes, color=color_blue, fontsize=fs_text + 1.8, ha="left", va="top")
            else:
                ax.set_title(f"{group}   (Total: {int(total_count_g)})", fontsize=fs_title + 0.85, y=1.05, ha="center")
                psi_color = "red" if psi_val > 0.1 else "black"
                ax.text(0.48, 1.015, f"PSI: {psi_val:.2f}   |", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_title + 0.85, color=psi_color)
                ax.text(0.52, 1.015, f"Miss: {g_miss_str}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="#555555")

        return fig

    @staticmethod
    def plot_feature_binning_risk_trend(
        df_detail: pd.DataFrame | pl.DataFrame,
        feature: str,
        group_col: str = "month",
        target_name: str = "Target",
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        dpi: int | None = 150,
    ) -> None:
        """
        绘制特征分箱风险趋势图 (支持有标签/无标签模式自适应)。

        该图表集成了特征的：
        - 样本分布 (Counts)
        - 坏率走势 (Bad Rate)
        - 跨期一致性 (RiskCorr)
        - 统计指标 (IV, KS, AUC, PSI)

        Parameters
        ----------
        df_detail : pd.DataFrame | pl.DataFrame
            评估明细数据表，需包含 'feature', 'bin_index', 'count' 等列。
        feature : str
            目标特征名。
        group_col : str
            分组维度列名（如月份、客群）。
        target_name : str
            目标变量名称，用于标题显示。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            当前特征绘图使用的 RC 参考坏率表；传入后图中 RC 与报告口径保持一致。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        dpi : int | None
            绘图分辨率。

        Returns
        -------
        None
            函数通过 IPython display 展示图表，不返回业务对象。

        Examples
        --------
        >>> df_detail = pd.DataFrame({
        ...     "feature": ["age", "age"],
        ...     "month": ["202601", "202602"],
        ...     "bin_index": [0, 0],
        ...     "bin_label": ["all", "all"],
        ...     "count": [100, 120],
        ...     "psi_bin": [0.0, 0.01],
        ... })
        >>> MarsPlotter.plot_feature_binning_risk_trend(df_detail, "age", dpi=80) is None
        True
        """
        fig: plt.Figure | None = MarsPlotter._build_feature_binning_risk_figure(
            df_detail=df_detail,
            feature=feature,
            group_col=group_col,
            target_name=target_name,
            risk_corr_reference_df=risk_corr_reference_df,
            show_risk=show_risk,
        )
        if fig is None:
            return

        MarsPlotter._show_scrollable(fig, dpi=dpi or 150)

    @staticmethod
    def plot_feature_binning_risk_trend_batch(
        df_detail: pd.DataFrame | pl.DataFrame,
        features: list[str],
        group_col: str = "month",
        target_name: str = "Target",
        target_key: str | None = None,
        dpi: int = 150,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        risk_corr_baseline: RiskCorrBaseline = "total",
    ) -> None:
        """
        批量绘制多个特征的分箱风险趋势图。

        支持按指定指标（IV/KS/AUC）对特征进行排序展示。

        Parameters
        ----------
        df_detail : pd.DataFrame | pl.DataFrame
            评估明细数据表。
        features : list[str]
            待绘图的特征名称列表。
        group_col : str
            分组维度列。
        target_name : str
            目标名。
        target_key : str | None
            批量绘图时当前目标列的标识；仅用于日志与标题上下文。
        dpi : int
            图像分辨率。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        sort_by : str
            排序依据指标，可选 'iv', 'ks', 'auc'。
        ascending : bool
            是否升序排列（默认降序，即最重要的特征排在最前面）。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            批量绘图时共用的 RC 参考坏率表。
        risk_corr_baseline : RiskCorrBaseline
            批量绘图阶段默认生效的 RC 基准。

        Returns
        -------
        None
            函数逐个展示图表，不返回业务对象。

        Examples
        --------
        >>> df_detail = pd.DataFrame({
        ...     "feature": ["age", "income"],
        ...     "month": ["202601", "202601"],
        ...     "bin_index": [0, 0],
        ...     "bin_label": ["all", "all"],
        ...     "count": [100, 120],
        ...     "psi_bin": [0.0, 0.01],
        ... })
        >>> MarsPlotter.plot_feature_binning_risk_trend_batch(
        ...     df_detail,
        ...     ["age", "income"],
        ...     dpi=80,
        ...     sort_by="",
        ... ) is None
        True
        """
        df_detail = MarsPlotter._as_pandas_detail_frame(df_detail)

        # 重置交互式容器的显示标记
        display(HTML("<script>window.MARS_PLOTTER_HINT_SHOWN = undefined;</script>"))

        sorted_features = sort_batch_features(
            df_detail,
            features=features,
            group_col=group_col,
            sort_by=sort_by,
            ascending=ascending,
        )

        # 批量入口只负责顺序调度，单图渲染继续复用同一个公开入口。
        for feat in sorted_features:
            MarsPlotter.plot_feature_binning_risk_trend(
                df_detail=df_detail,
                feature=feat,
                group_col=group_col,
                target_name=target_name,
                risk_corr_reference_df=risk_corr_reference_df,
                show_risk=show_risk,
                dpi=dpi
            )
