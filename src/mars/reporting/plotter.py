"""MARS 风控特征趋势图绘制与 HTML 渲染工具。"""

from __future__ import annotations

import base64
import unicodedata
import uuid
from collections.abc import Mapping
from io import BytesIO
from typing import Literal, Protocol, Union, cast

import numpy as np
import pandas as pd
import polars as pl
from IPython.display import HTML, display

from mars.compute import RiskCorrBaseline
from mars.core.constants import DIVISION_EPSILON
from mars.reporting._matplotlib import _resolve_cjk_font_family, ensure_matplotlib_environment
from mars.reporting._time_range import TimeRange, normalize_time_range

ensure_matplotlib_environment()

import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backend_bases import RendererBase  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.layout_engine import LayoutEngine  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402


class _RiskTrendSummaryBox(FancyBboxPatch):
    """用同一背景包围中文名和统计说明，并按导出 DPI 与 tight 裁剪重测边界。"""

    def __init__(self, fig: plt.Figure, summary: Text, name: Text) -> None:
        self._texts = (summary, name)
        super().__init__(
            (0, 0), 0, 0, boxstyle="round,pad=0.4",
            mutation_scale=float(summary.get_fontsize()) / 72,
            transform=fig.dpi_scale_trans,
            facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.8,
            zorder=summary.get_zorder() - 1, gid="mars-summary-box",
        )

    def _update_bounds(self, renderer: RendererBase | None) -> None:
        """按本次 renderer 测量字形，避免高分辨率导出沿用首次绘制的像素边界。"""
        content = Bbox.union([text.get_window_extent(renderer) for text in self._texts])
        bounds = content.transformed(self.get_transform().inverted())
        self.set_bounds(bounds.x0, bounds.y0, bounds.width, bounds.height)

    def draw(self, renderer: RendererBase) -> None:
        """在实际绘制时更新共同背景。"""
        self._update_bounds(renderer)
        super().draw(renderer)

    def get_window_extent(self, renderer: RendererBase | None = None) -> Bbox:
        """在 tight 边界收集时也使用当前字形尺寸。"""
        self._update_bounds(renderer)
        return super().get_window_extent(renderer)


class _FigureRenderer(Protocol):
    """补足 Matplotlib 类型声明中缺失的私有 renderer 获取协议。"""

    def _get_renderer(self) -> RendererBase:
        """获取当前后端的 renderer，不执行完整绘制。"""
        ...


class _RiskTrendFigure(Figure):
    """仅为趋势图在调用方改变 DPI 时提前更新画布尺寸，避免首帧沿用旧像素高度。"""

    def set_dpi(self, val: float) -> None:
        """设置 DPI 后按当前后端重测已安装的趋势图页眉。"""
        super().set_dpi(val)
        layout = self.get_layout_engine()
        if isinstance(layout, _RiskTrendHeader):
            layout.execute(self)


class _RiskTrendHeader(LayoutEngine):
    """以初始图形尺寸排列中文名和日期；导出时按目标后端重测，并保留 tight 裁剪布局。"""

    _adjust_compatible = True
    _colorbar_gridspec = False

    def __init__(
        self, fig: Figure, summary: Text, name: Text | None,
        dated_axes: list[tuple[Axes, Text]],
    ) -> None:
        super().__init__()
        self._size = (float(fig.get_figwidth()), float(fig.get_figheight()))
        self._axes: list[tuple[Axes, Bbox]] = [
            (ax, ax.get_position().frozen()) for ax in fig.axes
        ]
        self._summary_position = summary.get_position()
        self._summary = summary
        self._name = name
        self._dated_axes = dated_axes

    def execute(self, fig: Figure) -> None:
        """仅获取 renderer 并测量文字，不在重排内部触发 canvas.draw 或完整 Figure.draw。"""
        dpi = fig.dpi
        try:
            # Figure 的设施支持 SVG 等没有 canvas.get_renderer 的后端；内部只截获 renderer。
            renderer = cast(_FigureRenderer, fig)._get_renderer()
            self._layout(fig, renderer)
        finally:
            # SVG 获取 renderer 时会暂设 72 DPI，不能污染调用方或其他导出入口。
            fig.dpi = dpi

    def _layout(self, fig: Figure, renderer: RendererBase) -> None:
        """从初始图形尺寸计算文字所需高度与间距，保留坐标轴的英寸尺寸和文字属性。"""
        width, height = self._size
        summary = self._summary
        summary_x, summary_y = self._summary_position
        dpi = fig.dpi

        # 每次恢复同一物理基准，禁止把前次导出的高度和偏移再次叠加。
        fig.set_size_inches(width, height, forward=False)
        for ax, bounds in self._axes:
            ax.set_position(bounds)
        summary.set_position(self._summary_position)
        title_top = max(
            ax.title.get_window_extent(renderer).y1 for ax, _ in self._axes if ax.get_title()
        )
        padding = renderer.points_to_pixels(float(summary.get_fontsize()) * 0.4)
        summary_bottom = summary.get_window_extent(renderer).y0 - padding
        summary_gap = max(2 / 72, (summary_bottom - title_top) / dpi)

        # 日期沿用组名到指标的可见净距；业务标题、指标和绘图区不移动。
        header_top = title_top
        for ax, date in self._dated_axes:
            title_box = ax.title.get_window_extent(renderer)
            metric_top = max(
                text.get_window_extent(renderer).y1
                for text in ax.texts
                if text.get_transform() == ax.transAxes and text.get_position()[1] == 1.015
            )
            gap = max(0, title_box.y0 - metric_top)
            date.set_y(float(ax.transAxes.inverted().transform((0, title_box.y1 + gap))[1]))
            header_top = max(header_top, date.get_window_extent(renderer).y1)
        date_height = max(0, (header_top + summary_gap * dpi - summary_bottom) / dpi)

        name_height = 0.0
        if self._name is not None:
            # probe 与真实说明保持相同字体、行距及字面量解析规则。
            line = Text(
                text=summary.get_text().splitlines()[0],
                fontproperties=summary.get_fontproperties(), linespacing=1.6,
                parse_math=False,
            )
            line.set_figure(fig)
            line_gap = (
                summary.get_window_extent(renderer).height
                - 2 * line.get_window_extent(renderer).height
            )
            name_height = (self._name.get_window_extent(renderer).height + line_gap) / dpi

        new_height = height + date_height + name_height
        fig.set_size_inches(width, new_height, forward=False)
        for ax, bounds in self._axes:
            ax.set_position((
                bounds.x0, bounds.y0 * height / new_height,
                bounds.width, bounds.height * height / new_height,
            ))
        summary.set_y((summary_y * height + date_height) / new_height)
        if self._name is not None:
            self._name.set_position((
                summary_x, (summary_y * height + date_height + name_height) / new_height,
            ))
        else:
            summary.update_bbox_position_size(renderer)


class MarsPlotter:
    """
    将分箱评估明细绘制为风险趋势图或嵌入式 HTML。

    适用于 Notebook 展示和 HTML 报告生成。绘图读取已计算的明细，
    不拟合分箱规则；有标签时展示风险指标，无标签时展示分布与 PSI。

    Attributes
    ----------
    UNIT_WIDTH : int
        单个子图的基准宽度，单位为英寸。
    UNIT_HEIGHT : float
        单个子图的基准高度，单位为英寸。

    Examples
    --------
    >>> plotter = MarsPlotter()
    >>> isinstance(plotter.UNIT_WIDTH, (int, float))
    True
    """

    UNIT_WIDTH = 3  # 单个子图的基准宽度，单位为英寸。
    UNIT_HEIGHT = 2.75 # 单个子图的基准高度，单位为英寸。

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
        if isinstance(df_detail, pl.DataFrame):
            return df_detail.to_pandas()
        return df_detail

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
        display(
            HTML(
                MarsPlotter._build_image_html(
                    MarsPlotter._figure_to_base64(fig, dpi=dpi, close=True),
                ),
            ),
        )

    @staticmethod
    def _build_rollup_row_mask(df_detail: pd.DataFrame) -> pd.Series:
        """标记分箱明细中的汇总行。"""
        rollup_mask = pd.Series(False, index=df_detail.index, dtype=bool)
        if "bin_label" in df_detail.columns:
            rollup_mask = rollup_mask | (df_detail["bin_label"].astype(str) == "Total")
        if "bin_type" in df_detail.columns:
            rollup_mask = rollup_mask | (df_detail["bin_type"].astype(str) == "汇总组")
        return rollup_mask

    @staticmethod
    def _summarize_binning_metrics(df_detail: pd.DataFrame) -> tuple[float, float, float]:
        """汇总面板 IV 与 AUC，KS 优先取已有 ks 的最大值，否则取 ks_bin。"""
        iv_value = float(df_detail["iv_bin"].sum()) if "iv_bin" in df_detail.columns else 0.0
        ks_column = "ks" if "ks" in df_detail.columns else "ks_bin"
        ks_value = float(df_detail[ks_column].max()) if ks_column in df_detail.columns else 0.0
        auc_value = float(df_detail["auc_bin"].sum()) if "auc_bin" in df_detail.columns else 0.0
        if auc_value < 0.5:
            auc_value = 1 - auc_value
        return iv_value, ks_value, auc_value

    @staticmethod
    def _calculate_panel_risk_corr(
        panel_df: pd.DataFrame,
        reference_df: pd.DataFrame | None,
        undefined_value: float = 1.0,
    ) -> float:
        """按正常箱与参考箱坏率计算 Spearman RC；参考不足或无法定义时返回调用方指定的缺省值。"""
        if reference_df is None or reference_df.empty:
            return undefined_value

        current_df = panel_df[panel_df["bin_index"] >= 0].copy()
        if current_df.empty:
            return undefined_value

        merged_df = current_df.merge(
            reference_df[["feature", "bin_index", "base_br"]],
            on=["feature", "bin_index"],
            how="left",
        )
        merged_df = merged_df.dropna(subset=["bad_rate", "base_br"])
        if merged_df.empty:
            return undefined_value

        current_values = merged_df["bad_rate"].to_numpy(dtype=float)
        baseline_values = merged_df["base_br"].to_numpy(dtype=float)
        if len(current_values) <= 1:
            return undefined_value
        if (
            np.std(current_values) <= DIVISION_EPSILON
            or np.std(baseline_values) <= DIVISION_EPSILON
        ):
            return undefined_value

        corr_value = merged_df["bad_rate"].corr(merged_df["base_br"], method="spearman")
        if np.isnan(corr_value):
            return undefined_value
        return float(corr_value)

    @staticmethod
    def _normalize_show_risk(
        show_risk: str,
    ) -> Literal["count", "amt", "both"]:
        """规范化公开的风险线展示模式。"""
        normalized = str(show_risk).strip().lower()
        if normalized not in {"count", "amt", "both"}:
            raise ValueError("`show_risk` only supports 'count', 'amt', or 'both'.")
        return cast(Literal["count", "amt", "both"], normalized)

    @staticmethod
    def _has_amount_risk_columns(df_detail: pd.DataFrame) -> bool:
        """判断明细表是否具备可绘制的金额风险列。"""
        required_cols = {"amt_bad_rate", "lift_amt"}
        if not required_cols.issubset(df_detail.columns):
            return False
        return bool(df_detail["amt_bad_rate"].notna().any())

    @staticmethod
    def _has_labeled_target_contract(df_detail: pd.DataFrame) -> bool:
        """校验有标签明细的已表现样本计数契约。"""
        has_label_values = any(
            column in df_detail.columns and df_detail[column].notna().any()
            for column in ("bad", "bad_rate")
        )
        if "observed_count" not in df_detail.columns:
            if has_label_values:
                raise ValueError(
                    "Labeled risk trend data requires `observed_count` to distinguish "
                    "observed samples from total samples."
                )
            return False

        has_observed_count = df_detail["observed_count"].notna().any()
        if has_label_values and not has_observed_count:
            raise ValueError(
                "Labeled risk trend data requires non-null `observed_count` values."
            )
        if has_observed_count and "bad" not in df_detail.columns:
            raise ValueError("Labeled risk trend data requires the `bad` column.")
        return bool(has_observed_count)

    @staticmethod
    def _figure_to_base64(fig: plt.Figure, dpi: int = 150, close: bool = True) -> str:
        """将 Matplotlib 图表序列化为 Base64 PNG 字符串。"""
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        buffer.seek(0)
        image_text = base64.b64encode(buffer.read()).decode("utf-8")
        if close:
            plt.close(fig)
        return image_text

    @staticmethod
    def _build_image_html(img_str: str) -> str:
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
            <img id="{image_id}" src="data:image/png;base64,{img_str}" title="双击图片：放大查看细节 / 缩小查看全貌" />
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

    @staticmethod
    def render_feature_binning_risk_trend_html(
        df_detail: Union[pd.DataFrame, pl.DataFrame],
        feature: str,
        group_col: str = "month",
        target_name: str = "Target",
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        dpi: int | None = 150,
        time_range: tuple[str, str] | None = None,
        *,
        include_total: bool = True,
    ) -> str:
        """
        将单个特征的分箱风险趋势图渲染为可嵌入 HTML 片段。

        Parameters
        ----------
        df_detail : Union[pd.DataFrame, pl.DataFrame]
            已计算的分箱评估明细表，需含特征、分组、箱编号、箱标签和样本数。
            有标签数据还需 ``observed_count`` 与 ``bad``，以区分已表现样本和全样本。
        feature : str
            需要绘制的英文特征 ID，按 ``feature`` 列精确匹配。
        group_col : str
            分组列名，默认 ``"month"``；每组生成一个面板，不作为日期范围推断依据。
        target_name : str
            图中显示的目标名称，默认 ``"Target"``；只控制文字，不筛选目标行。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            RC 参考坏率表，包含 ``feature``、``bin_index`` 和 ``base_br``。
            按特征与正常箱匹配；默认不提供参考，图中 RC 沿用 1.0 的回退值。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        dpi : int | None
            输出 PNG 的渲染分辨率。为 ``None`` 时使用默认值 ``150``。

        time_range : tuple[str, str] | None
            本次数据有效 ``time_col`` 的最小和最大日期，用于图中日期说明。
            必须显式提供合法范围；默认 None 会抛出 ``ValueError``，不从分组名推测日期。
        include_total : bool
            是否显示 Total 面板，默认显示；不改变全局统计、共享坐标范围或 RC 参考。
            关闭后没有其他面板时抛出 ``ValueError``。

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
        >>> html = MarsPlotter.render_feature_binning_risk_trend_html(
        ...     df_detail, "age", dpi=80, time_range=("202601", "202602")
        ... )
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
            time_range=time_range,
            include_total=include_total,
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
        time_range: tuple[str, str] | None = None,
        *,
        include_total: bool = True,
        display_name: str | None = None,
        group_time_ranges: Mapping[str | None, TimeRange | None] | None = None,
        preserve_undefined: bool = False,
    ) -> plt.Figure | None:
        """构建风险趋势图对象但不直接展示。"""
        normalized_time_range: TimeRange = normalize_time_range(time_range)
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

        df_feat: pd.DataFrame = df_detail[df_detail["feature"] == feature].copy()
        if df_feat.empty or group_col not in df_feat.columns:
            return None

        group_values = df_feat[group_col].astype(str)
        has_total_panel = "Total" in group_values.values
        rollup_mask = MarsPlotter._build_rollup_row_mask(df_feat)
        plot_df = df_feat.loc[~rollup_mask].copy()
        if has_total_panel:
            total_metric_df = plot_df[plot_df[group_col].astype(str) == "Total"].copy()
            total_rollup_df = df_feat[
                (df_feat[group_col].astype(str) == "Total") & rollup_mask
            ].copy()
        else:
            total_metric_df = plot_df.copy()
            total_rollup_df = df_feat.loc[rollup_mask].copy()

        if not total_rollup_df.empty and "total_count" in total_rollup_df.columns:
            total_count = int(total_rollup_df["total_count"].iloc[0])
        elif "total_count" in total_metric_df.columns and not total_metric_df.empty:
            total_count = int(total_metric_df["total_count"].iloc[0])
        elif not total_metric_df.empty:
            total_count = int(total_metric_df["count"].sum())
        else:
            total_count = 0

        has_observed_risk_global = (
            "bad_rate" in total_metric_df.columns
            and total_metric_df["bad_rate"].notna().any()
        )
        has_labeled_target = MarsPlotter._has_labeled_target_contract(total_metric_df)

        if has_observed_risk_global:
            global_iv, global_ks, global_auc = MarsPlotter._summarize_binning_metrics(
                total_metric_df,
            )

        trend_str = "n.a."
        if has_observed_risk_global:
            trend_source_df = total_rollup_df if not total_rollup_df.empty else total_metric_df
            if "trend" in trend_source_df.columns:
                raw_trend = trend_source_df["trend"].iloc[0]
                if pd.notna(raw_trend) and str(raw_trend).lower() != "undefined":
                    trend_str = str(raw_trend)

            if trend_str == "n.a.":
                df_trend_calc = total_metric_df[
                    total_metric_df["bin_index"] >= 0
                ].sort_values("bin_index")
                if len(df_trend_calc) > 1:
                    x_arr = df_trend_calc["bin_index"].values
                    y_arr = df_trend_calc["bad_rate"].values
                    if np.std(y_arr) > DIVISION_EPSILON:
                        corr = np.corrcoef(x_arr, y_arr)[0, 1]
                        if corr >= 0.5:
                            trend_str = f"asc({corr:.2f})"
                        elif corr <= -0.5:
                            trend_str = f"desc({corr:.2f})"
                        else:
                            trend_str = f"n.a.({corr:.2f})"
                    else:
                        trend_str = "flat"

        missing_row = total_metric_df[total_metric_df["bin_index"] == -1]
        if not missing_row.empty and total_count > 0:
            miss_count = missing_row["count"].sum()
            miss_str = f"{miss_count / total_count:.2%}"
        elif preserve_undefined and total_count > 0:
            miss_str = "0.00%"
        else:
            miss_str = "nan%"

        groups: list[str | None] = sorted(
            group for group in plot_df[group_col].dropna().map(str).unique() if group != "Total"
        )
        if plot_df[group_col].isna().any():
            groups.append(None)
        all_groups = groups + (["Total"] if has_total_panel and include_total else [])
        time_range_label = f"[{normalized_time_range[0]} ~ {normalized_time_range[1]}]"

        feature_reference_pd: pd.DataFrame | None = None
        if reference_pd is not None and not reference_pd.empty:
            feature_reference_pd = reference_pd[
                reference_pd["feature"].astype(str) == feature
            ].copy()

        n_panels = len(all_groups)
        if n_panels == 0:
            if has_total_panel and not include_total:
                raise ValueError("`include_total=False` leaves no risk trend panels to plot.")
            return None

        total_width = MarsPlotter.UNIT_WIDTH * n_panels
        display_lines = (
            MarsPlotter._wrap_display_name(display_name, width=max(30, n_panels * 35))
            if display_name else []
        )
        total_height = MarsPlotter.UNIT_HEIGHT + 0.7
        fig = plt.figure(figsize=(total_width, total_height), FigureClass=_RiskTrendFigure)

        base_h = 2.5
        fs_title = base_h * 1.8 + 2
        fs_label = base_h * 1.5 + 1.5
        fs_text = base_h * 1.5 + 1

        gs = gridspec.GridSpec(
            1, n_panels,
            figure=fig,
            wspace=0.09,
            left=0.05, right=0.95, top=0.75, bottom=0.15,
        )

        if has_labeled_target:
            summary_str_1 = f"{feature},  {target_name},  Total: {int(total_count)},  {time_range_label}"
            if has_observed_risk_global:
                ks_text = f"{global_ks:.1f}" if np.isfinite(global_ks) else "n.a."
                summary_str_2 = f"IV: {global_iv:.3f},  KS: {ks_text},  AUC: {global_auc:.2f},  Missing: {miss_str},  Trend: {trend_str}"
            else:
                summary_str_2 = f"IV: n.a.,  KS: n.a.,  AUC: n.a.,  Missing: {miss_str},  Trend: n.a."
        else:
            summary_str_1 = f"{feature}  (Label-Free Mode),  Total: {int(total_count)},  {time_range_label}"
            summary_str_2 = f"Missing: {miss_str},  PSI Check Only"

        summary_text = fig.text(
            0.04, 0.94, summary_str_1 + "\n" + summary_str_2,
            fontsize=fs_title + 0.85, va="top", ha="left", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc", alpha=0.8),
        )
        summary_text.set_parse_math(False)
        dated_axes: list[tuple[Axes, Text]] = []

        global_max_count = 0.0
        global_max_bad = 0.0
        global_max_amt_bad = 0.0
        # Total 开关只影响可见面板。共享坐标范围仍使用原有全部组，避免缩放误变。
        scale_groups = groups + (["Total"] if has_total_panel else [])
        for group in scale_groups:
            tmp_df = MarsPlotter._select_group_rows(plot_df, group_col, group)
            if tmp_df.empty:
                continue

            tmp_counts = (
                tmp_df["count"] / tmp_df["count"].sum()
                if "count_dist" not in tmp_df.columns
                else tmp_df["count_dist"]
            )
            if len(tmp_counts) > 0:
                global_max_count = max(global_max_count, tmp_counts.max())

            if has_observed_risk_global and "bad_rate" in tmp_df.columns:
                tmp_bads = tmp_df["bad_rate"]
                if len(tmp_bads) > 0:
                    global_max_bad = max(global_max_bad, tmp_bads.max())
            if amount_risk_available and "amt_bad_rate" in tmp_df.columns:
                tmp_amt_bads = tmp_df["amt_bad_rate"].dropna()
                if len(tmp_amt_bads) > 0:
                    global_max_amt_bad = max(global_max_amt_bad, tmp_amt_bads.max())

        to_percent = FuncFormatter(lambda y, _: f"{y:.0%}")

        for i, group in enumerate(all_groups):
            ax = plt.subplot(gs[i])

            for spine in ax.spines.values():
                spine.set_linewidth(0.2)

            df_g = MarsPlotter._select_group_rows(plot_df, group_col, group).sort_values("bin_index")
            if df_g.empty:
                continue

            rc_val = MarsPlotter._calculate_panel_risk_corr(
                df_g,
                feature_reference_pd,
                undefined_value=float("nan") if preserve_undefined else 1.0,
            )

            has_observed_risk = (
                "bad_rate" in df_g.columns and df_g["bad_rate"].notna().any()
            )
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

            if has_observed_risk:
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
            psi_val = df_g["psi_bin"].sum(min_count=1 if preserve_undefined else 0) if "psi_bin" in df_g.columns else float("nan") if preserve_undefined else 0.0
            psi_text = f"{psi_val:.2f}" if np.isfinite(psi_val) else "n.a."

            g_miss_row = df_g[df_g["bin_index"] == -1]
            g_miss_str = f"{g_miss_row['count'].sum() / total_count_g:.0%}" if not g_miss_row.empty and total_count_g > 0 else "0%"
            group_label = group if group is not None else "Unknown group"

            if has_labeled_target and not has_observed_risk:
                total_bad = df_g["bad"].sum()
                total_observed = df_g["observed_count"].sum()
                ax.set_title(
                    f"{group_label} ({int(total_bad):,}/{int(total_observed):,}: "
                    f"n.a., n: {int(total_count_g):,})",
                    fontsize=fs_title + 0.85,
                    y=1.05,
                    ha="center",
                )
                psi_color = "red" if psi_val > 0.1 else "black"
                ax.text(0.48, 1.015, f"PSI: {psi_text}   |", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_title + 0.85, color=psi_color)
                ax.text(0.52, 1.015, f"Miss: {g_miss_str}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="#555555")
            elif has_observed_risk:
                total_bad = df_g["bad"].sum()
                total_observed = df_g["observed_count"].sum()
                avg_bad_rate = total_bad / total_observed
                ax.set_title(
                    f"{group_label} ({int(total_bad):,}/{int(total_observed):,}: "
                    f"{avg_bad_rate:.1%}, n: {int(total_count_g):,})",
                    fontsize=fs_title + 0.85,
                    y=1.05,
                    ha="center",
                )

                iv_val, ks_val, auc_val = MarsPlotter._summarize_binning_metrics(df_g)

                rc_str = f"RC:{rc_val:.2f}" if not np.isnan(rc_val) else "RC:n.a."
                rc_color = "red" if (not np.isnan(rc_val) and rc_val < 0.7) else "#555555"

                ks_text = f"{ks_val:.1f}" if np.isfinite(ks_val) else "n.a."
                perf_text = f"IV: {iv_val:.2f},  KS: {ks_text},  AUC: {auc_val:.2f},"
                ax.text(0.602, 1.015, perf_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_title + 0.85, color="black")
                ax.text(0.607, 1.015, f"  PSI: {psi_text},", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="red" if psi_val > 0.1 else "black")
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
                ax.set_title(f"{group_label}   (Total: {int(total_count_g)})", fontsize=fs_title + 0.85, y=1.05, ha="center")
                psi_color = "red" if psi_val > 0.1 else "black"
                ax.text(0.48, 1.015, f"PSI: {psi_text}   |", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_title + 0.85, color=psi_color)
                ax.text(0.52, 1.015, f"Miss: {g_miss_str}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs_title + 0.85, color="#555555")

            if group_time_ranges and group in group_time_ranges:
                group_range = group_time_ranges[group]
                range_label = (
                    f"[{group_range[0]} ~ {group_range[1]}]"
                    if group_range is not None else "[Date range: unknown]"
                )
                date_text = ax.text(
                    0.5, 1.1, range_label, transform=ax.transAxes,
                    fontproperties=ax.title.get_fontproperties(), color=ax.title.get_color(),
                    ha="center", va="bottom", clip_on=False, gid="mars-group-date",
                )
                date_text.set_parse_math(False)
                dated_axes.append((ax, date_text))
            ax.title.set_parse_math(False)

        if display_lines or dated_axes:
            MarsPlotter._layout_risk_trend_header(fig, summary_text, display_lines, dated_axes)

        return fig

    @staticmethod
    def _layout_risk_trend_header(
        fig: plt.Figure,
        summary: Text,
        display_lines: list[str],
        dated_axes: list[tuple[Axes, Text]],
    ) -> None:
        """安装仅用于趋势图新增行的布局，在显示及导出预绘时按目标 renderer 重测。"""
        name: Text | None = None
        if display_lines:
            name_font = summary.get_fontproperties().copy()
            cjk_family = _resolve_cjk_font_family()
            if cjk_family is not None:
                name_font.set_family(cjk_family)
            name = cast(Text, fig.text(
                *summary.get_position(), "\n".join(display_lines),
                fontproperties=name_font, color=summary.get_color(),
                ha="left", va="top", linespacing=1.6, gid="mars-display-name",
            ))
            name.set_parse_math(False)
            # 同一个圆角框包围中英文；padding、圆角、底色和边框保持原设计。
            summary.set_bbox(None)
            fig.add_artist(_RiskTrendSummaryBox(fig, summary, name))
        layout = _RiskTrendHeader(fig, summary, name, dated_axes)
        fig.set_layout_engine(layout)
        layout.execute(fig)

    @staticmethod
    def _select_group_rows(
        frame: pd.DataFrame, group_col: str, group: str | None,
    ) -> pd.DataFrame:
        """独立保留未知分组，避免 pandas nullable 字符串变成混合排序键或业务组别名。"""
        mask = frame[group_col].isna() if group is None else frame[group_col].eq(group)
        return frame.loc[mask]

    @staticmethod
    def _wrap_display_name(value: str, *, width: int) -> list[str]:
        """按中英文实际字符宽度换行中文名，避免长业务文本压住子图。"""
        lines: list[str] = []
        line = ""
        used_width = 0
        for character in value:
            char_width = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            if character == "\n" or used_width + char_width > width:
                if line:
                    lines.append(line)
                line = ""
                used_width = 0
                if character == "\n":
                    continue
            line += character
            used_width += char_width
        if line:
            lines.append(line)
        return lines

    @staticmethod
    def plot_feature_binning_risk_trend(
        df_detail: pd.DataFrame | pl.DataFrame,
        feature: str,
        group_col: str = "month",
        target_name: str = "Target",
        risk_corr_reference_df: pd.DataFrame | pl.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        dpi: int | None = 150,
        time_range: tuple[str, str] | None = None,
        *,
        include_total: bool = True,
    ) -> None:
        """
        在 Notebook 中展示单特征的分箱风险趋势图。

        图中展示样本分布、坏率、RC 及 IV、KS、AUC、PSI。
        无标签时只展示样本分布与 PSI；有标签但未表现的分组保留分布，风险指标显示未定义。

        Parameters
        ----------
        df_detail : pd.DataFrame | pl.DataFrame
            已计算的分箱明细，需含 ``feature``、分组列、``bin_index``、``bin_label`` 和 ``count``。
            有标签数据还需 ``observed_count`` 与 ``bad``，以区分已表现样本和全样本。
        feature : str
            需要绘制的英文特征 ID，按 ``feature`` 列精确匹配。
        group_col : str
            分组列名，默认 ``"month"``；每组生成一个面板，不作为日期范围推断依据。
        target_name : str
            图中显示的目标名称，默认 ``"Target"``；只控制文字，不筛选目标行。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            RC 参考坏率表，包含 ``feature``、``bin_index`` 和 ``base_br``。
            按特征与正常箱匹配；默认不提供参考，图中 RC 沿用 1.0 的回退值。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        dpi : int | None
            展示用 PNG 的分辨率，默认 150；None 也使用 150。

        time_range : tuple[str, str] | None
            本次数据有效 ``time_col`` 的最小和最大日期，用于图中日期说明。
            必须显式提供合法范围；默认 None 会抛出 ``ValueError``，不从分组名推测日期。
        include_total : bool
            是否显示 Total 面板，默认显示；不改变全局统计、共享坐标范围或 RC 参考。
            关闭后没有其他面板时抛出 ``ValueError``。

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
        >>> MarsPlotter.plot_feature_binning_risk_trend(
        ...     df_detail, "age", dpi=80, time_range=("202601", "202602")
        ... ) is None
        True
        """
        fig: plt.Figure | None = MarsPlotter._build_feature_binning_risk_figure(
            df_detail=df_detail,
            feature=feature,
            group_col=group_col,
            target_name=target_name,
            risk_corr_reference_df=risk_corr_reference_df,
            show_risk=show_risk,
            time_range=time_range,
            include_total=include_total,
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
        time_range: tuple[str, str] | None = None,
        *,
        include_total: bool = True,
    ) -> None:
        """
        批量绘制多个特征的分箱风险趋势图。

        读取已有明细，按指定指标（IV、KS 或 AUC）排列特征后逐个展示。

        Parameters
        ----------
        df_detail : pd.DataFrame | pl.DataFrame
            已计算的分箱评估明细表，字段要求同 ``plot_feature_binning_risk_trend``。
        features : list[str]
            待绘图的英文特征 ID 列表，绘制顺序由 ``sort_by`` 决定。
        group_col : str
            分组列名，默认 ``"month"``；用于分面及选择排序所需的 Total 行。
        target_name : str
            图中显示的目标名称，默认 ``"Target"``；只控制文字，不筛选目标行。
        target_key : str | None
            保留的目标标识参数，默认 None；当前实现不读取此值，也不据此筛选明细。
        dpi : int
            展示用 PNG 的分辨率，默认 150。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        sort_by : str
            特征排序指标，默认 ``"iv"``，支持 ``"iv"``、``"ks"`` 和 ``"auc"``。
            优先使用 Total 组；没有 Total 时使用全部明细。空字符串或其他值保留传入顺序。
        ascending : bool
            是否按排序指标升序排列，默认降序；数值大小不直接表示业务重要性。
        risk_corr_reference_df : pd.DataFrame | pl.DataFrame | None
            各特征共用的 RC 参考坏率表，包含 ``feature``、``bin_index`` 和 ``base_br``。
            默认不提供参考，图中 RC 沿用 1.0 的回退值。
        risk_corr_baseline : RiskCorrBaseline
            保留的基准参数，默认 ``"total"``；当前实现不读取此值。
            图中 RC 参考由 ``risk_corr_reference_df`` 决定。
        time_range : tuple[str, str] | None
            本次数据有效 ``time_col`` 的最小和最大日期，所有特征共用此说明。
            必须显式提供合法范围；默认 None 会抛出 ``ValueError``，不从分组名推测日期。
        include_total : bool
            是否显示 Total 面板，默认显示；不改变特征排序、全局统计或共享坐标范围。

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
        ...     time_range=("202601", "202601"),
        ... ) is None
        True
        """
        df_detail = MarsPlotter._as_pandas_detail_frame(df_detail)

        # 重置交互式容器的显示标记
        display(HTML("<script>window.MARS_PLOTTER_HINT_SHOWN = undefined;</script>"))

        if sort_by and sort_by.lower() in {"iv", "ks", "auc"}:
            sort_metric = sort_by.lower()
            feature_stats: list[dict[str, str | float]] = []
            for feat in features:
                df_feat = df_detail[df_detail["feature"] == feat]
                if df_feat.empty:
                    continue

                df_calc = (
                    df_feat[df_feat[group_col] == "Total"]
                    if "Total" in df_feat[group_col].values
                    else df_feat
                )
                score = 0.0
                if sort_metric == "iv":
                    score = float(df_calc["iv_bin"].sum())
                elif sort_metric == "ks":
                    score = MarsPlotter._summarize_binning_metrics(df_calc)[1]
                elif sort_metric == "auc":
                    score = float(df_calc["auc_bin"].sum())
                    if score < 0.5:
                        score = 1 - score

                feature_stats.append({"feature": feat, "score": score})

            df_stats = pd.DataFrame(feature_stats)
            sorted_features = (
                [
                    str(feature)
                    for feature in df_stats.sort_values(
                        by="score",
                        ascending=ascending,
                    )["feature"].tolist()
                ]
                if not df_stats.empty
                else features
            )
        else:
            sorted_features = features

        # 批量入口只负责顺序调度，单图渲染继续复用同一个公开入口。
        for feat in sorted_features:
            MarsPlotter.plot_feature_binning_risk_trend(
                df_detail=df_detail,
                feature=feat,
                group_col=group_col,
                target_name=target_name,
                risk_corr_reference_df=risk_corr_reference_df,
                show_risk=show_risk,
                dpi=dpi,
                time_range=time_range,
                include_total=include_total,
            )
