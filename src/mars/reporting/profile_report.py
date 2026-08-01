"""数据画像报告对象。"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Union

import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame
from mars.reporting._profile_excel import _ProfileExcelWriter
from mars.reporting._profile_html import write_profile_html


class ProfileData(NamedTuple):
    """
    画像报告底层数据对象集合。

    Attributes
    ----------
    overview : DataFrame
        特征概览宽表。
    dq_trends : dict of str to DataFrame
        数据质量指标的趋势宽表字典。
    stats_trends : dict of str to DataFrame
        统计分布指标的趋势宽表字典。
    comparisons : dict of str to DataFrame
        Schema drift 与 unseen rate 对比表字典。

    Examples
    --------
    >>> import polars as pl
    >>> data = ProfileData(
    ...     overview=pl.DataFrame(), dq_trends={}, stats_trends={}, comparisons={}
    ... )
    >>> data.dq_trends
    {}
    """

    overview: Union[pl.DataFrame, pd.DataFrame]
    dq_trends: Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    stats_trends: Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    comparisons: Dict[str, Union[pl.DataFrame, pd.DataFrame]]

class MarsProfileReport:
    """
    数据特征画像与质量评估报告容器。

    管理 `MarsDataProfiler` 产出的特征概览、数据质量趋势和统计分布趋势表，并提供表格读取、
    富文本视图和 Excel 导出方法。

    Attributes
    ----------
    overview_table : DataFrame
        全量特征概览表。

    dq_tables : dict of str to DataFrame
        数据质量指标趋势表字典。

    stats_tables : dict of str to DataFrame
        统计分布指标趋势表字典。
    comparison_tables : dict of str to DataFrame
        显式请求的 current/benchmark 对比表。
    report_meta : dict of str to Any
        本次运行的轻量元数据与结构化诊断。

    Notes
    -----
    `show_overview` 与 `show_trend` 使用 Pandas Styler 显示色带、数据条和 sparkline。
    `write_excel()` 导出带条件格式的电子表格。

    Examples
    --------
    >>> import polars as pl
    >>> overview = pl.DataFrame({"feature": ["age"], "missing_rate": [0.0]})
    >>> dq_tables = {"missing": pl.DataFrame({"feature": ["age"], "202601": [0.0]})}
    >>> report = MarsProfileReport(overview, dq_tables=dq_tables, stats_tables={})
    >>> overview_df, dq_dict, stat_dict, comparisons = report.get_profile_data()
    >>> overview_df.height
    1
    """

    def __init__(
        self,
        overview: Union[pl.DataFrame, pd.DataFrame],
        dq_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        stats_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        comparison_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]] | None = None,
        report_meta: Dict[str, Any] | None = None,
    ) -> None:
        """
        初始化画像报告容器。

        Parameters
        ----------
        overview : Union[pl.DataFrame, pd.DataFrame]
            特征概览宽表。
        dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            数据质量趋势表字典。
        stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势表字典。
        comparison_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]] | None
            Schema drift 与 unseen rate 对比表字典。
        report_meta : Dict[str, Any] | None
            报告运行元数据与可恢复诊断。
        """
        self.overview_table = overview
        self.dq_tables = dq_tables
        self.stats_tables = stats_tables
        self.comparison_tables = dict(comparison_tables or {})
        self.report_meta = dict(report_meta or {})

        # 建立索引：将所有指标名映射到对应的数据源类型 ('dq' 或 'stat')
        # 这允许我们在 show_trend 中快速定位
        self._metric_index: Dict[str, str] = {}
        for k in self.dq_tables.keys():
            self._metric_index[k] = "dq"
        for k in self.stats_tables.keys():
            self._metric_index[k] = "stat"
        for k in self.comparison_tables.keys():
            self._metric_index[k] = "comparison"

    def get_profile_data(self) -> ProfileData:
        """
        返回画像报告的原始数据对象。

        Returns
        -------
        ProfileData
            包含概览表、数据质量趋势表、统计趋势表和对比表的命名元组。

        Examples
        --------
        >>> import polars as pl
        >>> overview = pl.DataFrame({"feature": ["age"], "missing_rate": [0.0]})
        >>> report = MarsProfileReport(overview, dq_tables={}, stats_tables={})
        >>> report.get_profile_data().overview.height
        1
        """
        return ProfileData(
            overview=self.overview_table,
            dq_trends=self.dq_tables,
            stats_trends=self.stats_tables,
            comparisons=self.comparison_tables,
        )

    def _repr_html_(self) -> str:
        """返回 Jupyter 环境下的 HTML 摘要面板。"""
        df_ov = self.overview_table
        n_feats = len(df_ov) if hasattr(df_ov, "__len__") else df_ov.height

        dq_keys = list(self.dq_tables.keys())
        stat_keys = list(self.stats_tables.keys())

        # 样式定义 (Inline CSS for portability)
        # 胶囊样式，用于包裹指标名
        pill_style = (
            "background-color: #e8f4f8; color: #2980b9; border: 1px solid #bce0eb; "
            "padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin-right: 4px;"
        )
        # 代码块样式
        code_style = (
            "background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; "
            "font-family: monospace; color: #e74c3c; font-weight: bold;"
        )

        # 辅助函数：生成指标徽章列表
        def _fmt_pills(keys: list[str]) -> str:
            """将指标名称渲染为 HTML 胶囊标签。"""
            if not keys:
                return "<span style='color:#ccc'>None</span>"
            # 为了防止指标太多撑爆屏幕，限制显示数量 (例如只显示前 20 个，后面加 ...)
            display_keys = keys[:30]
            pills = "".join([f"<span style='{pill_style}'>'{k}'</span>" for k in display_keys])
            if len(keys) > 30:
                pills += f"<span style='color:#999; font-size:0.8em'> (+{len(keys)-30} more)</span>"
            return pills

        # 组装 HTML
        return f"""
        <div style="border: 1px solid #e0e0e0; border-left: 5px solid #2980b9; border-radius: 4px; background: white; max-width: 900px; font-family: 'Segoe UI', sans-serif;">

            <div style="padding: 12px 15px; background-color: #f8f9fa; border-bottom: 1px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold; color: #2c3e50; font-size: 1.1em;">
                    📊 Mars Data Profile
                </div>
                <div style="font-size: 0.85em; color: #7f8c8d;">
                    <span style="margin-left: 15px;">🏷️ Features: <b>{n_feats}</b></span>
                    <span style="margin-left: 15px;">🔍 DQ Metrics: <b>{len(dq_keys)}</b></span>
                    <span style="margin-left: 15px;">📉 Stat Metrics: <b>{len(stat_keys)}</b></span>
                </div>
            </div>

            <div style="padding: 15px;">

                <div style="margin-bottom: 15px;">
                    <div style="font-size: 0.8em; text-transform: uppercase; color: #95a5a6; font-weight: bold; margin-bottom: 5px;">Quick Actions</div>
                    <div style="display: flex; gap: 20px; font-size: 0.95em;">
                        <div>👉 <span style="{code_style}">.show_overview()</span> &nbsp;<span style="color:#555">View Full Report</span></div>
                        <div>💾 <span style="{code_style}">.write_excel()</span> &nbsp;<span style="color:#555">Export XLSX</span></div>
                        <div>📥 <span style="{code_style}">.get_profile_data()</span> &nbsp;<span style="color:#555">Get Raw Data</span></div>
                    </div>
                </div>

                <div style="border-top: 1px dashed #e0e0e0; padding-top: 12px;">
                    <div style="font-size: 0.8em; text-transform: uppercase; color: #95a5a6; font-weight: bold; margin-bottom: 8px;">
                        Trend Analysis <span style="font-weight:normal; text-transform:none; color:#bbb">(Use <code>.show_trend('metric_name')</code>)</span>
                    </div>

                    <div style="display: flex; margin-bottom: 8px; align-items: baseline;">
                        <div style="width: 80px; font-weight: bold; color: #27ae60; font-size: 0.9em;">DQ:</div>
                        <div style="flex: 1; line-height: 1.6;">
                            {_fmt_pills(dq_keys)}
                        </div>
                    </div>

                    <div style="display: flex; align-items: baseline;">
                        <div style="width: 80px; font-weight: bold; color: #2980b9; font-size: 0.9em;">Stats:</div>
                        <div style="flex: 1; line-height: 1.6;">
                            {_fmt_pills(stat_keys)}
                        </div>
                    </div>
                </div>

            </div>

            <div style="padding: 6px 15px; background-color: #fff8e1; border-top: 1px solid #fae5b0; font-size: 0.8em; color: #d35400;">
                💡 <b>Pro Tip:</b> Use <span style="{code_style}">.show_trend('psi')</span> to detect population stability drift.
            </div>
        </div>
        """

    def write_excel(
        self,
        path: str = "mars_report.xlsx",
        group_ascending: bool = True,
        sort_by: Union[str, List[str]] = "total",
        sort_ascending: bool = False,
    ) -> None:
        """
        导出画像 Excel 报告。

        Parameters
        ----------
        path : str
            输出 Excel 文件路径。
        group_ascending : bool
            趋势页中分组列的横向排序方向。
        sort_by : Union[str, List[str]]
            趋势页内部的排序依据。
        sort_ascending : bool
            趋势页内部是否按 ``sort_by`` 升序排列。

        Returns
        -------
        None
            方法仅产生文件写入副作用。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> overview = pl.DataFrame({"feature": ["age"], "missing_rate": [0.0]})
        >>> report = MarsProfileReport(overview, dq_tables={}, stats_tables={})
        >>> with TemporaryDirectory() as tmp:
        ...     report.write_excel(str(Path(tmp) / "profile.xlsx")) is None
        True
        """
        _ProfileExcelWriter(self).write_excel(
            path=path,
            group_ascending=group_ascending,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
        )

    def write_html(
        self,
        path: str = "mars_profile_report.html",
        *,
        report_name: str = "MARS Data Profile",
    ) -> None:
        """Export a self-contained interactive HTML profile report.

        Parameters
        ----------
        path : str
            Output HTML file path.
        report_name : str
            Report title displayed in the HTML header.

        Returns
        -------
        None
            The method only writes the report file.

        """
        write_profile_html(self, path=path, report_name=report_name)

    def _get_styler(
        self,
        df_input: object,
        title: str,
        cmap: str,
        sort_by: List[str] | None = None,
        sort_ascending: bool = False,
        subset_cols: List[str] | None = None,
        add_bars: bool = False,
        fmt_as_pct: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> pd.io.formats.style.Styler:
        """构建画像报告复用的 Pandas Styler。"""
        return _ProfileExcelWriter(self)._get_styler(
            df_input=df_input,
            title=title,
            cmap=cmap,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            subset_cols=subset_cols,
            add_bars=add_bars,
            fmt_as_pct=fmt_as_pct,
            vmin=vmin,
            vmax=vmax,
        )

    def show_overview(self,
                      features: Union[str, List[str]] | None = None,
                      sort_by: Union[str, List[str]] | None = None,
                      sort_ascending: bool = False) -> pd.io.formats.style.Styler:
        """
        展示特征概览宽表。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。
        sort_by : Union[str, List[str]] | None
            排序依据列。若为 ``None``，默认先按 ``dtype`` 再按 ``missing_rate`` 排序。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列。

        Returns
        -------
        pd.io.formats.style.Styler
            适合在 Jupyter 环境中直接渲染的样式化概览表。

        Examples
        --------
        >>> import polars as pl
        >>> overview = pl.DataFrame(
        ...     {
        ...         "feature": ["age"],
        ...         "dtype": ["Int64"],
        ...         "missing_rate": [0.0],
        ...         "zeros_rate": [0.0],
        ...         "unique_rate": [1.0],
        ...         "mode_rate": [0.25],
        ...     }
        ... )
        >>> report = MarsProfileReport(overview, dq_tables={}, stats_tables={})
        >>> hasattr(report.show_overview(features="age"), "to_html")
        True
        """
        # 转换为 Pandas 副本以进行切片
        df = to_pandas_frame(self.overview_table).copy()

        # 特征筛选逻辑
        if features is not None:
            if isinstance(features, str):
                features = [features]
            df = df[df["feature"].isin(features)]

        requested_sort = ["dtype"] + (
            ["missing_rate"]
            if sort_by is None
            else ([sort_by] if isinstance(sort_by, str) else sort_by)
        )
        available_sort = [column for column in requested_sort if column in df.columns]
        return self._get_styler(
            df,
            title="Dataset Overview",
            cmap="RdYlGn_r",
            sort_by=available_sort or None,
            sort_ascending=sort_ascending,
            # 指定哪些列应用“红绿灯”配色 (高值=红)
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "mode_rate"],
            fmt_as_pct=False # 概览表混合了多种类型，不强制全转百分比，由内部逻辑细分
        )

    def show_trend(self,
                   metric: str,
                   features: Union[str, List[str]] | None = None,
                   group_ascending: bool = True,
                   sort_by: Union[List[str], str] = "total",
                   sort_ascending: bool = False) -> pd.io.formats.style.Styler:
        """
        展示指定指标的分组趋势。

        Parameters
        ----------
        metric : str
            指标名称，例如 ``"missing"``、``"mean"`` 或 ``"psi"``。
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。
        group_ascending : bool
            分组列或时间切片列的横向排序方向。
        sort_by : Union[List[str], str]
            趋势表内部排序依据，可以是单列或多列列表。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列。

        Returns
        -------
        pd.io.formats.style.Styler
            样式化趋势热力表。

        Raises
        ------
        ValueError
            当 ``metric`` 不在当前报告支持的指标范围内时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> overview = pl.DataFrame({"feature": ["age"], "dtype": ["Int64"], "missing_rate": [0.0]})
        >>> trend = pl.DataFrame({"feature": ["age"], "dtype": ["Int64"], "2026-01": [0.0], "total": [0.0]})
        >>> report = MarsProfileReport(overview, dq_tables={"missing": trend}, stats_tables={})
        >>> hasattr(report.show_trend("missing", features="age"), "to_html")
        True
        """
        # 路由逻辑：查找指标属于哪个表
        source_type = self._metric_index.get(metric)
        if source_type is None:
            # 提供可用指标的快速提示，方便交互式探索。
            available = list(self._metric_index.keys())
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available[:10]}...")

        # 获取数据
        vmin: float | None
        vmax: float | None
        if source_type == "dq":
            df_raw = self.dq_tables[metric]
            # DQ 默认配置
            cmap = "RdYlGn_r"  # 红色代表高风险 (高缺失)
            fmt_pct = True     # DQ 指标通常是率 (Rate/Ratio)
            vmin, vmax = 0, 1  # 率通常在 0~1 之间

        else: # source_type 为 "stat"。
            df_raw = self.stats_tables[metric]
            # 统计指标默认配置
            cmap = "Blues"     # 蓝色代表数值高低 (中性)
            fmt_pct = False    # 统计值通常是绝对值
            vmin, vmax = None, None

        # 特殊指标覆盖配置
        if metric == "psi":
            cmap = "RdYlGn_r" # PSI 高了是坏事
            fmt_pct = False   # PSI 是数值不是百分比
            vmin, vmax = 0.0, 0.5 # 锚定阈值

        df = to_pandas_frame(df_raw).copy()

        # 特征筛选逻辑
        if features is not None:
            if isinstance(features, str):
                features = [features]
            df = df[df["feature"].isin(features)]

        # 排序
        df = df.sort_values(by=sort_by, ascending=sort_ascending)
        df = self._reorder_trend_cols(df, group_ascending=group_ascending)

        return self._get_styler(
            df,
            title=f"Trend Analysis: {metric}",
            cmap=cmap,
            fmt_as_pct=fmt_pct,
            vmin=vmin,
            vmax=vmax,
            add_bars=True # 所有趋势表都允许显示 CV 条
        )

    def _reorder_trend_cols(self, df: pd.DataFrame, group_ascending: bool) -> pd.DataFrame:
        """重新排列趋势表的列顺序。"""
        # 定义元数据列和末尾统计列
        meta_cols = ["feature", "dtype", "distribution", "mode_value"]
        stat_cols = ["total", "group_mean", "group_var", "group_cv"]

        # 识别中间的分组列（如时间列）
        all_cols = df.columns.tolist()
        group_cols = [c for c in all_cols if c not in meta_cols + stat_cols]

        # 排序分组列 (受 group_ascending 控制)
        group_cols_sorted = sorted(group_cols, reverse=not group_ascending)

        # 组合最终顺序
        final_order = (
            [c for c in meta_cols if c in all_cols]
            + group_cols_sorted
            + [c for c in stat_cols if c in all_cols]
        )
        return df[final_order]

