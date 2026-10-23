"""数据画像报告对象。"""

from __future__ import annotations

from html import escape
from typing import Any, Dict, List, NamedTuple, Union

import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame
from mars.reporting._metadata import FeatureMetadata
from mars.reporting._profile_excel import _ProfileExcelWriter
from mars.reporting._profile_html import write_profile_html
from mars.reporting._query import ReportFrame, _ReportQuery


class ProfileData(NamedTuple):
    """
    汇总画像报告的概览、趋势和对比表。

    Attributes
    ----------
    overview : DataFrame
        特征概览宽表。
    dq_trends : dict of str to DataFrame
        数据质量指标的趋势宽表字典。
    stats_trends : dict of str to DataFrame
        统计分布指标的趋势宽表字典。
    comparisons : dict of str to DataFrame
        字段类型变化（schema drift）与新增类别占比（unseen rate）的对比表字典。

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

class MarsProfileReport(_ReportQuery):
    """
    数据特征画像与质量评估报告容器。

    保存 `MarsDataProfiler` 已计算的概览、数据质量、统计分布和对比表。
    支持查询、Notebook 展示及 Excel、HTML 导出，查询时不重新计算统计。

    Attributes
    ----------
    overview_table : DataFrame
        全量特征概览表。

    dq_tables : dict of str to DataFrame
        数据质量指标趋势表字典。

    stats_tables : dict of str to DataFrame
        统计分布指标趋势表字典。
    comparison_tables : dict of str to DataFrame
        显式请求的当前样本与基准样本对比表。
    report_meta : dict of str to Any
        本次运行的实际参数、计算口径和结构化诊断。

    Notes
    -----
    `show_overview` 与 `show_trend` 使用 Pandas Styler 显示色带、数据条和 sparkline。
    `write_excel()` 导出带条件格式的电子表格，`write_html()` 导出离线单文件报告。

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
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
    ) -> None:
        """
        初始化画像报告容器。

        Parameters
        ----------
        overview : Union[pl.DataFrame, pd.DataFrame]
            已计算的特征概览宽表，每行对应一个特征。
        dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            指标名到数据质量趋势宽表的映射，分组值或时间段作为列。
        stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            指标名到统计趋势宽表的映射，分组值或时间段作为列。
        comparison_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]] | None
            字段类型变化与新增类别占比的对比表，默认不包含对比结果。
        report_meta : Dict[str, Any] | None
            本次运行的实际参数与诊断，默认使用空字典，不补算缺少的统计。
        feature_metadata : FeatureMetadata | None
            以英文特征 ID 为键的业务字典，或包含 ``feature`` 列的 Pandas/Polars 表。
            支持 ``display_name``、``description``、``data_source`` 和 ``unit``；
            默认不补充业务解释，已有身份列中的有效信息仍会保留。
        business_context : dict[str, Any] | None
            JSON 业务上下文，仅用于解释结果，不改变统计。
            标签定义按 ``labels`` 中的目标字段登记，未提供的信息标为 unknown。
        """
        self.overview_table = overview
        self.dq_tables = dq_tables
        self.stats_tables = stats_tables
        self.comparison_tables = dict(comparison_tables or {})
        self.report_meta = dict(report_meta or {})
        self._calculation_status: ReportFrame | None = None
        self._initialize_semantics(feature_metadata, business_context)

        # 按指标名定位数据质量、统计或对比表，供 show_trend 复用。
        self._metric_index: Dict[str, str] = {}
        for k in self.dq_tables.keys():
            self._metric_index[k] = "dq"
        for k in self.stats_tables.keys():
            self._metric_index[k] = "stat"
        for k in self.comparison_tables.keys():
            self._metric_index[k] = "comparison"

    def _query_tables(self) -> dict[str, ReportFrame]:
        """按公共目录命名画像表，保留各表的 Pandas/Polars 类型。"""
        return {"overview": self.overview_table,
                **({"calculation_status": self._calculation_status} if self._calculation_status is not None else {}),
                **{f"dq.{k}": v for k, v in self.dq_tables.items()},
                **{f"stats.{k}": v for k, v in self.stats_tables.items()},
                **{f"comparison.{k}": v for k, v in self.comparison_tables.items()}}

    def _query_metadata(self) -> dict[str, Any]:
        """返回画像运行元数据。"""
        return self.report_meta

    def get_profile_data(self) -> ProfileData:
        """
        返回报告中已计算的统计表。

        Returns
        -------
        ProfileData
            包含概览表、数据质量趋势表、统计趋势表和对比表的命名元组。
            这些表沿用报告中的对象，不包含原始样本。

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
        status_html = ""
        if self._calculation_status is not None:
            # 摘要显示有界状态证据；完整行仍由同一公共状态表查询。
            state: pd.DataFrame = self._display_frame(self.get_table("calculation_status", limit=10))
            status_html = (
                '<div style="margin-top:12px"><b>Calculation status</b>'
                '<div style="overflow:auto">'
                f'{state.to_html(index=False, escape=True)}</div>'
                '<code>get_table("calculation_status")</code></div>'
            )

        # 内嵌 CSS，使 Notebook 摘要无需加载外部样式。
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
            # 摘要最多显示 30 个指标名，其余只显示数量，避免面板过宽。
            display_keys = keys[:30]
            pills = "".join([f"<span style='{pill_style}'>'{escape(k)}'</span>" for k in display_keys])
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
                    <div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.95em;">
                        <div>👉 <span style="{code_style}">.show_overview()</span> &nbsp;<span style="color:#555">View Full Report</span></div>
                        <div>💾 <span style="{code_style}">.write_excel()</span> &nbsp;<span style="color:#555">Export XLSX</span></div>
                        <div>🌐 <span style="{code_style}">profile_report.write_html("profile_report.html")</span> &nbsp;<span style="color:#555">Export HTML</span></div>
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
                {status_html}

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
            趋势页分组列的横向排序方向，默认按列名升序排列。
        sort_by : Union[str, List[str]]
            趋势页特征行的排序字段，默认按 ``total`` 排序。
            可传入单列名或多列列表，字段必须存在于相应趋势表。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列特征行，默认降序。

        Returns
        -------
        None
            写入报告文件，不返回数据。

        Notes
        -----
        数值格式沿用实际指标单位；业务文字、身份、分组名和表头按原文保存，
        以 ``=`` 开头的字符串不会转换为公式。

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
        """导出可离线浏览的单文件画像 HTML 报告。

        报告包含元数据、概览、数据质量、统计和对比表，支持搜索及按原值排序。
        不生成图表，也不重新计算统计。

        Parameters
        ----------
        path : str
            输出文件路径，默认写入 ``mars_profile_report.html``；父目录必须存在。
        report_name : str
            HTML 页头和浏览器标签页的报告标题，默认使用 ``"MARS Data Profile"``。

        Returns
        -------
        None
            写入报告文件，不返回数据。

        Notes
        -----
        父目录不存在时，底层写入函数抛出 ``FileNotFoundError``；
        报告渲染或文件写入失败时抛出 ``RuntimeError``。

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
        fmt_as_pct: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        table_name: str | None = None,
    ) -> pd.io.formats.style.Styler:
        """构建画像报告复用的 Pandas Styler。"""
        return _ProfileExcelWriter(self)._get_styler(
            df_input=df_input,
            title=title,
            cmap=cmap,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            subset_cols=subset_cols,
            fmt_as_pct=fmt_as_pct,
            vmin=vmin,
            vmax=vmax,
            table_name=table_name,
        )

    def show_overview(self,
                      features: Union[str, List[str]] | None = None,
                      sort_by: Union[str, List[str]] | None = None,
                      sort_ascending: bool = False,
                      *, columns: list[str] | None = None,
                      limit: int | None = None,
                      sources: str | list[str] | None = None) -> pd.io.formats.style.Styler:
        """
        展示特征概览宽表，数值单位按真实指标字段定义格式化。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。
        sort_by : Union[str, List[str]] | None
            特征行的排序字段。默认按已有的 ``dtype``、``missing_rate`` 依次排序；
            显式指定时只使用所列字段，排序先于列投影和行数限制。
        sort_ascending : bool
            是否按排序字段升序排列，默认降序；也适用于默认排序字段。
        columns : list[str] | None
            展示字段。默认保留全部列；显式指定时只返回所列字段，并保持指定顺序。
        limit : int | None
            排序后的最大展示行数，默认不限制。设为 0 时返回保留列结构的空视图。
        sources : str | list[str] | None
            按已有特征来源筛选；来源未知时明确报错。

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
        effective_sort = sort_by
        if sort_by is None:
            effective_sort = [c for c in ("dtype", "missing_rate") if c in self.overview_table.columns] or None
        df = to_pandas_frame(self.get_table(
            "overview", features=features, columns=columns, sort_by=effective_sort,
            descending=not sort_ascending, limit=limit, sources=sources,
        ))
        df = self._display_frame(df)
        return self._get_styler(
            df,
            title="Dataset Overview",
            cmap="RdYlGn_r",
            # 指定哪些列应用“红绿灯”配色 (高值=红)
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "mode_rate"],
            table_name="overview",
        )

    def show_trend(self,
                   metric: str,
                   features: Union[str, List[str]] | None = None,
                   group_ascending: bool = True,
                   sort_by: Union[List[str], str] | None = None,
                   sort_ascending: bool = False,
                   *, columns: list[str] | None = None,
                   limit: int | None = None,
                   sources: str | list[str] | None = None) -> pd.io.formats.style.Styler:
        """
        展示指定指标的分组趋势，数值单位按所属指标定义格式化。

        Parameters
        ----------
        metric : str
            指标或比较表名称，例如 ``"missing"``、``"mean"``、``"psi"``、
            ``"schema"`` 或 ``"unseen"``。
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。
        group_ascending : bool
            分组列或时间切片列的横向排序方向，默认按列名升序。
            仅在 ``columns=None`` 时生效；身份列置前，``total`` 置后。
        sort_by : Union[List[str], str] | None
            表内排序依据，可以是单列或多列列表。``None`` 优先选择 ``total``；
            没有该列的比较表按 ``feature`` 排序，均遵守 ``sort_ascending``。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列。
        columns : list[str] | None
            展示字段。默认保留全部列并排列分组列；显式指定时保持所列字段及其顺序。
        limit : int | None
            排序后的最大展示行数，默认不限制。设为 0 时返回保留列结构的空视图。
        sources : str | list[str] | None
            按已登记来源筛选特征，默认不限制；与 ``features`` 取交集，未知来源报错。

        Returns
        -------
        pd.io.formats.style.Styler
            样式化表；文本类型与计算状态保留原值，空查询结果仍可渲染。
            对存在未计算状态的展示特征，在标题中列出状态与原因；原值仍在状态表中。
            业务分组名称不影响单位；比例展示百分比，AUC、PSI、IV、Lift、WOE
            保持数值口径，KS 展示 0–100 点。

        Raises
        ------
        ValueError
            当 ``metric`` 不受支持或显式排序列不存在时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> overview = pl.DataFrame({"feature": ["age"], "dtype": ["Int64"], "missing_rate": [0.0]})
        >>> trend = pl.DataFrame({"feature": ["age"], "dtype": ["Int64"], "2026-01": [0.0], "total": [0.0]})
        >>> report = MarsProfileReport(overview, dq_tables={"missing": trend}, stats_tables={})
        >>> hasattr(report.show_trend("missing", features="age"), "to_html")
        True
        """
        # 先确定指标所属的表，再复用公共查询的筛选、排序和投影。
        source_type = self._metric_index.get(metric)
        if source_type is None:
            # 提供可用指标的快速提示，方便交互式探索。
            available = list(self._metric_index.keys())
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available[:10]}...")

        # 获取数据
        vmin: float | None
        vmax: float | None
        if source_type == "dq":
            # DQ 默认配置
            cmap = "RdYlGn_r"  # 比例越高，颜色越接近红色。
            vmin, vmax = 0, 1  # 数据质量比例使用 0～1 的固定色阶。

        else: # 统计及比较表沿用各自的原生结果。
            # 统计指标默认配置
            cmap = "Blues"     # 颜色深浅只表示数值大小。
            vmin, vmax = None, None

        # 特殊指标覆盖配置
        if metric == "psi":
            cmap = "RdYlGn_r" # PSI 越高，颜色越接近红色。
            vmin, vmax = 0.0, 0.5 # 固定色阶便于比较不同切片。

        prefix = {"dq": "dq", "stat": "stats", "comparison": "comparison"}[source_type]
        # 按原始表 schema 选择默认排序，显式请求仍由公共查询校验。
        table_name = f"{prefix}.{metric}"
        if sort_by is None:
            table_columns: list[str] = list(self._query_tables()[table_name].columns)
            sort_by = next((column for column in ("total", "feature") if column in table_columns), None)
        df: pd.DataFrame = to_pandas_frame(self.get_table(
            table_name, features=features, columns=columns,
            sort_by=sort_by, descending=not sort_ascending, limit=limit, sources=sources,
        ))
        df = self._display_frame(df)
        if columns is None:
            df = self._reorder_trend_cols(df, group_ascending=group_ascending)

        title = f"Trend Analysis: {metric}"
        if self._calculation_status is not None and not df.empty:
            # 状态只随本次展示的特征出现，不改变趋势表的 schema 或 null 值。
            displayed_features: list[str] = (
                df["feature"].tolist() if "feature" in df.columns else
                to_pandas_frame(self.get_table(
                    table_name, features=features, columns=["feature"],
                    sort_by=sort_by, descending=not sort_ascending, limit=limit, sources=sources,
                ))["feature"].tolist()
            )
            state: pd.DataFrame = to_pandas_frame(self.get_table(
                "calculation_status", features=displayed_features,
                filters={"metric": metric}, columns=["status", "reason"],
            ))
            statuses = state.drop_duplicates().itertuples(index=False, name=None)
            details = "; ".join(escape(f"{status}/{reason}") for status, reason in statuses)
            if details:
                title += f" — {details}"
        return self._get_styler(
            df,
            title=title,
            cmap=cmap,
            table_name=table_name,
            vmin=vmin,
            vmax=vmax,
        )

    def _reorder_trend_cols(self, df: pd.DataFrame, group_ascending: bool) -> pd.DataFrame:
        """重新排列趋势表的列顺序。"""
        # 定义元数据列和末尾全量统计列
        meta_cols = ["target", "y", "feature", "display_name", "dtype", "data_source", "distribution", "mode_value"]
        stat_cols = ["total"]

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

