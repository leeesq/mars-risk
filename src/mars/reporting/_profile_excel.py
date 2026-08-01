"""数据画像 Excel 导出实现。"""

from __future__ import annotations

import importlib.util
from typing import Any, List, Union

import pandas as pd

from mars.compute import to_pandas_frame


class _ProfileExcelWriter:
    """画像报告 Excel 导出能力。"""

    def __init__(self, report: Any) -> None:
        self._report = report

    def __getattr__(self, name: str) -> Any:
        """将只读数据访问委托给 report 容器。"""
        return getattr(self._report, name)

    def write_excel(self: Any,
                    path: str = "mars_report.xlsx",
                    group_ascending: bool = True,
                    sort_by: Union[str, List[str]] = "total",
                    sort_ascending: bool = False) -> None:
        """
        导出画像 Excel 报告。

        Parameters
        ----------
        path : str
            输出文件路径。
        group_ascending : bool
            趋势页中分组列的横向排序方向。
        sort_by : Union[str, List[str]]
            趋势页内部的排序依据。
        sort_ascending : bool
            趋势页内部是否按 ``sort_by`` 升序排列。

        Notes
        -----
        该方法依赖 ``xlsxwriter`` 导出带样式的多工作表 Excel 文件。

        Raises
        ------
        ImportError
            基础安装缺少 ``xlsxwriter`` 时抛出。
        ValueError
            报告没有任何可导出内容时抛出。
        RuntimeError
            Excel 渲染或写入失败时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
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
        >>> with TemporaryDirectory() as tmp:
        ...     report.write_excel(str(Path(tmp) / "profile.xlsx")) is None
        True
        """
        # 1. 依赖检查
        if importlib.util.find_spec("xlsxwriter") is None:
            raise ImportError(
                "'xlsxwriter' is included in the base mars-risk installation; "
                "reinstall mars-risk if missing."
            )

        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                wrote_any = False
                if self.report_meta:
                    metadata = pd.DataFrame(
                        [
                            {
                                "key": key,
                                "value": value if isinstance(value, str) else repr(value),
                            }
                            for key, value in sorted(self.report_meta.items())
                        ]
                    )
                    metadata.to_excel(writer, sheet_name="Metadata", index=False)
                    wrote_any = True
                #--------------------------------------------------------
                # 1. 导出概览页 (Overview)
                #--------------------------------------------------------
                if not to_pandas_frame(self.overview_table).empty:
                    overview_styler = self.show_overview()
                    overview_styler.to_excel(writer, sheet_name="Overview", index=False)
                    wrote_any = True

                #--------------------------------------------------------
                # 2. 统一导出所有趋势页 (Trend & DQ)
                #--------------------------------------------------------
                dq_keys = list(self.dq_tables.keys())
                stat_keys = list(self.stats_tables.keys())
                all_metrics = dq_keys + stat_keys

                for metric in all_metrics:
                    # 保证导出的表结构与 Notebook 展示完全一致
                    styler = self.show_trend(
                        metric,
                        group_ascending=group_ascending,
                        sort_by=sort_by,
                        sort_ascending=sort_ascending
                    )

                    if styler is not None:
                        prefix = "DQ" if metric in self.dq_tables else "Trend"
                        sheet_name = f"{prefix}_{metric.capitalize()}"[:31]

                        styler.to_excel(writer, sheet_name=sheet_name, index=False)
                        wrote_any = True

                        # 确保条件格式锚定的列与导出的表完全吻合
                        self._apply_excel_formatting(
                            writer, sheet_name, metric,
                            group_ascending=group_ascending,
                            sort_by=sort_by,
                            sort_ascending=sort_ascending
                        )

                for name, comparison in self.comparison_tables.items():
                    sheet_name = f"Compare_{name.capitalize()}"[:31]
                    to_pandas_frame(comparison).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                    )
                    wrote_any = True

                if not wrote_any:
                    raise ValueError("Profile report contains no generated content to export.")

                # 3. 自动列宽调整
                for sheet in writer.sheets.values():
                    sheet.autofit()

        except Exception as exc:
            raise RuntimeError(f"Failed to export profile Excel to '{path}'.") from exc

    def _apply_excel_formatting(
        self: Any,
        writer: Any,
        sheet_name: str,
        metric: str,
        group_ascending: bool,
        sort_by: str | list[str],
        sort_ascending: bool,
    ) -> None:
        """为导出的趋势工作表应用条件格式。"""
        if metric in self.dq_tables:
            raw_df = self.dq_tables[metric]
        else:
            raw_df = self.stats_tables[metric]

        # 必须和 show_trend 的内部重排逻辑一模一样，否则 Excel 样式会错位
        df_pd: pd.DataFrame = to_pandas_frame(raw_df).copy()

        # 匹配行排序
        if sort_by in df_pd.columns or (isinstance(sort_by, list) and all(c in df_pd.columns for c in sort_by)):
            df_pd = df_pd.sort_values(by=sort_by, ascending=sort_ascending)

        # 匹配列排序
        df_pd = self._reorder_trend_cols(df_pd, group_ascending=group_ascending)

        # 动态识别趋势列范围，避免依赖固定列位置。
        meta_and_stat = set(["feature", "dtype", "distribution", "mode_value", "total", "group_mean", "group_var", "group_cv"])
        time_cols = [c for c in df_pd.columns if c not in meta_and_stat]

        if not time_cols:
            return

        worksheet = writer.sheets[sheet_name]

        # PSI 专用三色阶 (红绿灯)
        if metric == "psi":
            meta_cols = ["feature", "dtype", "distribution", "mode_value"]
            start_col = 0
            for i, col in enumerate(df_pd.columns):
                if col not in meta_cols:
                    start_col = i
                    break

            end_col = len(df_pd.columns) - 1

            worksheet.conditional_format(1, start_col, len(df_pd), end_col, {
                'type': '3_color_scale',
                'min_type': 'num', 'min_value': 0.05, 'min_color': '#63BE7B', # 绿色
                'mid_type': 'num', 'mid_value': 0.15, 'mid_color': '#FFEB84', # 黄色
                'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'  # 红色
            })

        # 稳定性 Data Bars (针对 group_cv)
        if "group_cv" in df_pd.columns:
            col_idx = df_pd.columns.get_loc("group_cv")
            worksheet.conditional_format(1, col_idx, len(df_pd), col_idx, {
                'type': 'data_bar',
                'bar_color': '#638EC6',
                'bar_solid': True,
                'min_type': 'num', 'min_value': 0,
                'max_type': 'num', 'max_value': 1
            })

    def _get_styler(
        self: Any,
        df_input: Any,
        title: str,
        cmap: str,
        sort_by: List[str] | None = None,
        sort_ascending: bool = False, # 统一内部 API 命名
        subset_cols: List[str] | None = None,
        add_bars: bool = False,
        fmt_as_pct: bool = False,
        vmin: float | None = None,
        vmax: float | None = None
    ) -> pd.io.formats.style.Styler:
        """生成统一的 Pandas Styler 样式对象。"""
        if df_input is None:
            raise ValueError("Requested profile table was not generated.")
        df: pd.DataFrame = to_pandas_frame(df_input)
        if sort_by is not None:
            df = df.sort_values(by=sort_by, ascending=sort_ascending) # 使用统一参数进行底层排序
        if df.empty:
            raise ValueError("Requested profile table is empty.")

        # 元数据排除列表
        exclude_meta: List[str] = [
            "feature", "dtype",
            "group_mean", "group_var", "group_cv",
            "distribution",
            "mode_value"
            ]

        # 确定色彩渐变范围
        if subset_cols:
            gradient_cols: List[str] = [c for c in subset_cols if c in df.columns]
        else:
            gradient_cols = [c for c in df.columns if c not in exclude_meta]

        styler = df.style.set_caption(f"<b>{title}</b>").hide(axis="index")

        # 应用热力图
        if gradient_cols:
            styler = styler.background_gradient(
                cmap=cmap,
                subset=gradient_cols,
                axis=None,
                vmin=vmin,
                vmax=vmax
            )

        # 应用数据条
        if add_bars and "group_cv" in df.columns:
            styler = styler.bar(subset=["group_cv"], color='#ff9999', vmin=0, vmax=1, width=90)
            styler = styler.format("{:.4f}", subset=["group_cv", "group_var"])

        # 数值格式化逻辑
        num_cols: pd.Index = df.select_dtypes(include=['number']).columns
        data_cols: List[str] = [c for c in num_cols if c not in ["group_var", "group_cv", "distribution"]]

        pct_format: str = "{:.2%}"
        float_format: str = "{:.2f}"

        if fmt_as_pct:
            if data_cols:
                styler = styler.format(pct_format, subset=data_cols)
        else:
            pct_cols: List[str] = [
                c for c in df.columns
                if ("rate" in c or "ratio" in c) and (c in num_cols)
            ]

            if pct_cols:
                styler = styler.format(pct_format, subset=pct_cols)

            float_cols: List[str] = [c for c in data_cols if c not in pct_cols]
            if float_cols:
                styler = styler.format(float_format, subset=float_cols)

        # 分布迷你图样式
        if "distribution" in df.columns:
            styler = styler.set_table_styles([
                {'selector': '.col_distribution', 'props': [
                    # 优先使用 Consolas (Win) 或 Menlo (Mac)，最后 fallback 到 monospace
                    ('font-family', '"Consolas", "Menlo", "Courier New", monospace'),
                    ('color', '#1f77b4'),
                    ('white-space', 'pre'), # [关键] 防止 HTML 自动压缩连续空格
                    ('font-weight', 'bold'),
                    ('text-align', 'left')
                ]}
            ], overwrite=False)

        # 全局表格外观
        styler = styler.set_table_styles([
            {
                'selector': 'th',
                'props': [('text-align', 'left'), ('background-color', '#f0f2f5'), ('color', '#333')]
            },
            {
                'selector': 'caption',
                'props': [('font-size', '1.2em'), ('padding', '10px 0'), ('color', '#2c3e50')]
            }
        ], overwrite=False)

        return styler
