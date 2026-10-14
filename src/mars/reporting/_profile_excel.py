"""数据画像 Excel 导出实现。"""

from __future__ import annotations

import importlib.util
from functools import partial
from numbers import Real
from typing import Any, List, Union

import numpy as np
import pandas as pd

from mars.compute import to_pandas_frame
from mars.reporting._metadata import export_report_semantics, is_metadata_column
from mars.reporting._semantics import _definition
from mars.utils.html import format_html_value


def _format_count(value: Any) -> str:
    """计数的整数值显示为整数，加权小数与缺失沿用共享格式。"""
    if isinstance(value, (int, np.integer)):
        return format_html_value(int(value), compact_float=True)
    if isinstance(value, Real) and float(value).is_integer():
        return format_html_value(int(float(value)), compact_float=True)
    return format_html_value(value, missing_text="—")


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
                for sheet_name, frame in export_report_semantics(self).items():
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)
                    wrote_any = True
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
                    self._format_excel_styler(overview_styler, "overview").to_excel(writer, sheet_name="Overview", index=False)
                    self._apply_number_formats(writer, "Overview", overview_styler.data, "overview")
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

                        table_name = f'{"dq" if metric in self.dq_tables else "stats"}.{metric}'
                        self._format_excel_styler(styler, table_name).to_excel(writer, sheet_name=sheet_name, index=False)
                        self._apply_number_formats(writer, sheet_name, styler.data, table_name)
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
                    self._apply_number_formats(writer, sheet_name, to_pandas_frame(comparison), f"comparison.{name}")
                    wrote_any = True

                if not wrote_any:
                    raise ValueError("Profile report contains no generated content to export.")

                # 3. 自动列宽调整
                for sheet in writer.sheets.values():
                    sheet.autofit()

        except Exception as exc:
            raise RuntimeError(f"Failed to export profile Excel to '{path}'.") from exc

    def _number_formats(self, frame: pd.DataFrame, table_name: str) -> dict[str, str]:
        """从实际字段语义取得 Excel 格式，不改变整数或加权小数的值。"""
        fields: dict[str, Any] = self._report.describe()["tables"][table_name]["fields"]
        metric = table_name.rsplit(".", 1)[-1]
        formats: dict[str, str] = {}
        for column in frame.columns:
            if is_metadata_column(column) or not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
                continue
            unit = fields[column]["unit"]
            if unit in {"identifier", "type", "category", "dimension", "state", "state_reason", "original_value", "display", "index", "ordinal"}:
                continue
            if unit in {"ratio", "ratio_difference"} and metric not in {"auc", "auc_bin"} and column not in {"auc", "auc_bin"}:
                number_format = "0.00%"
            elif str(fields[column]["dtype"]).lower().startswith(("int", "uint")) or (
                unit == "count" and frame[column].dropna().mod(1).eq(0).all()
            ):
                number_format = "#,##0"
            else:
                number_format = "#,##0.00"
            formats[column] = number_format
        return formats

    def _format_excel_styler(
        self, styler: pd.io.formats.style.Styler, table_name: str,
    ) -> pd.io.formats.style.Styler:
        """将数字格式合并进已有色阶样式，避免单元格样式覆盖列格式。"""
        for column, number_format in self._number_formats(styler.data, table_name).items():
            styler = styler.set_properties(subset=[column], **{"number-format": number_format})
        return styler

    def _apply_number_formats(
        self, writer: Any, sheet_name: str, frame: pd.DataFrame, table_name: str,
    ) -> None:
        """设置空单元格及无 Styler 的比较表列格式，保留原统计数值。"""
        for column, number_format in self._number_formats(frame, table_name).items():
            index = frame.columns.get_loc(column)
            writer.sheets[sheet_name].set_column(index, index, None, writer.book.add_format({"num_format": number_format}))

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
        if (isinstance(sort_by, str) and sort_by in df_pd.columns) or (isinstance(sort_by, list) and all(c in df_pd.columns for c in sort_by)):
            df_pd = df_pd.sort_values(by=sort_by, ascending=sort_ascending)

        # 匹配列排序
        df_pd = self._reorder_trend_cols(df_pd, group_ascending=group_ascending)

        # 动态识别趋势列范围，避免依赖固定列位置。
        time_cols = [c for c in df_pd.columns if not is_metadata_column(c) and c != "total"]

        if not time_cols:
            return

        worksheet = writer.sheets[sheet_name]

        # PSI 专用三色阶 (红绿灯)
        if metric == "psi":
            for column_index, column in enumerate(df_pd.columns):
                if is_metadata_column(column) or not pd.api.types.is_numeric_dtype(df_pd[column]):
                    continue
                worksheet.conditional_format(1, column_index, len(df_pd), column_index, {
                    'type': '3_color_scale',
                    'min_type': 'num', 'min_value': 0.05, 'min_color': '#63BE7B', # 绿色
                    'mid_type': 'num', 'mid_value': 0.15, 'mid_color': '#FFEB84', # 黄色
                    'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'  # 红色
                })

    def _get_styler(
        self: Any,
        df_input: Any,
        title: str,
        cmap: str,
        sort_by: List[str] | None = None,
        sort_ascending: bool = False, # 统一内部 API 命名
        subset_cols: List[str] | None = None,
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
        # 只对真实数值列应用渐变；文本状态、Boolean 和全空列保留原值。
        num_cols: pd.Index = df.select_dtypes(include=['number']).columns
        candidates: List[str] = subset_cols if subset_cols is not None else list(df.columns)
        gradient_cols: List[str] = [
            column for column in candidates
            if column in num_cols and not is_metadata_column(column) and np.isfinite(df[column].dropna()).any()
        ]

        styler = df.style.set_caption(f"<b>{title}</b>").hide(axis="index")

        # 应用热力图
        if gradient_cols:
            styler = styler.background_gradient(
                cmap=cmap,
                subset=gradient_cols,
                axis=None,
                vmin=vmin,
                vmax=vmax,
                gmap=df[gradient_cols].where(np.isfinite(df[gradient_cols])),
            )

        # 复用 HTML 数值格式；文本按 HTML 转义，比例的整数端点仍显示百分比。
        formatters: dict[str, Any] = {}
        for column in num_cols:
            if is_metadata_column(column):
                continue
            unit = _definition(column)["unit"]
            percent = fmt_as_pct or (
                column not in {"auc", "auc_bin"} and (
                    unit in {"ratio", "ratio_difference"}
                    or "rate" in column or "ratio" in column
                )
            )
            if not percent and unit in {"count", "count_or_weight"}:
                formatters[column] = _format_count
            elif not percent and pd.api.types.is_integer_dtype(df[column].dtype):
                formatters[column] = "{:,}"
            else:
                formatters[column] = partial(format_html_value, as_percent=percent, missing_text="—")
        styler = styler.format(formatters, na_rep="—", escape="html")

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
