"""分箱评估 Excel 导出实现。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from mars.compute import to_pandas_frame
from mars.reporting._metadata import export_report_semantics
from mars.utils.logger import logger


class _BinningExcelWriter:
    """分箱报告 Excel 导出能力。"""

    def __init__(self, report: Any) -> None:
        self._report = report

    def __getattr__(self, name: str) -> Any:
        """将只读数据访问委托给 report 容器。"""
        return getattr(self._report, name)

    @classmethod
    def _resolve_excel_template_path(
        cls: type[_BinningExcelWriter],
        file_name: str,
    ) -> Path:
        """解析 `mars.reporting.template` 下的 Excel 模板路径。"""
        reporting_spec = importlib.util.find_spec("mars.reporting")
        if reporting_spec is None or reporting_spec.origin is None:
            raise FileNotFoundError("无法定位 `mars.reporting` 模块，因此无法读取 Excel 模板。")

        reporting_dir = Path(reporting_spec.origin).resolve().parent
        template_path = reporting_dir / "template" / file_name
        if not template_path.exists():
            raise FileNotFoundError(f"找不到 Excel 模板文件: {template_path}")
        return template_path

    @classmethod
    def _read_detail_template_schema(
        cls: type[_BinningExcelWriter],
        template_path: Path,
        sheet_name: str,
    ) -> Tuple[List[str], int, int]:
        """读取明细 sheet 的模板列与表格列范围。"""
        import openpyxl
        from openpyxl.utils.cell import range_boundaries

        workbook = openpyxl.load_workbook(template_path)
        try:
            worksheet = workbook[sheet_name]
            if not worksheet.tables:
                raise ValueError(f"sheet `{sheet_name}` 缺少 Excel table，无法按模板导出明细表。")

            table = next(iter(worksheet.tables.values()))
            table_ref = getattr(table, "ref", None)
            if not isinstance(table_ref, str) or not table_ref:
                raise ValueError(
                    f"sheet `{sheet_name}` 的 Excel table 缺少 `ref`，无法读取模板列结构。"
                )

            first_col, _, last_col, _ = range_boundaries(table_ref)
            headers: List[str] = []
            for col_idx in range(first_col, last_col + 1):
                header_value = worksheet.cell(row=1, column=col_idx).value
                if header_value is None or not str(header_value).strip():
                    raise ValueError(
                        f"sheet `{sheet_name}` 第 1 行第 {col_idx} 列表头为空，无法按模板导出。"
                    )
                headers.append(str(header_value))

            return headers, first_col, last_col
        finally:
            workbook.close()

    @classmethod
    def _build_excel_detail_export_frame(
        cls: type[_BinningExcelWriter],
        detail_df: pd.DataFrame,
        template_headers: List[str],
    ) -> pd.DataFrame:
        """按模板列顺序构建 Excel 明细导出数据。"""
        missing_columns = [
            column_name
            for column_name in template_headers
            if column_name not in detail_df.columns
        ]
        if missing_columns:
            missing_display = ", ".join(missing_columns)
            raise ValueError(
                "Excel 模板列在 `detail_table` 中缺失："
                f" {missing_display}"
            )

        return detail_df.loc[:, template_headers].copy()

    def _additional_excel_tables(self) -> dict[str, pd.DataFrame]:
        """模板透视之外交付最终物化结果与口径，不依赖历史缓存。"""
        tables = {
            "最终汇总": to_pandas_frame(self.summary_table),
        }
        if "ks" in self.trend_tables:
            tables["最终KS趋势"] = to_pandas_frame(self.trend_tables["ks"])
        if self.report_meta.get("ks_method") == "raw":
            tables["KS口径"] = pd.DataFrame(
                list(self.report_meta["ks_source_by_feature"].items()),
                columns=["feature", "ks_method"],
            )
            tables["KS计算诊断"] = pd.DataFrame(self.report_meta["raw_ks_diagnostics"])
        tables.update(export_report_semantics(self))
        return tables

    @staticmethod
    def _prepare_openpyxl_pivots(workbook: Any) -> None:
        """验证最终 KS 的模板来源并清除旧缓存，透视由 Excel 打开后刷新。"""
        from openpyxl.utils.cell import range_boundaries

        for sheet in workbook:
            for pivot in sheet._pivots:
                cache = pivot.cache
                cache.refreshOnLoad = True
                cache.enableRefresh = True
                cache.records = None
                cache.id = None
                cache.recordCount = 0
                cache.saveData = False
                cache.missingItemsLimit = 0
                for field in pivot.dataFields:
                    source = cache.cacheFields[field.fld].name
                    if source == "ks_bin":
                        raise ValueError("Excel 模板 KS 透视必须读取最终 ks，不能读取 ks_bin。")
                    if source == "ks":
                        field.name = (
                            "avg(KS最终，0–100点)" if field.subtotal == "average"
                            else "KS（最终，0–100点）"
                        )
                        # 只替换表头文本，不移动模板的布局和样式。
                        for row in sheet:
                            for cell in row:
                                if cell.value in {"KS", "avg(KS) "}:
                                    cell.value = field.name

                # 旧模板的缓存值不能作为本次结果交付，保留表头、筛选区与样式。
                first_col, first_row, last_col, last_row = range_boundaries(pivot.location.ref)
                for row in sheet.iter_rows(
                    min_row=first_row + pivot.location.firstDataRow,
                    max_row=last_row, min_col=first_col, max_col=last_col,
                ):
                    for cell in row:
                        cell.value = None

    @staticmethod
    def _refresh_xlwings_pivots(workbook: Any) -> None:
        """原生 Excel 刷新扩展后的同一模板，KS 透视直接读取已计算的最终 KS。"""
        if sys.platform.startswith("darwin"):
            # xlwings 的 Mac 原生对象使用 AppleScript 字典，不能调用 Windows COM 方法。
            workbook.api.refresh_all()
            for cache in workbook.api.pivot_caches.get():
                cache.missing_items_limit.set(0)
                cache.refresh()
            for sheet in workbook.sheets:
                for pivot in sheet.api.pivot_tables.get():
                    for field in pivot.data_fields.get():
                        if field.source_name.get() == "ks":
                            caption = (
                                "avg(KS最终，0–100点)" if str(field.name.get()).startswith("avg(")
                                else "KS（最终，0–100点）"
                            )
                            field.name.set(caption)
                    pivot.refresh_table()
            workbook.app.calculate()
            return
        workbook.api.RefreshAll()
        workbook.app.api.CalculateUntilAsyncQueriesDone()
        for sheet in workbook.sheets:
            pivots = sheet.api.PivotTables()
            for pivot_index in range(1, pivots.Count + 1):
                pivot = pivots.Item(pivot_index)
                # 不保留模板中已不存在的特征/目标/分组筛选项。
                pivot.PivotCache().MissingItemsLimit = 0
                pivot.PivotCache().Refresh()
                fields = pivot.DataFields
                for field_index in range(1, fields.Count + 1):
                    field = fields.Item(field_index)
                    if field.SourceName == "ks":
                        field.Caption = (
                            "avg(KS最终，0–100点)" if field.Function == -4106
                            else "KS（最终，0–100点）"
                        )
                pivot.RefreshTable()
        workbook.app.api.CalculateFullRebuild()

    @staticmethod
    def _write_xlwings_detail_rows(
        worksheet: Any, rows: list[list[Any]], *, start_row: int, first_col: int, last_col: int,
    ) -> None:
        """按确定区域分块写明细，避免大矩阵经过 xlwings 的隐式 Range.resize。"""
        from openpyxl.utils import get_column_letter

        first_letter = get_column_letter(first_col)
        last_letter = get_column_letter(last_col)
        for offset in range(0, len(rows), 5000):
            chunk = rows[offset:offset + 5000]
            first_row = start_row + offset
            last_row = first_row + len(chunk) - 1
            region = worksheet.range(f"{first_letter}{first_row}:{last_letter}{last_row}")
            if sys.platform.startswith("darwin"):
                region.value = chunk
            else:
                region.api.Value = tuple(tuple(row) for row in chunk)

    def write_excel(self: Any, path: str = "mars_bin_report.xlsx", engine: str = "openpyxl") -> None:
        """
        导出评估 Excel 报告。

        raw 与 binned 使用同一引擎模板。raw 的透视 KS 数据字段读取最终
        ``ks``，箱级 ``ks_bin`` 保持分箱口径；附加最终汇总、趋势及诊断表。
        openpyxl 不计算或刷新透视，最终物化结果可直接读取，透视由 Excel 打开后刷新。

        Parameters
        ----------
        path : str
            导出的 Excel 文件路径。
        engine : str
            写入 Excel 的底层引擎。
            - "auto": 自动检测，Win/Mac 下优先尝试 xlwings，若失败或在 Linux 下则回退至 openpyxl。
            - "xlwings": 强制使用 xlwings 引擎 (依赖本地安装的 Excel 应用程序，格式保留最完美)。
            - "openpyxl": 强制使用 openpyxl 引擎 (无需安装 Excel，跨平台兼容性好)。

        Raises
        ------
        ValueError
            当 ``engine`` 不在支持列表中时抛出。
        RuntimeError
            当底层 Excel 导出流程失败时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12], "ks": [18.0]})
        >>> detail = pl.DataFrame({"feature": ["age"], "bin_index": [0], "count": [100]})
        >>> report = MarsBinningReport(summary, {}, detail)
        >>> with TemporaryDirectory() as tmp:
        ...     report.write_excel(str(Path(tmp) / "evaluation.xlsx"), engine="openpyxl") is None
        True
        """
        valid_engines = ["auto", "xlwings", "openpyxl"]
        if engine not in valid_engines:
            raise ValueError(f"不支持的 engine: '{engine}'，请从 {valid_engines} 中选择。")

        # 从首个数据行写入，不能把模板的示例箱行当成本次评估数据。
        start_write_row = 2
        sheet_name = "分组明细"
        template_name_xlwings = "mars_bin_report_win_mac.xlsx"
        template_name_openpyxl = "mars_bin_report_linux.xlsx"
        is_gui_env = sys.platform.startswith("win") or sys.platform.startswith("darwin")
        use_xlwings = engine == "xlwings" or (engine == "auto" and is_gui_env)

        if use_xlwings:
            try:
                import xlwings as xw

                template_path = self._resolve_excel_template_path(template_name_xlwings)
            except Exception as exc:
                if engine == "xlwings":
                    raise RuntimeError(
                        "强制使用 xlwings 导出失败，请确认本机已安装 Excel 与 xlwings。"
                        f"\n错误详情: {exc}"
                    ) from exc
                logger.warning("xlwings 不可用，自动降级为 openpyxl 导出: %s", exc)
                use_xlwings = False

        if not use_xlwings:
            template_path = self._resolve_excel_template_path(template_name_openpyxl)

        template_headers, first_col, last_col = self._read_detail_template_schema(
            template_path=template_path,
            sheet_name=sheet_name,
        )
        detail_pd: pd.DataFrame = to_pandas_frame(self.detail_table).copy()
        # 两个模板都显式保留最终 KS；分箱口径复用现有箱级结果，不重新分箱。
        if "ks" not in detail_pd.columns:
            if self.report_meta.get("ks_method") == "raw":
                raise ValueError("raw KS 报告缺少已计算的 detail.ks，不能用 ks_bin 代替。")
            keys = ["y", "feature", self.detail_group_col or "mars_group"]
            detail_pd["ks"] = detail_pd.groupby(keys, dropna=False)["ks_bin"].transform("max")
        export_pd = self._build_excel_detail_export_frame(
            detail_df=detail_pd,
            template_headers=template_headers,
        )
        rows: List[List[Any]] = export_pd.astype(object).where(pd.notna(export_pd), None).values.tolist()
        final_row = start_write_row + len(rows) - 1
        table_last_row = max(final_row, start_write_row - 1)
        from openpyxl.utils import get_column_letter

        table_ref = (
            f"{get_column_letter(first_col)}1:"
            f"{get_column_letter(last_col)}{table_last_row}"
        )

        if use_xlwings:
            app = None
            workbook = None
            fallback_openpyxl = False
            try:
                app = xw.App(visible=False, add_book=False)
                app.display_alerts = False
                app.screen_updating = False

                workbook = app.books.open(str(template_path))
                worksheet = workbook.sheets[sheet_name]
                self._write_xlwings_detail_rows(
                    worksheet, [template_headers], start_row=1,
                    first_col=first_col, last_col=last_col,
                )
                self._write_xlwings_detail_rows(
                    worksheet, rows, start_row=start_write_row,
                    first_col=first_col, last_col=last_col,
                )

                table = worksheet.tables[0] if worksheet.tables else None
                if table is None:
                    raise ValueError(f"sheet `{sheet_name}` 缺少 Excel table，无法按模板导出明细表。")
                table.resize(
                    worksheet.range(table_ref)
                )

                last_used_row = worksheet.used_range.last_cell.row
                if last_used_row > table_last_row:
                    worksheet.range(f"{table_last_row + 1}:{last_used_row}").delete()

                for name, frame in self._additional_excel_tables().items():
                    sheet = workbook.sheets.add(name)
                    if frame.shape[1]:
                        sheet.activate()
                        last_letter = get_column_letter(frame.shape[1])
                        extra_rows = [list(frame.columns)] + (
                            frame.astype(object).where(pd.notna(frame), None).values.tolist()
                        )
                        self._write_xlwings_detail_rows(
                            sheet, extra_rows, start_row=1, first_col=1, last_col=frame.shape[1],
                        )
                        region = sheet.range(f"A1:{last_letter}{len(frame) + 1}")
                        if not frame.empty:
                            sheet.tables.add(source=region, table_style_name="TableStyleMedium2")
                        header = sheet.range(f"A1:{last_letter}1")
                        header.color = "#1F4E78"
                        header.font.color = "#FFFFFF"
                        header.font.bold = True
                        region.columns.autofit()
                        sheet.range("A2").select()
                        if sys.platform.startswith("darwin"):
                            workbook.api.windows[1].freeze_panes.set(True)
                        else:
                            app.api.ActiveWindow.FreezePanes = True
                self._refresh_xlwings_pivots(workbook)
                workbook.save(path)
                logger.info("Exported binning report via xlwings: %s", path)
            except Exception as exc:
                if engine == "auto":
                    logger.warning("xlwings 导出失败，自动降级为 openpyxl: %s", exc)
                    fallback_openpyxl = True
                else:
                    raise RuntimeError(f"xlwings 导出过程中发生错误: {exc}") from exc
            finally:
                if workbook is not None:
                    workbook.close()
                if app is not None:
                    app.quit()
            if fallback_openpyxl:
                self.write_excel(path, engine="openpyxl")
            return

        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.worksheet.table import Table, TableStyleInfo

        workbook = openpyxl.load_workbook(template_path)
        try:
            worksheet = workbook[sheet_name]
            for column_offset, header in enumerate(template_headers):
                worksheet.cell(row=1, column=first_col + column_offset, value=header)
            for row_offset, row_data in enumerate(rows):
                current_row = start_write_row + row_offset
                for col_offset, value in enumerate(row_data):
                    worksheet.cell(
                        row=current_row,
                        column=first_col + col_offset,
                        value=value,
                    )

            if not worksheet.tables:
                raise ValueError(f"sheet `{sheet_name}` 缺少 Excel table，无法按模板导出明细表。")

            table = next(iter(worksheet.tables.values()))
            if not hasattr(table, "ref"):
                raise ValueError(f"sheet `{sheet_name}` 的 Excel table 缺少 `ref`，无法更新表格范围。")
            table.ref = table_ref
            table.autoFilter.ref = table_ref

            if worksheet.max_row > table_last_row:
                worksheet.delete_rows(table_last_row + 1, worksheet.max_row - table_last_row)

            self._prepare_openpyxl_pivots(workbook)
            worksheet.freeze_panes = "A2"
            for table_index, (name, frame) in enumerate(self._additional_excel_tables().items()):
                sheet = workbook.create_sheet(name)
                sheet.append(list(frame.columns))
                for record in frame.astype(object).where(pd.notna(frame), None).itertuples(index=False, name=None):
                    sheet.append(list(record))
                sheet.freeze_panes = "A2"
                if frame.shape[1]:
                    ref = f"A1:{get_column_letter(frame.shape[1])}{max(1, sheet.max_row)}"
                    sheet.auto_filter.ref = ref
                    if not frame.empty:
                        extra_table = Table(displayName=f"mars_final_{table_index}", ref=ref)
                        extra_table.tableStyleInfo = TableStyleInfo(
                            name="TableStyleMedium2", showRowStripes=True,
                        )
                        sheet.add_table(extra_table)
                    for cell in sheet[1]:
                        cell.fill = PatternFill("solid", fgColor="1F4E78")
                        cell.font = Font(color="FFFFFF", bold=True)
                    for column_index, column in enumerate(frame.columns, start=1):
                        sheet.column_dimensions[get_column_letter(column_index)].width = (
                            28 if column in {"feature", "display_name", "data_source"} else 18
                        )
                        if pd.api.types.is_numeric_dtype(frame[column]):
                            for column_cells in sheet.iter_cols(
                                min_col=column_index, max_col=column_index, min_row=2,
                            ):
                                for cell in column_cells:
                                    cell.number_format = "0.00"
            workbook.save(path)
            logger.info("Exported binning report via openpyxl: %s", path)
        finally:
            workbook.close()
        return
