"""分箱评估 Excel 导出实现。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from mars.compute import to_pandas_frame
from mars.reporting._metadata import export_report_semantics, is_metadata_column
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

    @staticmethod
    def _lookup_excel_ks(
        frame: pd.DataFrame,
        values: pd.DataFrame,
        *,
        target_col: str,
        primary_target: Any,
        require_total: bool = False,
    ) -> pd.Series:
        """按原表特征/目标行匹配分箱 KS，保留行序与空值；汇总必须有 Total 证据。"""
        if "feature" not in frame.columns:
            raise ValueError("原生 Excel KS 导出需要表中的 feature 身份列。")
        frame_target = next((col for col in ("target", "y") if col in frame.columns), None)
        keys = ["feature"]
        source: pd.DataFrame = values[["feature", target_col, "ks"]]
        if frame_target is not None:
            keys.append(frame_target)
            source = source.rename(columns={target_col: frame_target})
        else:
            source = source.loc[source[target_col] == primary_target]
        matched: pd.DataFrame = frame[keys].merge(
            source[keys + ["ks"]], on=keys, how="left", sort=False,
            validate="many_to_one", indicator=True,
        )
        if require_total and (matched["_merge"] != "both").any():
            raise ValueError("原生 Excel 最终汇总缺少对应 feature/target 的 Total 分箱 KS 明细。")
        return matched["ks"]

    def _prepare_excel_export(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """从物化结果构建独立的分箱 KS 导出视图，两条引擎共用且不修改源报告。"""
        detail: pd.DataFrame = to_pandas_frame(self.detail_table)
        # 历史模板使用 mars_group；已保存实际分组契约的报告优先使用契约列。
        group_col = self.detail_group_col or "mars_group"
        target_col = "y" if "y" in detail.columns else "target"
        keys = ["feature", target_col, group_col]
        missing = [col for col in [*keys, "ks_bin"] if col not in detail.columns]
        if missing:
            raise ValueError(
                "原生 Excel 分箱 KS 导出缺少 detail_table 必需字段："
                + ", ".join(missing)
            )

        # 无条件覆盖导出 ks，避免生成未观测分类组合，并保留 null 组与全空 KS。
        detail["ks"] = detail.groupby(
            keys, dropna=False, observed=True, sort=False,
        )["ks_bin"].transform("max")
        grouped: pd.DataFrame = detail[keys + ["ks"]].drop_duplicates(subset=keys)
        targets = self.report_meta.get("targets") or []
        observed_targets = detail[target_col].drop_duplicates().tolist()
        if targets:
            primary_target = targets[0]
        elif len(observed_targets) == 1:
            primary_target = observed_targets[0]
        else:
            raise ValueError("原生 Excel KS 趋势需要 report_meta.targets 指明首目标。")
        if primary_target not in observed_targets:
            raise ValueError("原生 Excel KS 导出首目标在 detail_table 中不存在。")

        summary: pd.DataFrame = to_pandas_frame(self.summary_table)
        if "ks" not in summary.columns:
            raise ValueError("原生 Excel 最终汇总缺少 summary_table.ks。")
        totals = grouped.loc[grouped[group_col] == "Total"]
        summary["ks"] = self._lookup_excel_ks(
            summary, totals, target_col=target_col, primary_target=primary_target,
            require_total=True,
        ).to_numpy()
        tables: dict[str, pd.DataFrame] = {"最终汇总": summary}
        if "ks" in self.trend_tables:
            trend: pd.DataFrame = to_pandas_frame(self.trend_tables["ks"])
            groups = grouped[group_col].astype("string").fillna("null")
            # 只替换现有分组列，首目标契约、身份列、元数据及行列顺序全部沿用源趋势。
            for column in trend.columns:
                if is_metadata_column(column):
                    continue
                values = grouped.loc[groups == column]
                trend[column] = self._lookup_excel_ks(
                    trend, values, target_col=target_col, primary_target=primary_target,
                ).to_numpy()
            tables["最终KS趋势"] = trend

        source_method = self.report_meta.get("ks_method", "binned")
        tables["KS口径"] = pd.DataFrame(
            [
                ("source_report_ks_method", source_method),
                ("exported_ks_method", "binned"),
                ("exported_ks_source", f"max(detail.ks_bin) over (feature, {target_col}, {group_col})"),
                ("ks_unit", "0–100 点"),
                ("pivot_ks", "分箱 KS（0–100点）；综合指标保留模板 average 聚合"),
                ("source_report_semantics", "ReportSemantics、CalculationStatus 保留源报告身份、参数及计算状态"),
            ], columns=["key", "value"],
        )
        if source_method == "raw":
            tables["KS口径"] = pd.concat([
                tables["KS口径"], pd.DataFrame([
                    ("source_ks_by_feature", str(self.report_meta.get("ks_source_by_feature", {}))),
                    ("source_raw_diagnostics", "源报告raw KS计算诊断仅解释源 raw KS，不是 Excel 分箱 KS 诊断"),
                ], columns=["key", "value"]),
            ], ignore_index=True)
            tables["源报告raw KS计算诊断"] = pd.DataFrame(
                self.report_meta.get("raw_ks_diagnostics", []),
            )
        tables.update(export_report_semantics(self))
        # 模板字段保持原名与 22 列布局，业务分组列仅在导出副本中映射。
        detail["mars_group"] = detail[group_col]
        detail["y"] = detail[target_col]
        return detail, tables

    @staticmethod
    def _prepare_openpyxl_pivots(workbook: Any) -> None:
        """验证分箱 KS 导出列并清除旧缓存，透视由 Excel 打开后刷新。"""
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
                        raise ValueError("Excel 模板 KS 透视必须读取导出分组分箱 ks，不能直接读取逐箱 ks_bin。")
                    if source == "ks":
                        field.name = (
                            "avg(分箱 KS，0–100点)" if field.subtotal == "average"
                            else "分箱 KS（0–100点）"
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
        """原生 Excel 刷新同一模板，KS 透视读取已聚合的导出分箱 KS。"""
        if sys.platform.startswith("darwin"):
            # xlwings 的 Mac 原生对象使用 AppleScript 字典，不能调用 Windows COM 方法。
            workbook.api.refresh_all()
            for cache in workbook.api.pivot_caches.get():
                cache.missing_items_limit.set(0)
                cache.refresh()
            for sheet in workbook.sheets:
                for pivot in sheet.api.pivot_tables.get():
                    for field in pivot.data_fields.get():
                        if field.source_name.get() == "ks_bin":
                            raise ValueError("Excel 模板 KS 透视必须读取导出分组分箱 ks，不能直接读取逐箱 ks_bin。")
                        if field.source_name.get() == "ks":
                            caption = (
                                "avg(分箱 KS，0–100点)" if str(field.name.get()).startswith("avg(")
                                else "分箱 KS（0–100点）"
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
                    if field.SourceName == "ks_bin":
                        raise ValueError("Excel 模板 KS 透视必须读取导出分组分箱 ks，不能直接读取逐箱 ks_bin。")
                    if field.SourceName == "ks":
                        field.Caption = (
                            "avg(分箱 KS，0–100点)" if field.Function == -4106
                            else "分箱 KS（0–100点）"
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

        raw 与 binned 使用同一引擎模板，主要 KS 均为分箱 KS（0–100点）。
        导出副本按特征、目标和实际分组聚合 ``max(detail.ks_bin)``，写入 ``ks``
        供透视、最终汇总和首目标趋势读取；不重新拟合或修改源报告的 KS 及状态。
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
            当 ``engine`` 不在支持列表中，或缺少分箱 KS 所需字段或 Total 明细时抛出。
        RuntimeError
            当底层 Excel 导出流程失败时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from mars.analysis import profile_risk
        >>> frame = pl.DataFrame({"age": list(range(12)), "bad": [0, 1] * 6})
        >>> report = profile_risk(frame, target="bad", features=["age"], n_bins=2).report
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
        detail_pd, additional_tables = self._prepare_excel_export()
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

                for name, frame in additional_tables.items():
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
                    # cell(..., value=None) 不会清除模板示例值，必须显式写入空值。
                    cell = worksheet.cell(
                        row=current_row,
                        column=first_col + col_offset,
                    )
                    cell.value = value

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
            for table_index, (name, frame) in enumerate(additional_tables.items()):
                sheet = workbook.create_sheet(name)
                sheet.append(list(frame.columns))
                for record in frame.astype(object).where(pd.notna(frame), None).itertuples(index=False, name=None):
                    sheet.append(list(record))
                sheet.freeze_panes = "A2"
                if frame.shape[1]:
                    ref = f"A1:{get_column_letter(frame.shape[1])}{max(1, sheet.max_row)}"
                    # 有原生 Table 时由其提供筛选，重复的 worksheet 筛选会使 Excel 拒绝打开。
                    if not frame.empty:
                        extra_table = Table(displayName=f"mars_final_{table_index}", ref=ref)
                        extra_table.tableStyleInfo = TableStyleInfo(
                            name="TableStyleMedium2", showRowStripes=True,
                        )
                        sheet.add_table(extra_table)
                    else:
                        sheet.auto_filter.ref = ref
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
