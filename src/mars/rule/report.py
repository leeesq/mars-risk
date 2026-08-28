"""规则挖掘 HTML 与 Excel 报告。"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd
import polars as pl

from mars.compute import FrameLike, to_pandas_table, to_polars_frame


@dataclass(frozen=True)
class MarsRuleReport:
    """规则挖掘的结构化报告与显式导出器。

    Parameters
    ----------
    summary_table : polars.DataFrame
        挖掘状态、候选数量和验证状态汇总。
    detail_tables : Mapping[str, polars.DataFrame]
        候选审计、评估、切片和可选高级分析表。
    metadata : Mapping[str, Any]
        已解析策略、数据角色和运行版本。
    caption : str
        Notebook 与文件报告标题。
    """

    summary_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    detail_tables: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    caption: str = "MARS Rule Mining Report"

    @classmethod
    def from_benchmark(
        cls,
        benchmark: Union[FrameLike, Mapping[str, Any], Sequence[Mapping[str, Any]]],
        *,
        caption: str = "MARS Rule Benchmark Report",
    ) -> MarsRuleReport:
        """从 benchmark 记录构造可导出的结构化报告。

        Parameters
        ----------
        benchmark : Union[FrameLike, Mapping[str, Any], Sequence[Mapping[str, Any]]]
            单条记录、记录序列或 Pandas/Polars 表。
        caption : str
            报告标题。

        Returns
        -------
        MarsRuleReport
            包含 benchmark 明细和行数汇总的报告。

        Raises
        ------
        TypeError
            benchmark 不是支持的表或记录结构时抛出。
        """
        try:
            benchmark_table: pl.DataFrame = _benchmark_to_frame(benchmark)
        except TypeError as exc:
            raise TypeError(
                "benchmark 必须是 DataFrame、mapping 或 mapping 序列。"
            ) from exc
        return cls(
            summary_table=pl.DataFrame([{"benchmark_rows": benchmark_table.height}]),
            detail_tables={"benchmark": benchmark_table},
            metadata={"report_type": "benchmark"},
            caption=caption,
        )

    def write_excel(
        self,
        path: Union[str, Path] = "mars_rule_report.xlsx",
        *,
        engine: str | None = None,
    ) -> None:
        """把报告写入多工作表 Excel。

        Parameters
        ----------
        path : Union[str, Path]
            输出工作簿路径；父目录会自动创建。
        engine : str | None
            可选 Pandas ExcelWriter 引擎。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_table = pd.DataFrame(
            [{"key": str(key), "value": json.dumps(value, ensure_ascii=False, default=str)}
             for key, value in self.metadata.items()]
        )
        with pd.ExcelWriter(output_path, engine=engine) as writer:
            to_pandas_table(self.summary_table).to_excel(writer, sheet_name="summary", index=False)
            metadata_table.to_excel(writer, sheet_name="metadata", index=False)
            used_sheet_names = {"summary", "metadata"}
            for name, table in self.detail_tables.items():
                sheet_name: str = _safe_sheet_name(str(name), used_sheet_names)
                used_sheet_names.add(sheet_name)
                to_pandas_table(table).to_excel(writer, sheet_name=sheet_name, index=False)

    def write_html(
        self,
        path: Union[str, Path] = "mars_rule_report.html",
    ) -> Path:
        """写出自包含 HTML 规则报告。

        Parameters
        ----------
        path : Union[str, Path]
            输出 HTML 路径；父目录会自动创建。

        Returns
        -------
        Path
            实际写出的文件路径。
        """
        output_path: Path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render_html(), encoding="utf-8")
        return output_path

    def render_html(self) -> str:
        """渲染不落盘的自包含 HTML 字符串。

        Returns
        -------
        str
            完整且对用户字段执行 HTML 转义的文档。
        """
        sections = [
            f"<h1>{html.escape(self.caption)}</h1>",
            "<h2>Summary</h2>",
            to_pandas_table(self.summary_table).to_html(index=False, escape=True),
            "<h2>Metadata</h2>",
            f"<pre>{html.escape(json.dumps(dict(self.metadata), ensure_ascii=False, indent=2, default=str))}</pre>",
        ]
        for name, table in self.detail_tables.items():
            sections.append(f"<h2>{html.escape(str(name).replace('_', ' ').title())}</h2>")
            sections.append(to_pandas_table(table).to_html(index=False, escape=True))
        document: str = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#202124}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #ddd;padding:6px;text-align:right}}th{{background:#f4f5f7}}
pre{{background:#f7f7f8;padding:12px;overflow:auto}}</style></head>
<body>{body}</body></html>""".format(
            title=html.escape(self.caption),
            body="\n".join(sections),
        )
        return document


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """生成合法且不重复的 Excel 工作表名称。"""
    cleaned: str = "".join("_" if char in "[]:*?/\\" else char for char in name).strip("'")
    base: str = cleaned[:31] or "table"
    candidate: str = base
    counter: int = 2
    while candidate in used:
        suffix: str = f"_{counter}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        counter += 1
    return candidate


def _benchmark_to_frame(
    benchmark: Union[FrameLike, Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> pl.DataFrame:
    """把 benchmark 支持类型规范为 Polars 表。"""
    if isinstance(benchmark, (pl.DataFrame, pd.DataFrame)):
        return to_polars_frame(benchmark)
    if isinstance(benchmark, Mapping):
        return pl.DataFrame([dict(benchmark)])
    if isinstance(benchmark, Sequence) and not isinstance(benchmark, (str, bytes)):
        if any(not isinstance(row, Mapping) for row in benchmark):
            raise TypeError("benchmark 序列中的每个元素都必须是 mapping。")
        return pl.DataFrame([dict(row) for row in benchmark])
    raise TypeError("benchmark 必须是 DataFrame、mapping 或 mapping 序列。")
