"""规则挖掘 HTML 与 Excel 报告。"""

from __future__ import annotations

import html
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd
import polars as pl

from mars.compute import FrameLike, to_pandas_table, to_polars_frame
from mars.reporting._metadata import FeatureMetadata, export_semantics
from mars.reporting._query import ReportFrame, _ReportQuery
from mars.reporting._serialization import encode
from mars.rule._report_semantics import _rule_snapshot


class MarsRuleReport(_ReportQuery):
    """规则挖掘的结构化报告与显式导出器。

    Parameters
    ----------
    summary_table : pl.DataFrame | None
        挖掘状态、候选数量和验证状态汇总。
    detail_tables : Mapping[str, pl.DataFrame] | None
        候选审计、评估、切片和可选高级分析表。
    metadata : Mapping[str, Any] | None
        已解析策略、数据角色和运行版本。
    caption : str
        Notebook 与文件报告标题。
    feature_metadata : FeatureMetadata | None
        英文特征 ID 对应的业务名、来源、定义和单位。
    business_context : dict[str, Any] | None
        标签定义、样本范围、币种及证据来源；缺失解释保持 unknown。

    Attributes
    ----------
    report_id : str
        构造时生成并在查询、保存与恢复后保持的报告身份。
    report_type : str
        rule 或 rule_benchmark；后者不包含规则资格。
    format_version : int
        公共 marsreport 格式版本，目前为 1。

    Notes
    -----
    Rule 仍为 Experimental。报告只保存已有证据，不重建 RuleSet 或赋予部署权限。
    业务元数据、上下文或规则表达式校验失败传播 ValueError。

    Examples
    --------
    >>> report = MarsRuleReport()
    >>> report.describe()["report_type"]
    'rule'
    """

    def __init__(
        self,
        summary_table: pl.DataFrame | None = None,
        detail_tables: Mapping[str, pl.DataFrame] | None = None,
        metadata: Mapping[str, Any] | None = None,
        caption: str = "MARS Rule Mining Report",
        *,
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
    ) -> None:
        snapshot = _rule_snapshot(
            summary_table if summary_table is not None else pl.DataFrame(),
            dict(detail_tables or {}),
            dict(metadata or {}),
            feature_metadata,
            business_context,
        )
        self._tables = snapshot._query_tables()
        self._description = snapshot.describe()
        self.report_id = snapshot.report_id
        self.report_type = snapshot.report_type
        self.format_version = snapshot.format_version
        self.feature_metadata = snapshot.feature_metadata
        self.business_context = snapshot.business_context
        self.source = snapshot.source
        self.report_meta = snapshot.report_meta
        self.summary_table = self._tables["summary"]
        self.detail_tables = {
            name: frame for name, frame in self._tables.items() if name != "summary"
        }
        self.metadata = self.report_meta
        self.caption = caption

    def _query_tables(self) -> dict[str, ReportFrame]:
        """提供公共统计表，查询不访问挖掘工作流。"""
        return self._tables

    def _query_metadata(self) -> dict[str, Any]:
        """提供真实挖掘、验证及高级分析参数。"""
        return self.report_meta

    def describe(self) -> dict[str, Any]:
        """取得包含规则关联、单位和状态的公共目录。

        Returns
        -------
        dict[str, Any]
            独立目录副本；无规则和未计算状态保持原义。

        Examples
        --------
        >>> MarsRuleReport().describe()["format_version"]
        1
        """
        description = deepcopy(self._description)
        description.update(
            parameters=deepcopy(self.metadata),
            feature_metadata=deepcopy(self.feature_metadata),
            business_context=deepcopy(self.business_context),
            source=deepcopy(self.source),
        )
        return description

    def show_table(self, name: str, **query: Any) -> pd.DataFrame:
        """查询小表并附上业务显示名。

        Parameters
        ----------
        name : str
            公共表名。
        **query : Any
            get_table 查询条件、投影和分页参数。

        Returns
        -------
        pd.DataFrame
            独立可读表，英文 ID 保留。

        Examples
        --------
        >>> report.show_table("candidates", limit=5)  # doctest: +SKIP
        """
        return self._display_frame(self.get_table(name, **query))

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

        Examples
        --------
        >>> MarsRuleReport.from_benchmark({"seconds": 1.25}).report_type
        'rule_benchmark'
        """
        try:
            benchmark_table: pl.DataFrame = _benchmark_to_frame(benchmark)
        except TypeError as exc:
            raise TypeError("benchmark 必须是 DataFrame、mapping 或 mapping 序列。") from exc
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

        Examples
        --------
        >>> report.write_excel("rules.xlsx")  # doctest: +SKIP
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_table = pd.DataFrame(
            [
                {"key": str(key), "value": json.dumps(value, ensure_ascii=False, default=str)}
                for key, value in self.metadata.items()
            ]
        )
        with pd.ExcelWriter(output_path, engine=engine) as writer:
            to_pandas_table(self.summary_table).to_excel(writer, sheet_name="summary", index=False)
            metadata_table.to_excel(writer, sheet_name="metadata", index=False)
            used_sheet_names = {"summary", "metadata"}
            for name, table in self.detail_tables.items():
                sheet_name: str = _safe_sheet_name(str(name), used_sheet_names)
                used_sheet_names.add(sheet_name)
                to_pandas_table(table).to_excel(writer, sheet_name=sheet_name, index=False)
            for name, table in export_semantics(self.describe()).items():
                sheet_name = _safe_sheet_name(name, used_sheet_names)
                used_sheet_names.add(sheet_name)
                table.to_excel(writer, sheet_name=sheet_name, index=False)

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

        Examples
        --------
        >>> report.write_html("rules.html")  # doctest: +SKIP
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

        Examples
        --------
        >>> MarsRuleReport().render_html().startswith("<!doctype html>")
        True
        """
        sections = [
            f"<h1>{html.escape(self.caption)}</h1>",
            "<h2>Summary</h2>",
            to_pandas_table(self.summary_table).to_html(index=False, escape=True),
            "<h2>Metadata</h2>",
            f"<pre>{html.escape(encode(self.describe()))}</pre>",
        ]
        for name, table in self.detail_tables.items():
            sections.append(f"<h2>{html.escape(str(name).replace('_', ' ').title())}</h2>")
            preview = table.head(100) if name == "candidates" else table
            sections.append(to_pandas_table(preview).to_html(index=False, escape=True))
            if preview.height < table.height:
                sections.append(
                    f"<p>候选预览 100 / {table.height} 行；完整证据使用 get_table('candidates')、Excel 或 .marsreport 导出。</p>"
                )
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
