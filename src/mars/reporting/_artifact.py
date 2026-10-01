"""可携带报告契约与单文件 JSON/Parquet 持久化。"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import zipfile
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, NoReturn, Protocol, cast, runtime_checkable

import pandas as pd
import polars as pl

from ._metadata import export_semantics
from ._query import ReportFrame, _ReportQuery
from ._serialization import encode, json_safe

pa = importlib.import_module("pyarrow")
pq = importlib.import_module("pyarrow.parquet")


def _invalid_json_constant(value: str) -> NoReturn:
    """拒绝 JSON 标准之外的非有限字面量，文件只能使用显式浮点标记。"""
    raise ValueError(f"Invalid marsreport JSON constant: {value!r}.")


def _copy_frame(frame: ReportFrame) -> ReportFrame:
    """隔离原生容器；Pandas object 列内的嵌套值也不共享可变对象。"""
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    result = frame.copy(deep=True)
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(deepcopy)
    return result


def _validate_description(description: dict[str, Any]) -> None:
    """校验可携带契约的最小完整性；未知元数据对象明确拒绝。"""
    required = {
        "report_id",
        "report_type",
        "format_version",
        "tables",
        "parameters",
        "feature_metadata",
        "business_context",
        "source",
    }
    if not isinstance(description, dict) or required - set(description):
        raise ValueError("Report description is missing required contract fields.")
    if type(description["format_version"]) is not int or description["format_version"] != 1:
        raise ValueError("Unsupported report format_version (supported: 1).")
    if not isinstance(description["report_id"], str) or not description["report_id"]:
        raise ValueError("Report report_id must be a non-empty persistent identifier.")
    if not isinstance(description["report_type"], str) or any(
        not isinstance(description[key], dict)
        for key in ("parameters", "feature_metadata", "business_context", "source")
    ):
        raise ValueError("Report type must be a string and metadata sections must be dictionaries.")
    if not isinstance(description["tables"], dict) or any(
        not isinstance(name, str)
        or not isinstance(entry, dict)
        or not {"rows", "grain", "fields"}.issubset(entry)
        for name, entry in description["tables"].items()
    ):
        raise ValueError("Report tables require names, rows, grain and field definitions.")
    json_safe(description)


@runtime_checkable
class Report(Protocol):
    """外部消费者共享的结构化报告契约；不要求具体分析器或报告类型。

    Attributes
    ----------
    report_id : str
        持久产物标识，区别于 Agent 会话登记句柄。
    format_version : int
        当前文件格式版本。
    report_type : str
        分析报告类型。

    Examples
    --------
    >>> isinstance(load_report("analysis.marsreport"), Report)  # doctest: +SKIP
    True
    """

    report_id: str
    format_version: int
    report_type: str

    def describe(self) -> dict[str, Any]:
        """返回身份、目录、参数、定义、业务上下文和诊断。"""
        ...

    def get_table(
        self,
        name: str,
        *,
        features: str | list[str] | None = None,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        sort_by: str | list[str] | None = None,
        descending: bool = False,
        offset: int = 0,
        limit: int | None = None,
        sources: str | list[str] | None = None,
    ) -> ReportFrame:
        """通过公开参数读取独立原生表。"""
        ...

    def get_feature(self, feature: str, *, limit: int = 100) -> dict[str, Any]:
        """取得单特征元信息和证据表。"""
        ...

    def search_features(
        self, query: str = "", *, sources: str | list[str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """检索候选并返回稳定英文标识。"""
        ...

    def query_page(self, name: str, **query: Any) -> dict[str, Any]:
        """取得分页计数与持久引用。"""
        ...

    def to_ai_context(
        self,
        *,
        tables: list[str] | None = None,
        features: str | list[str] | None = None,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        max_chars: int = 16000,
        sources: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        descending: bool = False,
        offset: int = 0,
        queries: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """输出字符预算内的有效 JSON 摘要。"""
        ...

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """保存完整分析产物。"""
        ...


class ReportSnapshot(_ReportQuery):
    """无需原始数据或会话的公共报告快照。

    Parameters
    ----------
    tables : dict[str, ReportFrame]
        表名到原生统计表的映射；构造时复制容器。
    description : dict[str, Any]
        Report.describe 返回的完整目录和元信息。

    Raises
    ------
    ValueError
        描述不符合契约或表目录与传入统计表不一致。

    Notes
    -----
    恢复仅还原统计事实，不重建分箱器或执行计算。查询保留原表后端。

    Examples
    --------
    >>> restored = load_report("analysis.marsreport")  # doctest: +SKIP
    >>> restored.get_table("summary", limit=10)  # doctest: +SKIP
    """

    def __init__(self, tables: dict[str, ReportFrame], description: dict[str, Any]) -> None:
        _validate_description(description)
        if set(tables) != set(description["tables"]):
            raise ValueError("ReportSnapshot tables differ from the public directory.")
        self._tables = {name: _copy_frame(table) for name, table in tables.items()}
        self._description: dict[str, Any] = json_safe(description)
        self.report_id = description["report_id"]
        self.format_version = description["format_version"]
        self.report_type = description["report_type"]
        self.feature_metadata = deepcopy(description.get("feature_metadata", {}))
        self.business_context = deepcopy(description.get("business_context", {}))
        self.source = deepcopy(description.get("source", {}))
        self.report_meta: dict[str, Any] = deepcopy(description.get("parameters", {}))
        self.feature_data_source = {
            f: m.get("data_source") or "UNMAPPED" for f, m in self.feature_metadata.items()
        }

    def _query_tables(self) -> dict[str, ReportFrame]:
        """恢复全部原生统计表，不访问分析器。"""
        return self._tables

    def _query_metadata(self) -> dict[str, Any]:
        """保留原计算配置与诊断。"""
        return self.report_meta

    def describe(self) -> dict[str, Any]:
        """返回保存时的完整语义目录副本。

        Returns
        -------
        dict[str, Any]
            保留原始类型名、持久标识、单位和诊断的说明。

        Examples
        --------
        >>> restored.describe()["report_id"]  # doctest: +SKIP
        """
        description = deepcopy(self._description)
        description.update(
            feature_metadata=deepcopy(self.feature_metadata),
            business_context=deepcopy(self.business_context),
            parameters=deepcopy(self.report_meta),
            source=deepcopy(self.source),
        )
        return cast(Dict[str, Any], json_safe(description))

    def show_table(self, name: str, **query: Any) -> pd.DataFrame:
        """查询后附上可读名；英文标识仍可直接复制。

        Parameters
        ----------
        name : str
            公共表名。
        **query : Any
            get_table 查询参数。

        Returns
        -------
        pd.DataFrame
            附有显示名的独立小表；缺名回退英文名。

        Examples
        --------
        >>> restored.show_table("summary", limit=5)  # doctest: +SKIP
        """
        return self._display_frame(self.get_table(name, **query))

    def write_html(self, path: str, *, report_name: str = "MARS Report Snapshot") -> None:
        """导出包含全部公共表和业务元信息的 HTML。

        Parameters
        ----------
        path : str
            HTML 路径；父目录必须存在。
        report_name : str
            报告标题。

        Examples
        --------
        >>> restored.write_html("restored.html")  # doctest: +SKIP
        """
        from ._profile_html import write_profile_html

        write_profile_html(self, path=path, report_name=report_name)

    def write_excel(self, path: str) -> None:
        """导出所有公共表及集中的业务元信息工作表。

        Parameters
        ----------
        path : str
            Excel 路径；父目录必须存在。

        Examples
        --------
        >>> restored.write_excel("restored.xlsx")  # doctest: +SKIP
        """
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in export_semantics(self.describe()).items():
                frame.to_excel(writer, sheet_name=name, index=False)
            for index, name in enumerate(self._tables):
                self.show_table(name).to_excel(
                    writer, sheet_name=f"{index:03d}_{name}"[:31], index=False
                )


def snapshot_report(report: object) -> ReportSnapshot:
    """通过公共接口隔离完整统计表和业务元信息。

    Parameters
    ----------
    report : object
        满足公共契约的报告；允许外部实现。

    Returns
    -------
    ReportSnapshot
        独立快照，保留持久标识和外部来源。

    Raises
    ------
    ValueError
        对象不满足契约或目录不完整时抛出。

    Examples
    --------
    >>> isolated = snapshot_report(report)  # doctest: +SKIP
    """
    if not isinstance(report, Report):
        raise ValueError("report must implement the public Report contract.")
    description = report.describe()
    _validate_description(description)
    if description.get("report_id") != report.report_id or description.get("format_version") != 1:
        raise ValueError("Report identity or format_version is invalid.")
    return ReportSnapshot(
        {name: report.get_table(name) for name in description["tables"]}, description
    )


def _pandas_arrow(frame: pd.DataFrame) -> Any:
    """保留索引和类别；浮点 NaN 不转换为 Arrow null。"""
    table = pa.Table.from_pandas(frame, preserve_index=True)
    for i, column in enumerate(frame.columns):
        series = frame[column]
        if pa.types.is_floating(table.schema.field(i).type):
            array = pa.array(series.array, type=table.schema.field(i).type, from_pandas=False)
            table = table.set_column(i, table.schema.field(i), array)
    return table


def save_report(report: Report, path: str | Path, *, overwrite: bool = False) -> None:
    """逐表写 ZIP；仅在全部完成后原子安装目标文件。"""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {destination}")
    description = report.describe()
    _validate_description(description)
    if description.get("format_version") != 1:
        raise ValueError("Unsupported report format_version; expected 1.")
    # 同目录临时文件保证替换不跨文件系统；失败时清理临时产物。
    handle, temporary = tempfile.mkstemp(
        prefix=".marsreport-", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        manifest: dict[str, Any] = {
            "format": "marsreport",
            "format_version": 1,
            "description": description,
            "tables": {},
            "normalization": {
                "pandas_index": "Arrow pandas metadata, preserve_index=True",
                "float": "NaN/Infinity preserved in Parquet; pre-existing Pandas float missing/NaN ambiguity retained",
                "date": "Parquet native date/datetime",
                "category": "Parquet dictionary and backend metadata",
            },
        }
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for i, name in enumerate(description["tables"]):
                frame = report.get_table(name)
                if any(not isinstance(c, str) for c in frame.columns) or len(
                    set(frame.columns)
                ) != len(frame.columns):
                    raise ValueError(f"Table {name!r} requires unique string column names.")
                filename = f"tables/{i:04d}.parquet"
                with archive.open(filename, "w", force_zip64=True) as stream:
                    if isinstance(frame, pl.DataFrame):
                        frame.write_parquet(stream)
                    else:
                        pq.write_table(_pandas_arrow(frame), stream)
                manifest["tables"][name] = {
                    "path": filename,
                    "backend": "polars" if isinstance(frame, pl.DataFrame) else "pandas",
                    "backend_version": pl.__version__
                    if isinstance(frame, pl.DataFrame)
                    else pd.__version__,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "schema": {
                        str(c): str(t)
                        for c, t in (
                            frame.schema.items()
                            if isinstance(frame, pl.DataFrame)
                            else frame.dtypes.items()
                        )
                    },
                }
            archive.writestr("manifest.json", encode(manifest))
        if overwrite:
            os.replace(temporary, destination)
        else:
            # 硬链接实现无覆盖安装；并发创建同名文件也不会被覆盖。
            os.link(temporary, destination)
    except (pa.ArrowException, pl.exceptions.PolarsError, TypeError) as exc:
        raise ValueError(f"Cannot encode report tables: {exc}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_report(path: str | Path) -> ReportSnapshot:
    """从单个 marsreport 文件恢复完整公共报告，无需原始宽表。

    Parameters
    ----------
    path : str | Path
        含 manifest.json 和 Parquet 表的 ZIP 文件。

    Returns
    -------
    ReportSnapshot
        保留表类型、schema、顺序、索引及持久身份的报告。

    Raises
    ------
    ValueError
        格式、清单、表或版本无效时抛出。

    Notes
    -----
    文件访问错误传播 OSError；父目录和路径由调用者管理。

    Examples
    --------
    >>> restored = load_report("analysis.marsreport")  # doctest: +SKIP
    >>> restored.get_feature("income")  # doctest: +SKIP
    """
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(
                archive.read("manifest.json"), parse_constant=_invalid_json_constant
            )
            if (
                manifest.get("format") != "marsreport"
                or type(manifest.get("format_version")) is not int
                or manifest.get("format_version") != 1
            ):
                raise ValueError(
                    "Invalid marsreport format or unsupported format_version (supported: 1)."
                )
            description = manifest["description"]
            _validate_description(description)
            if description["format_version"] != 1 or set(manifest["tables"]) != set(
                description["tables"]
            ):
                raise ValueError("Invalid report version or missing table declarations.")
            tables: dict[str, ReportFrame] = {}
            for name, entry in manifest["tables"].items():
                data = BytesIO(archive.read(entry["path"]))
                if entry["backend"] == "polars":
                    frame = pl.read_parquet(data)
                elif entry["backend"] == "pandas":
                    frame = pq.read_table(data).to_pandas()
                else:
                    raise ValueError(f"Unsupported table backend: {entry['backend']!r}.")
                schema = {
                    str(c): str(t)
                    for c, t in (
                        frame.schema.items()
                        if isinstance(frame, pl.DataFrame)
                        else frame.dtypes.items()
                    )
                }
                if (
                    len(frame) != entry["rows"]
                    or list(frame.columns) != entry["columns"]
                    or schema != entry["schema"]
                ):
                    raise ValueError(
                        f"Table {name!r} schema, row count or column order differs from manifest."
                    )
                tables[name] = frame
        return ReportSnapshot(tables, description)
    except (
        zipfile.BadZipFile,
        KeyError,
        json.JSONDecodeError,
        pa.ArrowException,
        pl.exceptions.PolarsError,
        TypeError,
        AttributeError,
    ) as exc:
        raise ValueError(f"Invalid or incomplete marsreport: {exc}") from exc
