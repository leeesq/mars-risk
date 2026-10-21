"""定义公共报告契约，并用单个 ZIP 文件保存 JSON 清单和 Parquet 统计表。"""

from __future__ import annotations

import ast
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

from ._excel import _preserve_openpyxl_strings
from ._metadata import export_semantics
from ._query import ReportFrame, _ReportQuery
from ._semantics import _normalize_state_definitions
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
    """供外部程序查询、引用和保存统计结果的公共报告契约。

    报告按表名提供已计算结果，无需原始数据或特定分析器。
    保存后的持久身份与查询参数可用于跨进程回放证据。

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
        """返回持久身份、表目录、字段定义、实际参数、业务上下文和诊断。"""
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
        """筛选、排序、选列并分页，返回与原表后端一致的独立容器。"""
        ...

    def get_feature(self, feature: str, *, limit: int = 100) -> dict[str, Any]:
        """读取单特征元数据和关联表；limit=0 保留空表结构与省略计数。"""
        ...

    def search_features(
        self, query: str = "", *, sources: str | list[str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """按文本和来源查找特征，返回英文 ID 及业务元数据，默认最多 20 项。"""
        ...

    def query_page(self, name: str, **query: Any) -> dict[str, Any]:
        """取得分页计数与持久引用；limit 默认 10，只接受正整数或 None。"""
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
        """按 Unicode 字符预算生成 JSON 证据；零行查询保留可回放引用。"""
        ...

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """保存完整统计表、语义目录和持久身份，默认拒绝覆盖已有文件。"""
        ...


class ReportSnapshot(_ReportQuery):
    """保存公共统计表及其解释信息的独立报告快照。

    可查询、分页、生成 AI 上下文和导出文件，无需原始样本或分析会话。

    Parameters
    ----------
    tables : dict[str, ReportFrame]
        公共表名到 Pandas/Polars 统计表的映射，构造时复制容器及 Pandas object 列中的可变值。
    description : dict[str, Any]
        ``Report.describe()`` 返回的完整目录和元数据，必须与 ``tables`` 的表名一致。

    Raises
    ------
    ValueError
        描述不符合契约或表目录与传入统计表不一致。

    Notes
    -----
    快照只恢复已保存的统计，不重建分箱器、模型或拟合状态，也不重算新样本。
    查询保留原表后端；画像和分箱的历史表会在内存中补齐身份列。

    Examples
    --------
    >>> restored = load_report("analysis.marsreport")  # doctest: +SKIP
    >>> restored.get_table("summary", limit=10)  # doctest: +SKIP
    """

    def __init__(self, tables: dict[str, ReportFrame], description: dict[str, Any]) -> None:
        _validate_description(description)
        if set(tables) != set(description["tables"]):
            raise ValueError("ReportSnapshot tables differ from the public directory.")
        for name, entry in description["tables"].items():
            relation = entry.get("feature_relation")
            if relation is None:
                continue
            if (
                not isinstance(relation, dict)
                or set(relation) != {"table", "key", "feature", "roles"}
                or any(not isinstance(relation[c], str) for c in ("table", "key", "feature"))
                or relation["table"] not in tables
                or not isinstance(relation["roles"], dict)
                or not relation["roles"]
                or any(
                    not isinstance(c, str) or not isinstance(role, str)
                    for c, role in relation["roles"].items()
                )
                or any(c not in tables[name].columns for c in relation["roles"])
                or any(
                    relation[c] not in tables[relation["table"]].columns for c in ("key", "feature")
                )
            ):
                raise ValueError(f"Invalid feature_relation for table {name!r}.")
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
        # load_report 已按磁盘 manifest 严格校验，之后仅在独立内存容器中补齐身份列。
        self._normalize_feature_tables()

    def _query_tables(self) -> dict[str, ReportFrame]:
        """读取快照中的全部原生统计表，不访问分析器。"""
        return self._tables

    def _query_metadata(self) -> dict[str, Any]:
        """保留原计算配置与诊断。"""
        return self.report_meta

    def describe(self) -> dict[str, Any]:
        """返回保存时的完整语义目录副本。

        Returns
        -------
        dict[str, Any]
            包含原报告类型、持久身份、表目录、单位、实际参数、上下文和诊断的独立说明。

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
        _normalize_state_definitions(description)
        return cast(Dict[str, Any], json_safe(description))

    def show_table(self, name: str, **query: Any) -> pd.DataFrame:
        """查询并转换为 Pandas 表，缺少显示名时使用英文 ID。

        Parameters
        ----------
        name : str
            ``describe()["tables"]`` 中的公共表名。
        **query : Any
            传给 ``get_table`` 的筛选、排序、选列和分页参数，默认返回整表。
            显式指定 ``columns`` 时只返回所列字段，不额外添加显示名列。

        Returns
        -------
        pd.DataFrame
            独立 Pandas 表。已有显示名列中的空值回退为对应英文 ID，不修改统计值。

        Examples
        --------
        >>> restored.show_table("summary", limit=5)  # doctest: +SKIP
        """
        return self._display_frame(self.get_table(name, **query))

    def write_html(self, path: str, *, report_name: str = "MARS Report Snapshot") -> None:
        """将全部已保存统计与业务元数据导出为离线 HTML。

        Parameters
        ----------
        path : str
            HTML 路径；父目录必须存在。
        report_name : str
            报告标题，默认使用 ``"MARS Report Snapshot"``。

        Notes
        -----
        模型分交叉快照使用专用交互页面；其他快照使用通用表格页面。
        导出不重新计算统计，也不恢复原报告的所有展示方法。

        Examples
        --------
        >>> restored.write_html("restored.html")  # doctest: +SKIP
        """
        if self.report_type == "score_cross":
            from mars.analysis.score_cross_view import write_score_cross_html

            write_score_cross_html(self, path, report_name=report_name)
            return
        from ._profile_html import write_profile_html

        write_profile_html(self, path=path, report_name=report_name)

    def write_excel(self, path: str) -> None:
        """将所有公共表的当前值与业务元数据写入静态 Excel 工作簿。

        Parameters
        ----------
        path : str
            Excel 路径；父目录必须存在。

        Notes
        -----
        每张公共表对应一个工作表，统计值沿用快照语义。
        不使用分箱报告的原生透视模板，也不依赖 Excel 刷新透视缓存。

        Examples
        --------
        >>> restored.write_excel("restored.xlsx")  # doctest: +SKIP
        """
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in export_semantics(self.describe()).items():
                frame.to_excel(writer, sheet_name=name, index=False)
                _preserve_openpyxl_strings(writer.sheets[name], frame)
            for index, name in enumerate(self._tables):
                frame = self.show_table(name)
                sheet_name = f"{index:03d}_{name}"[:31]
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                _preserve_openpyxl_strings(writer.sheets[sheet_name], frame)


def snapshot_report(report: object) -> ReportSnapshot:
    """通过公共接口复制完整统计表、语义目录和业务元数据。

    Parameters
    ----------
    report : object
        实现 ``Report`` 契约的报告，也接受外部实现；无需原始样本。

    Returns
    -------
    ReportSnapshot
        与原对象隔离的快照，保留报告类型、持久身份及来源，不重新计算指标。

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


def _decode_enum_categories(encoded: bytes) -> list[str]:
    """解析 Polars Enum 元数据的 UTF-8 字节长度前缀，保留类别顺序和空字符串。"""
    categories: list[str] = []
    position = 0
    while position < len(encoded):
        separator = encoded.find(b";", position)
        length_bytes = encoded[position:separator]
        if separator < 0 or not length_bytes.isdigit():
            raise ValueError("Invalid Polars Enum category metadata.")
        end = separator + 1 + int(length_bytes)
        if end > len(encoded):
            raise ValueError("Truncated Polars Enum category metadata.")
        categories.append(encoded[separator + 1:end].decode("utf-8"))
        position = end
    return categories


def _restore_polars_enums(
    frame: pl.DataFrame, data: BytesIO, declared_schema: dict[str, str],
) -> pl.DataFrame:
    """按真实 Parquet Enum 标记恢复旧 Polars 退化类型，随后仍严格核对完整 schema。"""
    categorical = [
        column for column, dtype in frame.schema.items() if isinstance(dtype, pl.Categorical)
    ]
    if not categorical:
        return frame
    data.seek(0)
    arrow_schema = pq.read_schema(data)
    casts: list[pl.Expr] = []
    for column in categorical:
        metadata = arrow_schema.field(column).metadata or {}
        if b"_PL_ENUM_VALUES2" in metadata:
            categories = _decode_enum_categories(metadata[b"_PL_ENUM_VALUES2"])
        elif metadata.get(b"POLARS.CATEGORICAL_TYPE") == b"ENUM":
            # 旧 Parquet 非空字典保留未使用类别；零行文件没有 row group 或字典值。
            if frame.height:
                data.seek(0)
                dictionaries = pq.read_table(data, columns=[column]).column(0).chunks
                categories = dictionaries[0].dictionary.to_pylist()
                if any(chunk.dictionary.to_pylist() != categories for chunk in dictionaries):
                    raise ValueError(f"Inconsistent Parquet Enum dictionaries: {column!r}.")
            else:
                declared = declared_schema.get(column, "")
                prefix = "Enum(categories="
                if not declared.startswith(prefix) or not declared.endswith(")"):
                    continue
                # 格式 1 的旧零行文件只在 manifest 保存完整类别；literal_eval 不执行代码。
                try:
                    categories = ast.literal_eval(declared[len(prefix):-1])
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Invalid saved Enum categories: {column!r}.") from exc
        else:
            continue
        if (
            not isinstance(categories, list)
            or any(not isinstance(value, str) for value in categories)
            or len(categories) != len(set(categories))
        ):
            raise ValueError(f"Invalid saved Enum categories: {column!r}.")
        casts.append(pl.col(column).cast(pl.Enum(categories)))
    return frame.with_columns(casts) if casts else frame


def save_report(report: Report, path: str | Path, *, overwrite: bool = False) -> None:
    """逐表写入同目录临时 ZIP，全部完成后再原子写入目标路径。"""
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
    """从单个 .marsreport 文件恢复已保存的公共统计表和语义目录。

    Parameters
    ----------
    path : str | Path
        包含 ``manifest.json`` 和 Parquet 表的 ZIP 文件，当前仅支持格式版本 1。

    Returns
    -------
    ReportSnapshot
        保留原报告类型、表后端、schema、顺序、索引及持久身份的快照。
        无需原始宽表、分箱器或分析会话，也不重新计算统计。

    Raises
    ------
    ValueError
        格式、清单、表或版本无效时抛出。

    Notes
    -----
    加载时校验清单、表 schema、行数和列顺序，不支持跨格式版本迁移。
    文件访问错误传播 OSError；父目录和路径由调用方管理。

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
                    frame = _restore_polars_enums(frame, data, entry["schema"])
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
