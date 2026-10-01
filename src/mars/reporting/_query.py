"""报告结果查询与有预算的 AI 上下文；只消费已有表，不执行指标计算。"""

from __future__ import annotations

import operator
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Union, cast
from uuid import uuid4

import pandas as pd
import polars as pl

from mars._compat import polars_is_in
from mars.reporting._metadata import FeatureMetadata, normalize_business_context, normalize_metadata
from mars.reporting._semantics import _definition
from mars.reporting._serialization import JSON_RULES, table_rows
from mars.reporting._serialization import encode as _encode
from mars.reporting._serialization import json_safe as _json_safe

ReportFrame = Union[pl.DataFrame, pd.DataFrame]


def _filter_value(frame: ReportFrame, column: str, value: Any) -> Any:
    """按真实字段类型解析 ISO 日期引用；普通字符串列不做转换。"""
    if isinstance(value, list):
        return [_filter_value(frame, column, item) for item in value]
    if not isinstance(value, (str, date, datetime)):
        return value
    dtype = frame.schema[column] if isinstance(frame, pl.DataFrame) else frame[column].dtype
    try:
        if isinstance(frame, pl.DataFrame):
            if dtype == pl.Date and isinstance(value, str):
                return date.fromisoformat(value)
            if isinstance(dtype, pl.Datetime) and isinstance(value, str):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"Filter requires an ISO-8601 value for {column!r}.") from exc
    return value


def _names(value: str | list[str] | None, parameter: str) -> list[str] | None:
    """校验字符串或无重复字符串列表。"""
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or any(not isinstance(v, str) for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{parameter} must be a string or a list of unique strings.")
    return values


def query_table(
    frame: ReportFrame,
    *,
    features: str | list[str] | None = None,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    sort_by: str | list[str] | None = None,
    descending: bool = False,
    offset: int = 0,
    limit: int | None = None,
    _copy_result: bool = True,
) -> ReportFrame:
    """在原生表上执行受校验的筛选、排序、投影和分页，并返回独立容器。"""
    if (
        type(offset) is not int
        or offset < 0
        or (limit is not None and (type(limit) is not int or limit < 0))
    ):
        raise ValueError("offset and limit must be non-negative integers (limit may be None).")
    if type(descending) is not bool:
        raise ValueError("descending must be a bool.")
    selected = _names(features, "features")
    projected = _names(columns, "columns")
    ordering = _names(sort_by, "sort_by")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("filters must be a dictionary of column conditions.")
    conditions = list((filters or {}).items())
    if selected is not None:
        conditions.append(("feature", {"op": "in", "value": selected}))
    required = [*(projected or []), *(ordering or []), *[c for c, _ in conditions]]
    invalid = [c for c in required if c not in frame.columns]
    if invalid:
        raise ValueError(f"Unknown columns: {invalid}. Available: {list(frame.columns)}")
    result = frame
    comparisons = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }
    for column, condition in conditions:
        op, value = "eq", condition
        if isinstance(condition, dict):
            if set(condition) not in [{"op", "value"}, {"op"}]:
                raise ValueError(f"Invalid filter for {column}: use op and value.")
            op, value = condition["op"], condition.get("value")
            if not isinstance(op, str):
                raise ValueError("Filter op must be a string.")
            if op not in {"is_null", "is_not_null"} and "value" not in condition:
                raise ValueError(f"Filter {op!r} requires value.")
        if not isinstance(op, str) or op not in {
            *comparisons,
            "in",
            "not_in",
            "is_null",
            "is_not_null",
        }:
            raise ValueError(f"Unsupported filter operator: {op!r}.")
        value = _filter_value(result, column, value)
        series = pl.col(column) if isinstance(result, pl.DataFrame) else result[column]
        if op in {"in", "not_in"}:
            if not isinstance(value, list) or any(isinstance(v, (dict, list)) for v in value):
                raise ValueError(f"Filter {op} requires a scalar list.")
            mask = (
                polars_is_in(series, pl.Series(value))
                if isinstance(result, pl.DataFrame)
                else series.isin(value)
            )
            if op == "not_in":
                mask = ~mask
        elif op in {"is_null", "is_not_null"} or value is None:
            if op not in {"eq", "ne", "is_null", "is_not_null"}:
                raise ValueError("None only supports eq, ne and null checks.")
            mask = series.is_null() if isinstance(result, pl.DataFrame) else series.isna()
            if op in {"ne", "is_not_null"}:
                mask = ~mask
        else:
            if not isinstance(value, (str, int, float, bool, date, datetime)):
                raise ValueError("Filter value must be a scalar.")
            try:
                mask = comparisons[op](series, value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Filter is incompatible with column {column!r}.") from exc
        try:
            result = result.filter(mask) if isinstance(result, pl.DataFrame) else result.loc[mask]
        except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            raise ValueError(f"Filter is incompatible with column {column!r}.") from exc
    if ordering:
        result = (
            result.sort(ordering, descending=descending, nulls_last=True, maintain_order=True)
            if isinstance(result, pl.DataFrame)
            else result.sort_values(ordering, ascending=not descending, kind="stable")
        )
    if projected is not None and not projected:
        raise ValueError("columns must not be empty.")
    if isinstance(result, pl.DataFrame):
        page = result.slice(offset, limit)
        return (page.select(projected) if projected is not None else page).clone()
    page = result.iloc[offset : None if limit is None else offset + limit]
    if projected is not None:
        page = page.loc[:, projected]
    return page.copy() if _copy_result else page


class _ReportQuery:
    """两个领域报告共享的轻量结果查询，不持有原始数据。"""

    def _initialize_semantics(
        self,
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
        legacy: dict[str, str] | None = None,
    ) -> None:
        """集中保存业务定义；稳定英文标识不进入统计计算。"""
        features: list[str] = []
        inferred: dict[str, str] = dict(legacy or {})
        for table in self._query_tables().values():
            if "feature" not in table.columns:
                continue
            values = (
                table["feature"].unique(maintain_order=True).to_list()
                if isinstance(table, pl.DataFrame)
                else table["feature"].unique().tolist()
            )
            features.extend(f for f in values if isinstance(f, str) and f not in features)
            if "data_source" in table.columns:
                rows = (
                    table.select("feature", "data_source").unique().to_dicts()
                    if isinstance(table, pl.DataFrame)
                    else table[["feature", "data_source"]].drop_duplicates().to_dict("records")
                )
                for row in rows:
                    feature, source = row["feature"], row["data_source"]
                    if isinstance(source, str) and source != "UNMAPPED":
                        if inferred.get(feature) not in (None, source):
                            raise ValueError(f"Conflicting data_source for feature {feature!r}.")
                        inferred[feature] = source
        self.feature_metadata = normalize_metadata(
            feature_metadata, features, inferred, legacy_direction="feature_to_source"
        )
        self.feature_data_source = {
            f: m.get("data_source") or "UNMAPPED" for f, m in self.feature_metadata.items()
        }
        self.business_context = {
            "label_definition": "unknown",
            "observation_window": "unknown",
            "currency": "unknown",
            **normalize_business_context(business_context),
        }
        self.business_context_source = {
            key: "user_provided" if key in (business_context or {}) else "unknown"
            for key in self.business_context
        }
        self.report_id = str(uuid4())
        self.format_version = 1
        self.source: dict[str, Any] = {"kind": "provided_statistics", "producer": "unknown"}
        self.report_type = type(self).__name__

    def search_features(
        self,
        query: str = "",
        *,
        sources: str | list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按英文标识、显示名、定义和来源检索，重复中文名返回所有候选。

        Parameters
        ----------
        query : str
            不区分大小写的包含查询；空字符串匹配全部。
        sources : str | list[str] | None
            来源筛选，语义同 get_table。
        limit : int
            非负返回数量上限。

        Returns
        -------
        list[dict[str, Any]]
            明确包含 feature 和原始业务元数据的候选列表。

        Raises
        ------
        ValueError
            查询或数量无效时抛出。

        Examples
        --------
        >>> report.search_features("月收入")  # doctest: +SKIP
        """
        if not isinstance(query, str) or type(limit) is not int or limit < 0:
            raise ValueError("query must be a string and limit a non-negative integer.")
        allowed = self._source_features(sources)
        needle = query.casefold()
        return [
            {"feature": f, **deepcopy(m)}
            for f, m in self.feature_metadata.items()
            if (allowed is None or f in allowed)
            and any(needle in str(v).casefold() for v in [f, *m.values()] if v is not None)
        ][:limit]

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """逐表保存完整报告为单个 JSON/Parquet ZIP 产物。

        Parameters
        ----------
        path : str | Path
            目标文件；父目录必须存在。
        overwrite : bool
            默认拒绝覆盖；True 原子替换已存在文件。

        Notes
        -----
        编码失败传播 ValueError；拒绝覆盖传播 FileExistsError；I/O 失败传播 OSError。
        失败不会留下可误认为完整报告的半成品。

        Examples
        --------
        >>> report.save("analysis.marsreport")  # doctest: +SKIP
        """
        from ._artifact import save_report

        save_report(self, path, overwrite=overwrite)

    def _display_frame(self, frame: ReportFrame) -> pd.DataFrame:
        """仅对已缩小的展示表附加可读名称，不污染原始统计行。"""
        from mars.compute import to_pandas_frame

        result = to_pandas_frame(frame).copy()
        if "feature" in result.columns and any(
            m.get("display_name") for m in self.feature_metadata.values()
        ):
            result.insert(
                1,
                "display_name",
                result["feature"].map(
                    lambda f: self.feature_metadata.get(f, {}).get("display_name") or f
                ),
            )
        return result

    def _query_metadata(self) -> dict[str, Any]:
        """读取两类报告各自保存的元数据。"""
        raise NotImplementedError

    def _query_tables(self) -> dict[str, ReportFrame]:
        """由具体报告提供规范表名。"""
        raise NotImplementedError

    def _source_features(self, sources: str | list[str] | None) -> list[str] | None:
        """仅消费已有来源信息，不推断特征来源。"""
        names = _names(sources, "sources")
        if names is None:
            return None
        source_map: dict[str, str] = {
            f: m.get("data_source") or "UNMAPPED" for f, m in self.feature_metadata.items()
        }
        if not source_map:
            raise ValueError("Feature source information is unknown in this report.")
        unknown = set(names) - set(source_map.values())
        if unknown:
            raise ValueError(f"Unknown feature sources: {sorted(unknown)}.")
        return [feature for feature, source in source_map.items() if source in names]

    def describe(self) -> dict[str, Any]:
        """返回报告目录、字段口径、实际参数和限制，不重新计算指标。

        Returns
        -------
        dict[str, Any]
            结构化说明；未登记的业务单位、标签定义和观察窗口标为 unknown。

        Examples
        --------
        >>> report.describe()["tables"]  # doctest: +SKIP
        """
        tables: dict[str, Any] = {}
        for name, frame in self._query_tables().items():
            schema = frame.schema if isinstance(frame, pl.DataFrame) else frame.dtypes.to_dict()
            metric = name.split(".", 1)[1] if "." in name else None
            if name == "missing_by_day":
                metric = "missing"
            tables[name] = {
                "rows": len(frame),
                "index": {
                    "kind": type(frame.index).__name__,
                    "names": list(frame.index.names),
                    "queryable": False,
                }
                if isinstance(frame, pd.DataFrame)
                else None,
                "grain": "target/feature/group/bin"
                if name == "detail"
                else (
                    "feature (days in columns)"
                    if name == "missing_by_day"
                    else "target/feature/bin/reference"
                    if name == "risk_corr_reference"
                    else "feature (groups in columns)"
                ),
                "fields": {
                    str(c): {
                        "dtype": str(dtype),
                        **_definition(
                            metric
                            if metric and c not in {"feature", "dtype", "data_source"}
                            else str(c)
                        ),
                    }
                    for c, dtype in schema.items()
                },
            }
            for column, field in tables[name]["fields"].items():
                if name in {
                    "detail",
                    "risk_corr_reference",
                } and column == self._query_metadata().get("group_col"):
                    field.update(unit="dimension", meaning="实际计算分组字段；取值保留原始分组标识")
                if field["unit"] == "count_or_weight":
                    field["unit"] = (
                        "weight_sum" if self._query_metadata().get("weights_col") else "count"
                    )
                if (
                    column in {"avg_amt", "tot_amt", "observed_amt", "bad_amt", "good_amt"}
                    and self.business_context.get("currency", "unknown") != "unknown"
                ):
                    field["business_currency"] = self.business_context["currency"]
            if name == "calculation_status":
                tables[name]["grain"] = "target/feature/group/metric"
            if name == "summary":
                tables[name]["grain"] = "target/feature" if "target" in frame.columns else "feature"
            if name == "overview" or name == "comparison.schema":
                tables[name]["grain"] = "feature"
            if name.startswith("trend."):
                targets = self._query_metadata().get("targets", [])
                tables[name]["target"] = targets[0] if targets else "unknown"
            if name == "missing_by_day":
                for column, field in tables[name]["fields"].items():
                    if column not in {"feature", "dtype"}:
                        field["meaning"] = (
                            "自然日原始缺失值数/当日样本数，含 Null、NaN 和配置缺失值；不使用样本权重，total 为全量比例"
                        )
            if self._query_metadata().get("ks_method") == "raw":
                for column, field in tables[name]["fields"].items():
                    if column == "ks" or name == "trend.ks" and column not in {"feature", "dtype"}:
                        field["meaning"] = (
                            "原始数值排序的经验 KS×100；类别回退分箱 KS，逐特征来源与失败见 ks_source_by_feature / raw_ks_diagnostics"
                        )
            if name == "trend.lift":
                for column, field in tables[name]["fields"].items():
                    if column not in {"feature", "dtype", "data_source"}:
                        field["meaning"] = "当前分组箱 Lift 的最大值；未表现分组为空"
            if name == "comparison.unseen":
                for column, field in tables[name]["fields"].items():
                    if column in {
                        "benchmark_unique_count",
                        "valid_count",
                        "unseen_count",
                        "unseen_unique_count",
                    }:
                        field.update(
                            unit="count",
                            meaning="有效类别样本或不同值计数；详细状态见 status / reason",
                        )
                    elif column in {
                        "status",
                        "reason",
                        "is_categorical",
                        "current_dtype",
                        "benchmark_dtype",
                    }:
                        field.update(
                            unit="unknown", meaning="计算状态、原因或字段类型，不是 unseen rate"
                        )
        context = deepcopy(self.business_context)
        labels = context.setdefault("labels", {})
        targets = self._query_metadata().get("targets") or [
            self._query_metadata().get("target_requested")
        ]
        for target in targets:
            if target:
                labels[target] = {
                    "definition": "unknown",
                    "positive_class": "unknown",
                    "negative_class": "unknown",
                    "performance_window": "unknown",
                    **labels.get(target, {}),
                }
        return cast(
            Dict[str, Any],
            _json_safe(
                {
                    "report_type": self.report_type,
                    "report_id": self.report_id,
                    "format_version": self.format_version,
                    "feature_metadata": deepcopy(self.feature_metadata),
                    "source": deepcopy(self.source),
                    "context_source": {
                        "business_context": deepcopy(self.business_context_source),
                        "parameters": "calculation",
                    },
                    "serialization": deepcopy(JSON_RULES),
                    "tables": tables,
                    "calculation_state": {
                        "table": "calculation_status" if "calculation_status" in tables else None,
                        "diagnostics": "parameters.diagnostics/fit_failures/raw_ks_diagnostics",
                        "scope": "feature/target/group/metric; profile status contains exceptional outcomes only, not all trend cells",
                    },
                    "parameters": deepcopy(self._query_metadata()),
                    "business_context": context,
                    "limitations": [
                        "只组织已有统计，未计算表不在目录中；失败和跳过见 parameters.diagnostics / fit_failures。",
                        "Null 不是数值 0；风险指标需结合标签表现状态，样本分布使用全样本。",
                        "比较前核对样本范围、权重、缺失/特殊箱、拟合来源、排序口径及参考来源；关联不代表因果。",
                    ],
                }
            ),
        )

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
        """查询已有结果，返回可继续加工且与原表类型一致的 DataFrame。

        Parameters
        ----------
        name : str
            describe 目录中的规范表名。
        features : str | list[str] | None
            特征名称，None 为全部。
        columns : list[str] | None
            投影字段；在筛选、排序之后投影。
        filters : dict[str, Any] | None
            字段到标量相等条件或 {op, value}；支持 eq/ne/lt/le/gt/ge/in/not_in/is_null/is_not_null。
        sort_by : str | list[str] | None
            排序字段。
        descending : bool
            是否降序；空值置后。
        offset : int
            非负起始行号。
        limit : int | None
            非负最大行数；None 不限制。排序配合 limit 实现 Top-K。
        sources : str | list[str] | None
            已登记的特征来源名称；来源未知时明确报错。

        Returns
        -------
        ReportFrame
            查询结果的独立容器；不重新计算指标。

        Raises
        ------
        ValueError
            表名、字段、操作符、来源或分页参数无效时抛出。

        Examples
        --------
        >>> report.get_table("summary", sort_by="iv", descending=True, limit=10)  # doctest: +SKIP
        """
        tables = self._query_tables()
        if name not in tables:
            raise ValueError(f"Unknown table {name!r}. Available: {list(tables)}")
        allowed = self._source_features(sources)
        if allowed is not None:
            requested = _names(features, "features")
            features = allowed if requested is None else [f for f in requested if f in allowed]
        return query_table(
            tables[name],
            features=features,
            columns=columns,
            filters=filters,
            sort_by=sort_by,
            descending=descending,
            offset=offset,
            limit=limit,
        )

    def query_page(self, name: str, **query: Any) -> dict[str, Any]:
        """按 get_table 参数取得分页证据、总匹配数及持久引用。

        Parameters
        ----------
        name : str
            公共表目录中的表名。
        **query : Any
            get_table 的关键字参数；默认 offset=0、limit=10。

        Returns
        -------
        dict[str, Any]
            原生 data、计数、截断原因、next_offset 和可重放的 reference。

        Notes
        -----
        查询无效时传播 get_table 的 ValueError。

        Examples
        --------
        >>> page = report.query_page("summary", limit=5)  # doctest: +SKIP
        """
        options = {"offset": 0, "limit": 10, **query}
        page = self.get_table(name, **options)
        features = options.get("features")
        allowed = self._source_features(options.get("sources"))
        if allowed is not None:
            requested = _names(features, "features")
            features = allowed if requested is None else [f for f in requested if f in allowed]
        # 计数只筛选原生表；Pandas 未筛选整表时不生成副本。
        matched = query_table(
            self._query_tables()[name],
            features=features,
            filters=options.get("filters"),
            _copy_result=False,
        )
        total = len(matched)
        offset = options["offset"]
        remaining = max(total - offset - len(page), 0)
        return {
            "data": page,
            "total_rows": total,
            "returned_rows": len(page),
            "truncated": remaining > 0,
            "omitted_rows": remaining,
            "omission_reason": "pagination" if remaining else None,
            "next_offset": offset + len(page) if remaining else None,
            "reference": {"report_id": self.report_id, "table": name, "query": deepcopy(options)},
        }

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
        """生成问题相关的预算内摘要；完整报告仍是可分页的事实依据。

        Parameters
        ----------
        tables : list[str] | None
            表目录名称；默认第一张表，queries 给定时默认其键。
        features : str | list[str] | None
            原始英文标识。
        columns : list[str] | None
            共同列投影；宽趋势可按日期列选择范围。
        filters : dict[str, Any] | None
            get_table 的筛选条件。
        limit : int
            每表行数上限。
        max_chars : int
            最终 JSON 的 Unicode 字符预算，不等于 token 数。
        sources : str | list[str] | None
            来源筛选。
        sort_by : str | list[str] | None
            排序列。
        descending : bool
            是否降序。
        offset : int
            每表起始位置。
        queries : dict[str, dict[str, Any]] | None
            每表的 get_table 参数覆盖共同参数，支持异构表查询。

        Returns
        -------
        str
            标准 JSON；定义仅出现一次，省略记录包含数量、原因及继续查询定位。

        Raises
        ------
        ValueError
            查询无效或预算无法容纳最小必要说明时抛出。

        Examples
        --------
        >>> report.to_ai_context(queries={"summary": {"limit": 3}})  # doctest: +SKIP
        """
        if type(max_chars) is not int or max_chars < 512:
            raise ValueError("max_chars must be an integer >= 512.")
        available = self._query_tables()
        if queries is not None and (
            not isinstance(queries, dict)
            or any(name not in available or not isinstance(q, dict) for name, q in queries.items())
        ):
            raise ValueError("queries must map available table names to get_table options.")
        selected = _names(tables, "tables")
        selected = (
            list(queries)
            if selected is None and queries
            else list(available)[:1]
            if selected is None
            else selected
        )
        if not selected or any(name not in available for name in selected):
            raise ValueError(f"tables must select names from {list(available)}.")
        description = self.describe()
        table_definitions = description["tables"]
        description["tables"] = {}
        description["feature_metadata"] = {}
        payload: dict[str, Any] = {
            "description": description,
            "evidence": [],
            "omitted": [],
            "serialization": description.pop("serialization", deepcopy(JSON_RULES)),
            "budget": {"max_chars": max_chars, "unit": "unicode_characters"},
        }
        omitted = payload["omitted"]
        for name in selected:
            options: dict[str, Any] = dict(
                features=features,
                columns=columns,
                filters=filters,
                limit=limit,
                sources=sources,
                sort_by=sort_by,
                descending=descending,
                offset=offset,
            )
            options.update((queries or {}).get(name, {}))
            page = self.query_page(name, **options)
            frame = page["data"]
            entry = table_definitions[name]
            wide = name.startswith(("trend.", "dq.", "stats.")) or name == "missing_by_day"
            identifiers = {"feature", "dtype", "data_source", "target"}
            # 日期是维度，指标定义仅保存一次；证据仍使用原表列名以便精确重放。
            if wide:
                dimension_fields = [c for c in frame.columns if c not in identifiers]
                definition = (
                    entry["fields"].get(dimension_fields[0], {}) if dimension_fields else {}
                )
                entry = {
                    "rows": entry["rows"],
                    "grain": entry["grain"],
                    "target": entry.get("target", "unknown"),
                    "value_definition": definition,
                    "dimension": "column names are group/date identifiers",
                    "fields": {c: entry["fields"][c] for c in frame.columns if c in identifiers},
                }
            else:
                entry["fields"] = {c: entry["fields"][c] for c in frame.columns}
            description["tables"][name] = entry
            rows = table_rows(frame)
            item = {
                "reference": name,
                "report_id": self.report_id,
                "query": page["reference"]["query"],
                "total_rows": page["total_rows"],
                "returned_rows": len(rows),
                "rows": rows,
                "next_offset": page["next_offset"],
                "truncated": page["truncated"],
            }
            payload["evidence"].append(item)
            selected_features = (
                frame["feature"].to_list()
                if isinstance(frame, pl.DataFrame) and "feature" in frame.columns
                else frame["feature"].tolist()
                if "feature" in frame.columns
                else _names(options.get("features"), "features") or []
            )
            description["feature_metadata"].update(
                {f: deepcopy(self.feature_metadata.get(f, {})) for f in selected_features}
            )
            if page["omitted_rows"]:
                omitted.append(
                    {
                        "reference": name,
                        "report_id": self.report_id,
                        "rows": page["omitted_rows"],
                        "reason": "row_limit",
                        "next_offset": page["next_offset"],
                    }
                )
        # 大型说明块独立裁剪；每次省略完整参数值并保留 describe 定位。
        for key, value in list(description["parameters"].items()):
            if len(_encode(value)) > max_chars // 8:
                description["parameters"].pop(key)
                omitted.append(
                    {
                        "reference": f"describe.parameters.{key}",
                        "count": 1,
                        "reason": "description_budget",
                    }
                )

        for feature, entry in description["feature_metadata"].items():
            for key, value in list(entry.items()):
                if len(_encode(value)) > max_chars // 8:
                    entry.pop(key)
                    omitted.append(
                        {
                            "reference": f"describe.feature_metadata.{feature}.{key}",
                            "count": 1,
                            "reason": "description_budget",
                        }
                    )
        if len(_encode(description["business_context"])) > max_chars // 3:
            count = len(description["business_context"])
            description["business_context"] = {}
            omitted.append(
                {
                    "reference": "describe.business_context",
                    "count": count,
                    "reason": "description_budget",
                }
            )

        def marker(item: dict[str, Any], kind: str, locator: Any) -> dict[str, Any]:
            """累计预算省略计数，避免为每个日期或行添加对象。"""
            found = next(
                (
                    m
                    for m in omitted
                    if m.get("reference") == item["reference"] and m.get("kind") == kind
                ),
                None,
            )
            if found is None:
                found = {
                    "reference": item["reference"],
                    "report_id": self.report_id,
                    "kind": kind,
                    "reason": "output_budget",
                    "rows" if kind == "rows" else "columns": 0,
                    "continue_at": locator,
                }
                omitted.append(found)
            return found

        while len(_encode(payload)) > max_chars:
            candidates = [item for item in payload["evidence"] if item["rows"]]
            # 宽表先移除末尾完整时间段，保证单特征仍有有效证据。
            wide_items = [
                item
                for item in candidates
                if item["reference"].startswith(("trend.", "dq.", "stats."))
                or item["reference"] == "missing_by_day"
            ]
            wide_item = max(wide_items, key=lambda item: len(_encode(item["rows"])), default=None)
            dimensions = (
                [
                    c
                    for c in wide_item["rows"][0]
                    if c not in {"feature", "dtype", "data_source", "target"}
                ]
                if wide_item
                else []
            )
            if wide_item is not None and len(dimensions) > 1:
                column = dimensions[-1]
                m = marker(
                    wide_item, "columns", {"column": column, "query": "get_table(columns=...)"}
                )
                m["columns"] += 1
                m["continue_at"]["column"] = column
                for row in wide_item["rows"]:
                    row.pop(column)
                wide_item["truncated"] = True
                continue
            if candidates:
                item = max(candidates, key=lambda entry: len(_encode(entry["rows"])))
                item["rows"].pop()
                item["returned_rows"] -= 1
                item["next_offset"] = item["query"]["offset"] + item["returned_rows"]
                item["truncated"] = True
                m = marker(item, "rows", {"offset": item["next_offset"]})
                m["rows"] += 1
                m["continue_at"]["offset"] = item["next_offset"]
                continue
            # 无证据可裁剪后只能省略整个说明块；必需身份、目录、计数和引用保留。
            block = next(
                (
                    key
                    for key in ("parameters", "feature_metadata", "business_context", "limitations")
                    if description.get(key)
                ),
                None,
            )
            if block is None:
                raise ValueError(
                    "max_chars cannot contain minimum report description; increase budget."
                )
            count = len(description[block])
            description[block] = {} if block != "limitations" else []
            omitted.append(
                {"reference": f"describe.{block}", "count": count, "reason": "description_budget"}
            )
        return _encode(payload)

    def get_feature(self, feature: str, *, limit: int = 100) -> dict[str, Any]:
        """关联读取单特征概览、分箱和趋势，并明确缺失的数据类别。

        Parameters
        ----------
        feature : str
            单个特征名称。
        limit : int
            每表最大行数，默认 100；完整表可继续 get_table 分页。

        Returns
        -------
        dict[str, Any]
            tables、omitted_rows 和 unavailable；返回原生 DataFrame，不自动渲染。

        Raises
        ------
        ValueError
            特征不存在或分页参数无效时抛出。

        Examples
        --------
        >>> report.get_feature("age")["tables"]  # doctest: +SKIP
        """
        results: dict[str, ReportFrame] = {}
        omitted_rows: dict[str, int] = {}
        matched = False
        for name, table in self._query_tables().items():
            if "feature" in table.columns:
                page = self.query_page(name, features=feature, limit=limit)
                matched = matched or page["total_rows"] > 0
                results[name] = page["data"]
                omitted_rows[name] = page["omitted_rows"]
        if not matched:
            raise ValueError(f"Unknown feature: {feature!r}.")
        unavailable = [
            kind
            for kind in ["overview", "summary", "detail", "trend"]
            if not any(
                (
                    name == kind
                    or name.startswith(f"{kind}.")
                    or kind == "trend"
                    and name.startswith(("dq.", "stats."))
                )
                and len(table)
                for name, table in results.items()
            )
        ]
        return {
            "feature": feature,
            "report_id": self.report_id,
            "metadata": deepcopy(self.feature_metadata.get(feature, {})),
            "tables": results,
            "omitted_rows": omitted_rows,
            "unavailable": unavailable,
        }
