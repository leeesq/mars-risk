"""报告结果查询与有预算的 AI 上下文；只消费已有表，不执行指标计算。"""

from __future__ import annotations

import json
import math
import operator
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl

from mars._compat import polars_is_in

ReportFrame = Union[pl.DataFrame, pd.DataFrame]

# 定义来自 compute/binning、profiling/metrics；业务单位未登记时不能推断。
_DEFINITIONS: dict[str, tuple[str, str]] = {
    "missing": ("ratio", "缺失箱样本量/全样本量；画像包括 Null、NaN 和配置缺失值"),
    "zeros": ("ratio", "原始零值数/全样本数，非数值字段为 0"),
    "unique": ("ratio", "原始不同值数/全样本数，含缺失；overview 超百万行使用近似去重"),
    "mode": ("ratio", "原始最高频值数/全样本数，含缺失"),
    "pct": ("ratio", "箱样本量/分组全样本量，配置权重时为权重和之比"),
    "bad_rate": ("ratio", "bad/observed_count；未表现标签不计入分母"),
    "base_br": ("ratio", "RC 参考箱的 bad/observed_count"),
    "cum_bad_rate": ("ratio", "累计 bad/累计 observed_count"),
    "amt_bad_rate": ("ratio", "bad_amt/observed_amt；未表现金额不计入分母，分母非正时为空"),
    "ks": ("points_0_100", "排序后累计坏样本与好样本分布最大绝对差 ×100"),
    "ks_bin": ("points_0_100", "当前箱累计坏样本与好样本分布绝对差 ×100"),
    "auc": ("ratio", "按配置排序的分箱 ROC 梯形面积；报告归一到不小于 0.5"),
    "auc_bin": ("ratio", "分箱 ROC 梯形面积贡献；汇总行进行 AUC 归一"),
    "iv": ("dimensionless", "各箱 (bad_dist-good_dist)×WOE 之和，复用稳定化规则"),
    "iv_bin": ("dimensionless", "(bad_dist-good_dist)×WOE"),
    "woe": ("dimensionless", "稳定化的 ln(bad_dist/good_dist)"),
    "psi": ("dimensionless", "各箱 (actual-expected)×ln(actual/expected) 之和；参考及箱范围见参数"),
    "psi_bin": ("dimensionless", "当前箱 PSI 贡献；缺失箱和特殊箱由参数控制"),
    "lift": ("dimensionless", "箱坏率/总体坏率，复用稳定化规则"),
    "lift_min": ("dimensionless", "Total 正常箱 Lift 的最小值"),
    "lift_max": ("dimensionless", "Total 正常箱 Lift 的最大值"),
    "lift_amt": ("dimensionless", "箱金额坏率/总体金额坏率，复用稳定化规则"),
    "risk_corr": ("correlation", "正常箱坏率与 RC 参考坏率的 Spearman 相关系数，参考来源见参数"),
    "mono": ("correlation", "Total 正常箱索引与坏率的 Spearman 相关系数；无法定义时沿用历史占位 1"),
    "count": ("count_or_weight", "全样本数；配置 weights_col 时为样本权重和"),
    "observed_count": ("count_or_weight", "已表现标签样本数或权重和；无标签时未计算"),
    "bad": ("count_or_weight", "已表现坏样本数或权重和"),
    "good": ("count_or_weight", "observed_count-bad"),
    "total_count": ("count_or_weight", "分组全样本数或权重和"),
    "cum_count": ("count_or_weight", "按分箱索引展示顺序的累计全样本量或权重和"),
    "cum_observed_count": ("count_or_weight", "按分箱索引展示顺序的累计已表现样本量或权重和"),
    "cum_bad": ("count_or_weight", "按分箱索引展示顺序的累计坏样本量或权重和"),
    "avg_amt": (
        "unknown_business_unit",
        "tot_amt/count；配置样本权重时 count 为权重和；业务币种未知",
    ),
    "observed_amt": ("unknown_business_unit", "已表现样本金金额和；业务币种未知"),
    "tot_amt": ("unknown_business_unit", "全样本金金额和；业务币种未知"),
    "good_amt": ("unknown_business_unit", "好样本金金额和；业务币种未知"),
    "bad_amt": ("unknown_business_unit", "坏样本金金额和；业务币种未知"),
    "unseen": ("ratio", "当前有效类别中不在基准有效类别的样本比例"),
    "unseen_rate": ("ratio", "当前有效类别中不在基准有效类别的样本比例"),
    "feature": ("identifier", "原输入特征名称"),
    "dtype": ("type", "原输入字段的数据类型"),
    "data_source": ("category", "调用方登记的特征来源；未映射字段可能标为 UNMAPPED"),
    "source": ("category", "已保存的参考来源，例如 total、first_group 或 benchmark_df"),
    "bin_index": ("index", "分箱规则索引；缺失和特殊箱为负，汇总行由 bin_type 标识"),
    "bin_label": ("category", "来自拟合分箱规则的箱标签"),
    "bin_type": ("category", "正常、缺失、特殊或汇总箱类型"),
    "trend": ("category", "Total 正常箱 WOE 形态；无标签时未计算"),
    "y": ("identifier", "目标列标识；真实目标及无标签状态见参数"),
    "target": ("identifier", "真实目标列标识"),
    "mode_value": ("original_value", "原始最高频取值的字符串表示，含缺失"),
    "distribution": ("display", "已有 sparkline 展示；抽样配置见参数，不用于精确统计"),
}
for _metric, _meaning in {
    "mean": "非加权有效值均值",
    "median": "有效值中位数",
    "sum": "有效值和，空有效集沿用 0",
    "std": "有效值样本标准差（ddof=1）",
    "min": "有效值最小值",
    "max": "有效值最大值",
    "p25": "有效值 25% 分位数（nearest 插值）",
    "p75": "有效值 75% 分位数（nearest 插值）",
    "skew": "有效值偏度（bias=True）",
    "kurtosis": "有效值超额峰度（fisher=True, bias=True）",
}.items():
    _DEFINITIONS[_metric] = (
        "dimensionless" if _metric in {"skew", "kurtosis"} else "unknown_business_unit",
        f"{_meaning}；排除 Null、NaN、配置缺失值和特殊值；非数值字段未计算",
    )


def _definition(column: str) -> dict[str, str]:
    """映射显式登记的指标变体；其余字段明确标为未知。"""
    name = column
    for suffix in ["_rate", "_min", "_max"]:
        if name.endswith(suffix) and name not in _DEFINITIONS:
            name = name[: -len(suffix)]
            break
    if name == "rc":
        name = "risk_corr"
    unit, meaning = _DEFINITIONS.get(name, ("unknown", "描述字段或未登记指标；含义未知"))
    if name != column and column.endswith("_max"):
        meaning += "；跨有效分组最大值"
    elif name != column and column.endswith("_min"):
        meaning += "；跨有效分组最小值"
    return {"unit": unit, "meaning": meaning}


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
    if projected is not None:
        if not projected:
            raise ValueError("columns must not be empty.")
        result = (
            result.select(projected)
            if isinstance(result, pl.DataFrame)
            else result.loc[:, projected]
        )
    if isinstance(result, pl.DataFrame):
        return result.slice(offset, limit).clone()
    return result.iloc[offset : None if limit is None else offset + limit].copy()


def _json_safe(value: Any) -> Any:
    """日期转 ISO，非有限数保留字符串标记，空值为 null，未知对象拒绝序列化。"""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else np.datetime_as_string(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported context value type: {type(value).__name__}.")


def _encode(value: Any) -> str:
    """输出标准紧凑 JSON，不允许裸 NaN。"""
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class _ReportQuery:
    """两个领域报告共享的轻量结果查询，不持有原始数据。"""

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
        source_map: dict[str, str] = getattr(self, "feature_data_source", {})
        main = next(iter(self._query_tables().values()))
        if not source_map and {"feature", "data_source"}.issubset(main.columns):
            rows = (
                main.select("feature", "data_source").to_dicts()
                if isinstance(main, pl.DataFrame)
                else main[["feature", "data_source"]].to_dict("records")
            )
            source_map = {r["feature"]: r["data_source"] for r in rows}
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
        return {
            "report_type": type(self).__name__,
            "tables": tables,
            "parameters": deepcopy(self._query_metadata()),
            "business_context": {
                "label_definition": "unknown",
                "observation_window": "unknown",
                "currency": "unknown",
            },
            "limitations": [
                "只组织已有统计，未计算表不在目录中；失败和跳过见 parameters.diagnostics / fit_failures。",
                "Null 不是数值 0；风险指标需结合标签表现状态，样本分布使用全样本。",
                "比较前核对样本范围、权重、缺失/特殊箱、拟合来源、排序口径及参考来源；关联不代表因果。",
            ],
        }

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

    def to_ai_context(
        self,
        *,
        tables: list[str] | None = None,
        features: str | list[str] | None = None,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        max_chars: int = 16000,
    ) -> str:
        """生成可复制或写入文件的紧凑 JSON 上下文，不调用 LLM。

        Parameters
        ----------
        tables : list[str] | None
            规范表名列表；默认仅 overview 或 summary。
        features : str | list[str] | None
            按特征缩小范围。
        columns : list[str] | None
            所选表共同的字段投影。
        filters : dict[str, Any] | None
            与 get_table 相同的筛选条件。
        limit : int
            每表最大行数，默认 10。
        max_chars : int
            完整 JSON 字符预算，默认 16000；不等于 token 数。

        Returns
        -------
        str
            有效 JSON；省略项明确记录，日期为 ISO，非有限数为 NaN/Infinity/-Infinity 字符串，空值为 null。

        Raises
        ------
        ValueError
            参数无效、预算无法容纳说明或值无法序列化时抛出。

        Examples
        --------
        >>> import json
        >>> json.loads(report.to_ai_context(features="age"))  # doctest: +SKIP
        """
        if type(max_chars) is not int or max_chars < 512 or type(limit) is not int or limit < 0:
            raise ValueError("max_chars must be >= 512 and limit must be non-negative integers.")
        available = self._query_tables()
        selected = _names(tables, "tables")
        selected = list(available)[:1] if selected is None else selected
        if not selected or any(name not in available for name in selected):
            raise ValueError(f"tables must select names from {list(available)}.")
        description = self.describe()
        omitted: list[dict[str, Any]] = []
        # 大型参数值保留定位信息，避免特征列表吃掉整个预算；完整说明始终可 describe。
        for key, value in description["parameters"].items():
            if len(_encode(value)) > max_chars // 8:
                description["parameters"][key] = {
                    "status": "omitted",
                    "reference": f"report_meta.{key}",
                }
                omitted.append({"reference": f"report_meta.{key}", "reason": "description_budget"})
        for name, table in description["tables"].items():
            if name not in selected:
                table["field_count"] = len(table.pop("fields"))
            elif columns is not None:
                table["fields"] = {c: v for c, v in table["fields"].items() if c in columns}
        payload: dict[str, Any] = {
            "description": description,
            "evidence": [],
            "omitted": omitted,
            "serialization": {
                "date": "ISO-8601",
                "non_finite": "NaN/Infinity/-Infinity strings",
                "null": "missing or uncomputed; see parameters",
                "zero": "numeric zero",
            },
            "budget": {"max_chars": max_chars, "unit": "unicode_characters"},
        }
        for name in selected:
            queried = self.get_table(name, features=features, columns=columns, filters=filters)
            page = query_table(queried, limit=limit)
            rows = page.to_dicts() if isinstance(page, pl.DataFrame) else page.to_dict("records")
            payload["evidence"].append(
                {
                    "reference": name,
                    "query": {
                        "table": name,
                        "features": features,
                        "columns": columns,
                        "filters": filters,
                        "offset": 0,
                    },
                    "total_rows": len(queried),
                    "returned_rows": len(rows),
                    "rows": rows,
                }
            )
            if len(queried) > len(rows):
                omitted.append(
                    {"reference": name, "rows": len(queried) - len(rows), "reason": "row_limit"}
                )
        # 逐表移除完整行并记录数量，绝不截断 JSON 文本。
        while len(_encode(payload)) > max_chars:
            candidates = [item for item in payload["evidence"] if item["rows"]]
            if not candidates:
                raise ValueError(
                    "max_chars cannot contain report description; increase budget or narrow tables/columns."
                )
            item = max(candidates, key=lambda entry: len(_encode(entry["rows"])))
            item["rows"].pop()
            item["returned_rows"] -= 1
            marker = next(
                (
                    m
                    for m in omitted
                    if m.get("reference") == item["reference"]
                    and m.get("reason") == "output_budget"
                ),
                None,
            )
            if marker is None:
                marker = {"reference": item["reference"], "rows": 0, "reason": "output_budget"}
                omitted.append(marker)
            marker["rows"] += 1
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
        for name, table in self._query_tables().items():
            if "feature" in table.columns:
                matched = self.get_table(name, features=feature)
                results[name] = query_table(matched, limit=limit)
                omitted_rows[name] = max(len(matched) - len(results[name]), 0)
        if not any(len(table) for table in results.values()):
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
            "tables": results,
            "omitted_rows": omitted_rows,
            "unavailable": unavailable,
        }
