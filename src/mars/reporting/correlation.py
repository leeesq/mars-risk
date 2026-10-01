"""复用筛选器真实矩阵与事件的可携带相关性报告。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ._artifact import Report, ReportSnapshot
from ._query import query_table
from ._result import _result_report


def _correlation_report(selector: Any) -> CorrelationReport:
    """将唯一 signed 矩阵转换为规范上三角，随后由调用方释放 dense 缓存。"""
    candidates = selector._corr_candidates
    matrix = selector._corr_matrix
    n = len(candidates)
    a, b = np.triu_indices(n, 1) if matrix is not None else (np.array([], dtype=int),) * 2
    names = np.asarray(candidates, dtype=str)
    values = matrix[a, b] if matrix is not None else np.array([], dtype=float)
    pairs = pl.DataFrame(
        {
            "feature_a": names[a],
            "feature_b": names[b],
            "correlation": values,
            "abs_correlation": np.abs(values),
            "status": np.where(np.isfinite(values), "valid", "unavailable"),
        }
    ).with_columns(
        [
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
            for c in ("correlation", "abs_correlation")
        ]
    )
    if n:
        pairs = pairs.with_columns(pl.col("feature_a", "feature_b").cast(pl.Enum(candidates)))
    catalog: list[dict[str, Any]] = []
    for feature in selector._corr_input_features:
        i = candidates.index(feature) if feature in candidates else None
        diagonal = float(matrix[i, i]) if matrix is not None and i is not None else None
        events = [r for r in selector.report_records_ if r["feature"] == feature]
        drops = [r for r in events if r["status"] == "Dropped"]
        catalog.append(
            {
                "feature": feature,
                "candidate_order": i,
                "participated": matrix is not None and i is not None,
                "diagonal": diagonal if diagonal is not None and np.isfinite(diagonal) else None,
                "diagonal_status": "valid"
                if diagonal is not None and np.isfinite(diagonal)
                else "unavailable"
                if i is not None and matrix is not None
                else "uncomputed",
                "selected": feature in selector.selected_features_,
                "exit_stage": drops[-1]["stage"] if drops else None,
            }
        )
    feature_table = pl.DataFrame(
        catalog,
        schema_overrides={
            "feature": pl.String,
            "candidate_order": pl.Int64,
            "diagonal": pl.Float64,
            "exit_stage": pl.String,
        },
    )
    decisions = pl.DataFrame(
        selector._corr_events,
        schema={
            "event_order": pl.Int64,
            "feature": pl.String,
            "trigger_feature": pl.String,
            "action": pl.String,
            "correlation": pl.Float64,
            "abs_correlation": pl.Float64,
            "threshold": pl.Float64,
            "operator": pl.String,
            "priority_metric": pl.String,
            "feature_priority": pl.Float64,
            "trigger_priority": pl.Float64,
            "tie_rule": pl.String,
            "white_list_protected": pl.Boolean,
        },
    ).with_columns(pl.col(pl.Float64).fill_nan(None))
    selection = pl.DataFrame(
        selector.report_records_,
        schema={
            "feature": pl.String,
            "data_source": pl.String,
            "status": pl.String,
            "stage": pl.String,
            "reason": pl.String,
            "value": pl.Float64,
            "description": pl.String,
        },
    )
    tables = {
        "features": feature_table,
        "pairs": pairs,
        "correlation_decisions": decisions,
        "selection": selection,
    }
    if getattr(selector, "_funnel_stats", None):
        tables["funnel"] = pl.DataFrame(selector._funnel_stats)
    parameters = {
        **selector._corr_parameters,
        "candidate_scope": candidates,
        "status": selector._corr_status,
        "sample_count_definition": "global correlation input rows; not pairwise overlap",
        "abs_correlation": "abs(signed correlation), materialized for native sorting",
        "feature_filter": "pairs: any endpoint union; independent sources and features conditions AND",
        "related_source_filter": "peer source only",
        "unknown_feature": "table query returns empty; dedicated operations raise ValueError",
    }
    snapshot = _result_report(
        "correlation",
        tables,
        parameters,
        selector._corr_input_features,
        getattr(selector, "feature_metadata", None),
        getattr(selector, "business_context", None),
        roles={
            "pairs": {"feature_a": "left endpoint", "feature_b": "right endpoint"},
            "correlation_decisions": {"feature": "decision subject", "trigger_feature": "trigger"},
        },
        definitions={
            "correlation": "signed correlation; representation/method in parameters",
            "diagonal": "actual engine diagonal; never synthesized as 1",
        },
    )
    return CorrelationReport(snapshot._tables, snapshot.describe())


def get_correlation_matrix(report: Report, features: list[str] | None = None) -> pd.DataFrame:
    """从保存的上三角和实际对角恢复诱导子矩阵，不重新计算。

    Parameters
    ----------
    report : Report
        原始报告或 load_report 的通用快照。
    features : list[str] | None
        原始 ID；None 使用整个候选集合，大集合应显式限定。

    Returns
    -------
    pd.DataFrame
        请求顺序的 signed 矩阵；不可用值为 NaN。

    Raises
    ------
    ValueError
        类型错误、重复 ID 或 ID 未进入相关阶段。

    Examples
    --------
    >>> get_correlation_matrix(restored, ["income", "loan_count"])  # doctest: +SKIP
    """
    if report.report_type != "correlation":
        raise ValueError("Expected a correlation report.")
    candidates = report.describe()["parameters"]["candidate_scope"]
    selected = list(candidates) if features is None else list(features)
    if len(set(selected)) != len(selected) or set(selected) - set(candidates):
        raise ValueError("features must be unique correlation candidate IDs.")
    matrix = pd.DataFrame(np.nan, index=selected, columns=selected)
    diagonal = report.get_table("features", features=selected)
    for row in diagonal.iter_rows(named=True):
        matrix.loc[row["feature"], row["feature"]] = row["diagonal"]
    # 先限定两端为所选集合，再转换有限子矩阵；不转出整个 pairs。
    pairs = report.get_table(
        "pairs",
        filters={
            "feature_a": {"op": "in", "value": selected},
            "feature_b": {"op": "in", "value": selected},
        },
    )
    for row in pairs.iter_rows(named=True):
        matrix.loc[row["feature_a"], row["feature_b"]] = row["correlation"]
        matrix.loc[row["feature_b"], row["feature_a"]] = row["correlation"]
    return matrix


def get_related_features(
    report: Report,
    feature: str,
    *,
    sources: str | list[str] | None = None,
    offset: int = 0,
    limit: int = 20,
) -> pl.DataFrame:
    """按绝对相关降序返回 peers；来源只过滤 peer，同值保持候选顺序。

    Parameters
    ----------
    report : Report
        correlation 报告，支持加载后的通用快照。
    feature : str
        查询中心的原始 ID。
    sources : str | list[str] | None
        peer 来源；未知来源报错。与中心 ID 条件采用交集。
    offset : int
        非负分页起点。
    limit : int
        最大行数。

    Returns
    -------
    pl.DataFrame
        peer ID、展示名、来源、signed 值、最终入选及真实触发事件。

    Raises
    ------
    ValueError
        报告类型、ID、来源或分页无效。

    Examples
    --------
    >>> get_related_features(restored, "income", limit=5)  # doctest: +SKIP
    """
    if report.report_type != "correlation":
        raise ValueError("Expected a correlation report.")
    description = report.describe()
    if feature not in description["parameters"]["candidate_scope"]:
        raise ValueError(f"Unknown correlation candidate: {feature!r}.")
    metadata = description["feature_metadata"]
    pairs = report.get_table("pairs", features=feature).with_columns(
        pl.when(pl.col("feature_a") == feature)
        .then(pl.col("feature_b").cast(pl.String))
        .otherwise(pl.col("feature_a").cast(pl.String))
        .alias("peer_feature")
    )
    if sources is not None:
        names = [sources] if isinstance(sources, str) else sources
        known = {m.get("data_source") or "UNMAPPED" for m in metadata.values()}
        if set(names) - known:
            raise ValueError("Unknown peer sources.")
        pairs = pairs.filter(
            pl.col("peer_feature").is_in(
                [f for f, m in metadata.items() if (m.get("data_source") or "UNMAPPED") in names]
            )
        )
    pairs = query_table(
        pairs, sort_by="abs_correlation", descending=True, offset=offset, limit=limit
    )
    peers = pairs["peer_feature"].to_list()
    directory = report.get_table("features", features=peers).select(
        pl.col("feature").alias("peer_feature"), "selected", "exit_stage"
    )
    pairs = (
        pairs.with_row_index("_order")
        .join(directory, on="peer_feature", how="left")
        .sort("_order")
        .drop("_order")
    )
    events = report.get_table("correlation_decisions", filters={"action": "drop"})
    triggers: dict[str, list[int]] = {p: [] for p in peers}
    reasons: dict[str, list[str]] = {p: [] for p in peers}
    for event in events.iter_rows(named=True):
        if feature in (event["feature"], event["trigger_feature"]):
            peer = event["trigger_feature"] if event["feature"] == feature else event["feature"]
            if peer in triggers:
                triggers[peer].append(event["event_order"])
                reasons[peer].append(
                    f"drop {event['feature']} triggered by {event['trigger_feature']}; abs={event['abs_correlation']} {event['operator']} {event['threshold']}"
                )
    return pairs.with_columns(
        pl.Series(
            "peer_display_name", [metadata.get(p, {}).get("display_name") or p for p in peers]
        ),
        pl.Series(
            "peer_source", [metadata.get(p, {}).get("data_source") or "UNMAPPED" for p in peers]
        ),
        pl.Series("decision_event_ids", [triggers[p] for p in peers], dtype=pl.List(pl.Int64)),
        pl.Series("decision_reasons", [reasons[p] for p in peers], dtype=pl.List(pl.String)),
    )


def show_correlation_matrix(
    report: Report,
    features: list[str] | None = None,
    *,
    max_features: int = 30,
) -> Any:
    """展示有限 signed 矩阵，固定 [-1, 1] 色标，空值与零分开。

    Parameters
    ----------
    report : Report
        原始或加载后的报告。
    features : list[str] | None
        展示候选 ID；默认候选顺序。
    max_features : int
        展示上限；截断数量保留在 caption，原报告不变。

    Returns
    -------
    Any
        Notebook 可显示、可 to_html/to_excel 的 Pandas Styler。

    Raises
    ------
    ValueError
        上限非正整数或矩阵查询无效。

    Examples
    --------
    >>> show_correlation_matrix(restored, ["income", "loan_count"])  # doctest: +SKIP
    """
    if type(max_features) is not int or max_features < 1:
        raise ValueError("max_features must be positive.")
    selected = report.describe()["parameters"]["candidate_scope"] if features is None else features
    matrix = get_correlation_matrix(report, selected[:max_features])
    metadata = report.describe()["feature_metadata"]
    labels = {
        f: f"{f} [{metadata.get(f, {}).get('display_name') or f}; {metadata.get(f, {}).get('data_source') or 'UNMAPPED'}]"
        for f in matrix.index
    }
    matrix = matrix.rename(index=labels, columns=labels)
    return (
        matrix.style.format("{:.3f}", na_rep="unavailable")
        .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1, axis=None)
        .highlight_null(color="#eee")
        .format_index(escape="html", axis=0)
        .format_index(escape="html", axis=1)
        .set_caption(
            f"Signed correlation; omitted features: {max(0, len(selected) - max_features)}"
        )
    )


class CorrelationReport(ReportSnapshot):
    """相关性独立快照；加载后使用同名公共函数获得相同专用操作。

    Parameters
    ----------
    tables : dict[str, Any]
        列式结果表。
    description : dict[str, Any]
        完整语义目录。

    Examples
    --------
    >>> report = selector.get_correlation_report()  # doctest: +SKIP
    >>> report.get_matrix(["income"])  # doctest: +SKIP
    """

    def get_matrix(self, features: list[str] | None = None) -> pd.DataFrame:
        """从保存结果恢复 signed 诱导子矩阵，不访问原始数据。

        Parameters
        ----------
        features : list[str] | None
            原始候选 ID，None 使用全部；大集合建议显式限定。

        Returns
        -------
        pd.DataFrame
            请求顺序的矩阵，不可用值为 NaN；保留实际对角。

        Raises
        ------
        ValueError
            重复或未知候选 ID。

        Examples
        --------
        >>> report.get_matrix(["income", "loan_count"])  # doctest: +SKIP
        """  # 异常由共用查询函数传播；pydoclint noqa: DOC502
        return get_correlation_matrix(self, features)

    def get_related(
        self,
        feature: str,
        *,
        sources: str | list[str] | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> pl.DataFrame:
        """取得与中心特征相关的 peers，来源只过滤另一端。

        Parameters
        ----------
        feature : str
            原始候选 ID，不以中文名作为键。
        sources : str | list[str] | None
            peer 来源，与中心条件采用交集。
        offset : int
            非负起始行号。
        limit : int
            最大返回数，按绝对相关降序，同值保持候选顺序。

        Returns
        -------
        pl.DataFrame
            signed 值、peer 业务元数据、最终状态与真实事件证据。

        Raises
        ------
        ValueError
            ID、来源或分页参数无效。

        Examples
        --------
        >>> report.get_related("income", limit=10)  # doctest: +SKIP
        """  # 异常由共用查询函数传播；pydoclint noqa: DOC502
        return get_related_features(self, feature, sources=sources, offset=offset, limit=limit)

    def show_matrix(self, features: list[str] | None = None, *, max_features: int = 30) -> Any:
        """显示固定 [-1,1] 发散色标的小矩阵，未知值与有效零分开。

        Parameters
        ----------
        features : list[str] | None
            原始候选 ID；None 使用候选顺序。
        max_features : int
            正数显示上限；省略数量在 caption 中提示，快照不截断。

        Returns
        -------
        Any
            支持 Notebook/to_html/to_excel 的 Styler，含原始 ID、展示名及来源。

        Raises
        ------
        ValueError
            上限或候选 ID 无效。

        Examples
        --------
        >>> report.show_matrix(max_features=20)  # doctest: +SKIP
        """  # 异常由共用展示函数传播；pydoclint noqa: DOC502
        return show_correlation_matrix(self, features, max_features=max_features)
