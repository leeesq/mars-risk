"""保存筛选器的相关矩阵和决策记录，并提供查询与展示方法。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ._artifact import Report, ReportSnapshot
from ._query import query_table
from ._result import _result_report


def _correlation_report(selector: Any) -> CorrelationReport:
    """从有符号相关矩阵提取上三角和决策记录，随后由调用方释放稠密矩阵缓存。"""
    candidates = selector._corr_candidates
    matrix = selector._corr_matrix
    n = len(candidates)
    a, b = np.triu_indices(n, 1) if matrix is not None else (np.array([], dtype=int),) * 2
    values = matrix[a, b] if matrix is not None else np.array([], dtype=float)
    # 直接以物理索引构造 Enum，避免平方级复制长名称的 NumPy Unicode 数组。
    # 空候选仍沿用 String schema；规范上三角、类别顺序和实际系数不变。
    endpoint_dtype = pl.Enum(candidates) if n else pl.String
    pairs = (
        pl.DataFrame(
            {
                "feature_a": pl.Series(a.astype(np.uint32)).cast(endpoint_dtype),
                "feature_b": pl.Series(b.astype(np.uint32)).cast(endpoint_dtype),
                "correlation": values,
            }
        )
        .with_columns(
            pl.col("correlation").abs().alias("abs_correlation"),
            pl.when(pl.col("correlation").is_finite())
            .then(pl.lit("valid"))
            .otherwise(pl.lit("unavailable"))
            .alias("status"),
        )
        .with_columns(
            [
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
                for c in ("correlation", "abs_correlation")
            ]
        )
    )
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
    """从已保存的上三角与实际对角值恢复所选特征的相关子矩阵。

    Parameters
    ----------
    report : Report
        原始相关性报告，或 ``load_report()`` 恢复的相关性快照；不访问原始样本。
    features : list[str] | None
        进入相关性阶段的英文候选 ID，决定返回矩阵的行列及顺序。
        默认使用全部候选；候选较多时宜显式限定范围。

    Returns
    -------
    pd.DataFrame
        按请求顺序排列的有符号相关矩阵，不可用值为 NaN。
        对角线沿用已保存的实际结果，不补为 1。

    Raises
    ------
    ValueError
        报告类型不符、候选 ID 重复或 ID 未进入相关性阶段时抛出。

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
    # 先筛选两端均在所选范围内的特征对，避免转换完整 pairs 表。
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
    """按相关系数绝对值降序返回其他特征及其筛选证据。

    来源条件只筛选另一端特征，不限制中心特征；相关系数绝对值相同时保留候选顺序。

    Parameters
    ----------
    report : Report
        ``report_type="correlation"`` 的原始报告或通用快照。
    feature : str
        作为查询中心的英文候选 ID，必须已进入相关性阶段。
    sources : str | list[str] | None
        另一端特征的业务来源，默认不限制；来源未知时报错。
    offset : int
        按相关系数绝对值排序后跳过的行数，默认 0，必须为非负整数。
    limit : int
        最大返回行数，默认 20，必须为非负整数；0 返回空表。

    Returns
    -------
    pl.DataFrame
        另一端特征的 ID、显示名、来源、有符号相关系数、最终入选状态及实际淘汰事件。
        决策记录来自原筛选过程，不根据相关系数重新生成。

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
    """展示指定数量特征的有符号相关矩阵，使用固定 [-1, 1] 色标。

    空值单独着色，有效零值保留为 0，不重新计算相关系数。

    Parameters
    ----------
    report : Report
        原始相关性报告或加载后的相关性快照。
    features : list[str] | None
        展示候选的英文 ID 及顺序，默认使用全部候选的原顺序。
    max_features : int
        最多展示的特征数，默认 30，必须为正整数。
        仅展示所选列表的前若干项，省略数量显示在表标题中，不截断原报告。

    Returns
    -------
    Any
        可在 Notebook 中显示的 Pandas Styler，也可调用 ``to_html()`` 或 ``to_excel()``。

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
    """保存相关矩阵、候选状态和筛选决策的独立报告快照。

    ``load_report()`` 返回通用快照，可用本模块同名公共函数继续执行矩阵和邻居查询。

    Parameters
    ----------
    tables : dict[str, Any]
        候选目录、上三角特征对和筛选记录等已计算结果表。
    description : dict[str, Any]
        与结果表对应的完整语义目录，含方法、候选范围、实际参数和业务元数据。

    Examples
    --------
    >>> report = selector.get_correlation_report()  # doctest: +SKIP
    >>> report.get_matrix(["income"])  # doctest: +SKIP
    """

    def get_matrix(self, features: list[str] | None = None) -> pd.DataFrame:
        """从保存结果恢复所选特征的有符号相关子矩阵，不访问原始样本。

        Parameters
        ----------
        features : list[str] | None
            英文候选 ID，决定矩阵行列及顺序；默认使用全部候选。
            候选较多时宜显式限定范围。

        Returns
        -------
        pd.DataFrame
            按请求顺序排列的矩阵，不可用值为 NaN；对角线沿用保存的实际结果。

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
        """查询与中心特征相关的其他特征，按相关系数绝对值降序返回。

        Parameters
        ----------
        feature : str
            原始候选 ID，不以中文名作为键。
        sources : str | list[str] | None
            仅筛选另一端特征的业务来源，默认不限制；未知来源报错。
        offset : int
            排序后跳过的行数，默认 0，必须为非负整数。
        limit : int
            最大返回行数，默认 20；0 返回空表。相关系数绝对值相同时保留候选顺序。

        Returns
        -------
        pl.DataFrame
            有符号相关系数、另一端特征的业务元数据、最终入选状态和实际筛选事件。

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
        """用固定 [-1, 1] 色标展示相关矩阵，空值与有效零值分别着色。

        Parameters
        ----------
        features : list[str] | None
            英文候选 ID 及展示顺序，默认使用全部候选的原顺序。
        max_features : int
            最多展示的特征数，默认 30，必须为正整数。
            仅展示所选列表的前若干项，省略数量显示在表标题中，不截断快照。

        Returns
        -------
        Any
            包含英文 ID、显示名和来源的 Pandas Styler，支持 Notebook 展示及 HTML/Excel 导出。

        Raises
        ------
        ValueError
            上限或候选 ID 无效。

        Examples
        --------
        >>> report.show_matrix(max_features=20)  # doctest: +SKIP
        """  # 异常由共用展示函数传播；pydoclint noqa: DOC502
        return show_correlation_matrix(self, features, max_features=max_features)
