"""规则交互与累计边际贡献分析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, islice
from typing import Any, Dict, List, Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

from mars.compute import FrameLike, to_polars_frame
from mars.rule.contracts import MarsRuleSet


@dataclass(frozen=True)
class MarsRuleAnalysis:
    """保存按需执行的规则高级分析结果。

    Parameters
    ----------
    interaction_table : polars.DataFrame
        两两命中重叠和组合风险表，按需包含金额与客户指标。
    cumulative_table : polars.DataFrame
        按 RuleSet 顺序累计覆盖与边际贡献表，按需包含金额与客户指标。
    bootstrap_table : polars.DataFrame
        可选 top-k Lift bootstrap 区间；默认分析不会执行重采样。
    metadata : Mapping[str, Any]
        分析目标、字段配置、配对预算和样本量。
    """

    interaction_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    cumulative_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    bootstrap_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def analyze_rule_set(
    rule_set: MarsRuleSet,
    df: FrameLike,
    *,
    target: str,
    amount_col: str | None = None,
    customer_col: str | None = None,
    max_pairs: int = 5000,
    bootstrap_repeats: int = 0,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> MarsRuleAnalysis:
    """计算规则交互和累计贡献。

    Parameters
    ----------
    rule_set : MarsRuleSet
        已排序的最终规则集。
    df : FrameLike
        分析样本，不会保存在结果中。
    target : str
        二分类目标列。
    amount_col : str | None
        可选金额列；提供后计算交互、累计及边际金额指标。
    customer_col : str | None
        可选客户列；提供后计算交互、累计及边际去重客户指标。
    max_pairs : int
        交互分析的最大规则对数量。
    bootstrap_repeats : int
        最终规则 Lift 重采样次数；``0`` 表示关闭。
    confidence_level : float
        bootstrap 双侧区间置信水平。
    random_state : int
        bootstrap 随机种子。

    Returns
    -------
    MarsRuleAnalysis
        交互、累计表与运行元数据。

    Raises
    ------
    ValueError
        配对预算非法、输入缺列或目标非二分类时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.rule import MarsRule, MarsRuleSet
    >>> rules = MarsRuleSet([MarsRule("x >= 2"), MarsRule("x >= 3")])
    >>> frame = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1], "amt": [10, 20, 30]})
    >>> result = analyze_rule_set(rules, frame, target="y", amount_col="amt")
    >>> result.cumulative_table.height
    2
    """
    if max_pairs < 1:
        raise ValueError("max_pairs 必须至少为 1。")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats 不能为负数。")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level 必须位于 (0.5, 1)。")
    frame: pl.DataFrame = to_polars_frame(df)
    required: List[str] = [*rule_set.required_features, target]
    required.extend(column for column in (amount_col, customer_col) if column)
    missing: List[str] = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"规则分析缺少必需列：{sorted(set(missing))}。")

    transformed = rule_set.transform(frame)
    assert isinstance(transformed, pl.DataFrame)
    target_alias: str = "__mars_analysis_target"
    transformed = (
        transformed
        .with_columns(pl.col(target).cast(pl.Float64, strict=False).alias(target_alias))
        .filter(pl.col(target_alias).is_not_null() & pl.col(target_alias).is_not_nan())
    )
    values: List[float] = sorted(float(value) for value in transformed[target_alias].unique())
    if not values:
        raise ValueError(f"目标列 {target!r} 没有可分析的非空标签。")
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError(f"目标列 {target!r} 必须是 0/1 二分类。")

    target_values: npt.NDArray[np.float64] = np.asarray(
        transformed[target_alias].to_numpy(),
        dtype=float,
    )
    amount_values: npt.NDArray[np.float64] | None = None
    if amount_col is not None:
        amount_values = np.asarray(
            transformed.select(pl.col(amount_col).cast(pl.Float64, strict=False))
            .to_series()
            .to_numpy(),
            dtype=float,
        )
    customer_values: npt.NDArray[np.int64] | None = (
        _factorize_customers(transformed[customer_col])
        if customer_col is not None
        else None
    )
    rule_masks: Dict[str, npt.NDArray[np.bool_]] = {
        rule.rule_id: np.asarray(
            transformed[f"rule__{rule.rule_id}"].to_numpy(),
            dtype=bool,
        )
        for rule in rule_set.rules
    }
    base_mask: npt.NDArray[np.bool_] = np.ones(transformed.height, dtype=bool)
    base: Dict[str, Any] = _population_metrics_from_mask(
        base_mask,
        target_values=target_values,
        amount_values=amount_values,
        customer_values=customer_values,
    )
    single_metrics: Dict[str, Dict[str, Any]] = {
        rule.rule_id: _population_metrics_from_mask(
            rule_masks[rule.rule_id],
            target_values=target_values,
            amount_values=amount_values,
            customer_values=customer_values,
            base=base,
        )
        for rule in rule_set.rules
    }
    interaction_rows: List[Dict[str, Any]] = []
    pair_iterator = islice(combinations(rule_set.rules, 2), max_pairs)
    for left, right in pair_iterator:
        left_metrics: Dict[str, Any] = single_metrics[left.rule_id]
        right_metrics: Dict[str, Any] = single_metrics[right.rule_id]
        intersection_mask: npt.NDArray[np.bool_] = (
            rule_masks[left.rule_id] & rule_masks[right.rule_id]
        )
        intersection_metrics: Dict[str, Any] = _population_metrics_from_mask(
            intersection_mask,
            target_values=target_values,
            amount_values=amount_values,
            customer_values=customer_values,
            base=base,
        )
        union_count: int = (
            int(left_metrics["sample_count"])
            + int(right_metrics["sample_count"])
            - int(intersection_metrics["sample_count"])
        )
        interaction_rows.append(
            {
                "rule_a": left.rule_id,
                "rule_b": right.rule_id,
                **_prefix_metrics(left_metrics, "rule_a"),
                **_prefix_metrics(right_metrics, "rule_b"),
                **_prefix_metrics(intersection_metrics, "intersection"),
                "union_count": union_count,
                "iou": _safe_div(intersection_metrics["sample_count"], union_count),
                "combo_gain_lift": _combo_gain_lift(
                    intersection_metrics.get("lift"),
                    left_metrics.get("lift"),
                    right_metrics.get("lift"),
                ),
            }
        )

    cumulative_rows: List[Dict[str, Any]] = []
    previous_mask: npt.NDArray[np.bool_] = np.zeros(transformed.height, dtype=bool)
    for rank, rule in enumerate(rule_set.rules, start=1):
        current_mask: npt.NDArray[np.bool_] = rule_masks[rule.rule_id]
        cumulative_mask: npt.NDArray[np.bool_] = previous_mask | current_mask
        marginal_mask: npt.NDArray[np.bool_] = current_mask & ~previous_mask
        cumulative_metrics: Dict[str, Any] = _population_metrics_from_mask(
            cumulative_mask,
            target_values=target_values,
            amount_values=amount_values,
            customer_values=customer_values,
            base=base,
        )
        marginal_metrics: Dict[str, Any] = _population_metrics_from_mask(
            marginal_mask,
            target_values=target_values,
            amount_values=amount_values,
            customer_values=customer_values,
            base=base,
        )
        cumulative_rows.append(
            {
                "rank": rank,
                "added_rule_id": rule.rule_id,
                **_prefix_metrics(cumulative_metrics, "cumulative"),
                **_prefix_metrics(marginal_metrics, "marginal"),
            }
        )
        previous_mask = cumulative_mask

    bootstrap_table: pl.DataFrame = _bootstrap_lifts(
        rule_masks,
        target_values,
        repeats=bootstrap_repeats,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    return MarsRuleAnalysis(
        interaction_table=_rows_to_frame(interaction_rows),
        cumulative_table=_rows_to_frame(cumulative_rows),
        bootstrap_table=bootstrap_table,
        metadata={
            "target": target,
            "amount_col": amount_col,
            "customer_col": customer_col,
            "max_pairs": max_pairs,
            "sample_count": base["sample_count"],
            "pair_count": len(interaction_rows),
            "bootstrap_repeats": bootstrap_repeats,
            "confidence_level": confidence_level,
            "random_state": random_state,
        },
    )


def _population_metrics_from_mask(
    mask: npt.NDArray[np.bool_],
    *,
    target_values: npt.NDArray[np.float64],
    amount_values: npt.NDArray[np.float64] | None,
    customer_values: npt.NDArray[np.int64] | None,
    base: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """在一次物化的命中矩阵上计算样本、金额和客户指标。"""
    sample_count: int = int(mask.sum())
    event_mask: npt.NDArray[np.bool_] = mask & (target_values == 1.0)
    event_count: int = int(event_mask.sum())
    event_rate: float | None = _safe_div(event_count, sample_count)
    amount_total: float | None = None
    event_amount: float | None = None
    if amount_values is not None:
        amount_total = float(np.nansum(amount_values[mask]))
        event_amount = float(np.nansum(amount_values[event_mask]))
    customer_count: int | None = None
    event_customer_count: int | None = None
    if customer_values is not None:
        customer_count = _unique_count(customer_values[mask])
        event_customer_count = _unique_count(customer_values[event_mask])
    amount_event_rate: float | None = _safe_div(event_amount, amount_total)
    customer_event_rate: float | None = _safe_div(event_customer_count, customer_count)
    metrics: Dict[str, Any] = {
        "sample_count": sample_count,
        "event_count": event_count,
        "coverage": None,
        "event_rate": event_rate,
        "lift": None,
        "amount_total": amount_total,
        "event_amount": event_amount,
        "amount_coverage": None,
        "amount_event_rate": amount_event_rate,
        "amount_lift": None,
        "customer_count": customer_count,
        "event_customer_count": event_customer_count,
        "customer_coverage": None,
        "customer_event_rate": customer_event_rate,
        "customer_lift": None,
    }
    if base is None:
        return metrics
    metrics.update(
        {
            "coverage": _safe_div(sample_count, base["sample_count"]),
            "lift": _safe_div(event_rate, base["event_rate"]),
            "amount_coverage": _safe_div(amount_total, base["amount_total"]),
            "amount_lift": _safe_div(amount_event_rate, base["amount_event_rate"]),
            "customer_coverage": _safe_div(customer_count, base["customer_count"]),
            "customer_lift": _safe_div(customer_event_rate, base["customer_event_rate"]),
        }
    )
    return metrics


def _factorize_customers(series: pl.Series) -> npt.NDArray[np.int64]:
    """一次性把客户标识编码为保持类型和缺失语义的整数。"""
    mapping: Dict[Any, int] = {}
    codes: npt.NDArray[np.int64] = np.empty(len(series), dtype=np.int64)
    for index, value in enumerate(series.to_list()):
        key: Any = (
            ("__missing__",)
            if value is None or isinstance(value, float) and np.isnan(value)
            else (type(value).__name__, value)
        )
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[index] = mapping[key]
    return codes


def _unique_count(values: npt.NDArray[np.int64]) -> int:
    """在预编码客户数组上计算去重数量。"""
    return int(np.unique(values).size)


def _bootstrap_lifts(
    rule_masks: Mapping[str, npt.NDArray[np.bool_]],
    target_values: npt.NDArray[np.float64],
    *,
    repeats: int,
    confidence_level: float,
    random_state: int,
) -> pl.DataFrame:
    """对最终规则执行确定性非参数 Lift bootstrap。"""
    if repeats == 0 or len(target_values) == 0:
        return pl.DataFrame()
    rng = np.random.default_rng(random_state)
    samples: Dict[str, List[float]] = {rule_id: [] for rule_id in rule_masks}
    row_count: int = len(target_values)
    for _ in range(repeats):
        indices: npt.NDArray[np.int64] = rng.integers(0, row_count, size=row_count)
        sampled_target: npt.NDArray[np.float64] = target_values[indices]
        base_rate: float = float(sampled_target.mean())
        if base_rate == 0.0:
            continue
        for rule_id, mask in rule_masks.items():
            sampled_mask: npt.NDArray[np.bool_] = mask[indices]
            if not sampled_mask.any():
                continue
            event_rate: float = float(sampled_target[sampled_mask].mean())
            samples[rule_id].append(event_rate / base_rate)
    tail: float = (1.0 - confidence_level) / 2.0
    rows: List[Dict[str, Any]] = []
    for rule_id, values in samples.items():
        if not values:
            continue
        array: npt.NDArray[np.float64] = np.asarray(values, dtype=float)
        rows.append(
            {
                "rule_id": rule_id,
                "repeat_count": len(values),
                "lift_ci_lower": float(np.quantile(array, tail)),
                "lift_median": float(np.quantile(array, 0.5)),
                "lift_ci_upper": float(np.quantile(array, 1.0 - tail)),
            }
        )
    return _rows_to_frame(rows)


def _prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """为人群指标添加稳定前缀。"""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _combo_gain_lift(
    combined_lift: Any,
    left_lift: Any,
    right_lift: Any,
) -> float | None:
    """计算组合 Lift 相对两条单规则最佳值的增益。"""
    if combined_lift is None or left_lift is None or right_lift is None:
        return None
    return float(combined_lift) - max(float(left_lift), float(right_lift))


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    """未定义除法返回 null。"""
    if numerator is None or denominator is None or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _rows_to_frame(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """把分析行转换为 Polars 表。"""
    return pl.DataFrame(rows) if rows else pl.DataFrame()
