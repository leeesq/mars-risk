"""规则验证的置信区间、精确检验与多重检验校正。"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import hypergeom

from mars.rule.contracts import RuleDirection


def add_statistical_metrics(
    table: pl.DataFrame,
    *,
    direction: RuleDirection,
    confidence_level: float,
) -> pl.DataFrame:
    """为固定长表增加单侧置信界、精确 p 值和 BH q 值。"""
    if table.is_empty():
        return table
    records: List[Dict[str, Any]] = [dict(row) for row in table.iter_rows(named=True)]
    totals: Dict[Tuple[str, str, str, str], Mapping[str, Any]] = {
        (
            str(row["dataset"]),
            str(row["target"]),
            str(row["slice"]),
            str(row["rule_id"]),
        ): row
        for row in records
        if row["group"] == "total"
    }
    for row in records:
        row.update(
            {
                "event_rate_ci_lower": None,
                "event_rate_ci_upper": None,
                "lift_ci_lower": None,
                "lift_ci_upper": None,
                "p_value": None,
                "q_value": None,
            }
        )
    hit_indices: List[int] = [
        index for index, row in enumerate(records) if row["group"] == "hit"
    ]
    if not hit_indices:
        return pl.DataFrame(records)
    population_sizes: List[int] = []
    population_events: List[int] = []
    sample_sizes: List[int] = []
    sample_events: List[int] = []
    base_rates: List[float] = []
    group_keys: List[Tuple[str, str, str]] = []
    for index in hit_indices:
        record: Dict[str, Any] = records[index]
        key: Tuple[str, str, str, str] = (
            str(record["dataset"]),
            str(record["target"]),
            str(record["slice"]),
            str(record["rule_id"]),
        )
        total: Mapping[str, Any] = totals[key]
        population_sizes.append(int(total["sample_count"]))
        population_events.append(int(total["event_count"]))
        sample_sizes.append(int(record["sample_count"]))
        sample_events.append(int(record["event_count"]))
        base_rates.append(float(total["event_rate"] or 0.0))
        group_keys.append(key[:3])

    population_array: npt.NDArray[np.int64] = np.asarray(population_sizes, dtype=np.int64)
    population_event_array: npt.NDArray[np.int64] = np.asarray(
        population_events,
        dtype=np.int64,
    )
    sample_array: npt.NDArray[np.int64] = np.asarray(sample_sizes, dtype=np.int64)
    sample_event_array: npt.NDArray[np.int64] = np.asarray(
        sample_events,
        dtype=np.int64,
    )
    lower_array, upper_array = _wilson_arrays(
        sample_event_array,
        sample_array,
        confidence_level=confidence_level,
    )
    if direction == "high_risk":
        p_values: npt.NDArray[np.float64] = np.asarray(
            hypergeom.sf(
                sample_event_array - 1,
                population_array,
                population_event_array,
                sample_array,
            ),
            dtype=float,
        )
    else:
        p_values = np.asarray(
            hypergeom.cdf(
                sample_event_array,
                population_array,
                population_event_array,
                sample_array,
            ),
            dtype=float,
        )

    p_value_groups: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    for position, index in enumerate(hit_indices):
        record = records[index]
        sample_count: int = sample_sizes[position]
        base_rate: float = base_rates[position]
        lower: float | None = (
            float(lower_array[position]) if sample_count > 0 else None
        )
        upper: float | None = (
            float(upper_array[position]) if sample_count > 0 else None
        )
        p_value: float | None = (
            float(p_values[position]) if sample_count > 0 else None
        )
        record.update(
            {
                "event_rate_ci_lower": lower,
                "event_rate_ci_upper": upper,
                "lift_ci_lower": _safe_div(lower, base_rate),
                "lift_ci_upper": _safe_div(upper, base_rate),
                "p_value": p_value,
            }
        )
        if p_value is not None:
            p_value_groups.setdefault(group_keys[position], []).append((index, p_value))

    for indexed_values in p_value_groups.values():
        q_values: List[float] = benjamini_hochberg(
            [value for _, value in indexed_values]
        )
        for (record_index, _), q_value in zip(indexed_values, q_values):
            records[record_index]["q_value"] = q_value
    return pl.DataFrame(records)


def add_empty_statistical_metrics(table: pl.DataFrame) -> pl.DataFrame:
    """在未请求统计计算时追加固定的可空统计列。"""
    if table.is_empty():
        return table
    return table.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in (
                "event_rate_ci_lower",
                "event_rate_ci_upper",
                "lift_ci_lower",
                "lift_ci_upper",
                "p_value",
                "q_value",
            )
        ]
    )


def _wilson_arrays(
    event_counts: npt.NDArray[np.int64],
    sample_counts: npt.NDArray[np.int64],
    *,
    confidence_level: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """向量化计算二项事件率的单侧 Wilson 界。"""
    z_value: float = NormalDist().inv_cdf(confidence_level)
    z_squared: float = z_value * z_value
    safe_counts: npt.NDArray[np.float64] = np.maximum(sample_counts, 1).astype(float)
    proportions: npt.NDArray[np.float64] = event_counts / safe_counts
    denominator: npt.NDArray[np.float64] = 1.0 + z_squared / safe_counts
    center: npt.NDArray[np.float64] = proportions + z_squared / (2.0 * safe_counts)
    distance: npt.NDArray[np.float64] = z_value * np.sqrt(
        proportions * (1.0 - proportions) / safe_counts
        + z_squared / (4.0 * safe_counts * safe_counts)
    )
    lower: npt.NDArray[np.float64] = np.maximum(
        0.0,
        (center - distance) / denominator,
    )
    upper: npt.NDArray[np.float64] = np.minimum(
        1.0,
        (center + distance) / denominator,
    )
    return lower, upper


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """按输入顺序返回 Benjamini-Hochberg 校正 q 值。"""
    count: int = len(p_values)
    if count == 0:
        return []
    order: List[int] = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted: List[float] = [1.0] * count
    running: float = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index: int = order[reverse_rank]
        rank: int = reverse_rank + 1
        candidate: float = min(1.0, float(p_values[index]) * count / rank)
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    """执行统计量安全除法。"""
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)
