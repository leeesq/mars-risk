"""分箱评估内部类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import pandas as pd
import polars as pl

from mars.compute import RiskCorrBaseline
from mars.feature.binning.base import MarsBinnerBase

FrameLike = Union[pl.DataFrame, pd.DataFrame]


@dataclass(frozen=True)
class EvaluationRunContext:
    """保存单次评估运行的显式上下文，避免把运行状态挂到 evaluator 实例。"""

    working_df: pl.DataFrame
    benchmark_df: pl.DataFrame | None
    binner: MarsBinnerBase
    target: str
    original_target: str | None
    has_target: bool
    features: list[str]
    feature_source_map: dict[str, str]
    group_col: str
    profile_by: str | None
    dt_col: str | None
    output_kind: str
    feature_start_aware_reference: bool
    risk_corr_baseline: RiskCorrBaseline
    psi_include_missing: bool
    psi_include_special: bool
    weights_col: str | None
    amount_col: str | None
    batch_size: int


@dataclass(frozen=True)
class MetricFrames:
    """保存分箱评估的核心长表结果。"""

    group_stats_raw: pl.DataFrame
    total_stats_raw: pl.DataFrame
    expected_dist: pl.DataFrame
    metrics_groups: pl.DataFrame
    metrics_total: pl.DataFrame
    stats_long: pl.DataFrame
    monitor_metrics_groups: pl.DataFrame | None
    monitor_metrics_total: pl.DataFrame | None
    feature_start_reference: dict[str, Any] | None
    risk_corr_reference_table: pl.DataFrame
    risk_corr_reference_source: str
    monotonicity_df: pl.DataFrame
