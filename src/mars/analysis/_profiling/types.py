"""数据画像内部类型与默认指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import polars as pl

DEFAULT_DQ_METRICS: list[str] = ["missing", "zeros", "unique", "mode"]
DEFAULT_STAT_METRICS: list[str] = [
    "psi",
    "mean",
    "std",
    "min",
    "max",
    "p25",
    "median",
    "p75",
    "skew",
    "kurtosis",
]
DEFAULT_PROFILE_METRICS: list[str] = DEFAULT_DQ_METRICS + DEFAULT_STAT_METRICS
COMPARISON_METRICS: list[str] = ["schema", "unseen"]

ProfileBinMethod = Literal["quantile", "uniform"]
FrameDtype = Union[type, pl.DataType]


@dataclass(frozen=True)
class ProfileMetricSelection:
    """单次画像运行需要计算的指标集合。"""

    dq_metrics: list[str]
    stat_metrics: list[str]
    comparison_metrics: list[str]


@dataclass(frozen=True)
class ProfileRunContext:
    """单次画像运行上下文。"""

    df: pl.DataFrame
    working_df: pl.DataFrame
    features: list[str]
    dtype_map: dict[str, pl.DataType]
    group_col: str | None


@dataclass(frozen=True)
class ProfileComputeOptions:
    """画像计算底层策略参数。"""

    missing_values: list[Any]
    special_values: list[Any]
    psi_n_bins: int
    psi_bin_method: ProfileBinMethod
    psi_remove_empty_bins: bool
    psi_merge_small_bins: bool
    psi_min_bin_size: float
    psi_cv_ignore_threshold: float
    psi_batch_size: int
    overview_batch_size: int
    sparkline_bins: int
    sparkline_sample_size: int
    psi_include_missing: bool
    psi_include_special: bool
    categorical_features: list[str]
    diagnostics: list[dict[str, Any]]
