"""统计筛选器 fit 阶段内部流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import polars as pl

from mars.compute import RiskCorrBaseline, normalize_risk_corr_baseline


@dataclass(frozen=True)
class _StatsSelectorFitContext:
    """保存统计筛选单次 fit 的初始化结果。"""

    frame: pl.DataFrame
    candidate_features: list[str]
    current_features: list[str]
    valid_white_list: list[str]


class _StatsSelectorState(Protocol):
    """描述 fit 初始化逻辑需要读写的选择器状态。"""

    target: str | None
    features: list[str] | None
    feature_data_source: dict[str, list[str]]
    time_col: str | None
    profile_by: str | None
    white_list: list[str]
    black_list: list[str]
    max_samples: int | None
    feature_start_aware_reference: bool
    risk_corr_baseline: RiskCorrBaseline
    _funnel_stats: list[dict[str, Any]]
    _feature_iv_dict: dict[str, float]
    _feature_source_map: dict[str, str]

    def _ensure_polars_dataframe(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        """转换输入表为 Polars DataFrame。"""
        ...

    def _normalize_feature_data_source(self, features: list[str]) -> dict[str, str]:
        """规范化特征来源映射。"""
        ...

    def _record_funnel(
        self,
        stage: str,
        description: str,
        thresholds: dict[str, Any] | str,
        count_before: int,
        count_after: int,
    ) -> None:
        """记录筛选漏斗节点。"""
        ...


def _prepare_fit_context(
    selector: _StatsSelectorState,
    df: pl.DataFrame | pd.DataFrame,
    *,
    target: str,
    features: list[str] | None,
    feature_data_source: dict[str, list[str]] | None,
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
    white_list: list[str] | None,
    black_list: list[str] | None,
    max_samples: int | None,
    feature_start_aware_reference: bool | None,
    risk_corr_baseline: RiskCorrBaseline | None,
) -> _StatsSelectorFitContext:
    """初始化单次筛选运行状态并应用静态黑名单。"""
    selector.target = target
    selector.features = features
    selector.feature_data_source = feature_data_source or {}
    selector.time_col = time_col
    selector.profile_by = (time_grain or "month") if time_col else group_col
    selector.white_list = white_list if white_list else []
    selector.black_list = black_list if black_list else []
    selector.max_samples = max_samples
    selector.feature_start_aware_reference = (
        selector.feature_start_aware_reference
        if feature_start_aware_reference is None
        else bool(feature_start_aware_reference)
    )
    selector.risk_corr_baseline = normalize_risk_corr_baseline(
        risk_corr_baseline or selector.risk_corr_baseline,
    )

    frame = selector._ensure_polars_dataframe(df)
    selector._funnel_stats = []
    selector._feature_iv_dict = {}

    exclude_cols = {selector.target}
    if selector.time_col:
        exclude_cols.add(selector.time_col)
    if selector.profile_by:
        exclude_cols.add(selector.profile_by)

    source_features = selector.features if selector.features else frame.columns
    candidate_features = [
        col for col in source_features if col in frame.columns and col not in exclude_cols
    ]
    selector._feature_source_map = selector._normalize_feature_data_source(candidate_features)
    valid_white_list = [
        feature for feature in selector.white_list if feature in candidate_features
    ]
    current_features = [
        feature for feature in candidate_features if feature not in selector.black_list
    ]
    selector._record_funnel(
        "Init",
        "Blacklist & Exclusions",
        {"black_list_len": len(selector.black_list)},
        len(candidate_features),
        len(current_features),
    )
    return _StatsSelectorFitContext(
        frame=frame,
        candidate_features=candidate_features,
        current_features=current_features,
        valid_white_list=valid_white_list,
    )


def _force_white_list_features(
    current_features: list[str],
    *,
    valid_white_list: list[str],
) -> list[str]:
    """把仍存在于候选空间的白名单特征强制并入最终结果。"""
    selected_features = list(current_features)
    selected_set = set(selected_features)
    for feature in valid_white_list:
        if feature not in selected_set:
            selected_features.append(feature)
            selected_set.add(feature)
    return selected_features
