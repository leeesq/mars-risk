"""建模会话的特征增长调参辅助函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from mars.compute import FrameLike
from mars.modeling.contracts.feature_growth_result import MarsFeatureGrowthResult

if TYPE_CHECKING:
    from mars.modeling.workflows.session import MarsModelingSession


def session_tune_incrementally(
    session: MarsModelingSession,
    df: FrameLike,
    *,
    steps: Sequence[int] | None = None,
    feature_order: Sequence[str] | None = None,
    importance_table: pd.DataFrame | None = None,
    min_features: int = 10,
    max_features: int | None = None,
    step_size: int | None = None,
    mode: str = "prefix",
    selection_metric: str | None = None,
    **tune_kwargs: Any,
) -> MarsFeatureGrowthResult:
    """执行会话级特征增长调参，并回写最近一次结果。"""
    result = session.feature_growth_tuner.tune(
        df,
        steps=steps,
        feature_order=feature_order,
        importance_table=importance_table,
        min_features=min_features,
        max_features=max_features,
        step_size=step_size,
        mode=mode,
        selection_metric=selection_metric,
        **tune_kwargs,
    )
    session._last_feature_growth_run = result
    if result.best_run is not None:
        session.tuner.last_run = result.best_run
    return result
