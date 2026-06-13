"""建模会话的 replay 辅助函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from mars.compute import FrameLike
from mars.modeling.contracts.replay_result import MarsModelReplayResult
from mars.modeling.contracts.tuning_result import MarsModelTuningResult
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection

if TYPE_CHECKING:
    from mars.modeling.workflows.session import MarsModelingSession


def session_replay(
    session: MarsModelingSession,
    tuning_result: MarsModelTuningResult,
    df: FrameLike,
    *,
    top_k: int = 5,
    sort_metric: str = "ks",
    include_val: bool = True,
    trial_nums: Sequence[int] | None = None,
    retrain: bool = True,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    optimize_metric: str | None = None,
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
    metric_directions: Mapping[str, MetricDirection] | None = None,
    training_metric: str | None = None,
    backend_metric: Any | None = None,
    benchmark_col: str | None = None,
    benchmark_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    val_target: str | None = None,
    aux_targets: Sequence[str] | None = None,
    target_group_cols: Mapping[str, str] | None = None,
    psi_include_missing: bool = False,
) -> MarsModelReplayResult:
    """执行会话级 replay。"""
    return session.replay_runner.replay(
        tuning_result,
        df,
        top_k=top_k,
        sort_metric=sort_metric,
        include_val=include_val,
        trial_nums=trial_nums,
        retrain=retrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        optimize_metric=optimize_metric,
        metric_params=metric_params,
        custom_metrics=custom_metrics,
        metric_directions=metric_directions,
        training_metric=training_metric,
        backend_metric=backend_metric,
        benchmark_col=benchmark_col,
        benchmark_cols=benchmark_cols,
        time_col=time_col,
        val_target=val_target,
        aux_targets=aux_targets,
        target_group_cols=target_group_cols,
        psi_include_missing=psi_include_missing,
    )
