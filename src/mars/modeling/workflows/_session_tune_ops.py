"""建模会话的调参与 replay 辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from mars.compute import FrameLike
from mars.modeling.contracts.tuning_result import MarsModelTuningResult
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection

if TYPE_CHECKING:
    from mars.modeling.workflows.session import MarsModelingSession


def session_tune(
    session: MarsModelingSession,
    df: FrameLike,
    *,
    param_space: Mapping[str, Any] | None = None,
    max_diff: float = 3.0,
    use_oot_penalty: bool = False,
    n_trials: int = 50,
    startup_trials: int = 20,
    warmup_steps: int = 100,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
    metric_directions: Mapping[str, MetricDirection] | None = None,
    training_metric: str | None = None,
    backend_metric: Any | None = None,
    keep_top_n_models: int = 5,
    artifact_dir: str | Path | None = "modeling_artifacts",
    importance_methods: Sequence[Literal["native", "shap"]] = ("native",),
    shap_sample_size: int = 5000,
    shap_background_size: int = 1000,
    overwrite: bool = False,
) -> MarsModelTuningResult:
    """执行会话级单次调参。"""
    return session.tuner.tune(
        df,
        param_space=param_space,
        max_diff=max_diff,
        use_oot_penalty=use_oot_penalty,
        n_trials=n_trials,
        startup_trials=startup_trials,
        warmup_steps=warmup_steps,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        metric_params=metric_params,
        custom_metrics=custom_metrics,
        metric_directions=metric_directions,
        training_metric=training_metric,
        backend_metric=backend_metric,
        keep_top_n_models=keep_top_n_models,
        artifact_dir=artifact_dir,
        importance_methods=importance_methods,
        shap_sample_size=shap_sample_size,
        shap_background_size=shap_background_size,
        overwrite=overwrite,
    )
