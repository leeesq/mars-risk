"""建模会话的评估辅助函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

from mars.compute import FrameLike
from mars.modeling.contracts.report import MarsModelingReport
from mars.modeling.evaluation import MarsModelEvaluator

if TYPE_CHECKING:
    from mars.modeling.workflows.session import MarsModelingSession


def session_evaluate(
    session: MarsModelingSession,
    df: FrameLike,
    *,
    pred_col: str,
    benchmark_col: str | None = None,
    benchmark_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    val_target: str | None = None,
    aux_targets: Sequence[str] | None = None,
    target_group_cols: Mapping[str, str] | None = None,
    feature_cols: Sequence[str] | None = None,
    importance_table: pd.DataFrame | None = None,
    psi_include_missing: bool = False,
) -> MarsModelingReport:
    """基于会话默认规格生成模型评估报告。"""
    run = session.last_run
    resolved_feature_cols = (
        list(feature_cols)
        if feature_cols is not None
        else list(session.tuner.spec.features)
    )
    resolved_importance = importance_table
    if resolved_importance is None and run is not None:
        resolved_importance = run.importance_table.copy()
    evaluator = MarsModelEvaluator()
    report = evaluator.evaluate(
        df,
        pred_col=pred_col,
        group_col=session.tuner.spec.dataset_flag_col,
        target=session.tuner.spec.target,
        benchmark_col=benchmark_col,
        benchmark_cols=benchmark_cols,
        time_col=time_col,
        val_target=val_target,
        aux_targets=aux_targets,
        target_group_cols=target_group_cols,
        feature_cols=resolved_feature_cols,
        importance_table=resolved_importance,
        psi_include_missing=psi_include_missing,
    )
    if run is not None:
        report.metadata.update(
            {
                "history_table": run.history_table.copy(),
                "importance_table": (
                    resolved_importance.copy()
                    if resolved_importance is not None
                    else run.importance_table.copy()
                ),
                "training_config": dict(run.training_config),
                "library_versions": dict(run.library_versions),
                "backend_data_mode": run.backend_data_mode,
                "model_type": run.model_type,
                "optimize_metric": run.optimize_metric,
                "best_score": run.best_score,
                "best_iteration": run.best_iteration,
            }
        )
    if session._last_feature_growth_run is not None:
        report.metadata.update(
            {
                "feature_growth_summary": session._last_feature_growth_run.summary_table.copy(),
                "feature_growth_steps": list(session._last_feature_growth_run.steps),
                "feature_growth_best_step": session._last_feature_growth_run.best_step,
                "feature_growth_selection_metric": session._last_feature_growth_run.selection_metric,
                "feature_growth_metadata": dict(session._last_feature_growth_run.metadata),
            }
        )
    return report
