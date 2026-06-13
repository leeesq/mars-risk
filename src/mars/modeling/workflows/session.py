"""建模工作流会话入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from mars.compute import FrameLike
from mars.modeling.contracts.feature_growth_result import MarsFeatureGrowthResult
from mars.modeling.contracts.replay_result import MarsModelReplayResult
from mars.modeling.contracts.report import MarsModelingReport
from mars.modeling.contracts.tuning_result import MarsModelTuningResult
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection
from mars.modeling.workflows._session_evaluate_ops import session_evaluate
from mars.modeling.workflows._session_growth_ops import session_tune_incrementally
from mars.modeling.workflows._session_replay_ops import session_replay
from mars.modeling.workflows._session_slice_ops import session_slice
from mars.modeling.workflows._session_tune_ops import session_tune
from mars.modeling.workflows.feature_growth import MarsFeatureIncrementalTuner
from mars.modeling.workflows.replay import MarsModelReplayRunner
from mars.modeling.workflows.tuner import MarsModelTuner


class MarsModelingSession:
    """组织切分、调参、评估和 replay 的会话级门面。"""

    def __init__(
        self,
        *,
        model_type: str,
        features: Sequence[str],
        target: str,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Sequence[str] | None = None,
        optimize_metric: str = "ks",
        seed: int = 1206,
        lr_feature_mode: str = "numeric",
        lr_binning_type: str = "native",
        lr_binner_kwargs: Mapping[str, Any] | None = None,
        lr_binner: Any | None = None,
    ) -> None:
        """初始化建模会话。"""
        self.tuner = MarsModelTuner(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            lr_feature_mode=lr_feature_mode,
            lr_binning_type=lr_binning_type,
            lr_binner_kwargs=lr_binner_kwargs,
            lr_binner=lr_binner,
        )
        self.replay_runner = MarsModelReplayRunner()
        self.feature_growth_tuner = MarsFeatureIncrementalTuner(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            lr_feature_mode=lr_feature_mode,
            lr_binning_type=lr_binning_type,
            lr_binner_kwargs=lr_binner_kwargs,
            lr_binner=lr_binner,
        )
        self._last_feature_growth_run: MarsFeatureGrowthResult | None = None

    @property
    def last_run(self) -> MarsModelTuningResult | None:
        """返回最近一次调参结果。"""
        return self.tuner.last_run

    @property
    def best_model(self) -> Any:
        """返回最近一次调参得到的最优模型。"""
        return self.tuner.best_model

    @property
    def best_score(self) -> float | None:
        """返回最近一次调参的最优验证分数。"""
        return self.tuner.best_score

    @property
    def best_params(self) -> dict[str, Any] | None:
        """返回最近一次调参的最优参数。"""
        return self.tuner.best_params

    @property
    def history_table(self) -> pd.DataFrame:
        """返回最近一次调参 history 表。"""
        return self.tuner.history_table

    @property
    def last_feature_growth_run(self) -> MarsFeatureGrowthResult | None:
        """返回最近一次特征增长调参结果。"""
        return self._last_feature_growth_run

    def slice(
        self,
        df: FrameLike,
        *,
        time_col: str,
        split_ratios: Mapping[str, float],
        target: str | None = None,
        mode: str = "strict",
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """生成带 dataset flag 的切分样本。"""
        return session_slice(
            self,
            df,
            time_col=time_col,
            split_ratios=split_ratios,
            target=target,
            mode=mode,
            train_key=train_key,
            val_key=val_key,
            random_seed=random_seed,
        )

    def tune(
        self,
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
        """执行单次调参。"""
        return session_tune(
            self,
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

    def tune_incrementally(
        self,
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
        """按特征数量递增执行多轮调参。"""
        return session_tune_incrementally(
            self,
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

    def evaluate(
        self,
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
        """基于预测分生成模型评估报告。"""
        return session_evaluate(
            self,
            df,
            pred_col=pred_col,
            benchmark_col=benchmark_col,
            benchmark_cols=benchmark_cols,
            time_col=time_col,
            val_target=val_target,
            aux_targets=aux_targets,
            target_group_cols=target_group_cols,
            feature_cols=feature_cols,
            importance_table=importance_table,
            psi_include_missing=psi_include_missing,
        )

    def replay(
        self,
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
        """基于调参结果执行 replay、重训和重评估。"""
        return session_replay(
            self,
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
