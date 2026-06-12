"""建模工作流会话入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.feature_growth import MarsFeatureGrowthResult, MarsFeatureIncrementalTuner
from mars.modeling.metrics import MetricCallable, MetricDirection
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelReplayResult, MarsModelTuningResult
from mars.modeling.slicing import MarsModelDataSplitter
from mars.modeling.spec import SplitSpec
from mars.modeling.tuning import MarsModelReplayRunner, MarsModelTuner
from mars.utils.frame import FrameLike


class MarsModelingSession:
    """
    组织切分、调参、评估和 replay 的会话级入口。

    Attributes
    ----------
    tuner : MarsModelTuner
        单次调参入口。
    replay_runner : MarsModelReplayRunner
        Top-K 或指定 trial replay 入口。
    feature_growth_tuner : MarsFeatureIncrementalTuner
        逐步增加特征调参入口。
    last_feature_growth_run : MarsFeatureGrowthResult or None
        最近一次特征增长调参结果。

    Examples
    --------
    >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
    >>> session.tuner.spec.features
    ['age']
    """

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
        """
        初始化建模会话。

        Parameters
        ----------
        model_type : str
            模型后端类型。
        features : Sequence[str]
            建模特征列。
        target : str
            目标列名。
        dataset_flag_col : str
            样本切片标记列名。
        categorical_features : Sequence[str] | None
            类别特征列。
        optimize_metric : str
            调参优化指标。
        seed : int
            随机种子。
        lr_feature_mode : str
            LR 特征模式。
        lr_binning_type : str
            LR WOE 模式使用的分箱器类型，支持 ``native``、``optimal`` 和 ``lite_opt``。
        lr_binner_kwargs : Mapping[str, Any] | None
            构造 LR 分箱器时使用的参数。
        lr_binner : Any | None
            显式复用的 LR 分箱器实例。
        """
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
        """
        返回当前会话最近一次调参结果。

        Returns
        -------
        MarsModelTuningResult or None
            最近一次调参结果；若尚未运行调参，则返回 ``None``。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.last_run is None
        True
        """
        return self.tuner.last_run

    @property
    def best_model(self) -> Any:
        """
        返回最近一次调参运行中的最佳模型。

        Returns
        -------
        Any
            最近一次调参运行中的最佳模型；若尚无调参结果，则返回 ``None``。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.best_model is None
        True
        """
        return self.tuner.best_model

    @property
    def best_score(self) -> float | None:
        """
        返回最近一次调参运行中的最佳验证集分数。

        Returns
        -------
        float or None
            最近一次调参运行的最佳验证集分数；若尚无调参结果，则返回 ``None``。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.best_score is None
        True
        """
        return self.tuner.best_score

    @property
    def best_params(self) -> dict[str, Any] | None:
        """
        返回最近一次调参运行中的最佳参数集合。

        Returns
        -------
        dict of str to Any or None
            最近一次调参运行的最佳参数副本；若尚无调参结果，则返回 ``None``。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.best_params is None
        True
        """
        return self.tuner.best_params

    @property
    def history_table(self) -> pd.DataFrame:
        """
        返回最近一次调参运行的结构化 Trial 历史表。

        Returns
        -------
        pandas.DataFrame
            Trial 历史表；若尚无调参结果，则返回空表。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.history_table.empty
        True
        """
        return self.tuner.history_table

    @property
    def last_feature_growth_run(self) -> MarsFeatureGrowthResult | None:
        """
        返回最近一次逐步增加特征调参结果。

        Returns
        -------
        MarsFeatureGrowthResult or None
            最近一次特征增长调参结果；若尚未运行，则返回 ``None``。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> session.last_feature_growth_run is None
        True
        """
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
        """
        使用会话配置切分原始建模样本。

        Parameters
        ----------
        df : FrameLike
            原始建模样本。
        time_col : str
            时间列名。
        split_ratios : Mapping[str, float]
            数据集切分比例，合计必须为 1。
        target : str | None
            标签列；默认使用 session 的 target。
        mode : str
            时间严格切分或建模窗口内随机 validation 切分。
        train_key : str
            hybrid 模式训练集标识。
        val_key : str
            hybrid 模式验证集标识。
        random_seed : int
            hybrid 模式随机种子。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致、已追加 dataset flag 的数据框。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"apply_dt": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], "y": [0, 1, 0, 1]}
        ... )
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> out = session.slice(df, time_col="apply_dt", split_ratios={"train": 0.5, "val": 0.5})
        >>> "dataset_flag" in out.columns
        True
        """
        split_spec = SplitSpec(
            time_col=time_col,
            label_col=target or self.tuner.spec.target,
            mode=mode.lower(),
            train_key=train_key,
            val_key=val_key,
            random_seed=random_seed,
        )
        slicer = MarsModelDataSplitter()
        if split_spec.mode == "strict":
            return slicer.split_by_time_strictly(
                df,
                time_col=split_spec.time_col,
                target=split_spec.label_col,
                split_ratios=dict(split_ratios),
                dataset_flag_col=self.tuner.spec.dataset_flag_col,
            )
        if split_spec.mode == "hybrid":
            return slicer.split_hybrid_random_val(
                df,
                time_col=split_spec.time_col,
                target=split_spec.label_col,
                split_ratios=dict(split_ratios),
                dataset_flag_col=self.tuner.spec.dataset_flag_col,
                train_key=split_spec.train_key,
                val_key=split_spec.val_key,
                random_seed=split_spec.random_seed,
            )
        raise ValueError(f"Unsupported slice mode: {mode!r}. Expected 'strict' or 'hybrid'.")

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
        """
        调用调参工具训练并返回结构化调参结果。

        Parameters
        ----------
        df : FrameLike
            已带 train/val/OOT 标识的建模样本。
        param_space : Mapping[str, Any] | None
            后端搜索空间覆盖或扩展。
        max_diff : float
            泛化衰减阈值。
        use_oot_penalty : bool
            是否将 OOT 衰减纳入 trial 有效性约束。
        n_trials : int
            Optuna trial 数量。
        startup_trials : int
            剪枝开始前的预热 trial 数量。
        warmup_steps : int
            剪枝器预热步数。
        num_boost_round : int
            最大 boosting 轮数。
        early_stopping_rounds : int
            early stopping 轮数。
        metric_params : Mapping[str, Any] | None
            指标参数，例如 ``f1_threshold``。
        custom_metrics : Mapping[str, MetricCallable] | None
            用户自定义指标函数字典。
        metric_directions : Mapping[str, MetricDirection] | None
            指标排序方向。
        training_metric : str | None
            后端训练期监控指标。
        backend_metric : Any | None
            透传给模型后端的原生自定义 metric。
        keep_top_n_models : int
            调参阶段动态保留的最优模型数量。
        artifact_dir : str | Path | None
            调参产物根目录；``None`` 表示不落盘。
        importance_methods : Sequence[Literal["native", "shap"]]
            特征重要性计算方式。
        shap_sample_size : int
            计算 SHAP values 的最大样本量。
        shap_background_size : int
            SHAP 背景样本量。
        overwrite : bool
            保留参数，当前独立运行目录不会覆盖旧产物。

        Returns
        -------
        MarsModelTuningResult
            单次调参结果。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> callable(session.tune)
        True
        """
        return self.tuner.tune(
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

    def incremental_tune(
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
        """
        按特征数量逐步扩展并执行多轮调参。

        Parameters
        ----------
        df : FrameLike
            已带 train/val/OOT 标识的建模样本。
        steps : Sequence[int] | None
            显式指定每轮使用的前 N 个特征数量。
        feature_order : Sequence[str] | None
            人工指定的稳定特征顺序。
        importance_table : pd.DataFrame | None
            特征重要性表；若提供且未指定 ``feature_order``，按重要性或 rank 排序。
        min_features : int
            自动生成 step 时的起始特征数。
        max_features : int | None
            自动生成 step 时的最大特征数。
        step_size : int | None
            自动生成 step 时的步长。
        mode : str
            特征增长模式。当前版本只支持前缀扩展。
        selection_metric : str | None
            跨 step 选择推荐模型时使用的 validation 指标。
        **tune_kwargs : Any
            透传给 ``MarsModelTuner.tune`` 的参数。

        Returns
        -------
        MarsFeatureGrowthResult
            包含 step 汇总表、每个成功 step 的 tuning run 和推荐模型。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> callable(session.incremental_tune)
        True
        """
        result = self.feature_growth_tuner.tune(
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
        self._last_feature_growth_run = result
        if result.best_run is not None:
            self.tuner.last_run = result.best_run
        return result

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
        """
        基于预测分数生成模型评估报告。

        Parameters
        ----------
        df : FrameLike
            已包含预测分数的数据框。
        pred_col : str
            预测分数列名。
        benchmark_col : str | None
            覆盖会话默认基准分数列。
        benchmark_cols : Sequence[str] | None
            多个 benchmark 分数列。
        time_col : str | None
            覆盖会话默认时间列。
        val_target : str | None
            可选校验标签列。
        aux_targets : Sequence[str] | None
            多个辅助验证标签列。
        target_group_cols : Mapping[str, str] | None
            target 到独立切片列名的映射。
        feature_cols : Sequence[str] | None
            用于计算特征 PSI 的特征列。
        importance_table : pd.DataFrame | None
            特征重要性表。
        psi_include_missing : bool
            计算 `score_psi` 和 `feature_psi` 时是否纳入缺失值箱。

        Returns
        -------
        MarsModelingReport
            汇总指标、明细表和训练元数据。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> callable(session.evaluate)
        True
        """
        run = self.last_run
        resolved_feature_cols = list(feature_cols) if feature_cols is not None else list(self.tuner.spec.features)
        resolved_importance = importance_table
        if resolved_importance is None and run is not None:
            resolved_importance = run.importance_table.copy()
        evaluator = MarsModelEvaluator()
        report = evaluator.evaluate(
            df,
            pred_col=pred_col,
            group_col=self.tuner.spec.dataset_flag_col,
            target=self.tuner.spec.target,
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
                    "importance_table": resolved_importance.copy() if resolved_importance is not None else run.importance_table.copy(),
                    "training_config": dict(run.training_config),
                    "library_versions": dict(run.library_versions),
                    "backend_data_mode": run.backend_data_mode,
                    "model_type": run.model_type,
                    "optimize_metric": run.optimize_metric,
                    "best_score": run.best_score,
                    "best_iteration": run.best_iteration,
                }
            )
        if self._last_feature_growth_run is not None:
            report.metadata.update(
                {
                    "feature_growth_summary": self._last_feature_growth_run.summary_table.copy(),
                    "feature_growth_steps": list(self._last_feature_growth_run.steps),
                    "feature_growth_best_step": self._last_feature_growth_run.best_step,
                    "feature_growth_selection_metric": self._last_feature_growth_run.selection_metric,
                    "feature_growth_metadata": dict(self._last_feature_growth_run.metadata),
                }
            )
        return report

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
        """
        复用调参结果执行 Top-K 或指定 trial replay、重训和重评分。

        Parameters
        ----------
        tuning_result : MarsModelTuningResult
            调参阶段产出的结果对象。
        df : FrameLike
            需要重训和评分的数据。
        top_k : int
            未指定 ``trial_nums`` 时回放的 Top-K 数量。
        sort_metric : str
            选择 Top-K 的排序指标。
        include_val : bool
            排序时是否纳入 validation 指标。
        trial_nums : Sequence[int] | None
            显式指定要回放的 trial 编号。
        retrain : bool
            是否按 trial 参数重新训练；``False`` 时只使用已保留模型。
        num_boost_round : int
            重训使用的最大 boosting 轮数。
        early_stopping_rounds : int
            重训使用的 early stopping 轮数。
        optimize_metric : str | None
            覆盖 replay 后端优化指标。
        metric_params : Mapping[str, Any] | None
            指标参数。
        custom_metrics : Mapping[str, MetricCallable] | None
            用户自定义指标函数字典。
        metric_directions : Mapping[str, MetricDirection] | None
            指标排序方向。
        training_metric : str | None
            后端训练期监控指标。
        backend_metric : Any | None
            透传给模型后端的原生自定义 metric。
        benchmark_col : str | None
            单个 benchmark 分数列。
        benchmark_cols : Sequence[str] | None
            多个 benchmark 分数列。
        time_col : str | None
            原始时间列。
        val_target : str | None
            单个辅助验证目标。
        aux_targets : Sequence[str] | None
            多个辅助验证目标。
        target_group_cols : Mapping[str, str] | None
            target 到独立切片列名的映射。
        psi_include_missing : bool
            replay 评估报告计算 `score_psi` 和 `feature_psi` 时是否纳入缺失值箱。

        Returns
        -------
        MarsModelReplayResult
            replay 排名表、模型、评分数据和报告。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> callable(session.replay)
        True
        """
        return self.replay_runner.run(
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
