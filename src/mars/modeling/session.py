"""建模工作流会话入口。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.feature_growth import MarsFeatureGrowthResult, MarsFeatureIncrementalTuner
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelReplayResult, MarsModelTuningResult
from mars.modeling.slicing import MarsModelDataSplitter
from mars.modeling.spec import SplitSpec
from mars.modeling.tuning import MarsModelReplayRunner, MarsModelTuner
from mars.modeling.utils import FrameLike


class MarsModelingSession:
    """
    组织切分、调参、评估和 replay 的会话级入口。

    Parameters
    ----------
    model_type : {"xgb", "lgb", "cbt", "cat", "catboost"}
        底层模型后端类型。
    features : sequence of str
        参与训练和预测的特征列名。
    target : str
        二分类目标变量列名。
    dataset_flag_col : str, default "dataset_flag"
        数据集切片标识列，按包含 train/val/oot 的规则识别角色。
    categorical_features : sequence of str, optional
        需要按类别特征处理的列名。
    optimize_metric : {"auc", "ks"}, default "ks"
        调参和 replay 使用的优化指标。
    seed : int, default 1206
        随机种子。

    Attributes
    ----------
    tuner : MarsModelTuner
        单次调参入口。
    replay_runner : MarsModelReplayRunner
        Top-K replay 入口。
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
        df : pandas.DataFrame or polars.DataFrame
            原始建模样本。
        time_col : str
            时间列名。
        split_ratios : mapping of str to float
            数据集切分比例，合计必须为 1。
        target : str, optional
            标签列；默认使用 session 的 target。
        mode : {"strict", "hybrid"}, default "strict"
            时间严格切分或建模窗口内随机 validation 切分。
        train_key : str, default "train"
            hybrid 模式训练集标识。
        val_key : str, default "val"
            hybrid 模式验证集标识。
        random_seed : int, default 42
            hybrid 模式随机种子。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致、已追加 dataset flag 的数据框。

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

    def tune(self, df: FrameLike, **kwargs: Any) -> MarsModelTuningResult:
        """
        调用调参工具训练并返回结构化调参结果。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            已带 train/val/OOT 标识的建模样本。
        **kwargs : Any
            透传给 ``MarsModelTuner.tune`` 的调参参数。

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
        return self.tuner.tune(df, **kwargs)

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
        df : pandas.DataFrame or polars.DataFrame
            已带 train/val/OOT 标识的建模样本。
        steps : sequence of int, optional
            显式指定每轮使用的前 N 个特征数量。
        feature_order : sequence of str, optional
            人工指定的稳定特征顺序。
        importance_table : pandas.DataFrame, optional
            特征重要性表；若提供且未指定 ``feature_order``，按重要性或 rank 排序。
        min_features : int, default 10
            自动生成 step 时的起始特征数。
        max_features : int, optional
            自动生成 step 时的最大特征数。
        step_size : int, optional
            自动生成 step 时的步长。
        mode : {"prefix"}, default "prefix"
            特征增长模式。当前版本只支持前缀扩展。
        selection_metric : {"auc", "ks"}, optional
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
        time_col: str | None = None,
        val_target: str | None = None,
        feature_cols: Sequence[str] | None = None,
        importance_table: pd.DataFrame | None = None,
    ) -> MarsModelingReport:
        """
        基于预测分数生成模型评估报告。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            已包含预测分数的数据框。
        pred_col : str
            预测分数列名。
        benchmark_col : str, optional
            覆盖会话默认基准分数列。
        time_col : str, optional
            覆盖会话默认时间列。
        val_target : str, optional
            可选校验标签列。
        feature_cols : sequence of str, optional
            用于计算特征 PSI 的特征列。
        importance_table : pandas.DataFrame, optional
            特征重要性表。

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
            time_col=time_col,
            val_target=val_target,
            feature_cols=resolved_feature_cols,
            importance_table=resolved_importance,
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
        **kwargs: Any,
    ) -> MarsModelReplayResult:
        """
        复用调参结果执行 Top-K replay、重训和重评分。

        Parameters
        ----------
        tuning_result : MarsModelTuningResult
            调参阶段产出的结果对象。
        df : pandas.DataFrame or polars.DataFrame
            需要重训和评分的数据。
        **kwargs : Any
            透传给 ``MarsModelReplayRunner.run`` 的 replay 参数。

        Returns
        -------
        MarsModelReplayResult
            replay 排名、leaderboard、模型、评分数据和报告。

        Examples
        --------
        >>> session = MarsModelingSession(model_type="xgb", features=["age"], target="y")
        >>> callable(session.replay)
        True
        """
        return self.replay_runner.run(tuning_result, df, **kwargs)
