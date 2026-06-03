"""建模工作流会话入口。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.feature_growth import MarsFeatureGrowthRun, MarsFeatureIncrementalTuner
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelingRun, MarsReplayRun
from mars.modeling.slicing import MarsModelDataSlicer
from mars.modeling.spec import SplitSpec
from mars.modeling.tuning import MarsModelReplay, MarsModelTuner
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
    benchmark_col : str, optional
        评估时默认使用的基准分数列。
    time_col : str, optional
        评估时默认使用的时间列。
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
        benchmark_col: str | None = None,
        time_col: str | None = None,
    ) -> None:
        self.tuner = MarsModelTuner(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )
        self.replay_runner = MarsModelReplay(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )
        self.feature_growth_tuner = MarsFeatureIncrementalTuner(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )
        self._last_feature_growth_run: MarsFeatureGrowthRun | None = None

    @property
    def last_run(self) -> MarsModelingRun | None:
        """返回当前会话最近一次调参结果。"""
        return self.tuner.last_run

    @property
    def best_model(self) -> Any:
        """Return the best model from the latest tuning run."""
        return self.tuner.best_model

    @property
    def best_score(self) -> float | None:
        """Return the best validation score from the latest tuning run."""
        return self.tuner.best_score

    @property
    def best_params(self) -> dict[str, Any] | None:
        """Return the best parameter set from the latest tuning run."""
        return self.tuner.best_params

    @property
    def history_table(self) -> pd.DataFrame:
        """Return the structured history table from the latest tuning run."""
        return self.tuner.history_table

    @property
    def last_feature_growth_run(self) -> MarsFeatureGrowthRun | None:
        """返回最近一次逐步增加特征调参结果。"""
        return self._last_feature_growth_run

    def slice(
        self,
        df: FrameLike,
        *,
        time_col: str,
        split_ratios: Mapping[str, float],
        label_col: str | None = None,
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
        label_col : str, optional
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
        """
        split_spec = SplitSpec(
            time_col=time_col,
            label_col=label_col or self.tuner.spec.target,
            mode=mode.lower(),
            train_key=train_key,
            val_key=val_key,
            random_seed=random_seed,
        )
        slicer = MarsModelDataSlicer(
            df=df,
            time_col=split_spec.time_col,
            label_col=split_spec.label_col,
            dataset_flag_col=self.tuner.spec.dataset_flag_col,
        )
        if split_spec.mode == "strict":
            return slicer.split_by_time_strictly(dict(split_ratios))
        if split_spec.mode == "hybrid":
            return slicer.split_hybrid_random_val(
                dict(split_ratios),
                train_key=split_spec.train_key,
                val_key=split_spec.val_key,
                random_seed=split_spec.random_seed,
            )
        raise ValueError(f"Unsupported slice mode: {mode!r}. Expected 'strict' or 'hybrid'.")

    def tune(self, df: FrameLike, **kwargs: Any) -> MarsModelingRun:
        """调用调参工具训练并返回结构化调参结果。"""
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
    ) -> MarsFeatureGrowthRun:
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
        MarsFeatureGrowthRun
            包含 step 汇总表、每个成功 step 的 tuning run 和推荐模型。
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
        val_target_col: str | None = None,
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
        val_target_col : str, optional
            可选校验标签列。
        feature_cols : sequence of str, optional
            用于计算特征 PSI 的特征列。
        importance_table : pandas.DataFrame, optional
            特征重要性表。

        Returns
        -------
        MarsModelingReport
            汇总指标、明细表和训练元数据。
        """
        run = self.last_run
        resolved_feature_cols = list(feature_cols) if feature_cols is not None else list(self.tuner.spec.features)
        resolved_importance = importance_table
        if resolved_importance is None and run is not None:
            resolved_importance = run.importance_table.copy()
        evaluator = MarsModelEvaluator(
            group_col=self.tuner.spec.dataset_flag_col,
            target_col=self.tuner.spec.target,
            benchmark_col=benchmark_col if benchmark_col is not None else self.tuner.spec.benchmark_col,
            time_col=time_col if time_col is not None else self.tuner.spec.time_col,
            val_target_col=val_target_col,
            feature_cols=resolved_feature_cols,
            importance_table=resolved_importance,
        )
        report = evaluator.evaluate(df, pred_col=pred_col)
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
        run: MarsModelingRun,
        df: FrameLike,
        **kwargs: Any,
    ) -> MarsReplayRun:
        """复用调参结果执行 Top-K replay、重训和重评分。"""
        return self.replay_runner.run(run, df, **kwargs)
