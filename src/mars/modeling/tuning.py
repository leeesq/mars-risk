"""建模调参与 Top-K replay 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Type

import pandas as pd

from mars.modeling.backends import (
    MarsCatBoostStrategy,
    MarsLGBStrategy,
    MarsLogisticRegressionStrategy,
    MarsXGBStrategy,
)
from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.prediction import ModelPredictor
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelReplayResult, MarsModelTuningResult
from mars.modeling.spec import ModelingSpec, ReplaySpec
from mars.modeling.utils import FrameLike, collect_library_versions

BACKEND_MAP: Dict[str, Type[Any]] = {
    "xgb": MarsXGBStrategy,
    "lgb": MarsLGBStrategy,
    "cbt": MarsCatBoostStrategy,
    "cat": MarsCatBoostStrategy,
    "catboost": MarsCatBoostStrategy,
    "lr": MarsLogisticRegressionStrategy,
    "logit": MarsLogisticRegressionStrategy,
    "logistic": MarsLogisticRegressionStrategy,
    "logistic_regression": MarsLogisticRegressionStrategy,
}


def _build_spec(
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
) -> ModelingSpec:
    """
    校验建模配置并构造共享规格对象。

    Parameters
    ----------
    model_type : str
        模型后端类型。
    features : sequence of str
        特征列名。
    target : str
        目标列名。

    Returns
    -------
    ModelingSpec
        标准化后的建模配置。
    """
    spec = ModelingSpec(
        model_type=model_type.lower(),
        features=list(features),
        target=target,
        dataset_flag_col=dataset_flag_col,
        categorical_features=list(categorical_features or []),
        optimize_metric=optimize_metric.lower(),
        seed=int(seed),
        lr_feature_mode=str(lr_feature_mode).lower(),
        lr_binning_type=str(lr_binning_type).lower(),
        lr_binner_kwargs=dict(lr_binner_kwargs or {}),
        lr_binner=lr_binner,
    )
    if spec.model_type not in BACKEND_MAP:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. Expected one of {sorted(BACKEND_MAP)}."
        )
    if spec.optimize_metric not in {"auc", "ks"}:
        raise ValueError(
            f"Unsupported optimize_metric: {optimize_metric!r}. Expected one of ['auc', 'ks']."
        )
    if spec.lr_feature_mode not in {"numeric", "woe"}:
        raise ValueError("lr_feature_mode must be one of {'numeric', 'woe'}.")
    if spec.lr_binning_type not in {"native", "opt", "optimal"}:
        raise ValueError("lr_binning_type must be one of {'native', 'opt', 'optimal'}.")
    return spec


def _build_backend_from_spec(
    spec: ModelingSpec,
    df: FrameLike,
    *,
    param_space: Mapping[str, Any] | None = None,
    max_diff: float = 3.0,
    use_oot_penalty: bool = False,
    optimize_metric: str | None = None,
    seed: int | None = None,
) -> Any:
    """根据建模配置创建具体后端策略实例。"""
    backend_cls = BACKEND_MAP[spec.model_type]
    backend_kwargs: Dict[str, Any] = {
        "df": df,
        "features": spec.features,
        "target": spec.target,
        "optimize_metric": (optimize_metric or spec.optimize_metric).lower(),
        "param_space": param_space,
        "max_diff": max_diff,
        "seed": spec.seed if seed is None else int(seed),
        "use_oot_penalty": use_oot_penalty,
        "dataset_flag_col": spec.dataset_flag_col,
        "categorical_features": spec.categorical_features,
    }
    if backend_cls is MarsLogisticRegressionStrategy:
        backend_kwargs.update(
            {
                "lr_feature_mode": spec.lr_feature_mode,
                "lr_binning_type": spec.lr_binning_type,
                "lr_binner_kwargs": spec.lr_binner_kwargs,
                "lr_binner": spec.lr_binner,
            }
        )
    return backend_cls(**backend_kwargs)


class MarsModelTuner:
    """
    二分类风险模型调参工具。

    Parameters
    ----------
    model_type : str
        模型后端类型。
    features : sequence of str
        参与训练的特征列名。
    target : str
        目标变量列名。
    dataset_flag_col : str, default "dataset_flag"
        数据集切片标识列。
    categorical_features : sequence of str, optional
        需要按类别特征处理的列名。
    optimize_metric : {"auc", "ks"}, default "ks"
        trial 最终优化指标。
    seed : int, default 1206
        随机种子。

    Attributes
    ----------
    spec : ModelingSpec
        当前调参任务的建模规格。
    last_run : MarsModelTuningResult or None
        最近一次调参结果。

    Examples
    --------
    >>> tuner = MarsModelTuner(model_type="xgb", features=["age"], target="y")
    >>> tuner.spec.optimize_metric
    'ks'
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
        self.spec: ModelingSpec = _build_spec(
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
        self.last_run: MarsModelTuningResult | None = None

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
        >>> tuner = MarsModelTuner(model_type="xgb", features=["age"], target="y")
        >>> tuner.best_model is None
        True
        """
        return None if self.last_run is None else self.last_run.best_model

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
        >>> tuner = MarsModelTuner(model_type="xgb", features=["age"], target="y")
        >>> tuner.best_score is None
        True
        """
        return None if self.last_run is None else self.last_run.best_score

    @property
    def best_params(self) -> Dict[str, Any] | None:
        """
        返回最近一次调参运行中的最佳参数集合。

        Returns
        -------
        dict of str to Any or None
            最近一次调参运行的最佳参数副本；若尚无调参结果，则返回 ``None``。

        Examples
        --------
        >>> tuner = MarsModelTuner(model_type="xgb", features=["age"], target="y")
        >>> tuner.best_params is None
        True
        """
        if self.last_run is None:
            return None
        return dict(self.last_run.best_params)

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
        >>> tuner = MarsModelTuner(model_type="xgb", features=["age"], target="y")
        >>> tuner.history_table.empty
        True
        """
        if self.last_run is None:
            return pd.DataFrame()
        return self.last_run.history_table.copy()

    def _build_backend(
        self,
        df: FrameLike,
        *,
        param_space: Mapping[str, Any] | None = None,
        max_diff: float = 3.0,
        use_oot_penalty: bool = False,
        optimize_metric: str | None = None,
        seed: int | None = None,
    ) -> Any:
        """为单次调参或 replay 任务构建具体后端策略。"""
        return _build_backend_from_spec(
            self.spec,
            df,
            param_space=param_space,
            max_diff=max_diff,
            use_oot_penalty=use_oot_penalty,
            optimize_metric=optimize_metric,
            seed=seed,
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
        history_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> MarsModelTuningResult:
        """
        调优一个模型后端并返回可复用的建模调优结果。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            已经带有 train、validation、OOT 切片标记的建模样本。
        param_space : mapping, optional
            对后端搜索空间的覆盖或扩展。
        max_diff : float, default 3.0
            泛化衰减阈值，单位是百分点。
        use_oot_penalty : bool, default False
            是否将 OOT 衰减纳入 trial 有效性判断。
        n_trials : int, default 50
            Optuna 试验次数。
        startup_trials : int, default 20
            剪枝器开始工作前的预热试验次数。
        warmup_steps : int, default 100
            剪枝器预热步数。
        num_boost_round : int, default 500
            最大 boosting 轮数。
        early_stopping_rounds : int, default 50
            early stopping 轮数。
        history_path : str or pathlib.Path, optional
            trial 历史记录 CSV 路径；`None` 表示只保存在内存中，不落盘。
        overwrite : bool, default False
            当 `history_path` 已存在时，是否允许覆盖。

        Returns
        -------
        MarsModelTuningResult
            包含最佳模型、调参历史、训练配置和元数据的建模调优结果。
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for MarsModelTuner.tune. "
                "Install the optional extra with `pip install \"mars-risk[tuning]\"`."
            ) from exc

        backend = self._build_backend(
            df,
            param_space=param_space,
            max_diff=max_diff,
            use_oot_penalty=use_oot_penalty,
        )

        resolved_history_path: Path | None = None
        if history_path is not None:
            resolved_history_path = Path(history_path)
            if resolved_history_path.exists() and not overwrite:
                raise FileExistsError(
                    f"history_path already exists: {resolved_history_path}. "
                    "Pass overwrite=True to replace it."
                )
            if resolved_history_path.exists():
                resolved_history_path.unlink()

        backend.num_boost_round = int(num_boost_round)
        backend.early_stopping_rounds = int(early_stopping_rounds)
        backend.training_metric = backend.optimize_metric

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=backend.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=startup_trials,
                n_warmup_steps=warmup_steps,
            ),
        )
        study.optimize(
            lambda trial: backend.objective(trial, startup_trials, resolved_history_path),
            n_trials=n_trials,
        )

        if backend.best_model is None:
            raise RuntimeError("No valid trial satisfied the generalization constraints.")

        history_table = backend.build_history_table()
        best_trial_num = int(study.best_trial.number)
        best_trial_rows = history_table.loc[history_table["trial_num"] == best_trial_num]
        if best_trial_rows.empty:
            raise RuntimeError(f"Could not locate the best trial record for trial_num={best_trial_num}.")

        best_trial_row = best_trial_rows.iloc[-1]
        best_params = {
            key: best_trial_row[key]
            for key in backend.replay_param_keys
            if key in best_trial_row.index and pd.notna(best_trial_row[key])
        }
        training_config = {
            "n_trials": int(n_trials),
            "startup_trials": int(startup_trials),
            "warmup_steps": int(warmup_steps),
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": int(early_stopping_rounds),
            "max_diff": float(max_diff),
            "use_oot_penalty": bool(use_oot_penalty),
            "param_space": dict(param_space or {}),
            "training_metric": backend.training_metric,
            "history_path": str(resolved_history_path.resolve()) if resolved_history_path else None,
            "seed": int(backend.seed),
        }
        if isinstance(backend, MarsLogisticRegressionStrategy):
            training_config.update(
                {
                    "lr_feature_mode": self.spec.lr_feature_mode,
                    "lr_binning_type": self.spec.lr_binning_type,
                    "lr_binner_kwargs": dict(self.spec.lr_binner_kwargs),
                }
            )
        diagnostic_tables: Dict[str, pd.DataFrame] = {}
        extract_diagnostics = getattr(backend, "extract_diagnostics", None)
        if callable(extract_diagnostics):
            diagnostic_tables = extract_diagnostics(backend.best_model)
        result = MarsModelTuningResult(
            model_type=self.spec.model_type,
            optimize_metric=backend.optimize_metric,
            features=list(self.spec.features),
            target=self.spec.target,
            dataset_flag_col=self.spec.dataset_flag_col,
            categorical_features=list(self.spec.categorical_features),
            best_params=best_params,
            best_iteration=backend.get_best_iteration(backend.best_model),
            best_model=backend.best_model,
            best_score=backend.best_score,
            history_table=history_table.copy(),
            history_path=str(resolved_history_path.resolve()) if resolved_history_path else None,
            study=study,
            replay_candidates=list(backend.replay_param_keys),
            importance_table=backend.extract_importance(backend.best_model),
            diagnostic_tables=diagnostic_tables,
            training_config=training_config,
            library_versions=collect_library_versions(
                "polars",
                "pandas",
                "pyarrow",
                "xgboost",
                "lightgbm",
                "catboost",
                "optuna",
                "sklearn",
                "statsmodels",
            ),
            feature_schema=dict(backend.feature_schema),
            backend_data_mode=backend.backend_data_mode,
            category_levels=dict(getattr(backend, "category_levels", {})),
        )
        self.last_run = result
        return result


class MarsModelReplayRunner:
    """
    基于 `MarsModelTuningResult` 回放 Top-K 调参结果。

    `MarsModelReplayRunner` 不在构造函数中绑定模型类型、特征列或目标列，而是从
    :meth:`run` 传入的调优结果中读取建模规格。benchmark 分数、时间列和替代验证目标
    属于本次 replay 评估上下文，因此保留在方法入参中。

    Examples
    --------
    >>> replay = MarsModelReplayRunner()
    >>> callable(replay.run)
    True
    """

    def __init__(self) -> None:
        self.spec: ModelingSpec | None = None

    @staticmethod
    def _build_spec_from_result(tuning_result: MarsModelTuningResult) -> ModelingSpec:
        """从调优结果恢复 replay 所需的建模规格。"""
        training_config = dict(getattr(tuning_result, "training_config", {}) or {})
        return _build_spec(
            model_type=tuning_result.model_type,
            features=tuning_result.features,
            target=tuning_result.target,
            dataset_flag_col=tuning_result.dataset_flag_col,
            categorical_features=tuning_result.categorical_features,
            optimize_metric=tuning_result.optimize_metric,
            seed=int(training_config.get("seed", 1206)),
            lr_feature_mode=str(training_config.get("lr_feature_mode", "numeric")),
            lr_binning_type=str(training_config.get("lr_binning_type", "native")),
            lr_binner_kwargs=training_config.get("lr_binner_kwargs"),
        )

    def _build_backend(
        self,
        df: FrameLike,
        *,
        optimize_metric: str | None = None,
        seed: int | None = None,
    ) -> Any:
        """构建用于 replay 已调优参数集合的后端。"""
        spec = self.spec
        if spec is None:
            raise RuntimeError("Replay spec is unavailable before run(...) receives a tuning run.")
        return _build_backend_from_spec(
            spec,
            df,
            optimize_metric=optimize_metric,
            seed=seed,
        )

    def run(
        self,
        tuning_result: MarsModelTuningResult,
        df: FrameLike,
        *,
        top_k: int = 5,
        sort_metric: str = "ks",
        include_val: bool = True,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        optimize_metric: str | None = None,
        benchmark_col: str | None = None,
        time_col: str | None = None,
        val_target: str | None = None,
    ) -> MarsModelReplayResult:
        """
        回放 Top-K trial，并生成模型、打分数据和评估报告。

        Parameters
        ----------
        tuning_result : MarsModelTuningResult
            提供模型类型、特征列、目标列和样本切片配置的调优结果。
        df : pandas.DataFrame or polars.DataFrame
            用于重新训练和打分的样本表。
        top_k : int, default 5
            要回放的 trial 数量。
        sort_metric : {"auc", "ks"}, default "ks"
            leaderboard 排序指标。
        include_val : bool, default True
            是否将 validation 切片指标纳入平均排序。
        num_boost_round : int, default 500
            当调优结果中没有保存该配置时使用的最大 boosting 轮数。
        early_stopping_rounds : int, default 50
            当调优结果中没有保存该配置时使用的 early stopping 轮数。
        optimize_metric : str, optional
            覆盖 replay 后端使用的优化指标。
        benchmark_col : str, optional
            benchmark 或 champion 模型分数列名。
        time_col : str, optional
            原始时间列名，用于补充报告中的时间边界。
        val_target : str, optional
            替代验证目标列名。

        Returns
        -------
        MarsModelReplayResult
            包含 replay leaderboard、模型、打分数据和评估报告的结果对象。
        """
        self.spec = self._build_spec_from_result(tuning_result)
        spec = self.spec

        run_training_config = dict(getattr(tuning_result, "training_config", {}) or {})
        resolved_num_boost_round = (
            int(num_boost_round)
            if num_boost_round != 500 or "num_boost_round" not in run_training_config
            else int(run_training_config.get("num_boost_round", num_boost_round))
        )
        resolved_early_stopping_rounds = (
            int(early_stopping_rounds)
            if early_stopping_rounds != 50 or "early_stopping_rounds" not in run_training_config
            else int(run_training_config.get("early_stopping_rounds", early_stopping_rounds))
        )
        replay_spec = ReplaySpec(
            top_k=top_k,
            sort_metric=sort_metric.lower(),
            include_val=include_val,
            num_boost_round=resolved_num_boost_round,
            early_stopping_rounds=resolved_early_stopping_rounds,
            optimize_metric=(optimize_metric or spec.optimize_metric).lower(),
        )

        history_df = tuning_result.history_table.copy()
        valid_df = history_df[
            (history_df["trial_state"] == "COMPLETE") & history_df["is_valid"]
        ].copy()
        if valid_df.empty:
            raise ValueError("No valid completed trials are available for replay.")

        metric_suffix = f"_{replay_spec.sort_metric}"
        oot_cols = [col for col in valid_df.columns if "oot" in col.lower() and col.endswith(metric_suffix)]
        cols_to_mean = list(oot_cols)
        if replay_spec.include_val:
            val_cols = [col for col in valid_df.columns if col.lower() == f"val_{replay_spec.sort_metric}".lower()]
            cols_to_mean.extend(val_cols)
        if not cols_to_mean:
            raise ValueError(f"No ranking columns were found for sort_metric={replay_spec.sort_metric!r}.")

        valid_df["custom_mean_score"] = valid_df[cols_to_mean].mean(axis=1)
        ranking_table = valid_df.sort_values("custom_mean_score", ascending=False).head(replay_spec.top_k).copy()

        backend = self._build_backend(
            df,
            optimize_metric=replay_spec.optimize_metric,
            seed=spec.seed,
        )
        backend.num_boost_round = replay_spec.num_boost_round
        backend.early_stopping_rounds = replay_spec.early_stopping_rounds
        backend.training_metric = backend.optimize_metric

        evaluator = MarsModelEvaluator()

        models: Dict[str, Any] = {}
        scored_df = df
        reports: Dict[str, MarsModelingReport] = {}
        importance_tables: Dict[str, pd.DataFrame] = {}
        diagnostic_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        leaderboard_rows: list[dict[str, Any]] = []

        for rank, (_, row) in enumerate(ranking_table.iterrows(), start=1):
            trial_num = int(row["trial_num"])
            pure_params = {
                key: row[key]
                for key in tuning_result.replay_candidates
                if key in row.index and pd.notna(row[key])
            }
            model = backend.train_model(
                trial=None,
                params=pure_params,
                startup_trials=10**9,
                training_metric=backend.training_metric,
            )
            model_name = f"top{rank}_trial{trial_num}"
            models[model_name] = model
            importance_tables[model_name] = backend.extract_importance(model)
            extract_diagnostics = getattr(backend, "extract_diagnostics", None)
            if callable(extract_diagnostics):
                diagnostic_tables[model_name] = extract_diagnostics(model)

            pred_col = f"prob_{model_name}"
            bench = ModelPredictor(
                model,
                feature_list=spec.features,
                categorical_features=spec.categorical_features,
                category_levels=getattr(backend, "category_levels", {}),
            )
            scored_df = bench.predict(scored_df, pred_col=pred_col, inplace=False)
            reports[model_name] = evaluator.evaluate(
                scored_df,
                pred_col=pred_col,
                group_col=spec.dataset_flag_col,
                target=spec.target,
                benchmark_col=benchmark_col,
                time_col=time_col,
                val_target=val_target,
                feature_cols=spec.features,
                importance_table=importance_tables[model_name],
            )

            leaderboard_row = {
                "rank": rank,
                "model_name": model_name,
                "trial_num": trial_num,
                "custom_mean_score": float(row["custom_mean_score"]),
                "best_iteration": backend.get_best_iteration(model),
                "backend_data_mode": backend.backend_data_mode,
            }
            for column_name, value in row.items():
                if column_name == "custom_mean_score" or column_name == "trial_num":
                    continue
                if str(column_name).endswith(f"_{replay_spec.sort_metric}") or str(column_name).startswith("val_"):
                    leaderboard_row[str(column_name)] = value
            leaderboard_rows.append(leaderboard_row)

        leaderboard_table = pd.DataFrame(leaderboard_rows)
        if not leaderboard_table.empty:
            metric_columns = sorted(
                [
                    column
                    for column in leaderboard_table.columns
                    if column not in {
                        "rank",
                        "model_name",
                        "trial_num",
                        "custom_mean_score",
                        "best_iteration",
                        "backend_data_mode",
                    }
                ]
            )
            leaderboard_table = leaderboard_table[
                [
                    "rank",
                    "model_name",
                    "trial_num",
                    "custom_mean_score",
                    "best_iteration",
                    "backend_data_mode",
                    *metric_columns,
                ]
            ]

        return MarsModelReplayResult(
            model_type=spec.model_type,
            ranking_table=ranking_table,
            leaderboard_table=leaderboard_table,
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
            diagnostic_tables=diagnostic_tables,
        )
