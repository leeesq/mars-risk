"""建模调参与 Top-K replay 工具。"""

from __future__ import annotations

import os
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
from mars.modeling.results import MarsModelingRun, MarsReplayRun
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
    benchmark_col: str | None = None,
    time_col: str | None = None,
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
        benchmark_col=benchmark_col,
        time_col=time_col,
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
    """Create a backend strategy instance from a modeling spec."""
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
    benchmark_col : str, optional
        评估时默认基准分数列。
    time_col : str, optional
        评估时默认时间列。
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
            benchmark_col=benchmark_col,
            time_col=time_col,
            lr_feature_mode=lr_feature_mode,
            lr_binning_type=lr_binning_type,
            lr_binner_kwargs=lr_binner_kwargs,
            lr_binner=lr_binner,
        )
        self.last_run: MarsModelingRun | None = None

    @property
    def best_model(self) -> Any:
        """Return the best model from the latest tuning run."""
        return None if self.last_run is None else self.last_run.best_model

    @property
    def best_score(self) -> float | None:
        """Return the best validation score from the latest tuning run."""
        return None if self.last_run is None else self.last_run.best_score

    @property
    def best_params(self) -> Dict[str, Any] | None:
        """Return the best parameter set from the latest tuning run."""
        if self.last_run is None:
            return None
        return dict(self.last_run.best_params)

    @property
    def history_table(self) -> pd.DataFrame:
        """Return the structured history table from the latest tuning run."""
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
        """Build a concrete backend strategy for one tuning or replay job."""
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
        save_path: str = "tuner_history.csv",
    ) -> MarsModelingRun:
        """
        调参训练一个模型族并返回可复用结果对象。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            已带 train/val/oot 标识的建模样本。
        param_space : mapping, optional
            覆盖或扩展默认搜索空间。
        max_diff : float, default 3.0
            泛化衰减阈值，单位为百分点。
        use_oot_penalty : bool, default False
            是否把 OOT 衰减纳入 trial 有效性约束。
        n_trials : int, default 50
            Optuna trial 数量。
        startup_trials : int, default 20
            剪枝启动前 trial 数。
        warmup_steps : int, default 100
            剪枝预热步数。
        num_boost_round : int, default 500
            最大 boosting 轮数。
        early_stopping_rounds : int, default 50
            早停轮数。
        save_path : str, default "tuner_history.csv"
            trial 历史落盘路径。

        Returns
        -------
        MarsModelingRun
            调参结果、最佳模型、训练配置和元数据。
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

        if os.path.exists(save_path):
            os.remove(save_path)

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
        study.optimize(lambda trial: backend.objective(trial, startup_trials, save_path), n_trials=n_trials)

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
        run = MarsModelingRun(
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
            history_path=str(Path(save_path).resolve()),
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
        self.last_run = run
        return run


class MarsModelReplay:
    """
    对调参历史中的 Top-K trial 进行重训、重评分和评估。

    Notes
    -----
    replay 默认复用 tuning run 中保存的训练轮数和早停配置；用户显式传入参数时会覆盖。
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
            benchmark_col=benchmark_col,
            time_col=time_col,
            lr_feature_mode=lr_feature_mode,
            lr_binning_type=lr_binning_type,
            lr_binner_kwargs=lr_binner_kwargs,
            lr_binner=lr_binner,
        )

    def _build_backend(
        self,
        df: FrameLike,
        *,
        optimize_metric: str | None = None,
        seed: int | None = None,
    ) -> Any:
        """Build the backend used to replay tuned parameter sets."""
        return _build_backend_from_spec(
            self.spec,
            df,
            optimize_metric=optimize_metric,
            seed=seed,
        )

    def run(
        self,
        run: MarsModelingRun,
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
        val_target_col: str | None = None,
    ) -> MarsReplayRun:
        """
        执行 Top-K replay 并生成排行榜、模型和评估报告。

        Parameters
        ----------
        run : MarsModelingRun
            调参阶段产出的结果对象。
        df : pandas.DataFrame or polars.DataFrame
            需要重训和评分的数据。
        top_k : int, default 5
            回放 trial 数量。
        sort_metric : {"auc", "ks"}, default "ks"
            排名指标。
        include_val : bool, default True
            排名均值是否包含验证集。

        Returns
        -------
        MarsReplayRun
            replay 排名、leaderboard、模型、评分数据和报告。
        """
        if run.model_type != self.spec.model_type:
            raise ValueError(
                f"Run model_type {run.model_type!r} does not match replay model_type {self.spec.model_type!r}."
            )

        run_training_config = dict(getattr(run, "training_config", {}) or {})
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
            optimize_metric=(optimize_metric or self.spec.optimize_metric).lower(),
        )

        history_df = run.history_table.copy()
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
            seed=self.spec.seed,
        )
        backend.num_boost_round = replay_spec.num_boost_round
        backend.early_stopping_rounds = replay_spec.early_stopping_rounds
        backend.training_metric = backend.optimize_metric

        evaluator = MarsModelEvaluator(
            group_col=self.spec.dataset_flag_col,
            target_col=self.spec.target,
            benchmark_col=benchmark_col if benchmark_col is not None else self.spec.benchmark_col,
            time_col=time_col if time_col is not None else self.spec.time_col,
            val_target_col=val_target_col,
        )

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
                for key in run.replay_candidates
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
                feature_list=self.spec.features,
                categorical_features=self.spec.categorical_features,
                category_levels=getattr(backend, "category_levels", {}),
            )
            scored_df = bench.predict(scored_df, pred_col_name=pred_col, inplace=False)
            reports[model_name] = evaluator.evaluate(scored_df, pred_col=pred_col)

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

        return MarsReplayRun(
            model_type=self.spec.model_type,
            ranking_table=ranking_table,
            leaderboard_table=leaderboard_table,
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
            diagnostic_tables=diagnostic_tables,
        )
