"""建模单次调参与训练收口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from mars.compute import FrameLike
from mars.modeling.artifacts import create_artifact_path, write_json
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.registry import resolve_backend_name
from mars.modeling.contracts.specs import ModelingSpec
from mars.modeling.contracts.tuning_result import MarsModelTuningResult
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection
from mars.modeling.workflows._backend_factory import build_backend_from_spec
from mars.modeling.workflows._importance import compute_shap_importance
from mars.modeling.workflows._runtime_metadata import collect_library_versions
from mars.modeling.workflows._spec_builder import build_modeling_spec

_build_spec = build_modeling_spec


class MarsModelTuner:
    """
    二分类风险模型调参工具。

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
        """
        初始化 Modeling Pipeline 调参器。

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
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
        keep_top_n_models: int = 0,
    ) -> MarsBaseModelStrategy:
        """为单次调参任务构建具体后端策略。"""
        return build_backend_from_spec(
            self.spec,
            df,
            param_space=param_space,
            max_diff=max_diff,
            use_oot_penalty=use_oot_penalty,
            optimize_metric=optimize_metric,
            seed=seed,
            metric_params=metric_params,
            custom_metrics=custom_metrics,
            metric_directions=metric_directions,
            training_metric=training_metric,
            backend_metric=backend_metric,
            keep_top_n_models=keep_top_n_models,
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
        """
        调优一个模型后端并返回可复用的建模调优结果。

        Parameters
        ----------
        df : FrameLike
            已经带有 train、validation、OOT 切片标记的建模样本。
        param_space : Mapping[str, Any] | None
            对后端搜索空间的覆盖或扩展。
        max_diff : float
            泛化衰减阈值，单位是百分点。
        use_oot_penalty : bool
            是否将 OOT 衰减纳入 trial 有效性判断。
        n_trials : int
            Optuna 试验次数。
        startup_trials : int
            剪枝器开始工作前的预热试验次数。
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
            模型后端训练期监控指标。
        backend_metric : Any | None
            透传给模型后端原生训练接口的自定义 metric。
        keep_top_n_models : int
            调参过程中动态保留的最优模型数量。
        artifact_dir : str | Path | None
            调参产物根目录；``None`` 表示不落盘。
        importance_methods : Sequence[Literal["native", "shap"]]
            特征重要性计算方式。
        shap_sample_size : int
            计算 SHAP values 的最大样本量。
        shap_background_size : int
            SHAP 背景样本量。
        overwrite : bool
            保留参数；当前每次调参都会创建独立运行目录。

        Returns
        -------
        MarsModelTuningResult
            包含最佳模型、调参历史、训练配置和元数据的建模调优结果。

        Raises
        ------
        ValueError
            当指标、重要性方法或输入配置不合法时抛出。
        ImportError
            当当前功能依赖的可选组件不可用时抛出。
        RuntimeError
            当底层训练、评估或导出流程失败时抛出。
        """
        del overwrite
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for MarsModelTuner.tune. "
                "Install the optional extra with `pip install \"mars-risk[tuning]\"`."
            ) from exc

        normalized_importance_methods = tuple(dict.fromkeys(importance_methods))
        unsupported_importance_methods = set(normalized_importance_methods).difference(
            {"native", "shap"}
        )
        if unsupported_importance_methods:
            raise ValueError(
                f"Unsupported importance_methods: {sorted(unsupported_importance_methods)}. "
                "Expected 'native' or 'shap'."
            )

        run_id = uuid4().hex[:8]
        artifact_path = create_artifact_path(
            artifact_dir,
            model_type=self.spec.model_type,
            target=self.spec.target,
            optimize_metric=self.spec.optimize_metric,
            run_id=run_id,
        )
        resolved_history_path = artifact_path / "history.csv" if artifact_path is not None else None

        backend = self._build_backend(
            df,
            param_space=param_space,
            max_diff=max_diff,
            use_oot_penalty=use_oot_penalty,
            metric_params=metric_params,
            custom_metrics=custom_metrics,
            metric_directions=metric_directions,
            training_metric=training_metric,
            backend_metric=backend_metric,
            keep_top_n_models=keep_top_n_models,
        )

        backend.num_boost_round = int(num_boost_round)
        backend.early_stopping_rounds = int(early_stopping_rounds)

        training_config = {
            "run_id": run_id,
            "n_trials": int(n_trials),
            "startup_trials": int(startup_trials),
            "warmup_steps": int(warmup_steps),
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": int(early_stopping_rounds),
            "max_diff": float(max_diff),
            "use_oot_penalty": bool(use_oot_penalty),
            "param_space": dict(param_space or {}),
            "metric_params": dict(metric_params or {}),
            "custom_metrics": sorted((custom_metrics or {}).keys()),
            "metric_directions": dict(backend.metric_directions),
            "training_metric": backend.training_metric,
            "backend_metric": repr(backend_metric) if backend_metric is not None else None,
            "keep_top_n_models": int(keep_top_n_models),
            "importance_methods": list(normalized_importance_methods),
            "shap_sample_size": int(shap_sample_size),
            "shap_background_size": int(shap_background_size),
            "history_path": str(resolved_history_path.resolve()) if resolved_history_path else None,
            "artifact_path": str(artifact_path.resolve()) if artifact_path else None,
            "seed": int(backend.seed),
        }
        if resolve_backend_name(self.spec.model_type) == "lr":
            training_config.update(
                {
                    "lr_feature_mode": self.spec.lr_feature_mode,
                    "lr_binning_type": self.spec.lr_binning_type,
                    "lr_binner_kwargs": dict(self.spec.lr_binner_kwargs),
                }
            )

        if artifact_path is not None:
            write_json(artifact_path / "run_config.json", training_config)
            write_json(
                artifact_path / "metadata.json",
                {
                    "artifact_type": "mars_model_tuning_result",
                    "artifact_schema_version": 2,
                    "run_id": run_id,
                    "status": "running",
                    "model_type": self.spec.model_type,
                    "target": self.spec.target,
                    "features": list(self.spec.features),
                    "categorical_features": list(self.spec.categorical_features),
                    "optimize_metric": backend.optimize_metric,
                    "metric_names": list(backend.metric_names),
                    "metric_directions": dict(backend.metric_directions),
                    "training_config": training_config,
                    "feature_schema": dict(backend.feature_schema),
                    "backend_data_mode": backend.backend_data_mode,
                    "category_levels": dict(getattr(backend, "category_levels", {})),
                },
            )

        study = optuna.create_study(
            direction=backend.metric_directions[backend.optimize_metric],
            sampler=optuna.samplers.TPESampler(seed=backend.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=startup_trials,
                n_warmup_steps=warmup_steps,
            ),
        )
        try:
            study.optimize(
                lambda trial: backend.objective(trial, startup_trials, resolved_history_path),
                n_trials=n_trials,
            )
        except Exception as exc:
            if artifact_path is not None:
                write_json(
                    artifact_path / "failure.json",
                    {
                        "run_id": run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise RuntimeError(f"Model tuning failed: {exc}") from exc

        if backend.best_model is None:
            failure_message = "No valid trial satisfied the generalization constraints."
            if artifact_path is not None:
                write_json(
                    artifact_path / "failure.json",
                    {
                        "run_id": run_id,
                        "error_type": RuntimeError.__name__,
                        "error": failure_message,
                    },
                )
            raise RuntimeError(failure_message)

        history_table = backend.build_history_table()
        best_trial_num = int(study.best_trial.number)
        best_trial_rows = history_table.loc[history_table["trial_num"] == best_trial_num]
        if best_trial_rows.empty:
            raise RuntimeError(
                f"Could not locate the best trial record for trial_num={best_trial_num}."
            )

        best_trial_row = best_trial_rows.iloc[-1]
        best_params = {
            key: best_trial_row[key]
            for key in backend.replay_param_keys
            if key in best_trial_row.index and pd.notna(best_trial_row[key])
        }
        diagnostic_tables: Dict[str, pd.DataFrame] = {}
        extract_diagnostics = getattr(backend, "extract_diagnostics", None)
        if callable(extract_diagnostics):
            diagnostic_tables = extract_diagnostics(backend.best_model)
        native_importance = backend.extract_importance(backend.best_model)
        importance_tables: Dict[str, pd.DataFrame] = {"native": native_importance.copy()}
        primary_importance = native_importance
        if "shap" in normalized_importance_methods:
            shap_importance = compute_shap_importance(
                backend,
                backend.best_model,
                sample_size=shap_sample_size,
                background_size=shap_background_size,
            )
            importance_tables["shap"] = shap_importance.copy()
            primary_importance = shap_importance

        retained_model_table = pd.DataFrame(backend.retained_model_rows)
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
            importance_table=primary_importance,
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
            retained_models=dict(backend.retained_models),
            retained_model_table=retained_model_table,
            artifact_path=str(artifact_path.resolve()) if artifact_path else None,
            run_id=run_id,
            metric_names=list(backend.metric_names),
            metric_directions=dict(backend.metric_directions),
            importance_tables=importance_tables,
            metadata={
                "artifact_schema_version": 2,
                "run_id": run_id,
                "artifact_path": str(artifact_path.resolve()) if artifact_path else None,
                "retained_trial_nums": sorted(backend.retained_models),
            },
        )
        if artifact_path is not None:
            result.export_artifact(str(artifact_path))
        self.last_run = result
        return result
