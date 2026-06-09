"""建模调参与 replay 工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from mars.modeling.artifacts import write_json
from mars.modeling.backends import (
    MarsCatBoostStrategy,
    MarsLGBStrategy,
    MarsLogisticRegressionStrategy,
    MarsXGBStrategy,
)
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.metrics import MetricCallable, MetricDirection
from mars.modeling.prediction import ModelPredictor
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelReplayResult, MarsModelTuningResult
from mars.modeling.spec import ModelingSpec, ReplaySpec
from mars.modeling.utils import FrameLike, collect_library_versions

BACKEND_MAP: dict[str, type[MarsBaseModelStrategy]] = {
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
    features : Sequence[str]
        特征列名。
    target : str
        目标列名。
    dataset_flag_col : str
        建模样本切片标记列名。
    categorical_features : Sequence[str] | None
        类别特征列名。
    optimize_metric : str
        调参优化指标，可使用内置指标或后续传入的自定义指标名。
    seed : int
        随机种子。
    lr_feature_mode : str
        Logistic Regression 特征模式，支持 ``"numeric"`` 和 ``"woe"``。
    lr_binning_type : str
        LR WOE 模式使用的分箱器类型。
    lr_binner_kwargs : Mapping[str, Any] | None
        构造 LR 分箱器时使用的参数。
    lr_binner : Any | None
        已拟合或待复用的 LR 分箱器实例。

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
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
    metric_directions: Mapping[str, MetricDirection] | None = None,
    training_metric: str | None = None,
    backend_metric: Any | None = None,
    keep_top_n_models: int = 0,
) -> MarsBaseModelStrategy:
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
        "metric_params": metric_params,
        "custom_metrics": custom_metrics,
        "metric_directions": metric_directions,
        "training_metric": training_metric,
        "backend_metric": backend_metric,
        "keep_top_n_models": keep_top_n_models,
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


def _safe_artifact_part(value: Any) -> str:
    """将模型类型、target 或指标名转换为稳定的目录片段。"""
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_\-]+", "_", text)
    return text.strip("_") or "unknown"


def _create_artifact_path(
    artifact_dir: str | Path | None,
    *,
    model_type: str,
    target: str,
    optimize_metric: str,
    run_id: str,
) -> Path | None:
    """根据运行上下文创建独立 artifact 目录。"""
    if artifact_dir is None:
        return None
    base_dir = Path(artifact_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "_".join(
        [
            timestamp,
            _safe_artifact_part(model_type),
            _safe_artifact_part(target),
            _safe_artifact_part(optimize_metric),
            _safe_artifact_part(run_id),
        ]
    )
    run_path = base_dir / run_name
    run_path.mkdir(parents=True, exist_ok=False)
    return run_path


def _compute_shap_importance(
    backend: MarsBaseModelStrategy,
    model: Any,
    *,
    sample_size: int,
    background_size: int,
) -> pd.DataFrame:
    """
    基于训练样本计算 SHAP 重要性。

    Parameters
    ----------
    backend : MarsBaseModelStrategy
        已完成训练数据缓存的后端策略。
    model : Any
        已训练模型。
    sample_size : int
        用于计算 SHAP values 的最大样本量。
    background_size : int
        用于构建解释器背景样本的最大样本量。

    Returns
    -------
    pandas.DataFrame
        MARS 统一格式的 SHAP 重要性表。

    Raises
    ------
    ImportError
        当 ``shap`` 未安装时抛出。
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required when importance_methods includes 'shap'. "
            "Install it with `pip install shap` or remove 'shap' from importance_methods."
        ) from exc

    train_df = backend.data_dict["train"]
    feature_frame = backend._get_feature_frame(  # noqa: SLF001
        train_df,
        for_categorical_backend=bool(backend.categorical_features),
    )
    if sample_size > 0 and len(feature_frame) > sample_size:
        feature_frame = feature_frame.sample(n=int(sample_size), random_state=backend.seed)
    background = feature_frame
    if background_size > 0 and len(background) > background_size:
        background = background.sample(n=int(background_size), random_state=backend.seed)

    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(feature_frame).values
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_frame)

    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])
    if values.ndim == 3:
        values = values[:, :, -1]
    importance_values = np.nanmean(np.abs(values), axis=0)
    total = float(np.nansum(importance_values))
    if total <= 0.0:
        normalized = np.zeros_like(importance_values, dtype=float)
    else:
        normalized = importance_values / total
    return pd.DataFrame(
        {
            "feature": list(backend.features),
            "importance": normalized,
            "raw_importance": importance_values,
            "rank": np.arange(1, len(backend.features) + 1),
            "importance_type": "shap_mean_abs",
            "model_type": backend.__class__.__name__,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


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
            LR WOE 模式使用的分箱器类型。
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
        """为单次调参或 replay 任务构建具体后端策略。"""
        return _build_backend_from_spec(
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
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for MarsModelTuner.tune. "
                "Install the optional extra with `pip install \"mars-risk[tuning]\"`."
            ) from exc

        normalized_importance_methods = tuple(dict.fromkeys(importance_methods))
        unsupported_importance_methods = set(normalized_importance_methods).difference({"native", "shap"})
        if unsupported_importance_methods:
            raise ValueError(
                f"Unsupported importance_methods: {sorted(unsupported_importance_methods)}. "
                "Expected 'native' or 'shap'."
            )

        run_id = uuid4().hex[:8]
        artifact_path = _create_artifact_path(
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
        if isinstance(backend, MarsLogisticRegressionStrategy):
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
            raise

        if backend.best_model is None:
            exc = RuntimeError("No valid trial satisfied the generalization constraints.")
            if artifact_path is not None:
                write_json(
                    artifact_path / "failure.json",
                    {
                        "run_id": run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise exc

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
        diagnostic_tables: Dict[str, pd.DataFrame] = {}
        extract_diagnostics = getattr(backend, "extract_diagnostics", None)
        if callable(extract_diagnostics):
            diagnostic_tables = extract_diagnostics(backend.best_model)
        native_importance = backend.extract_importance(backend.best_model)
        importance_tables: Dict[str, pd.DataFrame] = {"native": native_importance.copy()}
        primary_importance = native_importance
        if "shap" in normalized_importance_methods:
            shap_importance = _compute_shap_importance(
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
            result.write_artifact(str(artifact_path))
        self.last_run = result
        return result


class MarsModelReplayRunner:
    """
    基于 `MarsModelTuningResult` 回放调参结果。

    `MarsModelReplayRunner` 不在构造函数中绑定模型类型、特征列或目标列，而是从
    :meth:`run` 传入的调优结果中读取建模规格。回放候选既可以按 Top-K 自动选择，
    也可以由调用者传入 trial 编号。benchmark 分数、时间列和辅助验证目标属于本次
    replay 评估上下文，因此保留在方法入参中。

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
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
    ) -> MarsBaseModelStrategy:
        """构建用于 replay 已调优参数集合的后端。"""
        spec = self.spec
        if spec is None:
            raise RuntimeError("Replay spec is unavailable before run(...) receives a tuning run.")
        return _build_backend_from_spec(
            spec,
            df,
            optimize_metric=optimize_metric,
            seed=seed,
            metric_params=metric_params,
            custom_metrics=custom_metrics,
            metric_directions=metric_directions,
            training_metric=training_metric,
            backend_metric=backend_metric,
        )

    def run(
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
    ) -> MarsModelReplayResult:
        """
        回放 Top-K 或指定 trial，并生成模型、打分数据和评估报告。

        Parameters
        ----------
        tuning_result : MarsModelTuningResult
            提供模型类型、特征列、目标列和样本切片配置的调优结果。
        df : FrameLike
            用于重新训练和打分的样本表。
        top_k : int
            要回放的 trial 数量。
        sort_metric : str
            replay 排行表排序指标。
        include_val : bool
            是否将 validation 切片指标纳入平均排序。
        trial_nums : Sequence[int] | None
            指定要 replay 的 trial 编号；传入后按给定顺序回放，``top_k`` 不参与选择。
        retrain : bool
            是否使用 trial 参数重新训练；``False`` 时只使用调参阶段已保留的模型。
        num_boost_round : int
            当调优结果中没有保存该配置时使用的最大 boosting 轮数。
        early_stopping_rounds : int
            当调优结果中没有保存该配置时使用的 early stopping 轮数。
        optimize_metric : str | None
            覆盖 replay 后端使用的优化指标。
        metric_params : Mapping[str, Any] | None
            指标参数，例如 ``f1_threshold``。
        custom_metrics : Mapping[str, MetricCallable] | None
            replay 重训时使用的自定义指标函数字典。
        metric_directions : Mapping[str, MetricDirection] | None
            指标排序方向；会影响 Top-K trial 选择和自定义指标 replay。
        training_metric : str | None
            模型后端训练期监控指标。
        backend_metric : Any | None
            透传给模型后端原生训练接口的自定义 metric。
        benchmark_col : str | None
            benchmark 或 champion 模型分数列名。
        benchmark_cols : Sequence[str] | None
            多个 benchmark 或 champion 模型分数列名。
        time_col : str | None
            原始时间列名，用于补充报告中的时间边界。
        val_target : str | None
            替代验证目标列名。
        aux_targets : Sequence[str] | None
            辅助验证目标列名；不参与训练，只进入 replay 评估报告。
        target_group_cols : Mapping[str, str] | None
            每个目标对应的独立样本切片列名，用于长短 y 表现期不一致的评估。

        Returns
        -------
        MarsModelReplayResult
            包含 replay 排行表、模型、打分数据和评估报告的结果对象。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

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

        metric_direction = dict(getattr(tuning_result, "metric_directions", {}) or {}).get(
            replay_spec.sort_metric,
            "maximize",
        )
        valid_df["custom_mean_score"] = valid_df[cols_to_mean].mean(axis=1)
        if trial_nums is not None:
            requested_trial_nums = [int(trial_num) for trial_num in trial_nums]
            available_trial_nums = set(valid_df["trial_num"].astype(int).tolist())
            missing_trial_nums = [
                trial_num
                for trial_num in requested_trial_nums
                if trial_num not in available_trial_nums
            ]
            if missing_trial_nums:
                raise ValueError(
                    f"Requested trial_nums are not valid completed trials: {missing_trial_nums}."
                )
            trial_order = {trial_num: order for order, trial_num in enumerate(requested_trial_nums)}
            ranking_table = (
                valid_df.loc[valid_df["trial_num"].astype(int).isin(requested_trial_nums)]
                .assign(_trial_order=lambda frame: frame["trial_num"].astype(int).map(trial_order))
                .sort_values("_trial_order")
                .drop(columns=["_trial_order"])
                .copy()
            )
        else:
            ranking_table = (
                valid_df.sort_values(
                    "custom_mean_score",
                    ascending=metric_direction == "minimize",
                )
                .head(replay_spec.top_k)
                .copy()
            )

        restored_metric_directions: dict[str, MetricDirection] = {
            key: cast(MetricDirection, value)
            for key, value in dict(tuning_result.metric_directions).items()
            if value in {"maximize", "minimize"}
        }
        backend = self._build_backend(
            df,
            optimize_metric=replay_spec.optimize_metric,
            seed=spec.seed,
            metric_params=metric_params or tuning_result.training_config.get("metric_params"),
            custom_metrics=custom_metrics,
            metric_directions=metric_directions or restored_metric_directions,
            training_metric=training_metric or tuning_result.training_config.get("training_metric"),
            backend_metric=backend_metric,
        )
        backend.num_boost_round = replay_spec.num_boost_round
        backend.early_stopping_rounds = replay_spec.early_stopping_rounds

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
            if retrain:
                model = backend.train_model(
                    trial=None,
                    params=pure_params,
                    startup_trials=10**9,
                    training_metric=backend.training_metric,
                )
            else:
                if trial_num not in tuning_result.retained_models:
                    raise ValueError(
                        f"trial_num={trial_num} was not retained during tuning. "
                        "Use retrain=True or increase keep_top_n_models."
                    )
                model = tuning_result.retained_models[trial_num]
            model_name = f"top{rank}_trial{trial_num}"
            models[model_name] = model
            importance_tables[model_name] = (
                backend.extract_importance(model)
                if retrain
                else tuning_result.importance_table.copy()
            )
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
                benchmark_cols=benchmark_cols,
                time_col=time_col,
                val_target=val_target,
                aux_targets=aux_targets,
                target_group_cols=target_group_cols,
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
