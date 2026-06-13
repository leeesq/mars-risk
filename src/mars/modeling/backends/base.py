"""建模后端共享基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import FrameLike, is_polars_dataframe
from mars.modeling.backends._dataset_mixin import BackendDatasetMixin
from mars.modeling.backends._history_mixin import BackendHistoryMixin
from mars.modeling.backends._objective_mixin import BackendObjectiveMixin
from mars.modeling.evaluation.metrics import (
    MetricCallable,
    MetricDirection,
    normalize_metric_directions,
    resolve_metric_names,
)


class MarsBaseModelStrategy(
    BackendDatasetMixin,
    BackendHistoryMixin,
    BackendObjectiveMixin,
    ABC,
):
    """MARS 二分类建模后端共享基类。"""

    SUPPORTED_OPTIMIZE_METRICS = {"auc", "ks", "f1"}
    NATIVE_TRAINING_METRICS = {"auc", "ks"}

    def __init__(
        self,
        df: FrameLike,
        features: Sequence[str],
        target: str,
        *,
        optimize_metric: str = "ks",
        param_space: Mapping[str, Any] | None = None,
        max_diff: float = 3.0,
        seed: int = 1206,
        use_oot_penalty: bool = False,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Sequence[str] | None = None,
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
        keep_top_n_models: int = 0,
    ) -> None:
        """初始化后端共享状态并校验输入。"""
        self._input_is_polars = is_polars_dataframe(df)
        if isinstance(df, pl.DataFrame):
            self.df_pl: pl.DataFrame | None = df.clone()
            self.df_pd: pd.DataFrame | None = None
            self.df_native: FrameLike = self.df_pl
            native_columns = list(self.df_pl.columns)
        elif isinstance(df, pd.DataFrame):
            self.df_pl = None
            self.df_pd = df.copy()
            self.df_native = self.df_pd
            native_columns = list(self.df_pd.columns)
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

        self.features = list(features)
        self.target = target
        self.optimize_metric = optimize_metric.lower()
        self.param_space = dict(param_space or {})
        self.max_diff = float(max_diff)
        self.seed = int(seed)
        self.use_oot_penalty = use_oot_penalty
        self.dataset_flag_col = dataset_flag_col
        self.categorical_features = list(categorical_features or [])
        self.metric_params = dict(metric_params or {})
        self.custom_metrics = {
            str(name).lower(): metric
            for name, metric in dict(custom_metrics or {}).items()
        }
        self.metric_names = resolve_metric_names(self.custom_metrics)
        self.metric_directions = normalize_metric_directions(
            self.metric_names,
            metric_directions,
        )
        self.training_metric = self._resolve_training_metric(training_metric)
        self.backend_metric = backend_metric
        self.keep_top_n_models = max(0, int(keep_top_n_models))

        if self.optimize_metric not in self.metric_names:
            raise ValueError(
                f"Unsupported optimize_metric: {optimize_metric!r}. "
                f"Expected one of {sorted(self.metric_names)}."
            )

        required_cols = set(self.features + [self.target, self.dataset_flag_col])
        missing_cols = required_cols.difference(native_columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        cat_missing = set(self.categorical_features).difference(self.features)
        if cat_missing:
            raise ValueError(
                "Categorical features must be included in features. "
                f"Missing from features: {sorted(cat_missing)}"
            )

        self.history: list[dict[str, Any]] = []
        self.all_models: dict[int, Any] = {}
        self.retained_models: dict[int, Any] = {}
        self.retained_model_rows: list[dict[str, Any]] = []
        self.best_model: Any = None
        self.best_score = self._initial_best_score()

        self.num_boost_round = 500
        self.early_stopping_rounds = 50
        self.backend_data_mode = "unset"
        self.category_levels: dict[str, list[Any]] = {}
        if self._input_is_polars:
            assert self.df_pl is not None
            self.feature_schema = {
                feature: str(self.df_pl.schema.get(feature))
                for feature in self.features
            }
        else:
            assert self.df_pd is not None
            self.feature_schema = {
                feature: str(self.df_pd.dtypes.get(feature))
                for feature in self.features
            }

        self._prepare_data()
        self._initialize_category_levels()
        self._build_backend_data()

    @abstractmethod
    def _build_backend_data(self) -> None:
        """构建后端专用缓存。"""

    @abstractmethod
    def get_default_space(self) -> dict[str, Any]:
        """返回当前后端的默认搜索空间。"""

    @abstractmethod
    def train_model(
        self,
        trial: Any,
        params: dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """训练单次 trial 模型。"""

    @abstractmethod
    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """对指定切片执行分数预测。"""

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """返回统一格式的特征重要性表。"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement extract_importance.")
