"""LightGBM 建模后端。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.common import (
    build_importance_table as _build_importance_table,
)
from mars.modeling.backends.common import (
    load_backend_module as _load_module,
)
from mars.modeling.backends.common import (
    load_optuna_callback as _load_optuna_callback,
)
from mars.modeling.backends.common import (
    validate_numeric_pandas as _validate_numeric_pandas,
)
from mars.modeling.backends.common import (
    validate_numeric_polars as _validate_numeric_polars,
)
from mars.modeling.metrics import lgb_ks_metric as _lgb_ks_metric


class MarsLGBStrategy(MarsBaseModelStrategy):
    """
    基于 LightGBM 原生接口的调参策略。

    Attributes
    ----------
    dataset_dict : dict of str to Any
        按切片缓存的 LightGBM ``Dataset`` 对象。
    predict_frame_dict : dict of str to Any
        按切片缓存的预测输入数据。
    backend_data_mode : str
        当前后端缓存采用的数据转换模式。

    Examples
    --------
    >>> strategy = object.__new__(MarsLGBStrategy)
    >>> "num_leaves" in strategy.get_default_space()
    True
    """

    def _build_backend_data(self) -> None:
        """构建 LightGBM 训练集缓存与预测特征缓存。"""
        lgb = _load_module("lightgbm")

        self.dataset_dict: Dict[str, Any] = {}
        self.predict_frame_dict: Dict[str, Any] = {}
        self._lgb_use_categorical = self._has_categorical_backend_features()
        if self._lgb_use_categorical:
            self.backend_data_mode = (
                "polars_to_pandas_category" if self._input_is_polars else "pandas_category"
            )
        else:
            self.backend_data_mode = (
                "polars_arrow_numeric" if self._input_is_polars else "pandas_numeric"
            )
        for name, df in self.data_dict.items():
            y = self._get_target_array(df)
            if self._lgb_use_categorical:
                X = self._get_feature_frame(df, for_categorical_backend=True)
                self.predict_frame_dict[name] = X
                self.dataset_dict[name] = lgb.Dataset(
                    X,
                    label=y,
                    categorical_feature=self.categorical_features or "auto",
                    free_raw_data=False,
                )
            else:
                if self._input_is_polars:
                    X_pl = self._get_feature_polars(df)
                    _validate_numeric_polars(X_pl, "MarsLGBStrategy")
                    X_arrow = self._get_feature_arrow(df)
                    self.predict_frame_dict[name] = X_arrow
                    self.dataset_dict[name] = lgb.Dataset(
                        X_arrow,
                        label=y,
                        feature_name=list(self.features),
                        free_raw_data=False,
                    )
                else:
                    X = self._get_feature_frame(df, for_categorical_backend=False)
                    _validate_numeric_pandas(X, "MarsLGBStrategy")
                    self.predict_frame_dict[name] = X
                    self.dataset_dict[name] = lgb.Dataset(
                        X,
                        label=y,
                        feature_name=list(self.features),
                        free_raw_data=False,
                    )

    def get_default_space(self) -> Dict[str, Any]:
        """
        返回 LightGBM 默认搜索空间。

        Returns
        -------
        dict of str to Any
            LightGBM 默认超参数搜索空间。

        Examples
        --------
        >>> strategy = object.__new__(MarsLGBStrategy)
        >>> "num_leaves" in strategy.get_default_space()
        True
        """
        return {
            "num_leaves": ("int", 15, 63),
            "learning_rate": ("float", 0.02, 0.2, 0.02),
            "feature_fraction": ("float", 0.5, 1.0, 0.1),
            "bagging_fraction": ("float", 0.5, 1.0, 0.1),
            "bagging_freq": ("int", 0, 5),
            "min_data_in_leaf": ("int", 10, 100, 10),
            "lambda_l1": ("float", 0.0, 5.0, 0.5),
            "lambda_l2": ("float", 0.0, 5.0, 0.5),
        }

    def train_model(
        self,
        trial: Any,
        params: Dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """
        训练单次 LightGBM Trial。

        Parameters
        ----------
        trial : Any
            当前 Optuna Trial。
        params : Dict[str, Any]
            当前 Trial 的确定性超参数。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        training_metric : str
            训练期监控指标。

        Returns
        -------
        Any
            训练完成的 LightGBM 模型。

        Examples
        --------
        >>> strategy = object.__new__(MarsLGBStrategy)
        >>> callable(strategy.train_model)
        True
        """
        lgb = _load_module("lightgbm")

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        if trial is not None and getattr(trial, "number", 0) >= startup_trials:
            pruning_callback_cls = _load_optuna_callback("lightgbm", "LightGBMPruningCallback")
            callbacks.append(pruning_callback_cls(trial, training_metric, valid_name="val"))

        # LightGBM 的 pruning / early stopping 也统一跟随 training_metric，
        # 从而让 KS 优化目标与训练过程解耦。
        train_params = {
            "objective": "binary",
            "metric": "None" if training_metric == "ks" else training_metric,
            "verbosity": -1,
            "seed": self.seed,
            "feature_pre_filter": False,
        }
        train_params.update(params)

        train_kwargs: Dict[str, Any] = {}
        if training_metric == "ks":
            train_kwargs["feval"] = _lgb_ks_metric

        return lgb.train(
            train_params,
            self.dataset_dict["train"],
            num_boost_round=self.num_boost_round,
            valid_sets=[self.dataset_dict["train"], self.dataset_dict["val"]],
            valid_names=["train", "val"],
            callbacks=callbacks,
            **train_kwargs,
        )

    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """
        对指定切片执行 LightGBM 分数预测。

        Parameters
        ----------
        model : Any
            已训练 LightGBM 模型。
        split_name : str
            切片名称。

        Returns
        -------
        numpy.ndarray
            预测分数数组。

        Examples
        --------
        >>> class Model:
        ...     def predict(self, frame, num_iteration=None):
        ...         return np.array([0.2, 0.9])
        >>> strategy = object.__new__(MarsLGBStrategy)
        >>> strategy.predict_frame_dict = {"val": pd.DataFrame({"age": [20, 40]})}
        >>> strategy.predict_scores(Model(), "val").tolist()
        [0.2, 0.9]
        """
        best_iteration = self.get_best_iteration(model)
        return np.asarray(
            model.predict(
                self.predict_frame_dict[split_name],
                num_iteration=best_iteration if best_iteration is not None else None,
            )
        )

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """
        返回标准化后的 LightGBM 特征重要性表。

        Parameters
        ----------
        model : Any
            已训练的 LightGBM 模型。

        Returns
        -------
        pandas.DataFrame
            MARS 统一格式的重要性表。

        Examples
        --------
        >>> class DummyLGBModel:
        ...     def feature_name(self) -> list[str]:
        ...         return ["age"]
        ...     def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        ...         return np.array([2.0])
        >>> strategy = object.__new__(MarsLGBStrategy)
        >>> strategy.features = ["age"]
        >>> importance = strategy.extract_importance(DummyLGBModel())
        >>> importance.loc[0, "feature"]
        'age'
        """
        feature_names = [str(name) for name in model.feature_name()]
        importance_values = model.feature_importance(importance_type="gain")
        importance_map = {
            feature: float(value)
            for feature, value in zip(feature_names, importance_values, strict=False)
        }
        return _build_importance_table(
            model_type="lgb",
            importance_type="gain",
            features=list(self.features),
            importance_map=importance_map,
        )
