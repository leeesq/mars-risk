"""CatBoost 建模后端。"""

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
from mars.modeling.backends.registry import register_backend
from mars.modeling.evaluation.metrics import CatBoostKSMetric


@register_backend("cbt", "cat", "catboost")
class MarsCatBoostStrategy(MarsBaseModelStrategy):
    """
    基于 CatBoost 原生接口的调参策略。

    Attributes
    ----------
    pool_dict : dict of str to Any
        按切片缓存的 CatBoost ``Pool`` 对象。
    predict_frame_dict : dict of str to pandas.DataFrame
        按切片缓存的预测输入表。
    backend_data_mode : str
        当前后端缓存采用的数据转换模式。

    Examples
    --------
    >>> strategy = object.__new__(MarsCatBoostStrategy)
    >>> "depth" in strategy.get_default_space()
    True
    """

    def _build_backend_data(self) -> None:
        """构建 CatBoost 的 `Pool` 缓存与预测特征缓存。"""
        catboost = _load_module("catboost")

        self.pool_dict: Dict[str, Any] = {}
        self.predict_frame_dict: Dict[str, pd.DataFrame] = {}
        if self._input_is_polars:
            self.backend_data_mode = (
                "polars_to_pandas_category"
                if self._has_categorical_backend_features()
                else "pandas_numeric"
            )
        else:
            self.backend_data_mode = (
                "pandas_category" if self._has_categorical_backend_features() else "pandas_numeric"
            )
        for name, df in self.data_dict.items():
            X = self._get_feature_frame(df, for_categorical_backend=True)
            y = self._get_target_array(df)
            self.predict_frame_dict[name] = X
            self.pool_dict[name] = catboost.Pool(X, y, cat_features=self.categorical_features)

    def get_default_space(self) -> Dict[str, Any]:
        """
        返回 CatBoost 默认搜索空间。

        Returns
        -------
        dict of str to Any
            CatBoost 默认超参数搜索空间。

        Examples
        --------
        >>> strategy = object.__new__(MarsCatBoostStrategy)
        >>> "depth" in strategy.get_default_space()
        True
        """
        return {
            "depth": ("int", 2, 5),
            "learning_rate": ("float", 0.02, 0.2, 0.02),
            "l2_leaf_reg": ("float", 1.0, 10.0, 1.0),
            "random_strength": ("float", 0.0, 2.0, 0.2),
            "bagging_temperature": ("float", 0.0, 5.0, 0.5),
            "border_count": ("int", 32, 255),
        }

    def train_model(
        self,
        trial: Any,
        params: Dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """
        训练单次 CatBoost Trial。

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
            训练完成的 CatBoost 模型。

        Examples
        --------
        >>> strategy = object.__new__(MarsCatBoostStrategy)
        >>> callable(strategy.train_model)
        True
        """
        catboost = _load_module("catboost")

        callbacks = []

        # CatBoost 的训练监控指标同样走 training_metric；
        # 外层若优化 KS，会提前把这里的训练指标切到 AUC。
        eval_metric = (
            self.backend_metric
            if self.backend_metric is not None
            else CatBoostKSMetric() if training_metric == "ks" else training_metric.upper()
        )
        train_params = {
            "loss_function": "Logloss",
            "eval_metric": eval_metric,
            "iterations": self.num_boost_round,
            "random_seed": self.seed,
            "verbose": False,
            "use_best_model": True,
            "od_type": "Iter",
            "od_wait": self.early_stopping_rounds,
            "allow_writing_files": False,
        }
        train_params.update(params)

        model = catboost.CatBoostClassifier(**train_params)
        model.fit(
            self.pool_dict["train"],
            eval_set=self.pool_dict["val"],
            callbacks=callbacks,
            verbose=False,
        )
        return model

    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """
        对指定切片执行 CatBoost 分数预测。

        Parameters
        ----------
        model : Any
            已训练 CatBoost 模型。
        split_name : str
            切片名称。

        Returns
        -------
        numpy.ndarray
            预测分数数组。

        Examples
        --------
        >>> class Model:
        ...     def predict_proba(self, frame):
        ...         return np.array([[0.8, 0.2], [0.1, 0.9]])
        >>> strategy = object.__new__(MarsCatBoostStrategy)
        >>> strategy.predict_frame_dict = {"val": pd.DataFrame({"age": [20, 40]})}
        >>> strategy.predict_scores(Model(), "val").tolist()
        [0.2, 0.9]
        """
        preds = model.predict_proba(self.predict_frame_dict[split_name])
        return np.asarray(preds[:, 1])

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """
        返回标准化后的 CatBoost 特征重要性表。

        Parameters
        ----------
        model : Any
            已训练的 CatBoost 模型。

        Returns
        -------
        pandas.DataFrame
            MARS 统一格式的重要性表。

        Examples
        --------
        >>> class DummyCatBoostModel:
        ...     def get_feature_importance(self, type: str = "FeatureImportance") -> np.ndarray:
        ...         return np.array([2.0])
        >>> strategy = object.__new__(MarsCatBoostStrategy)
        >>> strategy.features = ["age"]
        >>> importance = strategy.extract_importance(DummyCatBoostModel())
        >>> importance.loc[0, "feature"]
        'age'
        """
        importance_values = model.get_feature_importance(type="FeatureImportance")
        importance_map = {
            feature: float(value)
            for feature, value in zip(self.features, importance_values)
        }
        return _build_importance_table(
            model_type="cbt",
            importance_type="feature_importance",
            features=list(self.features),
            importance_map=importance_map,
        )
