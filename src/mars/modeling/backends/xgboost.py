"""XGBoost 建模后端。"""

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
from mars.modeling.metrics import xgb_ks_metric as _xgb_ks_metric


class MarsXGBStrategy(MarsBaseModelStrategy):
    """
    基于 XGBoost 原生接口的调参策略。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        继承自基类的建模数据集，需包含训练、验证和可选 OOT 切片标识。
    features : sequence of str
        参与 XGBoost 训练的特征列名。
    target : str
        二分类目标列名。
    categorical_features : sequence of str, optional
        需要交给 XGBoost 原生类别特征处理的字段名。

    Attributes
    ----------
    dmatrix_dict : dict of str to Any
        按切片缓存的 XGBoost ``DMatrix`` 对象。
    backend_data_mode : str
        当前后端缓存采用的数据转换模式。

    Examples
    --------
    >>> strategy = object.__new__(MarsXGBStrategy)
    >>> "max_depth" in strategy.get_default_space()
    True
    """

    def _build_backend_data(self) -> None:
        """构建 XGBoost 训练与预测所需的 `DMatrix` 缓存。"""
        xgb = _load_module("xgboost")

        self.dmatrix_dict: Dict[str, Any] = {}
        self._xgb_use_categorical = self._has_categorical_backend_features()
        if self._xgb_use_categorical:
            self.backend_data_mode = (
                "polars_to_pandas_category" if self._input_is_polars else "pandas_category"
            )
        else:
            self.backend_data_mode = (
                "polars_arrow_numeric" if self._input_is_polars else "pandas_numeric"
            )
        for name, df in self.data_dict.items():
            y = self._get_target_array(df)
            if self._xgb_use_categorical:
                X = self._get_feature_frame(df, for_categorical_backend=True)
                unsupported = [
                    col
                    for col in X.columns
                    if not (
                        pd.api.types.is_numeric_dtype(X[col])
                        or pd.api.types.is_bool_dtype(X[col])
                        or pd.api.types.is_categorical_dtype(X[col])
                    )
                ]
                if unsupported:
                    raise ValueError(
                        "MarsXGBStrategy requires non-numeric columns to be declared in categorical_features. "
                        f"Found unsupported columns: {unsupported}"
                    )
                self.dmatrix_dict[name] = xgb.DMatrix(
                    X,
                    label=y,
                    enable_categorical=True,
                )
            else:
                if self._input_is_polars:
                    X_pl = self._get_feature_polars(df)
                    _validate_numeric_polars(X_pl, "MarsXGBStrategy")
                    self.dmatrix_dict[name] = xgb.DMatrix(self._get_feature_arrow(df), label=y)
                else:
                    X = self._get_feature_frame(df, for_categorical_backend=False)
                    _validate_numeric_pandas(X, "MarsXGBStrategy")
                    self.dmatrix_dict[name] = xgb.DMatrix(X, label=y)

    def get_default_space(self) -> Dict[str, Any]:
        """
        返回 XGBoost 默认搜索空间。

        Returns
        -------
        dict of str to Any
            XGBoost 默认超参数搜索空间。

        Examples
        --------
        >>> strategy = object.__new__(MarsXGBStrategy)
        >>> "max_depth" in strategy.get_default_space()
        True
        """
        return {
            "max_depth": ("int", 2, 6),
            "eta": ("float", 0.02, 0.2, 0.02),
            "subsample": ("float", 0.5, 1.0, 0.1),
            "colsample_bytree": ("float", 0.5, 1.0, 0.1),
            "min_child_weight": ("int", 1, 20),
            "gamma": ("float", 0.0, 5.0, 0.5),
            "reg_alpha": ("float", 0.0, 5.0, 0.5),
            "reg_lambda": ("float", 0.0, 5.0, 0.5),
            "max_delta_step": ("int", 0, 10),
            "scale_pos_weight": ("float", 1.0, 3.0, 0.1),
        }

    def train_model(
        self,
        trial: Any,
        params: Dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """
        训练单次 XGBoost Trial。

        Parameters
        ----------
        trial : Any
            当前 Optuna Trial。
        params : dict of str to Any
            当前 Trial 的确定性超参数。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        training_metric : str
            训练期监控指标。

        Returns
        -------
        Any
            训练完成的 XGBoost 模型。

        Examples
        --------
        >>> strategy = object.__new__(MarsXGBStrategy)
        >>> callable(strategy.train_model)
        True
        """
        xgb = _load_module("xgboost")

        callbacks = []
        if trial is not None and getattr(trial, "number", 0) >= startup_trials:
            pruning_callback_cls = _load_optuna_callback("xgboost", "XGBoostPruningCallback")
            callbacks.append(pruning_callback_cls(trial, f"val-{training_metric}"))

        # XGBoost 的 early stopping 与 pruning 统一监控 training_metric；
        # 当外部优化目标是 KS 时，这里通常会传入 AUC 作为训练期代理指标。
        train_params = {
            "booster": "gbtree",
            "tree_method": "hist",
            "objective": "binary:logistic",
            "verbosity": 0,
            "seed": self.seed,
            "eval_metric": "auc" if training_metric == "ks" else training_metric,
        }
        train_params.update(params)

        train_kwargs: Dict[str, Any] = {}
        if training_metric == "ks":
            train_kwargs["custom_metric"] = _xgb_ks_metric

        model = xgb.train(
            train_params,
            self.dmatrix_dict["train"],
            num_boost_round=self.num_boost_round,
            evals=[(self.dmatrix_dict["train"], "train"), (self.dmatrix_dict["val"], "val")],
            maximize=True,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
            callbacks=callbacks,
            **train_kwargs,
        )
        try:
            model.set_attr(mars_backend_data_mode=self.backend_data_mode)
        except Exception:
            pass
        return model

    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """
        对指定切片执行 XGBoost 分数预测。

        Parameters
        ----------
        model : Any
            已训练 XGBoost 模型。
        split_name : str
            切片名称。

        Returns
        -------
        numpy.ndarray
            预测分数数组。

        Examples
        --------
        >>> class Model:
        ...     def predict(self, data, iteration_range=None):
        ...         return np.array([0.2, 0.9])
        >>> strategy = object.__new__(MarsXGBStrategy)
        >>> strategy.dmatrix_dict = {"val": object()}
        >>> strategy.predict_scores(Model(), "val").tolist()
        [0.2, 0.9]
        """
        best_iteration = self.get_best_iteration(model)
        iteration_range = (0, best_iteration + 1) if best_iteration is not None else None
        if iteration_range is None:
            return np.asarray(model.predict(self.dmatrix_dict[split_name]))
        return np.asarray(model.predict(self.dmatrix_dict[split_name], iteration_range=iteration_range))

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """
        返回标准化后的 XGBoost 特征重要性表。

        Parameters
        ----------
        model : Any
            已训练的 XGBoost 模型。

        Returns
        -------
        pandas.DataFrame
            MARS 统一格式的重要性表。

        Examples
        --------
        >>> class DummyXGBModel:
        ...     def get_score(self, importance_type: str = "gain") -> dict[str, float]:
        ...         return {"age": 2.0}
        >>> strategy = object.__new__(MarsXGBStrategy)
        >>> strategy.features = ["age"]
        >>> importance = strategy.extract_importance(DummyXGBModel())
        >>> importance.loc[0, "feature"]
        'age'
        """
        raw_importance = model.get_score(importance_type="gain")
        importance_map = {str(feature): float(value) for feature, value in raw_importance.items()}
        return _build_importance_table(
            model_type="xgb",
            importance_type="gain",
            features=list(self.features),
            importance_map=importance_map,
        )
