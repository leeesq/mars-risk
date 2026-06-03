"""XGBoost 建模后端。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mars.modeling.backends.base import MarsBaseModelTuner
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


class MarsXGBStrategy(MarsBaseModelTuner):
    """
    基于 XGBoost 原生接口的调参策略。
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
        """
        best_iteration = self.get_best_iteration(model)
        iteration_range = (0, best_iteration + 1) if best_iteration is not None else None
        if iteration_range is None:
            return np.asarray(model.predict(self.dmatrix_dict[split_name]))
        return np.asarray(model.predict(self.dmatrix_dict[split_name], iteration_range=iteration_range))

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """Return a normalized XGBoost feature importance table."""
        raw_importance = model.get_score(importance_type="gain")
        importance_map = {str(feature): float(value) for feature, value in raw_importance.items()}
        return _build_importance_table(
            model_type="xgb",
            importance_type="gain",
            features=list(self.features),
            importance_map=importance_map,
        )
