"""MARS 建模调参的后端策略实现。"""

from __future__ import annotations

from typing import Any, Dict
import importlib

import numpy as np
import pandas as pd

from mars.modeling.base import MarsBaseModelTuner


def _load_module(module_name: str) -> Any:
    """
    按需加载可选依赖模块。

    Parameters
    ----------
    module_name : str
        模块名称。

    Returns
    -------
    Any
        导入后的模块对象。

    Raises
    ------
    ImportError
        当依赖未安装时抛出，并提示安装可选 extras。
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for mars.modeling. "
            f"Install the optional extras, for example `pip install \"mars-risk[ml,tuning]\"`."
        ) from exc


def _load_optuna_callback(module_name: str, class_name: str) -> Any:
    """
    加载 Optuna 集成回调类。

    Parameters
    ----------
    module_name : str
        `optuna-integration` 子模块名称。
    class_name : str
        回调类名。

    Returns
    -------
    Any
        回调类对象。

    Raises
    ------
    ImportError
        当找不到对应回调类时抛出。
    """
    root_module = _load_module("optuna_integration")
    callback = getattr(root_module, class_name, None)
    if callback is not None:
        return callback

    submodule = _load_module(f"optuna_integration.{module_name}")
    callback = getattr(submodule, class_name, None)
    if callback is None:
        raise ImportError(
            f"Could not locate {class_name} from optuna-integration. "
            "Please install a compatible version of optuna-integration."
        )
    return callback


def _build_importance_table(
    *,
    model_type: str,
    importance_type: str,
    features: list[str],
    importance_map: Dict[str, float],
) -> pd.DataFrame:
    """Normalize backend-specific importance outputs into one table."""
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "importance": float(importance_map.get(feature, 0.0)),
                "importance_type": importance_type,
                "model_type": model_type,
            }
        )

    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1, dtype=int)
    return importance_df[["feature", "importance", "importance_type", "model_type", "rank"]]


class MarsXGBStrategy(MarsBaseModelTuner):
    """
    基于 XGBoost 原生接口的调参策略。
    """

    def _build_backend_data(self) -> None:
        """构建 XGBoost 训练与预测所需的 `DMatrix` 缓存。"""
        xgb = _load_module("xgboost")

        self.dmatrix_dict: Dict[str, Any] = {}
        for name, df in self.data_dict.items():
            X = self._get_feature_frame(df, for_categorical_backend=False)
            non_numeric = [
                col
                for col in X.columns
                if not (
                    pd.api.types.is_numeric_dtype(X[col])
                    or pd.api.types.is_bool_dtype(X[col])
                )
            ]
            if non_numeric:
                raise ValueError(
                    "MarsXGBStrategy requires numeric or boolean features only. "
                    f"Found unsupported columns: {non_numeric}"
                )

            self.dmatrix_dict[name] = xgb.DMatrix(X, label=self._get_target_array(df))

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
            "eval_metric": training_metric,
        }
        train_params.update(params)

        return xgb.train(
            train_params,
            self.dmatrix_dict["train"],
            num_boost_round=self.num_boost_round,
            evals=[(self.dmatrix_dict["train"], "train"), (self.dmatrix_dict["val"], "val")],
            maximize=True,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
            callbacks=callbacks,
        )

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


class MarsLGBStrategy(MarsBaseModelTuner):
    """
    基于 LightGBM 原生接口的调参策略。
    """

    def _build_backend_data(self) -> None:
        """构建 LightGBM 训练集缓存与预测特征缓存。"""
        lgb = _load_module("lightgbm")

        self.dataset_dict: Dict[str, Any] = {}
        self.predict_frame_dict: Dict[str, pd.DataFrame] = {}
        for name, df in self.data_dict.items():
            X = self._get_feature_frame(df, for_categorical_backend=True)
            y = self._get_target_array(df)
            self.predict_frame_dict[name] = X
            self.dataset_dict[name] = lgb.Dataset(
                X,
                label=y,
                categorical_feature=self.categorical_features or "auto",
                free_raw_data=False,
            )

    def get_default_space(self) -> Dict[str, Any]:
        """
        返回 LightGBM 默认搜索空间。

        Returns
        -------
        dict of str to Any
            LightGBM 默认超参数搜索空间。
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
        params : dict of str to Any
            当前 Trial 的确定性超参数。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        training_metric : str
            训练期监控指标。

        Returns
        -------
        Any
            训练完成的 LightGBM 模型。
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
            "metric": training_metric,
            "verbosity": -1,
            "seed": self.seed,
            "feature_pre_filter": False,
        }
        train_params.update(params)

        return lgb.train(
            train_params,
            self.dataset_dict["train"],
            num_boost_round=self.num_boost_round,
            valid_sets=[self.dataset_dict["train"], self.dataset_dict["val"]],
            valid_names=["train", "val"],
            callbacks=callbacks,
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
        """
        best_iteration = self.get_best_iteration(model)
        return np.asarray(
            model.predict(
                self.predict_frame_dict[split_name],
                num_iteration=best_iteration if best_iteration is not None else None,
            )
        )

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """Return a normalized LightGBM feature importance table."""
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


class MarsCatBoostStrategy(MarsBaseModelTuner):
    """
    基于 CatBoost 原生接口的调参策略。
    """

    def _build_backend_data(self) -> None:
        """构建 CatBoost 的 `Pool` 缓存与预测特征缓存。"""
        catboost = _load_module("catboost")

        self.pool_dict: Dict[str, Any] = {}
        self.predict_frame_dict: Dict[str, pd.DataFrame] = {}
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
        """
        return {
            "depth": ("int", 4, 8),
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
        params : dict of str to Any
            当前 Trial 的确定性超参数。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        training_metric : str
            训练期监控指标。

        Returns
        -------
        Any
            训练完成的 CatBoost 模型。
        """
        catboost = _load_module("catboost")

        callbacks = []

        # CatBoost 的训练监控指标同样走 training_metric；
        # 外层若优化 KS，会提前把这里的训练指标切到 AUC。
        train_params = {
            "loss_function": "Logloss",
            "eval_metric": training_metric.upper(),
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
        """
        preds = model.predict_proba(self.predict_frame_dict[split_name])
        return np.asarray(preds[:, 1])

    def extract_importance(self, model: Any) -> pd.DataFrame:
        """Return a normalized CatBoost feature importance table."""
        importance_values = model.get_feature_importance(type="FeatureImportance")
        importance_map = {
            feature: float(value)
            for feature, value in zip(self.features, importance_values, strict=False)
        }
        return _build_importance_table(
            model_type="cbt",
            importance_type="feature_importance",
            features=list(self.features),
            importance_map=importance_map,
        )
