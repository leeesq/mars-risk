"""与后端注册表对齐的预测适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.backends.registry import ensure_builtin_backends_registered, resolve_backend_name
from mars.utils.imports import optional_import as _optional_import


@dataclass(frozen=True)
class PredictionAdapter:
    """面向 Pandas/Polars 特征矩阵的预测适配器。"""

    name: str
    matches: Callable[[Any], bool]
    predict_pandas: Callable[[Any, pd.DataFrame], np.ndarray]
    predict_polars: Callable[[Any, pl.DataFrame], np.ndarray | None]


_PREDICTION_ADAPTERS: Dict[str, PredictionAdapter] = {}


def register_prediction_adapter(adapter: PredictionAdapter) -> PredictionAdapter:
    """按规范后端名称注册预测适配器。"""
    _PREDICTION_ADAPTERS[adapter.name] = adapter
    return adapter


def get_prediction_adapter(model_type: str) -> PredictionAdapter:
    """返回规范后端名称或别名对应的预测适配器。"""
    ensure_builtin_backends_registered()
    return _PREDICTION_ADAPTERS[resolve_backend_name(model_type)]


def _predict_proba_like(model: Any, X: pd.DataFrame) -> np.ndarray:
    """处理 sklearn 风格 `predict_proba` 接口的通用适配逻辑。"""
    preds = model.predict_proba(X)
    preds_arr = np.asarray(preds)
    if preds_arr.ndim == 2 and preds_arr.shape[1] >= 2:
        return np.asarray(preds_arr[:, 1])
    return np.ravel(preds_arr)


def _xgb_matches(model: Any) -> bool:
    """判断模型对象是否属于 XGBoost 家族。"""
    xgb = _optional_import("xgboost")
    return bool(
        xgb is not None
        and (
            isinstance(model, getattr(xgb, "Booster", tuple()))
            or isinstance(model, getattr(xgb, "XGBModel", tuple()))
        )
    )


def _xgb_predict_pandas(model: Any, X: pd.DataFrame) -> np.ndarray:
    """兼容 Booster 与 sklearn 包装器的 XGBoost Pandas 预测路径。"""
    xgb = _optional_import("xgboost")
    if xgb is None:
        raise ImportError("xgboost is required for XGBoost prediction.")
    if isinstance(model, getattr(xgb, "Booster", tuple())):
        dtest = xgb.DMatrix(X, enable_categorical=any(str(dtype) == "category" for dtype in X.dtypes))
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            return np.asarray(model.predict(dtest))
        return np.asarray(model.predict(dtest, iteration_range=(0, best_iteration + 1)))
    if isinstance(model, getattr(xgb, "XGBModel", tuple())):
        return _predict_proba_like(model, X)
    raise TypeError(f"Unsupported XGBoost model type: {type(model)!r}")


def _xgb_predict_polars(model: Any, X: pl.DataFrame) -> np.ndarray | None:
    """提供 XGBoost Booster 的 Polars 原生预测路径。"""
    xgb = _optional_import("xgboost")
    if xgb is None or not isinstance(model, getattr(xgb, "Booster", tuple())):
        return None
    dtest = xgb.DMatrix(X.to_arrow())
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return np.asarray(model.predict(dtest))
    return np.asarray(model.predict(dtest, iteration_range=(0, best_iteration + 1)))


def _lgb_matches(model: Any) -> bool:
    """判断模型对象是否属于 LightGBM 家族。"""
    lgb = _optional_import("lightgbm")
    return bool(
        lgb is not None
        and (
            isinstance(model, getattr(lgb, "Booster", tuple()))
            or isinstance(model, getattr(lgb, "LGBMModel", tuple()))
        )
    )


def _lgb_predict_pandas(model: Any, X: pd.DataFrame) -> np.ndarray:
    """兼容 Booster 与 sklearn 包装器的 LightGBM Pandas 预测路径。"""
    lgb = _optional_import("lightgbm")
    if lgb is None:
        raise ImportError("lightgbm is required for LightGBM prediction.")
    if isinstance(model, getattr(lgb, "Booster", tuple())):
        best_iteration = getattr(model, "best_iteration", None)
        return np.asarray(model.predict(X, num_iteration=best_iteration or None))
    if isinstance(model, getattr(lgb, "LGBMModel", tuple())):
        return _predict_proba_like(model, X)
    raise TypeError(f"Unsupported LightGBM model type: {type(model)!r}")


def _lgb_predict_polars(model: Any, X: pl.DataFrame) -> np.ndarray | None:
    """提供 LightGBM Booster 的 Polars 原生预测路径。"""
    lgb = _optional_import("lightgbm")
    if lgb is None or not isinstance(model, getattr(lgb, "Booster", tuple())):
        return None
    best_iteration = getattr(model, "best_iteration", None)
    return np.asarray(model.predict(X.to_arrow(), num_iteration=best_iteration or None))


def _cat_matches(model: Any) -> bool:
    """判断模型对象是否属于 CatBoost 家族。"""
    catboost = _optional_import("catboost")
    return bool(catboost is not None and isinstance(model, getattr(catboost, "CatBoost", tuple())))


def _cat_predict_pandas(model: Any, X: pd.DataFrame) -> np.ndarray:
    """走 CatBoost 的 `predict_proba` Pandas 预测路径。"""
    return _predict_proba_like(model, X)


def _cat_predict_polars(model: Any, X: pl.DataFrame) -> np.ndarray | None:
    """CatBoost 当前不提供独立的 Polars 原生预测路径。"""
    return None


def _lr_matches(model: Any) -> bool:
    """判断对象是否暴露 sklearn 风格的 `predict_proba` 接口。"""
    return hasattr(model, "predict_proba")


def _lr_predict_pandas(model: Any, X: pd.DataFrame) -> np.ndarray:
    """走 Logistic/sklearn 风格模型的 Pandas 预测路径。"""
    return _predict_proba_like(model, X)


def _lr_predict_polars(model: Any, X: pl.DataFrame) -> np.ndarray | None:
    """Logistic/sklearn 风格模型当前不提供独立的 Polars 原生预测路径。"""
    return None


register_prediction_adapter(
    PredictionAdapter(
        name="xgb",
        matches=_xgb_matches,
        predict_pandas=_xgb_predict_pandas,
        predict_polars=_xgb_predict_polars,
    )
)
register_prediction_adapter(
    PredictionAdapter(
        name="lgb",
        matches=_lgb_matches,
        predict_pandas=_lgb_predict_pandas,
        predict_polars=_lgb_predict_polars,
    )
)
register_prediction_adapter(
    PredictionAdapter(
        name="cbt",
        matches=_cat_matches,
        predict_pandas=_cat_predict_pandas,
        predict_polars=_cat_predict_polars,
    )
)
register_prediction_adapter(
    PredictionAdapter(
        name="lr",
        matches=_lr_matches,
        predict_pandas=_lr_predict_pandas,
        predict_polars=_lr_predict_polars,
    )
)
