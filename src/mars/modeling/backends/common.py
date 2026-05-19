"""树模型后端共享工具。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.utils import require_optional_module


def load_backend_module(module_name: str) -> Any:
    """
    加载树模型后端依赖。

    Parameters
    ----------
    module_name : str
        依赖模块名。

    Returns
    -------
    Any
        已导入模块。
    """
    return require_optional_module(module_name)


def load_optuna_callback(module_name: str, class_name: str) -> Any:
    """
    加载 optuna-integration 中的剪枝回调类。

    Parameters
    ----------
    module_name : str
        optuna-integration 子模块名。
    class_name : str
        回调类名。

    Returns
    -------
    Any
        回调类对象。

    Raises
    ------
    ImportError
        找不到兼容回调类时抛出。
    """
    root_module = require_optional_module("optuna_integration")
    callback = getattr(root_module, class_name, None)
    if callback is not None:
        return callback

    submodule = require_optional_module(f"optuna_integration.{module_name}")
    callback = getattr(submodule, class_name, None)
    if callback is None:
        raise ImportError(
            f"Could not locate {class_name} from optuna-integration. "
            "Please install a compatible version of optuna-integration."
        )
    return callback


def build_importance_table(
    *,
    model_type: str,
    importance_type: str,
    features: list[str],
    importance_map: Dict[str, float],
) -> pd.DataFrame:
    """
    将各后端的重要性输出标准化为统一表结构。

    Parameters
    ----------
    model_type : str
        模型后端名称。
    importance_type : str
        重要性类型，例如 ``gain``。
    features : list of str
        原始特征顺序。
    importance_map : dict of str to float
        后端返回的特征重要性映射。

    Returns
    -------
    pandas.DataFrame
        包含 feature、importance、importance_type、model_type、rank 的表。
    """
    rows = [
        {
            "feature": feature,
            "importance": float(importance_map.get(feature, 0.0)),
            "importance_type": importance_type,
            "model_type": model_type,
        }
        for feature in features
    ]
    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1, dtype=int)
    return importance_df[["feature", "importance", "importance_type", "model_type", "rank"]]


def validate_numeric_polars(X: pl.DataFrame, backend_name: str) -> None:
    """
    校验 Polars/Arrow 数值路径只接收数值或布尔特征。

    Parameters
    ----------
    X : polars.DataFrame
        特征数据框。
    backend_name : str
        后端名称，用于错误提示。
    """
    unsupported = [name for name, dtype in X.schema.items() if not (dtype.is_numeric() or dtype == pl.Boolean)]
    if unsupported:
        raise ValueError(
            f"{backend_name} Arrow-native path requires numeric or boolean features only. "
            f"Found unsupported columns: {unsupported}. Pass them as categorical_features when supported."
        )


def validate_numeric_pandas(X: pd.DataFrame, backend_name: str) -> None:
    """
    校验 Pandas 数值路径只接收数值或布尔特征。

    Parameters
    ----------
    X : pandas.DataFrame
        特征数据框。
    backend_name : str
        后端名称，用于错误提示。
    """
    unsupported = [
        col
        for col in X.columns
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]))
    ]
    if unsupported:
        raise ValueError(
            f"{backend_name} pandas numeric path requires numeric or boolean features only. "
            f"Found unsupported columns: {unsupported}. Pass them as categorical_features when supported."
        )
