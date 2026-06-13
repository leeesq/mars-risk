"""建模后端共享工具。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mars.utils.imports import require_optional_module

HISTORY_BASE_COLUMNS = ["trial_num", "trial_state", "is_valid", "val_diff", "max_oot_diff"]
METRIC_NAMES = ("auc", "ks", "f1")


def load_backend_module(module_name: str) -> Any:
    """
    加载建模后端依赖。

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
    加载 ``optuna_integration`` 中的剪枝回调。

    Parameters
    ----------
    module_name : str
        ``optuna_integration`` 子模块名。
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
    importance_map: dict[str, float],
) -> pd.DataFrame:
    """
    将各后端的重要性输出标准化为统一表结构。

    Parameters
    ----------
    model_type : str
        模型后端名。
    importance_type : str
        重要性类型，例如 ``gain``。
    features : list[str]
        原始特征顺序。
    importance_map : dict[str, float]
        后端返回的特征重要性映射。

    Returns
    -------
    pandas.DataFrame
        统一的重要性表。
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
    importance_df = importance_df.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1, dtype=int)
    return importance_df[["feature", "importance", "importance_type", "model_type", "rank"]]


def validate_numeric_polars(X: pl.DataFrame, backend_name: str) -> None:
    """
    校验 Polars/Arrow 数值路径只接收数值或布尔特征。

    Parameters
    ----------
    X : pl.DataFrame
        特征数据表。
    backend_name : str
        后端名，用于错误提示。

    Raises
    ------
    ValueError
        存在非数值且非布尔特征时抛出。
    """
    unsupported = [
        name
        for name, dtype in X.schema.items()
        if not (dtype.is_numeric() or dtype == pl.Boolean)
    ]
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
    X : pd.DataFrame
        特征数据表。
    backend_name : str
        后端名，用于错误提示。

    Raises
    ------
    ValueError
        存在非数值且非布尔特征时抛出。
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


def split_name_sort_key(split_name: str) -> tuple[int, int, str]:
    """
    生成 train/val/oot 友好的稳定排序键。

    Parameters
    ----------
    split_name : str
        数据切片名。

    Returns
    -------
    tuple[int, int, str]
        排序键，顺序为 train、val、oot*、其他。
    """
    normalized = str(split_name).strip().lower()
    if "train" in normalized:
        return (0, 0, normalized)
    if "val" in normalized:
        return (1, 0, normalized)
    if "oot" in normalized:
        match = re.search(r"(\d+)", normalized)
        return (2, int(match.group(1)) if match else 10**9, normalized)
    return (3, 0, normalized)


def normalize_dataset_flags(flags: pd.Series | pl.Series) -> pd.Series:
    """
    标准化数据集标识列。

    Parameters
    ----------
    flags : pd.Series | pl.Series
        原始数据集标识列。

    Returns
    -------
    pandas.Series
        去空格并转小写后的序列。
    """
    flags_pd = flags.to_pandas() if isinstance(flags, pl.Series) else flags
    return flags_pd.astype(str).str.strip().str.lower()


def validate_dataset_flag_roles(flags: pd.Series | pl.Series) -> None:
    """
    校验单个 dataset flag 不会同时命中多个保留角色。

    Parameters
    ----------
    flags : pd.Series | pl.Series
        原始或标准化后的数据集标识列。

    Raises
    ------
    ValueError
        任一唯一值同时包含多个角色关键字时抛出。
    """
    normalized = normalize_dataset_flags(flags)
    unique_flags = sorted(set(normalized.dropna().tolist()))
    conflicts: list[str] = []
    for flag in unique_flags:
        roles = [
            role
            for role, matched in {
                "train": "train" in flag,
                "val": "val" in flag,
                "oot": "oot" in flag,
            }.items()
            if matched
        ]
        if len(roles) > 1:
            conflicts.append(flag)
    if conflicts:
        raise ValueError(
            "Ambiguous dataset_flag values matched multiple split roles: "
            f"{conflicts}. Please rename them so each value contains only one of train/val/oot."
        )
