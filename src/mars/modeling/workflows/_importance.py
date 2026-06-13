"""建模重要性计算的内部工具。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mars.modeling.backends.base import MarsBaseModelStrategy


def compute_shap_importance(
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
