"""LR 模型包装对象。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame
from mars.feature.binning.base import MarsBinnerBase


@dataclass(slots=True)
class MarsLogisticModel:
    """可序列化的 LR 模型包装器。"""

    estimator: Any
    features: list[str]
    model_features: list[str]
    lr_feature_mode: str = "numeric"
    binner: MarsBinnerBase | None = None

    def transform_features(self, X: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """返回 LR 实际消费的数值特征矩阵。"""
        frame = to_pandas_frame(X)
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            raise ValueError(f"Input data is missing required LR features: {missing}")

        if self.lr_feature_mode == "numeric":
            numeric = frame.loc[:, self.model_features].copy()
            return numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if self.binner is None:
            raise ValueError("LR WOE mode requires a fitted binner attached to the model.")
        woe_frame = self.binner.transform(frame.loc[:, self.features], return_type="woe")
        woe_frame = to_pandas_frame(woe_frame)
        missing_woe = sorted(set(self.model_features).difference(woe_frame.columns))
        if missing_woe:
            raise ValueError(f"WOE transform did not produce required columns: {missing_woe}")
        return woe_frame.loc[:, self.model_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def predict_proba(self, X: pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """通过保存的预处理路径预测二分类概率。"""
        model_frame = self.transform_features(X)
        return np.asarray(self.estimator.predict_proba(model_frame))
