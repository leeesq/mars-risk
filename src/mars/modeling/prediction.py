"""训练后模型预测辅助器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.report import MarsModelingReport
from mars.modeling.utils import (
    FrameLike,
    is_polars_dataframe,
    optional_import as _optional_import,
    restore_frame_type,
    to_pandas_frame,
)

class ModelPredictor:
    """
    训练后模型预测辅助器。

    Parameters
    ----------
    model : Any
        已训练模型对象。
    feature_list : sequence of str
        预测所需特征列。
    categorical_features : sequence of str, optional
        需要固定类别字典的特征列。
    category_levels : dict, optional
        train split 中抽取的稳定类别字典。
    """

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        categorical_features: Optional[Sequence[str]] = None,
        category_levels: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.model: Any = model
        self.features: List[str] = list(feature_list)
        self.categorical_features: List[str] = list(categorical_features or [])
        self.category_levels: Dict[str, List[Any]] = {
            str(feature): list(levels)
            for feature, levels in dict(category_levels or {}).items()
        }

    def _safe_predict_logic(self, df: pd.DataFrame) -> np.ndarray:
        """按模型类型分发 Pandas 预测逻辑。"""
        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X = df.loc[:, self.features].copy()
        for feature in self.categorical_features:
            if feature in X.columns:
                categories = self.category_levels.get(feature)
                if categories is not None:
                    X[feature] = X[feature].astype(pd.CategoricalDtype(categories=categories))
                else:
                    X[feature] = X[feature].astype("category")

        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")
        catboost = _optional_import("catboost")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X, enable_categorical=bool(self.categorical_features))
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if xgb is not None and isinstance(self.model, getattr(xgb, "XGBModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X, num_iteration=best_iteration or None))

        if lgb is not None and isinstance(self.model, getattr(lgb, "LGBMModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if catboost is not None and isinstance(self.model, getattr(catboost, "CatBoost", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")

    def _safe_predict_logic_polars(self, df: pl.DataFrame) -> np.ndarray:
        """数值特征场景优先使用 Polars/Arrow 预测通道。"""
        if self.categorical_features:
            return self._safe_predict_logic(df.to_pandas())

        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X_arrow = df.select(self.features).to_arrow()
        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X_arrow)
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X_arrow, num_iteration=best_iteration or None))

        return self._safe_predict_logic(df.to_pandas())

    def predict(
        self,
        df: FrameLike,
        pred_col_name: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """
        对数据集评分并追加预测列。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            待评分数据。
        pred_col_name : str, default "pred_score"
            追加的预测列名。
        inplace : bool, default False
            Pandas 输入时是否原地追加。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的评分数据。
        """
        prefer_polars = is_polars_dataframe(df)
        if prefer_polars and not inplace and isinstance(df, pl.DataFrame):
            preds = self._safe_predict_logic_polars(df)
            return df.with_columns(pl.Series(pred_col_name, preds))
        df_pd = df if isinstance(df, pd.DataFrame) and inplace else to_pandas_frame(df)
        df_pd[pred_col_name] = self._safe_predict_logic(df_pd)
        return restore_frame_type(df_pd, prefer_polars)

    def evaluate(
        self,
        df: FrameLike,
        group_col: str,
        target_col: str,
        *,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
        benchmark_col: Optional[str] = None,
        pred_col_name: str = "pred_score",
    ) -> MarsModelingReport:
        """评分后立即生成评估报告。"""
        scored = self.predict(df, pred_col_name=pred_col_name, inplace=False)
        evaluator = MarsModelEvaluator(
            group_col=group_col,
            target_col=target_col,
            time_col=time_col,
            benchmark_col=benchmark_col,
            val_target_col=val_target_col,
        )
        return evaluator.evaluate(scored, pred_col=pred_col_name)
