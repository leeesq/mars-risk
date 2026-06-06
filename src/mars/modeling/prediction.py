"""训练后模型预测辅助器。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.report import MarsModelingReport
from mars.modeling.utils import (
    FrameLike,
    is_polars_dataframe,
    restore_frame_type,
    to_pandas_frame,
)
from mars.modeling.utils import (
    optional_import as _optional_import,
)


class ModelPredictor:
    """
    训练后模型预测辅助器。

    Attributes
    ----------
    model : Any
        已训练模型对象。
    features : list of str
        预测所需特征列。
    categorical_features : list of str
        需要固定类别字典的特征列。
    category_levels : dict of str to list
        train split 中抽取的稳定类别字典。

    Examples
    --------
    >>> predictor = ModelPredictor(model=object(), feature_list=["age"])
    >>> predictor.features
    ['age']
    """

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        categorical_features: Sequence[str] | None = None,
        category_levels: Dict[str, Sequence[Any]] | None = None,
    ) -> None:
        """
        初始化模型预测辅助器。

        Parameters
        ----------
        model : Any
            已训练模型对象。
        feature_list : Sequence[str]
            预测时需要使用的特征列。
        categorical_features : Sequence[str] | None
            需要固定类别字典的类别特征列。
        category_levels : Dict[str, Sequence[Any]] | None
            训练阶段抽取的类别取值字典。
        """
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

        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(X)
            preds_arr = np.asarray(preds)
            if preds_arr.ndim == 2 and preds_arr.shape[1] >= 2:
                return np.asarray(preds_arr[:, 1])
            return np.ravel(preds_arr)

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
        pred_col: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """
        为一份样本追加预测分列。

        Parameters
        ----------
        df : FrameLike
            待打分样本表。
        pred_col : str
            追加到结果表中的预测分列名。
        inplace : bool
            当输入是 pandas DataFrame 时，是否直接在原对象上写入预测分。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            追加了 `pred_col` 的打分结果，尽量保持输入表类型。
        """
        prefer_polars = is_polars_dataframe(df)
        if prefer_polars and not inplace and isinstance(df, pl.DataFrame):
            preds = self._safe_predict_logic_polars(df)
            return df.with_columns(pl.Series(pred_col, preds))
        df_pd = df if isinstance(df, pd.DataFrame) and inplace else to_pandas_frame(df)
        df_pd[pred_col] = self._safe_predict_logic(df_pd)
        return restore_frame_type(df_pd, prefer_polars)

    def evaluate(
        self,
        df: FrameLike,
        group_col: str,
        target: str,
        *,
        time_col: str | None = None,
        val_target: str | None = None,
        benchmark_col: str | None = None,
        pred_col: str = "pred_score",
    ) -> MarsModelingReport:
        """
        对样本打分并立即构建模型评估报告。

        Parameters
        ----------
        df : FrameLike
            待打分和评估的样本表。
        group_col : str
            已存在的样本分组列名。
        target : str
            二分类目标列名。
        time_col : str | None
            原始时间列名，用于补充报告中的时间边界。
        val_target : str | None
            替代验证目标列名。
        benchmark_col : str | None
            benchmark 或 champion 模型分数列名。
        pred_col : str
            追加并用于评估的预测分列名。

        Returns
        -------
        MarsModelingReport
            基于打分结果生成的模型评估报告。
        """
        scored = self.predict(df, pred_col=pred_col, inplace=False)
        return MarsModelEvaluator().evaluate(
            scored,
            pred_col=pred_col,
            group_col=group_col,
            target=target,
            time_col=time_col,
            benchmark_col=benchmark_col,
            val_target=val_target,
        )
