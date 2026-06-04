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

        Examples
        --------
        >>> class DummyModel:
        ...     def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...         probs = X["age"].to_numpy(dtype=float) / 100.0
        ...         return np.column_stack([1.0 - probs, probs])
        >>> predictor = ModelPredictor(DummyModel(), feature_list=["age"])
        >>> scored = predictor.predict(pd.DataFrame({"age": [20, 30]}))
        >>> scored["pred_score"].round(2).tolist()
        [0.2, 0.3]
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
        time_col: str | None = None,
        val_target_col: str | None = None,
        benchmark_col: str | None = None,
        pred_col_name: str = "pred_score",
    ) -> MarsModelingReport:
        """
        评分后立即生成评估报告。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            待评分并评估的数据。
        group_col : str
            数据集分组列，例如 ``"train"``、``"val"`` 或月份分组。
        target_col : str
            二分类真实标签列。
        time_col : str, optional
            时间列，用于补充时序明细。
        val_target_col : str, optional
            验证目标列，适用于目标字段分阶段落地的场景。
        benchmark_col : str, optional
            基准模型分数字段。
        pred_col_name : str, default "pred_score"
            写入预测分数的列名。

        Returns
        -------
        MarsModelingReport
            评分数据对应的建模评估报告。

        Examples
        --------
        >>> class DummyModel:
        ...     def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...         probs = X["age"].to_numpy(dtype=float) / 100.0
        ...         return np.column_stack([1.0 - probs, probs])
        >>> df = pd.DataFrame({
        ...     "age": [20, 80, 40, 60],
        ...     "target": [0, 1, 0, 1],
        ...     "sample": ["train", "train", "val", "val"],
        ... })
        >>> predictor = ModelPredictor(DummyModel(), feature_list=["age"])
        >>> report = predictor.evaluate(df, group_col="sample", target_col="target")
        >>> isinstance(report, MarsModelingReport)
        True
        """
        scored = self.predict(df, pred_col_name=pred_col_name, inplace=False)
        evaluator = MarsModelEvaluator(
            group_col=group_col,
            target_col=target_col,
            time_col=time_col,
            benchmark_col=benchmark_col,
            val_target_col=val_target_col,
        )
        return evaluator.evaluate(scored, pred_col=pred_col_name)
