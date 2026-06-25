"""MARS 特征筛选器实现模块。"""

from __future__ import annotations

import copy
from typing import Any, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.feature.selection.base import MarsBaseSelector
from mars.utils.imports import require_optional_module


class MarsImportanceSelector(MarsBaseSelector):
    """
    基于模型重要性或 SHAP 的特征筛选器。

    该选择器支持直接消费已有 importance table，也可以训练 sklearn/树模型
    读取 ``feature_importances_`` 或 ``coef_``。当 ``method="shap"`` 时，
    选择器计算 mean absolute SHAP value 并统一输出 MARS importance table。

    Attributes
    ----------
    selected_features_ : list of str
        最终入选特征。
    importance_table_ : pandas.DataFrame
        标准化后的重要性表。
    estimator_ : Any or None
        由选择器训练得到的 estimator；使用外部 importance table 时为 ``None``。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> importance = pd.DataFrame({"feature": ["age"], "importance": [1.0]})
    >>> selector = MarsImportanceSelector()
    >>> selector.fit(df[["age"]], df["y"], features=["age"], importance_table=importance).selected_features_
    ['age']
    """

    def __init__(
        self,
        estimator: Union[str, Any] = "lgbm",
        estimator_params: dict | None = None,
        method: Literal["importance", "shap", "rfe", "sfm"] = "importance",
        selection_mode: Literal["top_k", "threshold", "percentile"] = "top_k",
        selection_threshold: Union[int, float, str] = 50,
        cv: int = 3,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        """
        初始化重要性筛选器配置。

        Parameters
        ----------
        estimator : Union[str, Any]
            底层模型类型或实例。
        estimator_params : dict | None
            底层模型初始化参数。
        method : Literal['importance', 'shap', 'rfe', 'sfm']
            重要性筛选策略。
        selection_mode : Literal['top_k', 'threshold', 'percentile']
            特征保留模式。
        selection_threshold : Union[int, float, str]
            对应筛选模式下的阈值。
        cv : int
            交叉验证折数。
        n_jobs : int
            并行任务数量。
        random_state : int
            随机种子。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        super().__init__()
        self.estimator = estimator
        self.estimator_params = dict(estimator_params or {})
        self.method = str(method).lower()
        self.selection_mode = str(selection_mode).lower()
        self.selection_threshold = selection_threshold
        self.cv = int(cv)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        if self.method not in {"importance", "shap", "rfe", "sfm"}:
            raise ValueError("method must be one of {'importance', 'shap', 'rfe', 'sfm'}.")
        if self.selection_mode not in {"top_k", "threshold", "percentile"}:
            raise ValueError("selection_mode must be one of {'top_k', 'threshold', 'percentile'}.")

        self.importance_table_: pd.DataFrame = pd.DataFrame()
        self.estimator_: Any | None = None

    def _prepare_xy(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any | None,
        features: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
        """将输入数据转为 Pandas，并解析目标列与候选特征。"""
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

        raw_features = list(features) if features is not None else list(df.columns)
        if y is None:
            return df.loc[:, raw_features], None, raw_features

        target_series = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce")
        valid_mask = target_series.notna().to_numpy()
        if int(valid_mask.sum()) == 0:
            raise ValueError("Target contains no valid numeric labels.")
        return df.loc[valid_mask, raw_features], target_series.loc[valid_mask].astype(int), raw_features

    @staticmethod
    def _encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """编码混合类型特征，并保留编码列到原始特征的映射。"""
        encoded_parts: list[pd.DataFrame] = []
        mapping: dict[str, str] = {}
        for feature in X.columns:
            series = X[feature]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                encoded_col = pd.to_numeric(series, errors="coerce")
                fill_value = encoded_col.median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                encoded_parts.append(pd.DataFrame({feature: encoded_col.fillna(fill_value)}))
                mapping[feature] = feature
                continue

            dummies = pd.get_dummies(
                series.astype("string").fillna("__MISSING__"),
                prefix=feature,
                prefix_sep="__",
                dtype=float,
            )
            encoded_parts.append(dummies)
            for encoded_feature in dummies.columns:
                mapping[str(encoded_feature)] = feature

        if not encoded_parts:
            raise ValueError("At least one feature is required for MarsImportanceSelector.")
        encoded = pd.concat(encoded_parts, axis=1)
        return encoded.astype(float), mapping

    def _build_estimator(self) -> Any:
        """实例化受支持的 estimator，或复制用户传入的 estimator 对象。"""
        if not isinstance(self.estimator, str):
            return copy.deepcopy(self.estimator)

        estimator_name = self.estimator.lower()
        params = dict(self.estimator_params)
        if estimator_name in {"rf", "random_forest", "randomforest"}:
            ensemble = require_optional_module("sklearn.ensemble")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            return ensemble.RandomForestClassifier(**params)
        if estimator_name in {"extra_trees", "extratrees", "et"}:
            ensemble = require_optional_module("sklearn.ensemble")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            return ensemble.ExtraTreesClassifier(**params)
        if estimator_name in {"lr", "logit", "logistic", "logistic_regression"}:
            linear = require_optional_module("sklearn.linear_model")
            params.setdefault("solver", "liblinear")
            params.setdefault("random_state", self.random_state)
            return linear.LogisticRegression(**params)
        if estimator_name in {"lgb", "lgbm", "lightgbm"}:
            lgb = require_optional_module("lightgbm")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            params.setdefault("verbosity", -1)
            return lgb.LGBMClassifier(**params)
        if estimator_name in {"xgb", "xgboost"}:
            xgb = require_optional_module("xgboost")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            params.setdefault("eval_metric", "logloss")
            return xgb.XGBClassifier(**params)
        if estimator_name in {"cat", "catboost", "cbt"}:
            catboost = require_optional_module("catboost")
            params.setdefault("iterations", 100)
            params.setdefault("random_seed", self.random_state)
            params.setdefault("verbose", False)
            return catboost.CatBoostClassifier(**params)
        raise ValueError(
            "Unsupported estimator. Expected one of "
            "{'rf', 'extra_trees', 'lr', 'lgbm', 'xgb', 'cat'} or an estimator object."
        )

    @staticmethod
    def _aggregate_importance(
        encoded_features: Sequence[str],
        values: Sequence[float],
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> dict[str, float]:
        """将编码列级别的重要性聚合回原始特征名。"""
        importance_map = {feature: 0.0 for feature in raw_features}
        for encoded_feature, value in zip(encoded_features, values, strict=False):
            raw_feature = mapping.get(str(encoded_feature), str(encoded_feature))
            if raw_feature in importance_map:
                importance_map[raw_feature] += float(value)
        return importance_map

    def _build_importance_table(
        self,
        importance_map: Mapping[str, float],
        importance_type: str,
    ) -> pd.DataFrame:
        """将重要性映射标准化为 MARS importance table 结构。"""
        rows = [
            {
                "feature": feature,
                "importance": float(importance),
                "importance_type": importance_type,
                "model_type": str(
                    self.estimator
                    if isinstance(self.estimator, str)
                    else type(self.estimator).__name__
                ),
            }
            for feature, importance in importance_map.items()
        ]
        table = pd.DataFrame(rows)
        table = table.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1, dtype=int)
        return table[["feature", "importance", "importance_type", "model_type", "rank"]]

    def _importance_from_estimator(
        self,
        estimator: Any,
        X_encoded: pd.DataFrame,
        y: pd.Series,
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """拟合 estimator 并提取内置特征重要性或系数。"""
        estimator.fit(X_encoded, y)
        self.estimator_ = estimator
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
            importance_type = "feature_importance"
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.ravel(estimator.coef_)).astype(float)
            importance_type = "abs_coef"
        else:
            raise ValueError(
                "Estimator must expose feature_importances_ or coef_ for method='importance'."
            )
        importance_map = self._aggregate_importance(
            list(X_encoded.columns),
            values,
            mapping,
            raw_features,
        )
        return self._build_importance_table(importance_map, importance_type)

    def _importance_from_shap(
        self,
        estimator: Any,
        X_encoded: pd.DataFrame,
        y: pd.Series,
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """拟合 estimator 并计算 mean absolute SHAP value。"""
        shap = require_optional_module("shap")
        estimator.fit(X_encoded, y)
        self.estimator_ = estimator
        sample = X_encoded.head(min(len(X_encoded), 300))

        try:
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(sample)
        except Exception:
            explainer = shap.Explainer(estimator.predict_proba, sample)
            explanation = explainer(sample)
            values = getattr(explanation, "values", explanation)

        if isinstance(values, list):
            values_arr = np.asarray(values[-1])
        else:
            values_arr = np.asarray(values)
        if values_arr.ndim == 3:
            values_arr = values_arr[:, :, -1]
        mean_abs = np.abs(values_arr).mean(axis=0)
        importance_map = self._aggregate_importance(
            list(X_encoded.columns),
            mean_abs,
            mapping,
            raw_features,
        )
        return self._build_importance_table(importance_map, "mean_abs_shap")

    @staticmethod
    def _normalize_importance_table(
        table: pd.DataFrame | pl.DataFrame,
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """校验并标准化用户传入的重要性表。"""
        table_pd = table.to_pandas() if isinstance(table, pl.DataFrame) else table.copy()
        if "feature" not in table_pd.columns or "importance" not in table_pd.columns:
            raise ValueError("importance_table must contain 'feature' and 'importance' columns.")
        table_pd["feature"] = table_pd["feature"].astype(str)
        table_pd["importance"] = pd.to_numeric(table_pd["importance"], errors="coerce").fillna(0.0)
        table_pd = table_pd[table_pd["feature"].isin(set(raw_features))].copy()
        if "importance_type" not in table_pd.columns:
            table_pd["importance_type"] = "provided"
        if "model_type" not in table_pd.columns:
            table_pd["model_type"] = "provided"
        table_pd = table_pd.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
        table_pd["rank"] = np.arange(1, len(table_pd) + 1, dtype=int)
        return table_pd[["feature", "importance", "importance_type", "model_type", "rank"]]

    def _select_features(self, table: pd.DataFrame) -> list[str]:
        """按 top-k、绝对阈值或百分位阈值选择特征。"""
        if table.empty:
            return []
        if self.selection_mode == "top_k":
            k = max(int(float(self.selection_threshold)), 0)
            return table.head(k)["feature"].astype(str).tolist()
        if self.selection_mode == "threshold":
            threshold = float(self.selection_threshold)
            return table.loc[table["importance"] >= threshold, "feature"].astype(str).tolist()

        raw_threshold = self.selection_threshold
        if isinstance(raw_threshold, str) and raw_threshold.endswith("%"):
            percentile = float(raw_threshold.rstrip("%")) / 100.0
        else:
            threshold_value = float(raw_threshold)
            percentile = threshold_value / 100.0 if threshold_value > 1 else threshold_value
        percentile = min(max(percentile, 0.0), 1.0)
        keep_count = int(np.ceil(len(table) * percentile))
        return table.head(keep_count)["feature"].astype(str).tolist()

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any | None = None,
        *,
        features: Sequence[str] | None = None,
        importance_table: pd.DataFrame | pl.DataFrame | None = None,
    ) -> MarsImportanceSelector:
        """
        执行基于模型重要性或 SHAP 的特征筛选。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征表。若未显式传入 ``y``，则必须包含目标列。
        y : Any | None
            二分类目标数组。
        features : Sequence[str] | None
            本次参与筛选的特征列；不传时使用输入表中的全部候选列。
        importance_table : pd.DataFrame | pl.DataFrame | None
            预先计算好的重要性表，需包含 ``feature`` 与 ``importance`` 列。

        Returns
        -------
        MarsImportanceSelector
            已拟合的重要性筛选器实例。

        Raises
        ------
        NotImplementedError
            当当前选项尚未实现时抛出。
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> importance = pd.DataFrame({"feature": ["age"], "importance": [1.0]})
        >>> selector = MarsImportanceSelector().fit(X, importance_table=importance)
        >>> selector.selected_features_
        ['age']
        """
        if self.method in {"rfe", "sfm"}:
            raise NotImplementedError(
                f"MarsImportanceSelector method={self.method!r} is not implemented in v1."
            )

        self.report_records_ = []
        X_pd, y_series, raw_features = self._prepare_xy(X, y, features)
        self.n_features_in_ = len(raw_features)

        provided_table = importance_table
        if provided_table is not None:
            table = self._normalize_importance_table(provided_table, raw_features)
        else:
            if y_series is None:
                raise ValueError("MarsImportanceSelector.fit requires y when `importance_table` is not provided.")
            X_encoded, mapping = self._encode_features(X_pd)
            estimator = self._build_estimator()
            if self.method == "importance":
                table = self._importance_from_estimator(
                    estimator,
                    X_encoded,
                    y_series,
                    mapping,
                    raw_features,
                )
            else:
                table = self._importance_from_shap(
                    estimator,
                    X_encoded,
                    y_series,
                    mapping,
                    raw_features,
                )

        selected = self._select_features(table)
        selected_set = set(selected)
        self.importance_table_ = table.copy()
        self.selected_features_ = [feature for feature in raw_features if feature in selected_set]

        importance_lookup = dict(zip(table["feature"], table["importance"], strict=False))
        for feature in raw_features:
            status = "Selected" if feature in selected_set else "Dropped"
            reason = self.selection_mode if feature in selected_set else f"below_{self.selection_mode}"
            self._register_decision(
                feature,
                status=status,
                stage=self.method,
                reason=reason,
                value=float(importance_lookup.get(feature, 0.0)),
                desc="Feature selection based on normalized importance table.",
            )

        self._is_fitted = True
        return self
