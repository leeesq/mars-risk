"""MARS 特征筛选器实现模块。"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.core.constants import DIVISION_EPSILON
from mars.feature.selection.base import _MarsXYSelector
from mars.utils.imports import require_optional_module


class MarsLinearSelector(_MarsXYSelector):
    """
    面向传统 LR 建模的线性特征筛选器。

    该选择器按相关性过滤、VIF 过滤和逐步回归三个阶段收敛候选特征。
    输入可以是 Polars 或 Pandas；统计建模边界会转换为 Pandas/NumPy，
    以复用 statsmodels 的 Logit、AIC/BIC 和 VIF 实现。

    Attributes
    ----------
    selected_features_ : list of str
        最终入选特征。
    coef_table_ : pandas.DataFrame
        最终 Logit 模型的系数、标准误和 p-value。
    vif_table_ : pandas.DataFrame
        VIF 阶段的最终候选特征 VIF 表。
    stepwise_history_ : pandas.DataFrame
        逐步回归每一步的 add/drop 决策记录。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> selector = MarsLinearSelector(corr_thr=0.95)
    >>> selector.fit(df[["age"]], df["y"], features=["age"]).selected_features_
    ['age']
    """

    def __init__(
        self,
        enable_corr_filter: bool = True,
        corr_thr: float = 0.8,
        corr_method: str = "spearman",
        enable_vif_filter: bool = False,
        vif_threshold: float = 5.0,
        enable_stepwise: bool = False,
        stepwise_direction: str = "forward",
        stepwise_criterion: str = "aic",
        max_features: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化线性筛选器配置。

        Parameters
        ----------
        enable_corr_filter : bool
            是否启用相关性去重阶段。
        corr_thr : float
            相关性去重阈值。
        corr_method : str
            相关性计算方法。
        enable_vif_filter : bool
            是否启用 VIF 筛查阶段。
        vif_threshold : float
            VIF 阈值。
        enable_stepwise : bool
            是否启用逐步回归阶段。
        stepwise_direction : str
            逐步回归方向。
        stepwise_criterion : str
            逐步回归优化准则。
        max_features : int | None
            最终保留特征数上限。
        n_jobs : int
            并行任务数量。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        super().__init__()
        self.enable_corr_filter = bool(enable_corr_filter)
        self.corr_thr = float(corr_thr)
        self.corr_method = str(corr_method).lower()
        self.enable_vif_filter = bool(enable_vif_filter)
        self.vif_threshold = float(vif_threshold)
        self.enable_stepwise = bool(enable_stepwise)
        self.stepwise_direction = str(stepwise_direction).lower()
        self.stepwise_criterion = str(stepwise_criterion).lower()
        self.max_features = max_features
        self.n_jobs = int(n_jobs)

        if self.stepwise_direction not in {"forward", "backward", "both"}:
            raise ValueError("stepwise_direction must be one of {'forward', 'backward', 'both'}.")
        if self.stepwise_criterion not in {"aic", "bic"}:
            raise ValueError("stepwise_criterion must be one of {'aic', 'bic'}.")

        self.coef_table_: pd.DataFrame = pd.DataFrame()
        self.vif_table_: pd.DataFrame = pd.DataFrame()
        self.stepwise_history_: pd.DataFrame = pd.DataFrame()

    def _prepare_xy(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any,
        features: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """将输入表转换为干净的数值建模矩阵。"""
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

        if y is None:
            raise ValueError("MarsLinearSelector.fit requires `y`.")

        target_col = "__mars_target__"
        df[target_col] = np.asarray(y)
        candidate_features = list(features) if features is not None else [
            feature for feature in df.columns if feature != target_col
        ]
        numeric_data: dict[str, pd.Series] = {}
        for feature in candidate_features:
            series = pd.to_numeric(df[feature], errors="coerce")
            if series.notna().sum() == 0:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="precheck",
                    reason="non_numeric",
                    desc="Feature cannot be converted to numeric values.",
                )
                continue
            numeric_data[feature] = series

        target_series = pd.to_numeric(df[target_col], errors="coerce")
        clean = pd.DataFrame(numeric_data)
        clean[target_col] = target_series
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if clean.empty:
            raise ValueError("No complete numeric rows are available for MarsLinearSelector.")
        if clean[target_col].nunique() < 2:
            raise ValueError("MarsLinearSelector requires a binary target with both classes present.")

        features = [feature for feature in candidate_features if feature in clean.columns]
        return clean.loc[:, features], clean[target_col].astype(int), features

    @staticmethod
    def _target_strength(X: pd.DataFrame, y: pd.Series, features: Sequence[str]) -> dict[str, float]:
        """按特征与目标的一元绝对关联强度生成排序分值。"""
        strengths: dict[str, float] = {}
        for feature in features:
            corr = pd.Series(X[feature]).corr(y, method="spearman")
            strengths[feature] = 0.0 if pd.isna(corr) else float(abs(corr))
        return strengths

    def _apply_corr_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """在高度相关的特征对中剔除一侧特征。"""
        if not self.enable_corr_filter or len(features) <= 1:
            return list(features)

        corr = X.loc[:, features].corr(method=self.corr_method).abs()
        strengths = self._target_strength(X, y, features)
        dropped: set[str] = set()
        for left_idx, left_feature in enumerate(features):
            if left_feature in dropped:
                continue
            for right_feature in features[left_idx + 1 :]:
                if right_feature in dropped:
                    continue
                value = float(corr.loc[left_feature, right_feature])
                if pd.isna(value) or value < self.corr_thr:
                    continue
                drop_feature = (
                    right_feature
                    if strengths[left_feature] >= strengths[right_feature]
                    else left_feature
                )
                dropped.add(drop_feature)
                self._register_decision(
                    drop_feature,
                    status="Dropped",
                    stage="corr",
                    reason=f"corr_with_{left_feature if drop_feature == right_feature else right_feature}",
                    value=value,
                    desc=f"Absolute {self.corr_method} correlation exceeded {self.corr_thr:.4f}.",
                )
                if drop_feature == left_feature:
                    break
        return [feature for feature in features if feature not in dropped]

    @staticmethod
    def _compute_vif_table(X: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """计算当前候选特征集合的 VIF 表。"""
        if not features:
            return pd.DataFrame(columns=["feature", "vif"])
        if len(features) == 1:
            return pd.DataFrame([{"feature": str(features[0]), "vif": 1.0}])

        vif_module = require_optional_module("statsmodels.stats.outliers_influence")
        variance_inflation_factor = vif_module.variance_inflation_factor
        values = X.loc[:, list(features)].astype(float).to_numpy()
        rows = []
        for idx, feature in enumerate(features):
            try:
                vif_value = float(variance_inflation_factor(values, idx))
            except Exception:
                vif_value = float("inf")
            rows.append({"feature": str(feature), "vif": vif_value})
        return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)

    def _apply_vif_filter(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        """迭代剔除 VIF 最高且超过阈值的特征。"""
        if not self.enable_vif_filter or len(features) <= 1:
            self.vif_table_ = self._compute_vif_table(X, features)
            return list(features)

        remaining = list(features)
        while len(remaining) > 1:
            vif_table = self._compute_vif_table(X, remaining)
            max_row = vif_table.iloc[0]
            max_vif = float(max_row["vif"])
            if max_vif <= self.vif_threshold:
                self.vif_table_ = vif_table
                return remaining
            drop_feature = str(max_row["feature"])
            remaining.remove(drop_feature)
            self._register_decision(
                drop_feature,
                status="Dropped",
                stage="vif",
                reason="high_vif",
                value=max_vif,
                desc=f"VIF exceeded {self.vif_threshold:.4f}.",
            )
        self.vif_table_ = self._compute_vif_table(X, remaining)
        return remaining

    def _fit_logit_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: Sequence[str],
    ) -> tuple[float, Any | None]:
        """拟合 statsmodels Logit 并返回当前配置的信息准则值。"""
        sm = require_optional_module("statsmodels.api")
        design = X.loc[:, list(features)] if features else pd.DataFrame(index=X.index)
        design = sm.add_constant(design, has_constant="add")
        try:
            result = sm.Logit(y, design).fit(disp=False, maxiter=200)
        except Exception:
            return float("inf"), None
        return float(getattr(result, self.stepwise_criterion)), result

    def _apply_stepwise(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """执行 forward、backward 或双向 AIC/BIC 逐步筛选。"""
        if not self.enable_stepwise or not features:
            return list(features)

        history: list[dict[str, Any]] = []

        def record(action: str, feature: str | None, score: float, selected: Sequence[str]) -> None:
            """记录逐步回归每一步的特征集合与信息准则分数。"""
            history.append(
                {
                    "action": action,
                    "feature": feature,
                    "criterion": self.stepwise_criterion,
                    "score": score,
                    "n_features": len(selected),
                    "selected_features": json.dumps(list(selected), ensure_ascii=False),
                }
            )

        if self.stepwise_direction == "backward":
            selected = list(features)
        else:
            selected = []

        current_score, current_result = self._fit_logit_score(X, y, selected)
        record("start", None, current_score, selected)

        def try_add() -> bool:
            """尝试加入一个能继续降低信息准则的候选特征。"""
            nonlocal current_score, current_result, selected
            remaining = [feature for feature in features if feature not in selected]
            if self.max_features is not None and len(selected) >= int(self.max_features):
                return False
            candidates = []
            for feature in remaining:
                score, result = self._fit_logit_score(X, y, [*selected, feature])
                candidates.append((score, feature, result))
            if not candidates:
                return False
            best_score, best_feature, best_result = min(candidates, key=lambda item: item[0])
            if best_score + DIVISION_EPSILON >= current_score:
                return False
            selected.append(best_feature)
            current_score = best_score
            current_result = best_result
            record("add", best_feature, current_score, selected)
            return True

        def try_drop() -> bool:
            """尝试移除一个能继续降低信息准则的已选特征。"""
            nonlocal current_score, current_result, selected
            if len(selected) <= 1:
                return False
            candidates = []
            for feature in selected:
                trial_features = [item for item in selected if item != feature]
                score, result = self._fit_logit_score(X, y, trial_features)
                candidates.append((score, feature, result, trial_features))
            best_score, best_feature, best_result, best_features = min(
                candidates,
                key=lambda item: item[0],
            )
            if best_score + DIVISION_EPSILON >= current_score:
                return False
            selected = list(best_features)
            current_score = best_score
            current_result = best_result
            record("drop", best_feature, current_score, selected)
            return True

        if self.stepwise_direction == "forward":
            while try_add():
                pass
        elif self.stepwise_direction == "backward":
            while try_drop():
                pass
        else:
            changed = True
            while changed:
                changed = try_add()
                while try_drop():
                    changed = True

        self.stepwise_history_ = pd.DataFrame(history)
        selected_set = set(selected)
        for feature in features:
            if feature not in selected_set:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="stepwise",
                    reason=f"not_selected_by_{self.stepwise_criterion}",
                    desc=f"Excluded by {self.stepwise_direction} stepwise regression.",
                )
        if current_result is not None and selected:
            params = current_result.params.reindex(["const", *selected])
            pvalues = current_result.pvalues.reindex(["const", *selected])
            stderr = current_result.bse.reindex(["const", *selected])
            self.coef_table_ = pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "coefficient": float(params.get(feature, np.nan)),
                        "abs_coefficient": abs(float(params.get(feature, np.nan))),
                        "p_value": float(pvalues.get(feature, np.nan)),
                        "std_err": float(stderr.get(feature, np.nan)),
                    }
                    for feature in selected
                ]
            )
        else:
            self.coef_table_ = pd.DataFrame(
                columns=["feature", "coefficient", "abs_coefficient", "p_value", "std_err"]
            )
        return selected

    def _apply_max_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """在未启用 stepwise 或保留过多特征时应用最终 Top-N 限制。"""
        if self.max_features is None or len(features) <= int(self.max_features):
            return list(features)
        strengths = self._target_strength(X, y, features)
        ranked = sorted(features, key=lambda feature: strengths.get(feature, 0.0), reverse=True)
        selected = ranked[: int(self.max_features)]
        selected_set = set(selected)
        for feature in features:
            if feature not in selected_set:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="max_features",
                    reason="rank_cap",
                    value=float(strengths.get(feature, 0.0)),
                    desc=f"Feature rank exceeded max_features={self.max_features}.",
                )
        return [feature for feature in features if feature in selected_set]

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any,
        *,
        features: Sequence[str] | None = None,
    ) -> MarsLinearSelector:
        """
        执行相关性、VIF 与可选 stepwise 线性特征筛选。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征表。
        y : Any
            二分类目标数组。
        features : Sequence[str] | None
            本次参与筛选的特征列；不传时使用输入表中的全部候选列。

        Returns
        -------
        MarsLinearSelector
            已拟合的线性筛选器实例。

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> selector = MarsLinearSelector().fit(X, y)
        >>> selector.selected_features_
        ['age']
        """
        self.report_records_ = []
        X_numeric, target_series, features = self._prepare_xy(X, y, features)
        self.n_features_in_ = len(features)

        selected = self._apply_corr_filter(X_numeric, target_series, features)
        if self.enable_corr_filter:
            corr_frame = X_numeric.loc[:, features].corr(method=self.corr_method).abs()
            for feature in selected:
                other_features = [item for item in features if item != feature]
                max_corr = (
                    float(corr_frame.loc[feature, other_features].max())
                    if other_features
                    else 0.0
                )
                self._register_decision(
                    feature,
                    status="Checked",
                    stage="corr",
                    reason="within_threshold",
                    value=max_corr,
                    desc=f"Maximum absolute {self.corr_method} correlation stayed below threshold.",
                )

        selected = self._apply_vif_filter(X_numeric, selected)
        if self.enable_vif_filter and not self.vif_table_.empty:
            selected_set = set(selected)
            for row in self.vif_table_.to_dict("records"):
                feature = str(row["feature"])
                if feature not in selected_set:
                    continue
                self._register_decision(
                    feature,
                    status="Checked",
                    stage="vif",
                    reason="within_threshold",
                    value=float(row["vif"]),
                    desc=f"VIF stayed below {self.vif_threshold:.4f}.",
                )

        selected = self._apply_stepwise(X_numeric, target_series, selected)
        if self.enable_stepwise:
            for feature in selected:
                self._register_decision(
                    feature,
                    status="Selected",
                    stage="stepwise",
                    reason=f"selected_by_{self.stepwise_criterion}",
                    desc=f"Retained by {self.stepwise_direction} stepwise regression.",
                )

        selected = self._apply_max_features(X_numeric, target_series, selected)

        self.selected_features_ = [feature for feature in features if feature in set(selected)]
        for feature in self.selected_features_:
            self._register_decision(
                feature,
                status="Selected",
                stage="final",
                reason="kept",
                desc="Feature survived linear selector filters.",
            )

        if self.coef_table_.empty and self.selected_features_:
            _, result = self._fit_logit_score(X_numeric, target_series, self.selected_features_)
            if result is not None:
                params = result.params.reindex(["const", *self.selected_features_])
                self.coef_table_ = pd.DataFrame(
                    [
                        {
                            "feature": feature,
                            "coefficient": float(params.get(feature, np.nan)),
                            "abs_coefficient": abs(float(params.get(feature, np.nan))),
                            "p_value": float(result.pvalues.get(feature, np.nan)),
                            "std_err": float(result.bse.get(feature, np.nan)),
                        }
                        for feature in self.selected_features_
                    ]
                )

        self._is_fitted = True
        return self
