"""Logistic regression modeling backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.modeling.backends.base import MarsBaseModelTuner
from mars.modeling.backends.common import build_importance_table as _build_importance_table
from mars.modeling.backends.common import validate_numeric_pandas as _validate_numeric_pandas
from mars.modeling.utils import require_optional_module

LR_FEATURE_MODE = Literal["numeric", "woe"]
LR_BINNING_TYPE = Literal["native", "opt", "optimal"]


@dataclass(slots=True)
class MarsLogisticModel:
    """
    可序列化的 LR 模型包装器。

    Parameters
    ----------
    estimator : Any
        已训练的 ``sklearn.linear_model.LogisticRegression`` 实例。
    features : list of str
        用户传入的原始特征名。
    model_features : list of str
        LR 实际消费的数值特征名。WOE 模式下通常为 ``{feature}_woe``。
    lr_feature_mode : {"numeric", "woe"}, default "numeric"
        特征预处理模式。
    binner : MarsBinnerBase, optional
        WOE 模式下用于 replay/evaluate 的已拟合分箱器。
    """

    estimator: Any
    features: list[str]
    model_features: list[str]
    lr_feature_mode: str = "numeric"
    binner: MarsBinnerBase | None = None

    def _to_pandas(self, X: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

    def transform_features(self, X: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """Return the numeric feature matrix consumed by LogisticRegression."""
        frame = self._to_pandas(X)
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            raise ValueError(f"Input data is missing required LR features: {missing}")

        if self.lr_feature_mode == "numeric":
            numeric = frame.loc[:, self.model_features].copy()
            return numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if self.binner is None:
            raise ValueError("LR WOE mode requires a fitted binner attached to the model.")
        woe_frame = self.binner.transform(frame.loc[:, self.features], return_type="woe")
        if isinstance(woe_frame, pl.DataFrame):
            woe_frame = woe_frame.to_pandas()
        missing_woe = sorted(set(self.model_features).difference(woe_frame.columns))
        if missing_woe:
            raise ValueError(f"WOE transform did not produce required columns: {missing_woe}")
        return woe_frame.loc[:, self.model_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def predict_proba(self, X: pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities through the stored preprocessing path."""
        model_frame = self.transform_features(X)
        return np.asarray(self.estimator.predict_proba(model_frame))


class MarsLogisticRegressionStrategy(MarsBaseModelTuner):
    """
    面向传统银行评分卡体系的 LR 建模后端。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        已包含特征、目标列和样本切片标识的建模数据。
    features : sequence of str
        参与训练的原始特征名。
    target : str
        二分类目标列名。
    optimize_metric : {"auc", "ks"}, default "ks"
        Optuna trial 的优化指标。
    param_space : mapping, optional
        LR 超参搜索空间覆盖项。
    max_diff : float, default 3.0
        训练集与验证集指标衰减阈值，单位为百分点。
    seed : int, default 1206
        随机种子。
    use_oot_penalty : bool, default False
        是否将最差 OOT 衰减纳入 trial 有效性约束。
    dataset_flag_col : str, default "dataset_flag"
        样本切片标识列。
    categorical_features : sequence of str, optional
        WOE 模式下按类别分箱处理的特征。
    lr_feature_mode : {"numeric", "woe"}, default "numeric"
        ``"numeric"`` 表示输入已数值化或已 WOE 化；``"woe"`` 表示内部先分箱并转 WOE。
    lr_binning_type : {"native", "opt", "optimal"}, default "native"
        WOE 模式下内部使用的分箱器类型。
    lr_binner_kwargs : mapping, optional
        内部分箱器初始化参数。
    lr_binner : MarsBinnerBase, optional
        用户传入的已配置分箱器。传入后优先复用该实例进行拟合与转换。

    Raises
    ------
    ValueError
        当 LR 模式、分箱类型、样本切片或特征列配置非法时抛出。
    ImportError
        缺少 sklearn 或 statsmodels 等可选依赖时抛出。
    """

    SUPPORTED_FEATURE_MODES = {"numeric", "woe"}
    SUPPORTED_BINNING_TYPES = {"native", "opt", "optimal"}

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        features: Sequence[str],
        target: str,
        *,
        optimize_metric: str = "ks",
        param_space: Mapping[str, Any] | None = None,
        max_diff: float = 3.0,
        seed: int = 1206,
        use_oot_penalty: bool = False,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Sequence[str] | None = None,
        lr_feature_mode: LR_FEATURE_MODE = "numeric",
        lr_binning_type: LR_BINNING_TYPE = "native",
        lr_binner_kwargs: Mapping[str, Any] | None = None,
        lr_binner: MarsBinnerBase | None = None,
    ) -> None:
        self.lr_feature_mode = str(lr_feature_mode).lower()
        self.lr_binning_type = str(lr_binning_type).lower()
        self.lr_binner_kwargs = dict(lr_binner_kwargs or {})
        self.lr_binner = lr_binner
        self.model_features: list[str] = []

        if self.lr_feature_mode not in self.SUPPORTED_FEATURE_MODES:
            raise ValueError(
                "lr_feature_mode must be one of {'numeric', 'woe'}, "
                f"got {lr_feature_mode!r}."
            )
        if self.lr_binning_type not in self.SUPPORTED_BINNING_TYPES:
            raise ValueError(
                "lr_binning_type must be one of {'native', 'opt', 'optimal'}, "
                f"got {lr_binning_type!r}."
            )

        super().__init__(
            df=df,
            features=features,
            target=target,
            optimize_metric=optimize_metric,
            param_space=param_space,
            max_diff=max_diff,
            seed=seed,
            use_oot_penalty=use_oot_penalty,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
        )

    def _build_backend_data(self) -> None:
        """Prepare split-level pandas matrices for LR training and scoring."""
        self.raw_feature_frame_dict: dict[str, pd.DataFrame] = {}
        self.feature_frame_dict: dict[str, pd.DataFrame] = {}
        for split_name, split_df in self.data_dict.items():
            self.raw_feature_frame_dict[split_name] = self._get_feature_frame(
                split_df,
                for_categorical_backend=False,
            )

        if self.lr_feature_mode == "numeric":
            self.backend_data_mode = "pandas_numeric"
            self.model_features = list(self.features)
            for split_name, raw_frame in self.raw_feature_frame_dict.items():
                _validate_numeric_pandas(raw_frame, "MarsLogisticRegressionStrategy")
                self.feature_frame_dict[split_name] = (
                    raw_frame.loc[:, self.model_features]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                )
            return

        self.backend_data_mode = f"pandas_{self.lr_binning_type}_woe"
        self.model_features = [f"{feature}_woe" for feature in self.features]
        binner = self._resolve_binner()
        train_X = self.raw_feature_frame_dict["train"]
        train_y = self._get_target_array(self.data_dict["train"])
        binner.fit(train_X, train_y)
        binner.set_output("polars")
        self.lr_binner = binner

        for split_name, raw_frame in self.raw_feature_frame_dict.items():
            woe_frame = binner.transform(raw_frame, return_type="woe")
            if isinstance(woe_frame, pl.DataFrame):
                woe_frame = woe_frame.to_pandas()
            missing = sorted(set(self.model_features).difference(woe_frame.columns))
            if missing:
                raise ValueError(f"WOE transform did not produce required columns: {missing}")
            self.feature_frame_dict[split_name] = (
                woe_frame.loc[:, self.model_features]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
        binner._cache_X = None
        binner._cache_y = None

    def _resolve_binner(self) -> MarsBinnerBase:
        """Return a user supplied or internally constructed LR binner."""
        if self.lr_binner is not None:
            return self.lr_binner

        kwargs = {
            "features": list(self.features),
            "cat_features": list(self.categorical_features),
            **self.lr_binner_kwargs,
        }
        if self.lr_binning_type in {"opt", "optimal"}:
            return MarsOptimalBinner(**kwargs)
        return MarsNativeBinner(**kwargs)

    def get_default_space(self) -> dict[str, Any]:
        """
        Return the lightweight LR hyperparameter search space.

        Returns
        -------
        dict of str to Any
            Optuna-compatible search space for sklearn LogisticRegression.
        """
        return {
            "C": ("float", 0.1, 5.0, 0.1),
            "penalty": ("categorical", ["l1", "l2"]),
            "class_weight": ("categorical", [None, "balanced"]),
            "max_iter": 500,
        }

    def train_model(
        self,
        trial: Any,
        params: dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> MarsLogisticModel:
        """
        Train a sklearn LogisticRegression model.

        Parameters
        ----------
        trial : Any
            Current Optuna trial. Unused by LR v1.
        params : dict of str to Any
            Concrete LR hyperparameters.
        startup_trials : int
            Pruning warmup count. Unused by LR v1.
        training_metric : str
            Training metric name. Unused by LR v1.

        Returns
        -------
        MarsLogisticModel
            Serializable LR model wrapper.
        """
        sklearn_linear = require_optional_module("sklearn.linear_model")
        logistic_cls = sklearn_linear.LogisticRegression

        train_params = {
            "solver": "liblinear",
            "random_state": self.seed,
        }
        train_params.update(params)
        estimator = logistic_cls(**train_params)
        estimator.fit(self.feature_frame_dict["train"], self._get_target_array(self.data_dict["train"]))
        return MarsLogisticModel(
            estimator=estimator,
            features=list(self.features),
            model_features=list(self.model_features),
            lr_feature_mode=self.lr_feature_mode,
            binner=self.lr_binner if self.lr_feature_mode == "woe" else None,
        )

    def predict_scores(self, model: MarsLogisticModel, split_name: str) -> np.ndarray:
        """
        Predict positive-class probabilities for one split.

        Parameters
        ----------
        model : MarsLogisticModel
            Trained LR model wrapper.
        split_name : str
            Dataset split name.

        Returns
        -------
        numpy.ndarray
            Positive-class probabilities.
        """
        proba = model.estimator.predict_proba(self.feature_frame_dict[split_name])
        return np.asarray(proba[:, 1])

    def extract_importance(self, model: MarsLogisticModel) -> pd.DataFrame:
        """Return a normalized absolute-coefficient importance table."""
        coefficients = np.ravel(model.estimator.coef_)
        importance_map = {
            feature: float(abs(coef))
            for feature, coef in zip(self.features, coefficients, strict=False)
        }
        return _build_importance_table(
            model_type="lr",
            importance_type="abs_coef",
            features=list(self.features),
            importance_map=importance_map,
        )

    def extract_diagnostics(self, model: MarsLogisticModel) -> dict[str, pd.DataFrame]:
        """Return statsmodels coefficient and model summary diagnostics."""
        sm = require_optional_module("statsmodels.api")
        X = self.feature_frame_dict["train"].copy()
        y = self._get_target_array(self.data_dict["train"])
        X_const = sm.add_constant(X, has_constant="add")

        try:
            result = sm.Logit(y, X_const).fit(disp=False)
            params = result.params.reindex(["const", *self.model_features])
            pvalues = result.pvalues.reindex(["const", *self.model_features])
            stderr = result.bse.reindex(["const", *self.model_features])
            converged = bool(result.mle_retvals.get("converged", False))
            aic = float(result.aic)
            bic = float(result.bic)
        except Exception:
            params = pd.Series(
                [float(model.estimator.intercept_[0]), *np.ravel(model.estimator.coef_).tolist()],
                index=["const", *self.model_features],
            )
            pvalues = pd.Series(np.nan, index=params.index)
            stderr = pd.Series(np.nan, index=params.index)
            converged = False
            aic = np.nan
            bic = np.nan

        rows = []
        for output_feature, model_feature in zip(self.features, self.model_features, strict=False):
            coef = float(params.get(model_feature, np.nan))
            rows.append(
                {
                    "feature": output_feature,
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "p_value": float(pvalues.get(model_feature, np.nan)),
                    "std_err": float(stderr.get(model_feature, np.nan)),
                    "odds_ratio": float(np.exp(coef)) if np.isfinite(coef) else np.nan,
                }
            )

        model_summary = pd.DataFrame(
            [
                {
                    "aic": aic,
                    "bic": bic,
                    "nobs": int(len(y)),
                    "n_features": int(len(self.features)),
                    "converged": converged,
                    "lr_feature_mode": self.lr_feature_mode,
                    "lr_binning_type": self.lr_binning_type if self.lr_feature_mode == "woe" else None,
                }
            ]
        )
        return {
            "coefficients": pd.DataFrame(rows),
            "model_summary": model_summary,
        }
