"""LogisticRegression 建模后端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.common import build_importance_table as _build_importance_table
from mars.modeling.backends.common import validate_numeric_pandas as _validate_numeric_pandas
from mars.modeling.metrics import MetricCallable, MetricDirection
from mars.modeling.utils import require_optional_module

LR_FEATURE_MODE = Literal["numeric", "woe"]
LR_BINNING_TYPE = Literal["native", "opt", "optimal"]


@dataclass(slots=True)
class MarsLogisticModel:
    """
    可序列化的 LR 模型包装器。

    Attributes
    ----------
    estimator : Any
        已训练的 ``LogisticRegression`` 实例或兼容对象。
    features : list of str
        用户传入的原始特征名。
    model_features : list of str
        LR 实际消费的数值特征名。
    lr_feature_mode : str
        特征预处理模式。
    binner : MarsBinnerBase or None
        WOE 模式下复用的分箱器。

    Examples
    --------
    >>> model = MarsLogisticModel(estimator=object(), features=["age"], model_features=["age"])
    >>> model.features
    ['age']
    """

    estimator: Any
    features: list[str]
    model_features: list[str]
    lr_feature_mode: str = "numeric"
    binner: MarsBinnerBase | None = None

    def _to_pandas(self, X: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """将 Pandas 或 Polars 输入复制为 Pandas DataFrame。"""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

    def transform_features(self, X: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """
        返回 ``LogisticRegression`` 实际消费的数值特征矩阵。

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame
            待转换的原始特征数据。

        Returns
        -------
        pandas.DataFrame
            按 ``model_features`` 排列并已数值化的特征矩阵。

        Raises
        ------
        ValueError
            当输入缺少必要特征，或 WOE 模式缺少已拟合分箱器时抛出。

        Examples
        --------
        >>> model = MarsLogisticModel(estimator=object(), features=["age"], model_features=["age"])
        >>> model.transform_features(pd.DataFrame({"age": [20, 30]})).shape
        (2, 1)
        """
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
        """
        通过已保存的预处理路径预测正类概率。

        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame
            待评分特征数据。

        Returns
        -------
        numpy.ndarray
            ``estimator.predict_proba`` 的二维概率输出。

        Examples
        --------
        >>> class DummyEstimator:
        ...     def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...         probs = X["age"].to_numpy(dtype=float) / 100.0
        ...         return np.column_stack([1.0 - probs, probs])
        >>> model = MarsLogisticModel(
        ...     estimator=DummyEstimator(),
        ...     features=["age"],
        ...     model_features=["age"],
        ... )
        >>> probabilities = model.predict_proba(pd.DataFrame({"age": [20, 30]}))
        >>> probabilities.round(2).tolist()
        [[0.8, 0.2], [0.7, 0.3]]
        """
        model_frame = self.transform_features(X)
        return np.asarray(self.estimator.predict_proba(model_frame))


class MarsLogisticRegressionStrategy(MarsBaseModelStrategy):
    """
    面向传统银行评分卡体系的 LR 建模后端。

    Attributes
    ----------
    lr_feature_mode : str
        LR 特征预处理模式。
    lr_binning_type : str
        WOE 模式下使用的分箱器类型。
    lr_binner_kwargs : dict
        内部分箱器初始化参数。
    lr_binner : MarsBinnerBase or None
        用户传入或内部创建的分箱器。
    model_features : list of str
        LR 实际训练和预测使用的特征名。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "age": [20, 30, 40, 50, 60, 70],
    ...     "y": [0, 1, 0, 1, 0, 1],
    ...     "dataset_flag": ["train", "train", "train", "train", "val", "val"],
    ... })
    >>> strategy = MarsLogisticRegressionStrategy(df, features=["age"], target="y")
    >>> strategy.lr_feature_mode
    'numeric'
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
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
        keep_top_n_models: int = 0,
        lr_feature_mode: LR_FEATURE_MODE = "numeric",
        lr_binning_type: LR_BINNING_TYPE = "native",
        lr_binner_kwargs: Mapping[str, Any] | None = None,
        lr_binner: MarsBinnerBase | None = None,
    ) -> None:
        """
        初始化 LR 后端并校验数值/WOE 特征模式。

        除基类调参状态外，该方法还保存 LR 专属分箱配置，并确保
        ``lr_feature_mode`` 与 ``lr_binning_type`` 落在受支持集合中。

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            已包含特征、目标列和样本切片列的建模样本。
        features : Sequence[str]
            参与训练的特征列名。
        target : str
            主训练目标列名。
        optimize_metric : str
            trial 目标函数使用的优化指标，可以是内置指标或自定义指标名。
        param_space : Mapping[str, Any] | None
            LR 后端参数搜索空间覆盖项。
        max_diff : float
            train 与 validation 指标允许的最大泛化差异，单位是百分点。
        seed : int
            随机种子。
        use_oot_penalty : bool
            是否将 OOT 衰减纳入 trial 有效性判断。
        dataset_flag_col : str
            建模样本切片列名。
        categorical_features : Sequence[str] | None
            类别特征列名；LR numeric 模式要求这些列已完成数值化。
        metric_params : Mapping[str, Any] | None
            指标参数，例如 ``f1_threshold``。
        custom_metrics : Mapping[str, MetricCallable] | None
            自定义指标函数字典。
        metric_directions : Mapping[str, MetricDirection] | None
            指标排序方向，未配置时默认按 maximize 处理。
        training_metric : str | None
            模型后端训练期监控指标。
        backend_metric : Any | None
            预留给后端原生 metric 的透传入口；LR 后端当前不直接消费。
        keep_top_n_models : int
            调参过程中动态保留的最优模型数量。
        lr_feature_mode : LR_FEATURE_MODE
            LR 特征模式，支持 ``numeric`` 和 ``woe``。
        lr_binning_type : LR_BINNING_TYPE
            WOE 模式使用的分箱器类型。
        lr_binner_kwargs : Mapping[str, Any] | None
            构造 WOE 分箱器时使用的参数。
        lr_binner : MarsBinnerBase | None
            显式传入的已配置分箱器。

        Raises
        ------
        ValueError
            当 LR 特征模式、分箱类型或基类输入配置不合法时抛出。
        """
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
            metric_params=metric_params,
            custom_metrics=custom_metrics,
            metric_directions=metric_directions,
            training_metric=training_metric,
            backend_metric=backend_metric,
            keep_top_n_models=keep_top_n_models,
        )

    def _build_backend_data(self) -> None:
        """准备 LR 训练和评分所需的分切片 Pandas 特征矩阵。"""
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
        binner.fit(
            train_X,
            train_y,
            features=list(self.features),
            cat_features=list(self.categorical_features),
        )
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

    def _resolve_binner(self) -> MarsNativeBinner | MarsOptimalBinner:
        """返回用户传入或内部构建的 LR WOE 分箱器。"""
        if self.lr_binner is not None:
            return cast(MarsNativeBinner | MarsOptimalBinner, self.lr_binner)

        kwargs = dict(self.lr_binner_kwargs)
        if self.lr_binning_type in {"opt", "optimal"}:
            return MarsOptimalBinner(**kwargs)
        return MarsNativeBinner(**kwargs)

    def get_default_space(self) -> dict[str, Any]:
        """
        返回轻量级 LR 超参数搜索空间。

        Returns
        -------
        dict of str to Any
            兼容 Optuna 的 ``sklearn.linear_model.LogisticRegression`` 搜索空间。

        Examples
        --------
        >>> strategy = object.__new__(MarsLogisticRegressionStrategy)
        >>> strategy.get_default_space()["max_iter"]
        500
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
        训练单次 ``sklearn.linear_model.LogisticRegression`` 模型。

        Parameters
        ----------
        trial : Any
            当前 Optuna Trial。LR v1 暂不直接使用。
        params : dict[str, Any]
            当前 Trial 解析后的 LR 超参数。
        startup_trials : int
            剪枝预热 Trial 数量。LR v1 暂不直接使用。
        training_metric : str
            训练期监控指标名。LR v1 暂不直接使用。

        Returns
        -------
        MarsLogisticModel
            可序列化的 LR 模型包装器。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "age": [20, 30, 40, 50, 60, 70],
        ...     "y": [0, 1, 0, 1, 0, 1],
        ...     "dataset_flag": ["train", "train", "train", "train", "val", "val"],
        ... })
        >>> strategy = MarsLogisticRegressionStrategy(df, features=["age"], target="y")
        >>> params = {"C": 1.0, "penalty": "l2", "max_iter": 100}
        >>> lr_model = strategy.train_model(None, params, 0, "auc")
        >>> isinstance(lr_model, MarsLogisticModel)
        True
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
        对指定数据切片预测正类概率。

        Parameters
        ----------
        model : MarsLogisticModel
            已训练的 LR 模型包装器。
        split_name : str
            数据切片名称。

        Returns
        -------
        numpy.ndarray
            正类概率一维数组。

        Examples
        --------
        >>> class DummyEstimator:
        ...     def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...         probs = X["age"].to_numpy(dtype=float) / 100.0
        ...         return np.column_stack([1.0 - probs, probs])
        >>> strategy = object.__new__(MarsLogisticRegressionStrategy)
        >>> strategy.feature_frame_dict = {"val": pd.DataFrame({"age": [20, 30]})}
        >>> lr_model = MarsLogisticModel(DummyEstimator(), ["age"], ["age"])
        >>> scores = strategy.predict_scores(lr_model, "val")
        >>> scores.round(2).tolist()
        [0.2, 0.3]
        """
        proba = model.estimator.predict_proba(self.feature_frame_dict[split_name])
        return np.asarray(proba[:, 1])

    def extract_importance(self, model: MarsLogisticModel) -> pd.DataFrame:
        """
        返回按绝对系数归一化后的特征重要性表。

        Parameters
        ----------
        model : MarsLogisticModel
            已训练的 LR 模型包装器。

        Returns
        -------
        pandas.DataFrame
            MARS 统一格式的重要性表。

        Examples
        --------
        >>> class DummyEstimator:
        ...     coef_ = np.array([[0.4]])
        >>> strategy = object.__new__(MarsLogisticRegressionStrategy)
        >>> strategy.features = ["age"]
        >>> lr_model = MarsLogisticModel(DummyEstimator(), ["age"], ["age"])
        >>> importance = strategy.extract_importance(lr_model)
        >>> importance.loc[0, "feature"]
        'age'
        """
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
        """
        返回 statsmodels 系数诊断和模型摘要。

        Parameters
        ----------
        model : MarsLogisticModel
            已训练的 LR 模型包装器。

        Returns
        -------
        dict of str to pandas.DataFrame
            包含 ``coefficients`` 与 ``model_summary`` 两张诊断表。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "age": [20, 30, 40, 50, 60, 70],
        ...     "y": [0, 1, 0, 1, 0, 1],
        ...     "dataset_flag": ["train", "train", "train", "train", "val", "val"],
        ... })
        >>> strategy = MarsLogisticRegressionStrategy(df, features=["age"], target="y")
        >>> lr_model = strategy.train_model(None, {"C": 1.0, "penalty": "l2"}, 0, "auc")
        >>> diagnostics = strategy.extract_diagnostics(lr_model)
        >>> set(diagnostics)
        {'coefficients', 'model_summary'}
        """
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
