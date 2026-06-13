"""LogisticRegression 建模后端。"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame
from mars.feature.base import MarsBinnerBase
from mars.feature.lite_opt_binner import MarsLiteOptBinner
from mars.feature.native_binner import MarsNativeBinner
from mars.feature.optimal_binner import MarsOptimalBinner
from mars.modeling.backends._logistic_diagnostics import build_logistic_diagnostics
from mars.modeling.backends._logistic_model import MarsLogisticModel
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.common import build_importance_table as _build_importance_table
from mars.modeling.backends.common import validate_numeric_pandas as _validate_numeric_pandas
from mars.modeling.backends.registry import register_backend
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection
from mars.utils.imports import require_optional_module

LR_FEATURE_MODE = Literal["numeric", "woe"]
LR_BINNING_TYPE = Literal["native", "optimal", "lite_opt"]


@register_backend("lr", "logit", "logistic", "logistic_regression")
class MarsLogisticRegressionStrategy(MarsBaseModelStrategy):
    """面向评分卡链路的 LR 建模后端。"""

    SUPPORTED_FEATURE_MODES = {"numeric", "woe"}
    SUPPORTED_BINNING_TYPES = {"native", "optimal", "lite_opt"}

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
        """初始化 LR 后端。"""
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
                "lr_binning_type must be one of {'native', 'optimal', 'lite_opt'}, "
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
        """准备 LR 训练和评分所需的 Pandas 特征矩阵。"""
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
            woe_frame = to_pandas_frame(binner.transform(raw_frame, return_type="woe"))
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

    def _resolve_binner(self) -> MarsNativeBinner | MarsOptimalBinner | MarsLiteOptBinner:
        """返回用户传入或内部构建的 LR WOE 分箱器。"""
        if self.lr_binner is not None:
            return cast(MarsNativeBinner | MarsOptimalBinner | MarsLiteOptBinner, self.lr_binner)

        kwargs = dict(self.lr_binner_kwargs)
        if self.lr_binning_type == "optimal":
            return MarsOptimalBinner(**kwargs)
        if self.lr_binning_type == "lite_opt":
            return MarsLiteOptBinner(**kwargs)
        return MarsNativeBinner(**kwargs)

    def get_default_space(self) -> dict[str, Any]:
        """返回 LR 的默认搜索空间。"""
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
        """训练单次 LR 模型。"""
        del trial, startup_trials, training_metric
        sklearn_linear = require_optional_module("sklearn.linear_model")
        logistic_cls = sklearn_linear.LogisticRegression

        train_params = {"solver": "liblinear", "random_state": self.seed}
        train_params.update(params)
        estimator = logistic_cls(**train_params)
        estimator.fit(
            self.feature_frame_dict["train"],
            self._get_target_array(self.data_dict["train"]),
        )
        return MarsLogisticModel(
            estimator=estimator,
            features=list(self.features),
            model_features=list(self.model_features),
            lr_feature_mode=self.lr_feature_mode,
            binner=self.lr_binner if self.lr_feature_mode == "woe" else None,
        )

    def predict_scores(self, model: MarsLogisticModel, split_name: str) -> np.ndarray:
        """预测指定切片的正类概率。"""
        proba = model.estimator.predict_proba(self.feature_frame_dict[split_name])
        return np.asarray(proba[:, 1])

    def extract_importance(self, model: MarsLogisticModel) -> pd.DataFrame:
        """返回按绝对系数归一后的特征重要性表。"""
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
        """返回 LR 系数诊断表和模型摘要。"""
        return build_logistic_diagnostics(self, model)
