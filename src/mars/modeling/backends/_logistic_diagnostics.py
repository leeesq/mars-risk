"""LR 诊断表 helper。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mars.utils.imports import require_optional_module


def build_logistic_diagnostics(strategy: Any, model: Any) -> dict[str, pd.DataFrame]:
    """构建 LR 系数诊断表和模型摘要表。"""
    sm = require_optional_module("statsmodels.api")
    X = strategy.feature_frame_dict["train"].copy()
    y = strategy._get_target_array(strategy.data_dict["train"])
    X_const = sm.add_constant(X, has_constant="add")

    try:
        result = sm.Logit(y, X_const).fit(disp=False)
        params = result.params.reindex(["const", *strategy.model_features])
        pvalues = result.pvalues.reindex(["const", *strategy.model_features])
        stderr = result.bse.reindex(["const", *strategy.model_features])
        converged = bool(result.mle_retvals.get("converged", False))
        aic = float(result.aic)
        bic = float(result.bic)
    except Exception:
        params = pd.Series(
            [float(model.estimator.intercept_[0]), *np.ravel(model.estimator.coef_).tolist()],
            index=["const", *strategy.model_features],
        )
        pvalues = pd.Series(np.nan, index=params.index)
        stderr = pd.Series(np.nan, index=params.index)
        converged = False
        aic = np.nan
        bic = np.nan

    rows = []
    for output_feature, model_feature in zip(strategy.features, strategy.model_features, strict=False):
        coef = float(params.get(model_feature, np.nan))
        rows.append(
            {
                "feature": output_feature,
                "model_feature": model_feature,
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
                "n_features": int(len(strategy.features)),
                "converged": converged,
                "lr_feature_mode": strategy.lr_feature_mode,
                "lr_binning_type": strategy.lr_binning_type if strategy.lr_feature_mode == "woe" else None,
            }
        ]
    )
    return {"coefficients": pd.DataFrame(rows), "model_summary": model_summary}
