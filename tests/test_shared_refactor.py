from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.analysis import profile_risk
from mars.compute import to_pandas_frame, to_pandas_table, to_polars_frame
from mars.modeling import MarsModelEvaluator


def test_frame_helpers_preserve_pandas_and_polars_inputs() -> None:
    pandas_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    polars_df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})

    assert to_polars_frame(pandas_df).shape == (2, 2)
    assert to_pandas_frame(polars_df).shape == (2, 2)
    assert to_pandas_table(None).empty

    copied = to_pandas_frame(pandas_df)
    copied.loc[0, "x"] = 99
    assert pandas_df.loc[0, "x"] == 1


def test_pipeline_and_logistic_do_not_keep_local_frame_conversion_wheels() -> None:
    import mars.pipeline.steps as pipeline_steps
    from mars.modeling.backends.logistic import MarsLogisticModel
    from mars.pipeline import MarsModelingPipeline

    assert not hasattr(MarsModelingPipeline, "_to_polars")
    assert not hasattr(MarsLogisticModel, "_to_pandas")
    assert "_to_polars" not in vars(pipeline_steps)


def test_model_evaluator_reuses_binning_psi_without_special_value_parameter(
    sample_modeling_pd: pd.DataFrame,
) -> None:
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))
    df.loc[(df["dataset_flag"] == "val") & (df.index % 3 == 0), "pred_score"] = np.nan

    report = MarsModelEvaluator().evaluate(
        df,
        pred_col="pred_score",
        group_col="dataset_flag",
        target="target",
        psi_include_missing=True,
    )
    risk_profile = profile_risk(
        df,
        target="target",
        features=["pred_score"],
        group_col="dataset_flag",
        binning_type="native",
        binner_params={"method": "quantile", "n_bins": 10},
        psi_include_missing=True,
    )
    risk_trend = risk_profile.report.trend_tables["psi"]
    risk_score_psi = float(
        risk_trend.loc[risk_trend["feature"] == "pred_score", "val"].iloc[0]
    )

    assert report.summary_table.loc["val", ("Target: target", "Score PSI")] == pytest.approx(
        risk_score_psi
    )
    assert "psi_include_special" not in inspect.signature(MarsModelEvaluator.evaluate).parameters


def test_modeling_artifact_path_helpers_are_centralized() -> None:
    import mars.modeling as modeling
    from mars.modeling.artifacts import create_artifact_path, safe_artifact_part, step_artifact_dir

    assert safe_artifact_part("Long Target@AUC") == "long_target_auc"
    assert create_artifact_path(None, model_type="lgb", target="y", optimize_metric="ks", run_id="x") is None
    assert step_artifact_dir("root", 12).endswith("features_12")
    assert "_create_artifact_path" not in vars(modeling)
    assert "_safe_artifact_part" not in vars(modeling)
    assert not hasattr(modeling.MarsFeatureIncrementalTuner, "_step_artifact_dir")
