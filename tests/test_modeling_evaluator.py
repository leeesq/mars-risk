from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mars.modeling import MarsModelingSession
from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.report import MarsModelingReport
from mars.modeling.results import MarsModelTuningResult


def test_modeling_session_evaluate_generates_report_and_can_export(sample_modeling_pd, tmp_path: Path):
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))

    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
    )
    report = session.evaluate(
        df,
        pred_col="pred_score",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )

    assert isinstance(report, MarsModelingReport)
    assert ("Target: target", "New KS") in report.summary_table.columns
    assert ("Target: target", "LogLoss") in report.summary_table.columns
    assert ("Target: target", "Brier") in report.summary_table.columns
    assert ("Target: target", "Score PSI") in report.summary_table.columns
    assert "decile_lift" in report.detail_tables
    assert "score_psi" in report.detail_tables
    assert "roc_curve" in report.detail_tables
    assert "ks_curve" in report.detail_tables
    assert "calibration_curve" in report.detail_tables
    assert "score_distribution" in report.detail_tables
    assert "feature_psi" in report.detail_tables
    assert "Gini" not in {col[-1] if isinstance(col, tuple) else col for col in report.summary_table.columns}
    assert ("Time Period", "Start Time") in report.summary_table.columns
    assert report.metadata["feature_cols"] == ["x1", "x2", "x3"]
    assert report.to_pandas().equals(report.summary_table)
    assert list(report.summary_table.index[:3]) == ["train", "val", "oot1"]

    export_path = tmp_path / "model_report.xlsx"
    report.write_excel(str(export_path))
    assert export_path.exists()

    styled = report.show_summary()
    assert hasattr(styled, "to_html")


def test_model_evaluator_generates_same_report_shape(sample_modeling_pd):
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))

    evaluator = MarsModelEvaluator()
    report = evaluator.evaluate(
        df,
        pred_col="pred_score",
        group_col="dataset_flag",
        target="target",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )

    assert isinstance(report, MarsModelingReport)
    assert ("Target: target", "New AUC") in report.summary_table.columns
    assert ("Target: target", "Top 10% Capture") in report.summary_table.columns
    assert not report.detail_tables["decile_lift"].empty
    assert "feature_psi" not in report.detail_tables
    assert list(report.summary_table.index[:3]) == ["train", "val", "oot1"]


def test_modeling_report_to_html_writes_single_file(sample_modeling_pd, tmp_path: Path):
    pytest.importorskip("matplotlib")
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))
    importance = pd.DataFrame(
        {
            "feature": ["x1", "x2", "x3"],
            "importance": [3.0, 2.0, 1.0],
            "importance_type": ["gain", "gain", "gain"],
            "model_type": ["xgb", "xgb", "xgb"],
            "rank": [1, 2, 3],
        }
    )

    report = MarsModelEvaluator().evaluate(
        df,
        pred_col="pred_score",
        group_col="dataset_flag",
        target="target",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
        feature_cols=["x1", "x2", "x3"],
        importance_table=importance,
    )
    report.metadata["feature_growth_summary"] = pd.DataFrame(
        {
            "feature_count": [1, 2, 3],
            "status": ["complete", "complete", "complete"],
            "train_ks": [45.0, 50.0, 52.0],
            "val_ks": [42.0, 48.0, 47.0],
            "selection_score": [42.0, 48.0, 47.0],
            "is_best": [False, True, False],
        }
    )
    report.metadata["feature_growth_selection_metric"] = "ks"
    report.metadata["feature_growth_best_step"] = 2

    scorecard = type(
        "DummyScorecard",
        (),
        {
            "pdo": 50,
            "base_score": 600,
            "base_odds": 20,
            "base_points": 512,
            "factor": 72.1,
            "offset": 384.2,
            "intercept": -0.5,
            "points_table": pd.DataFrame({"feature": ["x1"], "bin_index": [0], "points": [12]}),
        },
    )()
    html_path = report.to_html(tmp_path / "model_report.html", scorecard=scorecard)
    html_text = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    for section_name in [
        "Executive Summary",
        "Discrimination",
        "Calibration",
        "Stability",
        "Explainability",
        "Tuning Audit",
        "Feature Growth Audit",
        "Scorecard",
    ]:
        assert section_name in html_text
    assert "base_points" in html_text
    assert "data:image/png;base64" in html_text
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert "plotly" not in html_text.lower()
    assert "Gini" not in html_text


def test_modeling_session_evaluate_attaches_last_run_metadata(sample_modeling_pd):
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))
    importance = pd.DataFrame(
        {
            "feature": ["x1", "x2", "x3"],
            "importance": [3.0, 2.0, 1.0],
            "importance_type": ["gain", "gain", "gain"],
            "model_type": ["xgb", "xgb", "xgb"],
            "rank": [1, 2, 3],
        }
    )
    history = pd.DataFrame(
        {
            "trial_num": [0, 1],
            "trial_state": ["COMPLETE", "COMPLETE"],
            "is_valid": [True, True],
            "val_ks": [0.1, 0.2],
        }
    )
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
    )
    session.tuner.last_run = MarsModelTuningResult(
        model_type="xgb",
        optimize_metric="ks",
        features=["x1", "x2", "x3"],
        target="target",
        dataset_flag_col="dataset_flag",
        categorical_features=[],
        best_params={"max_depth": 3},
        best_iteration=12,
        best_model=None,
        best_score=0.2,
        history_table=history,
        history_path="history.csv",
        study=None,
        replay_candidates=["max_depth"],
        importance_table=importance,
        training_config={"training_metric": "ks"},
        library_versions={"xgboost": "test"},
        feature_schema={"x1": "float"},
        backend_data_mode="pandas_numeric",
    )

    report = session.evaluate(
        df,
        pred_col="pred_score",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )

    assert report.metadata["backend_data_mode"] == "pandas_numeric"
    assert report.metadata["training_config"] == {"training_metric": "ks"}
    assert report.metadata["library_versions"] == {"xgboost": "test"}
    assert report.metadata["history_table"].equals(history)
    assert report.metadata["importance_table"].equals(importance)
