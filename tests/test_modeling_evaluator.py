from pathlib import Path

import numpy as np

from mars.modeling import MarsModelEvaluator, MarsModelingReport, MarsModelingSession


def test_modeling_session_evaluate_generates_report_and_can_export(sample_modeling_pd, tmp_path: Path):
    df = sample_modeling_pd.copy()
    df["pred_score"] = 1 / (1 + np.exp(-(1.8 * df["x1"] - 1.0 * df["x2"] + 0.6 * df["x3"])))

    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )
    report = session.evaluate(df, pred_col="pred_score")

    assert isinstance(report, MarsModelingReport)
    assert ("Target: target", "New KS") in report.summary_table.columns
    assert ("Target: target", "LogLoss") in report.summary_table.columns
    assert ("Target: target", "Brier") in report.summary_table.columns
    assert ("Target: target", "Score PSI") in report.summary_table.columns
    assert "decile_lift" in report.detail_tables
    assert "score_psi" in report.detail_tables
    assert ("Time Period", "Start Time") in report.summary_table.columns
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

    evaluator = MarsModelEvaluator(
        group_col="dataset_flag",
        target_col="target",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )
    report = evaluator.evaluate(df, pred_col="pred_score")

    assert isinstance(report, MarsModelingReport)
    assert ("Target: target", "New AUC") in report.summary_table.columns
    assert ("Target: target", "Top 10% Capture") in report.summary_table.columns
    assert not report.detail_tables["decile_lift"].empty
    assert list(report.summary_table.index[:3]) == ["train", "val", "oot1"]
