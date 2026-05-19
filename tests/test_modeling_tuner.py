from pathlib import Path

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")

import mars
import mars.modeling as modeling
from mars.modeling import report as report_module
from mars.modeling import results as results_module
from mars.modeling import tuner as tuner_module
from mars.modeling import MarsModelTuner, MarsModelingRun, MarsModelingSession


@pytest.mark.parametrize("model_type", ["xgb", "lgb", "cbt"])
def test_modeling_session_tune_runs_for_all_backends(sample_modeling_pd, tmp_path: Path, model_type: str):
    session = MarsModelingSession(
        model_type=model_type,
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=11,
    )
    save_path = tmp_path / f"{model_type}_history.csv"
    result = session.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=2,
        startup_trials=1,
        warmup_steps=5,
        num_boost_round=30,
        early_stopping_rounds=8,
        save_path=str(save_path),
    )

    assert isinstance(result, MarsModelingRun)
    assert save_path.exists()
    assert session.last_run is result
    assert session.last_run.best_model is not None
    assert session.best_model is result.best_model
    assert session.best_score == result.best_score
    assert session.best_params == result.best_params
    assert session.history_table.equals(result.history_table)
    assert result.best_params
    assert set(result.importance_table.columns) == {"feature", "importance", "importance_type", "model_type", "rank"}
    assert not result.importance_table.empty
    assert "val_ks" in result.history_table.columns
    assert result.history_path.endswith(f"{model_type}_history.csv")
    assert result.replay_candidates
    assert result.training_config["training_metric"] == "ks"
    assert result.feature_schema
    assert result.library_versions
    if model_type in {"xgb", "lgb"}:
        assert result.backend_data_mode == "pandas_numeric"


def test_model_tuner_tune_matches_session_result_contract(sample_modeling_pd, tmp_path: Path):
    tuner = MarsModelTuner(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=21,
    )
    result = tuner.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / "tool_history.csv"),
    )

    assert isinstance(result, MarsModelingRun)
    assert tuner.last_run is result
    assert tuner.best_model is result.best_model
    assert tuner.best_params == result.best_params
    assert tuner.history_table.equals(result.history_table)


def test_modeling_run_artifact_roundtrip(sample_modeling_pd, tmp_path: Path):
    tuner = MarsModelTuner(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=22,
    )
    result = tuner.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / "artifact_history.csv"),
    )

    artifact_dir = result.write_artifact(str(tmp_path / "run_artifact"))
    loaded = MarsModelingRun.load_artifact(str(artifact_dir))

    assert loaded.study is None
    assert loaded.best_model is not None
    assert loaded.best_params == result.best_params
    assert loaded.history_table.equals(result.history_table)
    assert loaded.importance_table.equals(result.importance_table)
    assert loaded.training_config == result.training_config
    assert loaded.feature_schema == result.feature_schema
    assert loaded.backend_data_mode == result.backend_data_mode


def test_modeling_run_load_artifact_requires_metadata(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        MarsModelingRun.load_artifact(str(tmp_path / "missing_artifact"))


def test_modeling_public_exports_only_include_formal_api():
    assert "MarsModelingSession" in modeling.__all__
    assert "MarsModelTuner" in modeling.__all__
    assert "MarsModelEvaluator" in modeling.__all__
    assert "MarsModelReplay" in modeling.__all__
    assert "MarsModelDataSlicer" in modeling.__all__
    assert "MarsModelingRun" in modeling.__all__
    assert "MarsModelingReport" in modeling.__all__
    assert "MarsReplayRun" in modeling.__all__
    assert "MarsAutoModelTuner" not in modeling.__all__
    assert not hasattr(tuner_module, "MarsAutoModelTuner")
    assert not hasattr(report_module, "MarsModelEvaluationReport")
    assert not hasattr(results_module, "MarsTuningResult")
    assert not hasattr(results_module, "MarsPostTuningResult")
    assert "MarsModelTuner" in mars.__all__
    assert "MarsModelEvaluator" in mars.__all__
    assert "MarsModelReplay" in mars.__all__


def test_modeling_session_tune_records_oot_penalty_columns(sample_modeling_pd, tmp_path: Path):
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="auc",
        seed=12,
    )

    result = session.tune(
        sample_modeling_pd,
        use_oot_penalty=True,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / "oot_history.csv"),
    )

    assert "max_oot_diff" in result.history_table.columns
    assert result.best_score == session.last_run.best_score


def test_modeling_session_tune_raises_when_validation_split_missing(sample_modeling_pd):
    bad_df = sample_modeling_pd.loc[sample_modeling_pd["dataset_flag"] != "val"].copy()

    with pytest.raises(ValueError, match="validation"):
        MarsModelingSession(
            model_type="xgb",
            features=["x1", "x2", "x3"],
            target="target",
        ).tune(bad_df)


def test_modeling_session_tune_raises_for_invalid_optimize_metric():
    with pytest.raises(ValueError, match="optimize_metric"):
        MarsModelingSession(
            model_type="xgb",
            features=["x1", "x2", "x3"],
            target="target",
            optimize_metric="f1",
        )


def test_modeling_session_tune_uses_lowercase_contains_dataset_flags(sample_modeling_pd, tmp_path: Path):
    df = sample_modeling_pd.copy()
    df["dataset_flag"] = df["dataset_flag"].map(
        {
            "train": "SAMPLE_TRAIN_V1",
            "val": "Validation_Window",
            "oot1": "OOT_202403",
        }
    )

    result = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=33,
    ).tune(
        df,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / "contains_history.csv"),
    )

    assert "OOT_202403_ks" in result.history_table.columns


def test_modeling_session_tune_rejects_ambiguous_dataset_flags(sample_modeling_pd):
    df = sample_modeling_pd.copy()
    df.loc[df["dataset_flag"] == "train", "dataset_flag"] = "train_val_mix"

    with pytest.raises(ValueError, match="Ambiguous dataset_flag"):
        MarsModelingSession(
            model_type="xgb",
            features=["x1", "x2", "x3"],
            target="target",
        ).tune(df, n_trials=1)
