import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")
pytest.importorskip("statsmodels")

import mars
import mars.modeling as modeling
from mars.modeling import MarsModelingSession
from mars.modeling import report as report_module
from mars.modeling import results as results_module
from mars.modeling import tuning as tuning_module
from mars.modeling.feature_growth import MarsFeatureGrowthResult, MarsFeatureIncrementalTuner
from mars.modeling.metrics import CatBoostKSMetric, as_probability
from mars.modeling.results import MarsModelTuningResult
from mars.modeling.tuning import MarsModelTuner


@pytest.mark.parametrize("model_type", ["xgb", "lgb", "cbt"])
def test_modeling_session_tune_runs_for_all_backends(sample_modeling_pd, tmp_path: Path, model_type: str):
    session = MarsModelingSession(
        model_type=model_type,
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=11,
    )
    history_path = tmp_path / f"{model_type}_history.csv"
    result = session.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=2,
        startup_trials=1,
        warmup_steps=5,
        num_boost_round=30,
        early_stopping_rounds=8,
        history_path=str(history_path),
    )

    assert isinstance(result, MarsModelTuningResult)
    assert history_path.exists()
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
        history_path=str(tmp_path / "tool_history.csv"),
    )

    assert isinstance(result, MarsModelTuningResult)
    assert tuner.last_run is result
    assert tuner.best_model is result.best_model
    assert tuner.best_params == result.best_params
    assert tuner.history_table.equals(result.history_table)


def test_model_tuner_history_path_none_does_not_write_file(sample_modeling_pd, tmp_path: Path):
    tuner = MarsModelTuner(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=23,
    )
    result = tuner.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        history_path=None,
    )

    assert result.history_path is None
    assert not any(tmp_path.glob("*history*.csv"))


def test_model_tuner_rejects_existing_history_without_overwrite(sample_modeling_pd, tmp_path: Path):
    history_path = tmp_path / "existing_history.csv"
    history_path.write_text("trial_num\n", encoding="utf-8")
    tuner = MarsModelTuner(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=24,
    )

    with pytest.raises(FileExistsError, match="history_path"):
        tuner.tune(
            sample_modeling_pd,
            max_diff=20.0,
            n_trials=1,
            startup_trials=1,
            warmup_steps=3,
            num_boost_round=20,
            early_stopping_rounds=5,
            history_path=history_path,
        )


def test_logistic_regression_numeric_mode_tune_replay_and_artifact(sample_modeling_pd, tmp_path: Path):
    session = MarsModelingSession(
        model_type="lr",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="auc",
        seed=27,
        lr_feature_mode="numeric",
    )
    result = session.tune(
        sample_modeling_pd,
        max_diff=100.0,
        n_trials=2,
        startup_trials=1,
        warmup_steps=3,
        history_path=str(tmp_path / "lr_numeric_history.csv"),
    )

    assert result.model_type == "lr"
    assert result.best_model is not None
    assert result.importance_table["importance_type"].unique().tolist() == ["abs_coef"]
    assert {"coefficients", "model_summary"} == set(result.diagnostic_tables)
    assert result.diagnostic_tables["coefficients"]["feature"].tolist() == ["x1", "x2", "x3"]

    replay = session.replay_runner.run(result, sample_modeling_pd, top_k=1, sort_metric="auc")
    model_name = next(iter(replay.models))
    assert model_name.startswith("top1_trial")
    assert set(replay.diagnostic_tables[model_name]) == {"coefficients", "model_summary"}

    artifact_dir = result.write_artifact(str(tmp_path / "lr_numeric_artifact"))
    loaded = MarsModelTuningResult.load_artifact(str(artifact_dir))
    assert set(loaded.diagnostic_tables) == {"coefficients", "model_summary"}
    assert loaded.importance_table.equals(result.importance_table)


def test_logistic_regression_woe_mode_reuses_binner_in_artifact(sample_modeling_pd, tmp_path: Path):
    session = MarsModelingSession(
        model_type="logistic",
        features=["x1", "x2", "segment"],
        target="target",
        categorical_features=["segment"],
        optimize_metric="ks",
        seed=28,
        lr_feature_mode="woe",
        lr_binner_kwargs={"n_bins": 4, "n_jobs": 1},
    )
    result = session.tune(
        sample_modeling_pd,
        max_diff=100.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        history_path=str(tmp_path / "lr_woe_history.csv"),
    )

    assert result.model_type == "logistic"
    assert result.backend_data_mode == "pandas_native_woe"
    assert result.best_model.binner is not None
    assert result.best_model.lr_feature_mode == "woe"
    assert result.training_config["lr_feature_mode"] == "woe"
    assert set(result.importance_table["feature"]) == {"x1", "x2", "segment"}

    artifact_dir = result.write_artifact(str(tmp_path / "lr_woe_artifact"))
    loaded = MarsModelTuningResult.load_artifact(str(artifact_dir))
    assert loaded.best_model.binner is not None
    assert loaded.best_model.predict_proba(sample_modeling_pd.loc[:, loaded.features]).shape[0] == len(sample_modeling_pd)
    assert set(loaded.diagnostic_tables) == {"coefficients", "model_summary"}


def test_feature_incremental_tuner_resolves_steps_and_feature_order():
    tuner = MarsFeatureIncrementalTuner(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
    )

    assert tuner._resolve_steps(
        total_features=3,
        steps=[2, 2, 9, 0],
        min_features=10,
        max_features=None,
        step_size=None,
    ) == [1, 2, 3]
    assert tuner._resolve_steps(
        total_features=23,
        steps=None,
        min_features=5,
        max_features=12,
        step_size=None,
    ) == [5, 10, 12]

    importance = pd.DataFrame(
        {
            "feature": ["x3", "x1"],
            "rank": [1, 2],
        }
    )
    assert tuner._resolve_feature_order(feature_order=None, importance_table=importance) == ["x3", "x1", "x2"]

    with pytest.raises(ValueError, match="unknown"):
        tuner._resolve_feature_order(feature_order=["x1", "missing"], importance_table=None)


@pytest.mark.parametrize("model_type", ["xgb", "lgb", "cbt"])
def test_modeling_session_incremental_tune_runs_for_all_backends(sample_modeling_pd, tmp_path: Path, model_type: str):
    session = MarsModelingSession(
        model_type=model_type,
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=41,
    )

    result = session.incremental_tune(
        sample_modeling_pd,
        steps=[1, 3],
        max_diff=100.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        history_path=str(tmp_path / f"{model_type}_feature_growth.csv"),
    )

    assert isinstance(result, MarsFeatureGrowthResult)
    assert result.steps == [1, 3]
    assert set(result.runs) == {1, 3}
    assert result.best_run is not None
    assert result.best_run.features == result.best_features
    assert session.last_feature_growth_run is result
    assert session.last_run is result.best_run
    assert result.summary_table["feature_count"].tolist() == [1, 3]
    assert "val_ks" in result.summary_table.columns
    assert result.summary_table["is_best"].sum() == 1
    assert all(Path(path).exists() for path in result.summary_table["history_path"].dropna())


def test_feature_growth_run_artifact_roundtrip(sample_modeling_pd, tmp_path: Path):
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=42,
    )
    result = session.incremental_tune(
        sample_modeling_pd,
        steps=[2, 3],
        max_diff=100.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        history_path=str(tmp_path / "artifact_feature_growth.csv"),
    )

    artifact_dir = result.write_artifact(str(tmp_path / "feature_growth_artifact"))
    loaded = MarsFeatureGrowthResult.load_artifact(str(artifact_dir))

    assert loaded.best_step == result.best_step
    assert loaded.steps == result.steps
    assert loaded.feature_order == result.feature_order
    assert set(loaded.runs) == set(result.runs)
    assert loaded.summary_table["feature_count"].tolist() == result.summary_table["feature_count"].tolist()


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
        history_path=str(tmp_path / "artifact_history.csv"),
    )

    artifact_dir = result.write_artifact(str(tmp_path / "run_artifact"))
    loaded = MarsModelTuningResult.load_artifact(str(artifact_dir))

    assert loaded.study is None
    assert loaded.best_model is not None
    assert loaded.best_params == result.best_params
    assert loaded.history_table.equals(result.history_table)
    assert loaded.importance_table.equals(result.importance_table)
    assert loaded.training_config == result.training_config
    assert loaded.feature_schema == result.feature_schema
    assert loaded.backend_data_mode == result.backend_data_mode
    assert loaded.category_levels == result.category_levels


def test_modeling_run_load_artifact_requires_metadata(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        MarsModelTuningResult.load_artifact(str(tmp_path / "missing_artifact"))


def test_modeling_public_exports_only_include_formal_api():
    assert modeling.__all__ == ["MarsModelingSession"]
    assert hasattr(tuning_module, "MarsModelTuner")
    assert importlib.import_module("mars.modeling.feature_growth").MarsFeatureIncrementalTuner is MarsFeatureIncrementalTuner
    assert not hasattr(tuning_module, "MarsAutoModelTuner")
    assert not hasattr(report_module, "MarsModelEvaluationReport")
    assert not hasattr(results_module, "MarsTuningResult")
    assert not hasattr(results_module, "MarsPostTuningResult")
    assert "MarsModelingSession" in mars.__all__
    assert "MarsModelTuner" not in mars.__all__
    assert "MarsModelEvaluator" not in mars.__all__
    assert "MarsModelReplayRunner" not in mars.__all__


def test_modeling_old_module_paths_are_removed():
    for module_name in [
        "mars.modeling.base",
        "mars.modeling.data",
        "mars.modeling.strategies",
        "mars.modeling.tuner",
    ]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

    assert not hasattr(modeling, "MarsModelTuner")
    assert not hasattr(modeling, "MarsModelEvaluator")
    assert not hasattr(modeling, "MarsModelingReport")
    assert not hasattr(modeling, "MarsModelDataSplitter")


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
        history_path=str(tmp_path / "oot_history.csv"),
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
        history_path=str(tmp_path / "contains_history.csv"),
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


def test_catboost_ks_metric_handles_flattened_inputs_and_shape_errors():
    metric = CatBoostKSMetric()
    score, weight = metric.evaluate([[[-2.0, 0.0, 2.0, 3.0]]], [0, 0, 1, 1], None)

    assert weight == 1.0
    assert score > 0.0
    assert as_probability(np.array([[-2.0, 0.0, 2.0]])).shape == (3,)
    assert metric.evaluate([[]], [], None) == (0.0, 1.0)

    with pytest.raises(ValueError, match="mismatched"):
        metric.evaluate([[0.1, 0.2]], [0], None)
