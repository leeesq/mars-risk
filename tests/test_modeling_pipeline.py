from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")

from mars.modeling import MarsModelingSession, MarsModelReplayResult, MarsModelReplayRunner


def test_modeling_session_replay_retrains_and_scores(sample_modeling_df, tmp_path: Path):
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=15,
    )
    tuning_result = session.tune(
        sample_modeling_df,
        max_diff=20.0,
        n_trials=2,
        startup_trials=1,
        warmup_steps=5,
        num_boost_round=25,
        early_stopping_rounds=5,
        artifact_dir=tmp_path / "pipeline_artifacts",
    )

    result = session.replay(
        tuning_result,
        sample_modeling_df,
        time_col="biz_dt",
        benchmark_col="benchmark_score",
        top_k=1,
        sort_metric="ks",
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    assert isinstance(result, MarsModelReplayResult)
    assert tuning_result.backend_data_mode == "polars_arrow_numeric"
    assert len(result.models) == 1
    assert len(result.reports) == 1
    pred_cols = [col for col in result.scored_df.columns if str(col).startswith("prob_top1_trial")]
    assert pred_cols
    assert "custom_mean_score" in result.ranking_table.columns
    assert {"rank", "model_name", "trial_num", "custom_mean_score", "best_iteration"}.issubset(result.leaderboard_table.columns)
    assert "backend_data_mode" in result.leaderboard_table.columns
    assert list(result.importance_tables.keys()) == list(result.models.keys())


def test_model_replay_reuses_tuning_result(sample_modeling_df, tmp_path: Path):
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=16,
    )
    tuning_result = session.tune(
        sample_modeling_df,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        artifact_dir=tmp_path / "replay_artifacts",
    )
    replay = MarsModelReplayRunner()

    result = replay.replay(
        tuning_result,
        sample_modeling_df,
        time_col="biz_dt",
        benchmark_col="benchmark_score",
        top_k=1,
        sort_metric="ks",
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    assert isinstance(result, MarsModelReplayResult)
    assert len(result.models) == 1
    assert len(result.reports) == 1


def test_model_replay_builds_leaderboard_without_oot(sample_modeling_df, tmp_path: Path):
    no_oot_df = sample_modeling_df.filter(sample_modeling_df["dataset_flag"] != "oot1")
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=17,
    )
    tuning_result = session.tune(
        no_oot_df,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        artifact_dir=tmp_path / "no_oot_artifacts",
    )

    result = session.replay(
        tuning_result,
        no_oot_df,
        top_k=1,
        sort_metric="ks",
        include_val=True,
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    assert not result.leaderboard_table.empty
    assert "val_ks" in result.leaderboard_table.columns


def test_model_replay_artifact_roundtrip(sample_modeling_df, tmp_path: Path):
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1", "x2", "x3"],
        target="target",
        optimize_metric="ks",
        seed=18,
    )
    tuning_result = session.tune(
        sample_modeling_df,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        artifact_dir=tmp_path / "artifact_replay",
    )
    replay_result = session.replay(
        tuning_result,
        sample_modeling_df,
        time_col="biz_dt",
        benchmark_col="benchmark_score",
        top_k=1,
        sort_metric="ks",
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    artifact_dir = replay_result.export_artifact(str(tmp_path / "replay_artifact"))
    loaded = MarsModelReplayResult.from_artifact(str(artifact_dir))

    assert loaded.scored_df is None
    pd.testing.assert_frame_equal(loaded.ranking_table, replay_result.ranking_table, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.leaderboard_table, replay_result.leaderboard_table, check_dtype=False)
    assert set(loaded.models) == set(replay_result.models)
    assert set(loaded.reports) == set(replay_result.reports)
    assert set(loaded.importance_tables) == set(replay_result.importance_tables)
