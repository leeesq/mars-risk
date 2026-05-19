import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")

from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.prediction import ModelPredictor
from mars.modeling.results import MarsModelingRun
from mars.modeling.tuning import MarsModelTuner


@pytest.mark.parametrize(
    ("model_type", "features", "categorical_features"),
    [
        ("xgb", ["x1", "x2", "x3"], None),
        ("lgb", ["x1", "x2", "x3", "segment"], ["segment"]),
        ("cbt", ["x1", "x2", "x3", "segment"], ["segment"]),
    ],
)
def test_public_tooling_can_tune_and_evaluate_all_backends(
    model_type,
    features,
    categorical_features,
    sample_modeling_pd,
    tmp_path,
):
    tuner = MarsModelTuner(
        model_type=model_type,
        features=features,
        target="target",
        optimize_metric="auc",
        seed=13,
        categorical_features=categorical_features,
    )
    run = tuner.tune(
        sample_modeling_pd,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / f"{model_type}_tool_history.csv"),
    )

    evaluator = MarsModelEvaluator(
        group_col="dataset_flag",
        target_col="target",
        benchmark_col="benchmark_score",
        time_col="biz_dt",
    )

    assert run.best_model is not None
    assert run.best_iteration is None or isinstance(run.best_iteration, int)
    assert run.best_params
    assert "val_auc" in run.history_table.columns
    assert not run.importance_table.empty
    assert set(run.importance_table.columns) == {"feature", "importance", "importance_type", "model_type", "rank"}
    if categorical_features:
        assert "segment" in set(run.importance_table["feature"])
        assert run.backend_data_mode == "pandas_category"
    elif model_type in {"xgb", "lgb"}:
        assert run.backend_data_mode == "pandas_numeric"
    report = evaluator.evaluate(sample_modeling_pd.assign(pred_score=sample_modeling_pd["benchmark_score"]), pred_col="pred_score")
    assert ("Target: target", "New KS") in report.summary_table.columns


def test_lightgbm_categorical_levels_are_fixed_for_unseen_categories(sample_modeling_pd, tmp_path):
    df = sample_modeling_pd.copy()
    train_idx = df.index[df["dataset_flag"] == "train"]
    val_idx = df.index[df["dataset_flag"] == "val"]
    oot_idx = df.index[df["dataset_flag"] == "oot1"]
    df.loc[train_idx, "segment"] = ["A" if idx % 2 == 0 else "B" for idx in range(len(train_idx))]
    df.loc[val_idx, "segment"] = ["A" if idx % 2 == 0 else "B" for idx in range(len(val_idx))]
    df.loc[oot_idx, "segment"] = "UNSEEN_IN_TRAIN"

    tuner = MarsModelTuner(
        model_type="lgb",
        features=["x1", "x2", "x3", "segment"],
        target="target",
        optimize_metric="auc",
        seed=19,
        categorical_features=["segment"],
    )
    run = tuner.tune(
        df,
        max_diff=20.0,
        n_trials=1,
        startup_trials=1,
        warmup_steps=3,
        num_boost_round=20,
        early_stopping_rounds=5,
        save_path=str(tmp_path / "lgb_category_history.csv"),
    )

    assert run.category_levels == {"segment": ["A", "B"]}
    predictor = ModelPredictor(
        run.best_model,
        feature_list=run.features,
        categorical_features=run.categorical_features,
        category_levels=run.category_levels,
    )
    scored = predictor.predict(df, pred_col_name="score_with_unseen")

    assert "score_with_unseen" in scored.columns
    artifact_dir = run.write_artifact(str(tmp_path / "lgb_category_artifact"))
    loaded = MarsModelingRun.load_artifact(str(artifact_dir))
    assert loaded.category_levels == run.category_levels
