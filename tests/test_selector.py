import numpy as np
import pandas as pd
import polars as pl
import pytest

import mars
import mars.feature as feature_module
from mars.feature import MarsImportanceSelector, MarsLinearSelector, MarsStatsSelector


def test_feature_subpackage_exports_new_selectors_without_top_level_exports():
    assert "MarsLinearSelector" in feature_module.__all__
    assert "MarsImportanceSelector" in feature_module.__all__
    assert "MarsLinearSelector" not in mars.__all__
    assert "MarsImportanceSelector" not in mars.__all__


def _linear_selector_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=180)
    x2 = x1 + rng.normal(scale=0.03, size=180)
    x3 = rng.normal(size=180)
    x4 = rng.normal(size=180)
    score = 1.8 * x1 - 0.2 * x3 + rng.normal(scale=0.5, size=180)
    target = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": target})


def test_linear_selector_filters_corr_vif_and_stepwise():
    df = _linear_selector_df()
    selector = MarsLinearSelector(
        corr_thr=0.85,
        enable_vif_filter=True,
        vif_threshold=5.0,
        enable_stepwise=True,
        stepwise_direction="both",
        stepwise_criterion="aic",
        max_features=2,
    )

    selector.fit(df.drop(columns=["target"]), df["target"])
    transformed = selector.transform(pl.from_pandas(df))
    report = selector.get_report()

    assert "x1" in selector.selected_features_
    assert "x2" not in selector.selected_features_
    assert len(selector.selected_features_) <= 2
    assert set(transformed.columns) == set(selector.selected_features_)
    assert not selector.vif_table_.empty
    assert not selector.stepwise_history_.empty
    assert "corr" in set(report.get_column("stage").to_list())


def test_importance_selector_uses_existing_importance_table(sample_credit_df):
    importance = pd.DataFrame(
        {
            "feature": ["segment", "income", "age", "utilization"],
            "importance": [0.40, 0.30, 0.10, 0.05],
        }
    )
    selector = MarsImportanceSelector(
        method="importance",
        selection_mode="top_k",
        selection_threshold=2,
    )

    selector.fit(sample_credit_df, importance_table=importance)
    report = selector.get_report()

    assert selector.selected_features_ == ["income", "segment"]
    assert selector.importance_table_["rank"].tolist() == [1, 2, 3, 4]
    assert set(report.get_column("status").to_list()) == {"Dropped", "Selected"}


def test_importance_selector_importance_table_allows_missing_y(sample_credit_df):
    importance = pd.DataFrame(
        {
            "feature": ["income", "age"],
            "importance": [0.7, 0.3],
        }
    )
    selector = MarsImportanceSelector(
        method="importance",
        selection_mode="top_k",
        selection_threshold=1,
    )

    selector.fit(sample_credit_df.select(["income", "age"]), importance_table=importance)

    assert selector.selected_features_ == ["income"]


def test_importance_selector_training_requires_y(sample_credit_df):
    selector = MarsImportanceSelector(
        estimator="rf",
        method="importance",
        selection_mode="top_k",
        selection_threshold=1,
        random_state=19,
    )

    with pytest.raises(ValueError, match="requires y"):
        selector.fit(sample_credit_df.select(["income", "age"]))


def test_importance_selector_trains_estimator_for_feature_importance(sample_credit_pd):
    selector = MarsImportanceSelector(
        estimator="rf",
        estimator_params={"n_estimators": 30, "max_depth": 3},
        method="importance",
        selection_mode="top_k",
        selection_threshold=2,
        random_state=17,
    )

    selector.fit(sample_credit_pd.drop(columns=["target"]), sample_credit_pd["target"])

    assert len(selector.selected_features_) == 2
    assert set(selector.importance_table_.columns) == {
        "feature",
        "importance",
        "importance_type",
        "model_type",
        "rank",
    }


def test_importance_selector_shap_method(sample_credit_pd):
    pytest.importorskip("shap")
    selector = MarsImportanceSelector(
        estimator="rf",
        estimator_params={"n_estimators": 20, "max_depth": 3},
        method="shap",
        selection_mode="percentile",
        selection_threshold="50%",
        random_state=18,
    )

    selector.fit(sample_credit_pd.drop(columns=["target"]), sample_credit_pd["target"])

    assert selector.selected_features_
    assert selector.importance_table_["importance_type"].unique().tolist() == ["mean_abs_shap"]


@pytest.mark.parametrize("method", ["rfe", "sfm"])
def test_importance_selector_not_implemented_methods_raise(sample_credit_pd, method: str):
    selector = MarsImportanceSelector(method=method)

    with pytest.raises(NotImplementedError, match=method):
        selector.fit(sample_credit_pd.drop(columns=["target"]), sample_credit_pd["target"])


def test_stats_selector_records_feature_data_source_in_report(sample_credit_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 3, "min_bin_size": 0.1, "merge_small_bins": True},
    )

    selector.fit(
        sample_credit_df,
        target="target",
        features=["income", "utilization"],
        feature_data_source={"EXT_SOURCE_1": ["income"]},
    )
    report = selector.get_report()

    if isinstance(report, pl.DataFrame):
        source_map = {
            row["feature"]: row["data_source"]
            for row in report.select(["feature", "data_source"]).unique().to_dicts()
        }
    else:
        source_map = dict(zip(report["feature"], report["data_source"], strict=False))

    assert source_map["income"] == "EXT_SOURCE_1"
    assert source_map["utilization"] == "UNMAPPED"


def test_stats_selector_rejects_feature_data_source_outside_candidate_features(sample_credit_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 3},
    )

    with pytest.raises(ValueError, match="feature_data_source"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income", "utilization"],
            feature_data_source={"UNKNOWN": ["age"]},
        )


def test_stats_selector_trims_filtered_feature_data_source_for_eval_report(sample_credit_df):
    df = sample_credit_df.with_columns(pl.lit(None).alias("mostly_missing"))
    selector = MarsStatsSelector(
        missing_thr=0.5,
        psi_thr=None,
        corr_thr=None,
        skip_fine_scan=True,
        psi_include_missing=True,
        psi_include_special=True,
        rough_binning_params={
            "method": "quantile",
            "n_bins": 3,
            "min_bin_size": 0.1,
            "merge_small_bins": True,
        },
    )

    selector.fit(
        df,
        target="target",
        features=["income", "utilization", "mostly_missing"],
        feature_data_source={
            "APP": ["income"],
            "BUREAU": ["mostly_missing"],
        },
        group_col="month",
        white_list=["income", "utilization"],
    )
    report = selector.get_binning_report(df)
    decision_report = selector.get_report()

    decision_source_map = {
        row["feature"]: row["data_source"]
        for row in decision_report.select(["feature", "data_source"]).unique().to_dicts()
    }
    summary_source_map = {
        row["feature"]: row["data_source"]
        for row in report.summary_table.select(["feature", "data_source"]).to_dicts()
    }
    detail_source_map = {
        row["feature"]: row["data_source"]
        for row in report.detail_table.select(["feature", "data_source"]).unique().to_dicts()
    }

    assert decision_source_map["mostly_missing"] == "BUREAU"
    assert "mostly_missing" not in summary_source_map
    assert summary_source_map["income"] == "APP"
    assert summary_source_map["utilization"] == "UNMAPPED"
    assert detail_source_map["income"] == "APP"
    assert detail_source_map["utilization"] == "UNMAPPED"
    assert report.report_meta["psi_include_missing"] is True
    assert report.report_meta["psi_include_special"] is True


def test_stats_selector_preserves_selected_feature_order(sample_credit_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 3, "min_bin_size": 0.1, "merge_small_bins": True},
    )

    selector.fit(
        sample_credit_df,
        target="target",
        features=["income", "utilization"],
        white_list=["utilization"],
    )

    assert selector.selected_features_ == sorted(
        selector.selected_features_,
        key=["income", "utilization"].index,
    )


def test_stats_selector_propagates_feature_start_aware_reference(feature_start_aware_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 2, "min_bin_size": 0.05, "merge_small_bins": True},
    )

    selector.fit(
        feature_start_aware_df,
        target="target",
        features=["x"],
        time_col="biz_dt",
        time_grain="month",
        feature_start_aware_reference=True,
    )
    report = selector.get_binning_report(feature_start_aware_df)

    assert report.report_meta["feature_start_aware_reference"] is True
    assert report.report_meta["feature_start_reference_dates"] == {"x": "2024-02-15"}


def test_stats_selector_handles_notebook_mock_data_with_group_context() -> None:
    rng = np.random.default_rng(2028)
    rows = 360
    month_idx = np.arange(rows) // 120
    months = np.array(["2024-01", "2024-02", "2024-03"])[month_idx]
    predictive = rng.normal(loc=month_idx * 0.15, scale=1.0, size=rows)
    white_feature = predictive + rng.normal(scale=0.2, size=rows)
    noise = rng.normal(size=rows)
    high_missing = rng.normal(size=rows).astype(object)
    high_missing[:320] = None
    black_feature = rng.normal(size=rows)
    target = (predictive + rng.normal(scale=0.5, size=rows) > np.median(predictive)).astype(int)
    df = pl.DataFrame(
        {
            "month": months.tolist(),
            "predictive": predictive,
            "white_feature": white_feature,
            "noise": noise,
            "high_missing": high_missing.tolist(),
            "black_feature": black_feature,
            "target": target,
        }
    )

    selector = MarsStatsSelector(
        missing_thr=0.8,
        iv_thr=0.0,
        psi_thr=None,
        corr_thr=None,
        skip_fine_scan=True,
        rough_binning_params={
            "method": "quantile",
            "n_bins": 3,
            "min_bin_size": 0.05,
            "merge_small_bins": True,
        },
    )

    selector.fit(
        df,
        target="target",
        features=["predictive", "white_feature", "noise", "high_missing", "black_feature"],
        group_col="month",
        white_list=["white_feature"],
        black_list=["black_feature"],
        max_samples=300,
    )
    report = selector.get_report()

    assert "white_feature" in selector.selected_features_
    assert "black_feature" not in selector.selected_features_
    assert "high_missing" not in selector.selected_features_
    assert "white_feature" in set(report["feature"].to_list())
