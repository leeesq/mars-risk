import numpy as np
import pandas as pd
import polars as pl
import pytest

import mars
import mars.feature as feature_module
from mars.feature import MarsImportanceSelector, MarsLinearSelector
from mars.feature.selector import MarsStatsSelector


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
        target="target",
        corr_thr=0.85,
        enable_vif_filter=True,
        vif_threshold=5.0,
        enable_stepwise=True,
        stepwise_direction="both",
        stepwise_criterion="aic",
        max_features=2,
    )

    selector.fit(df)
    transformed = selector.transform(pl.from_pandas(df))
    report = selector.get_report()

    assert "x1" in selector.selected_features_
    assert "x2" not in selector.selected_features_
    assert len(selector.selected_features_) <= 2
    assert set(transformed.columns) == {*selector.selected_features_, "target"}
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
        target="target",
        method="importance",
        selection_mode="top_k",
        selection_threshold=2,
        importance_table=importance,
    )

    selector.fit(sample_credit_df)
    report = selector.get_report()

    assert selector.selected_features_ == ["income", "segment"]
    assert selector.importance_table_["rank"].tolist() == [1, 2, 3, 4]
    assert set(report.get_column("status").to_list()) == {"Dropped", "Selected"}


def test_importance_selector_trains_estimator_for_feature_importance(sample_credit_pd):
    selector = MarsImportanceSelector(
        target="target",
        estimator="rf",
        estimator_params={"n_estimators": 30, "max_depth": 3},
        method="importance",
        selection_mode="top_k",
        selection_threshold=2,
        random_state=17,
    )

    selector.fit(sample_credit_pd)

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
        target="target",
        estimator="rf",
        estimator_params={"n_estimators": 20, "max_depth": 3},
        method="shap",
        selection_mode="percentile",
        selection_threshold="50%",
        random_state=18,
    )

    selector.fit(sample_credit_pd)

    assert selector.selected_features_
    assert selector.importance_table_["importance_type"].unique().tolist() == ["mean_abs_shap"]


@pytest.mark.parametrize("method", ["rfe", "sfm"])
def test_importance_selector_not_implemented_methods_raise(sample_credit_pd, method: str):
    selector = MarsImportanceSelector(target="target", method=method)

    with pytest.raises(NotImplementedError, match=method):
        selector.fit(sample_credit_pd)


def test_stats_selector_records_feature_data_source_in_report(sample_credit_df):
    selector = MarsStatsSelector(
        target="target",
        features=["income", "utilization"],
        feature_data_source={"EXT_SOURCE_1": ["income"]},
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 3, "min_bin_size": 0.1, "merge_small_bins": True},
    )

    selector.fit(sample_credit_df)
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


def test_stats_selector_preserves_selected_feature_order(sample_credit_df):
    selector = MarsStatsSelector(
        target="target",
        features=["income", "utilization"],
        white_list=["utilization"],
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 3, "min_bin_size": 0.1, "merge_small_bins": True},
    )

    selector.fit(sample_credit_df)

    assert selector.selected_features_ == sorted(
        selector.selected_features_,
        key=["income", "utilization"].index,
    )


def test_stats_selector_propagates_feature_start_aware_baseline(feature_start_aware_df):
    selector = MarsStatsSelector(
        target="target",
        features=["x"],
        time_col="biz_dt",
        profile_by="month",
        feature_start_aware_baseline=True,
        skip_fine_scan=True,
        rough_binning_params={"method": "quantile", "n_bins": 2, "min_bin_size": 0.05, "merge_small_bins": True},
    )

    selector.fit(feature_start_aware_df)
    report, _ = selector.get_eval_report(feature_start_aware_df)

    assert report.report_meta["feature_start_aware_baseline"] is True
    assert report.report_meta["feature_start_baseline_dates"] == {"x": "2024-02-15"}
