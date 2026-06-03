import polars as pl

from mars.feature.selector import MarsStatsSelector


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
