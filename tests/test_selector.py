from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mars
import mars.feature as feature_module
from mars.feature import (
    MarsImportanceSelector,
    MarsLinearSelector,
    MarsNativeBinner,
    MarsStatsSelector,
)


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


@pytest.mark.optional_ml
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
    with pytest.raises(ValueError, match="importance.*shap"):
        MarsImportanceSelector(method=method)


def test_stats_selector_records_feature_data_source_in_report(sample_credit_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
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
        source_map = dict(zip(report["feature"], report["data_source"]))

    assert source_map["income"] == "EXT_SOURCE_1"
    assert source_map["utilization"] == "UNMAPPED"


def test_stats_selector_rejects_feature_data_source_outside_candidate_features(sample_credit_df):
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
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
        psi_thr=None,
        rc_thr=None,
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
    )
    report = selector.get_report()

    assert "white_feature" in selector.selected_features_
    assert "black_feature" not in selector.selected_features_
    assert "high_missing" not in selector.selected_features_
    assert "white_feature" in set(report["feature"].to_list())


@pytest.mark.parametrize(
    ("method", "expected_fit_rows"),
    [("quantile", 6), ("cart", 4)],
)
def test_stats_selector_rough_fit_scope_depends_on_supervision(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_fit_rows: int,
) -> None:
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 100.0, 101.0],
            "target": [0, 1, 0, 1, None, None],
        }
    )
    fit_rows: list[int] = []
    original_fit = MarsNativeBinner.fit

    def _capture_fit(
        self: MarsNativeBinner,
        X: pl.DataFrame | pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> MarsNativeBinner:
        """记录 Selector 送入粗分箱器的样本行数。"""
        fit_rows.append(len(X))
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(MarsNativeBinner, "fit", _capture_fit)
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        rough_lift_thr=100.0,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        rough_binning_params={"method": method, "n_bins": 2},
        n_jobs=1,
    )

    selector.fit(frame, target="target", features=["x"])

    assert fit_rows == [expected_fit_rows]


def test_stats_selector_allows_empty_result_after_observed_only_rough_metrics() -> None:
    frame = pl.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0, 100.0, 100.0, 101.0, 101.0],
            "target": [0, 1, 0, 1, None, None, None, None],
        }
    )
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=0.02,
        rough_lift_thr=100.0,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        rough_binning_params={"method": "quantile", "n_bins": 3},
        n_jobs=1,
    )

    fitted = selector.fit(frame, target="target", features=["x"])
    decision_report = selector.get_report()
    transformed = selector.transform(frame)

    assert fitted is selector
    assert selector.selected_features_ == []
    assert transformed.columns == ["target"]
    rough_decision = decision_report.filter(pl.col("stage") == "Rough_Scan").row(
        0,
        named=True,
    )
    assert rough_decision["status"] == "Dropped"
    assert rough_decision["value"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("frame_kind", ["polars", "pandas"])
def test_stats_selector_corr_uses_only_observed_target_rows(frame_kind: str) -> None:
    rng = np.random.default_rng(2030)
    observed_rows = 240
    unobserved_rows = 480
    observed_feature = rng.normal(size=observed_rows)
    event_probability = 1.0 / (1.0 + np.exp(-1.2 * observed_feature))
    observed_target = rng.binomial(1, event_probability)
    frame_data = {
        "x1": np.concatenate([observed_feature, rng.normal(size=unobserved_rows)]),
        "x2": np.concatenate([observed_feature, rng.normal(size=unobserved_rows)]),
        "target": [int(value) for value in observed_target] + [None] * unobserved_rows,
    }
    df: pl.DataFrame | pd.DataFrame
    if frame_kind == "polars":
        df = pl.DataFrame(frame_data)
    else:
        df = pd.DataFrame(frame_data)

    selector = MarsStatsSelector(
        skip_rough_scan=True,
        iv_thr=-1.0,
        lift_thr=None,
        psi_thr=None,
        rc_thr=None,
        corr_thr=0.8,
        binning_params={
            "n_bins": 4,
            "n_prebins": 12,
            "min_bin_size": 0.05,
            "min_bin_n_event": 1,
            "time_limit": 1,
            "n_jobs": 1,
        },
        n_jobs=1,
    )

    selector.fit(df, target="target", features=["x1", "x2"])
    decision_report = selector.get_report()
    if isinstance(decision_report, pd.DataFrame):
        decision_report = pl.from_pandas(decision_report)
    corr_decisions = decision_report.filter(pl.col("stage") == "Corr_Filter")

    assert len(selector.selected_features_) == 1
    assert corr_decisions.filter(pl.col("status") == "Dropped").height == 1
    assert selector._stability_report is not None
    assert selector._stability_report.report_meta["row_count"] == observed_rows + unobserved_rows


def test_stats_selector_rough_bins_use_benchmark_but_metrics_use_df() -> None:
    benchmark_df = pl.DataFrame(
        {
            "x": list(range(8)),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    df = pl.DataFrame(
        {
            "x": list(range(8)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        rough_binning_params={"method": "quantile", "n_bins": 2},
    )

    selector.fit(
        df,
        target="target",
        features=["x"],
        benchmark_df=benchmark_df,
    )

    finite_cuts = [cut for cut in selector._stage3_binner.bin_cuts_["x"] if np.isfinite(cut)]
    benchmark_iv = selector._stage3_binner.profile_bin_performance(
        benchmark_df,
        benchmark_df.get_column("target"),
        update_woe=False,
    ).get_column("IV").max()

    assert max(finite_cuts) < 8
    assert selector._feature_iv_dict["x"] == pytest.approx(0.0, abs=1e-9)
    assert benchmark_iv > selector._feature_iv_dict["x"]
    assert selector._stage3_binner.bin_woes_["x"]


def test_stats_selector_fine_bins_use_benchmark() -> None:
    rng = np.random.default_rng(2029)
    benchmark_values = np.linspace(0.0, 1.0, 120)
    evaluation_values = np.linspace(100.0, 101.0, 120)
    benchmark_df = pl.DataFrame(
        {
            "x": benchmark_values,
            "target": (
                benchmark_values + rng.normal(scale=0.25, size=benchmark_values.size) >= 0.5
            ).astype(int),
        }
    )
    df = pl.DataFrame(
        {
            "x": evaluation_values,
            "target": np.tile([0, 1], 60),
        }
    )
    selector = MarsStatsSelector(
        skip_rough_scan=True,
        iv_thr=-1.0,
        lift_thr=None,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        binning_params={
            "n_bins": 3,
            "min_bin_size": 0.05,
            "time_limit": 1,
        },
    )

    selector.fit(
        df,
        target="target",
        features=["x"],
        benchmark_df=benchmark_df,
    )
    finite_cuts = [cut for cut in selector._stage3_binner.bin_cuts_["x"] if np.isfinite(cut)]

    assert finite_cuts
    assert max(finite_cuts) < 2.0
    assert selector._feature_iv_dict["x"] == pytest.approx(0.0, abs=1e-9)


def test_stats_selector_benchmark_provides_full_psi_reference() -> None:
    benchmark_df = pl.DataFrame(
        {
            "x": [0.0] * 20 + [10.0] * 20,
            "target": [0, 1] * 20,
        }
    )
    df = pl.DataFrame(
        {
            "month": ["A"] * 20 + ["B"] * 20,
            "x": [0.0] * 20 + [10.0] * 20,
            "target": [0, 1] * 20,
        }
    )
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        psi_thr=100.0,
        rc_thr=None,
        corr_thr=None,
        rough_binning_params={"method": "quantile", "n_bins": 2},
    )

    selector.fit(
        df,
        target="target",
        features=["x"],
        group_col="month",
        benchmark_df=benchmark_df,
    )
    report = selector.get_binning_report(df, benchmark_df=benchmark_df)
    psi_row = report.trend_tables["psi"].row(0, named=True)
    detail_groups = set(report.detail_table.get_column("mars_group").to_list())

    assert psi_row["A"] > 0.0
    assert psi_row["B"] > 0.0
    assert "Benchmark" not in detail_groups
    assert report.report_meta["binning_fit_source"] == "benchmark_df"
    assert report.report_meta["benchmark_row_count"] == benchmark_df.height
    assert report.report_meta["row_count"] == df.height
    assert report.report_meta["selection_metric_source"] == "df"


def test_stats_selector_report_requires_benchmark_to_be_repassed() -> None:
    benchmark_df = pl.DataFrame(
        {
            "x": list(range(8)),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    df = benchmark_df.with_columns(pl.lit("A").alias("month"))
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        psi_thr=100.0,
        rc_thr=None,
        corr_thr=None,
        rough_binning_params={"method": "quantile", "n_bins": 2},
    ).fit(
        df,
        target="target",
        features=["x"],
        group_col="month",
        benchmark_df=benchmark_df,
    )
    cuts_before = list(selector._stage3_binner.bin_cuts_["x"])

    with pytest.raises(ValueError, match="must be provided to `get_binning_report`"):
        selector.get_binning_report(df)

    report = selector.get_binning_report(df, benchmark_df=benchmark_df)

    assert selector._stage3_binner.bin_cuts_["x"] == cuts_before
    assert report.report_meta["binning_fit_source"] == "benchmark_df"


def test_stats_selector_requires_group_context_when_stability_is_enabled(
    sample_credit_df: pl.DataFrame,
) -> None:
    selector = MarsStatsSelector(skip_fine_scan=True)

    with pytest.raises(ValueError, match="`group_col` or `time_col` is required"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income"],
        )


def test_stats_selector_rejects_time_grain_without_time_col(
    sample_credit_df: pl.DataFrame,
) -> None:
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
    )

    with pytest.raises(ValueError, match="`time_grain` requires `time_col`"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income"],
            time_grain="week",
        )


@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"x": [0.0, 1.0]}),
        pl.DataFrame({"x": [0.0, 1.0], "target": [None, None]}),
        pl.DataFrame({"x": [0.0, 1.0], "target": [0, 0]}),
    ],
)
def test_stats_selector_requires_valid_binary_target_in_df(df: pl.DataFrame) -> None:
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
    )

    with pytest.raises(ValueError, match="target|Target"):
        selector.fit(df, target="target", features=["x"])


def test_stats_selector_validates_benchmark_schema(sample_credit_df: pl.DataFrame) -> None:
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
    )
    benchmark_df = sample_credit_df.select(["income", "target"])

    with pytest.raises(ValueError, match="missing active feature columns.*utilization"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income", "utilization"],
            benchmark_df=benchmark_df,
        )


def test_stats_selector_rejects_empty_benchmark(sample_credit_df: pl.DataFrame) -> None:
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
    )
    benchmark_df = sample_credit_df.select(["income", "target"]).head(0)

    with pytest.raises(ValueError, match="must contain at least one row"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income"],
            benchmark_df=benchmark_df,
        )


def test_stats_selector_reports_benchmark_binning_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_df = pl.DataFrame(
        {
            "benchmark_marker": [1, 1, 1, 1],
            "x": [0.0, 1.0, 2.0, 3.0],
            "target": [0, 0, 1, 1],
        }
    )
    df = pl.DataFrame(
        {
            "x": [10.0, 11.0, 12.0, 13.0],
            "target": [0, 0, 1, 1],
        }
    )
    original_transform = MarsNativeBinner.transform

    def _drop_benchmark_bin(
        self: MarsNativeBinner,
        frame: pl.DataFrame,
        **kwargs: object,
    ) -> pl.DataFrame:
        transformed = original_transform(self, frame, **kwargs)
        if "benchmark_marker" in frame.columns:
            return transformed.drop("x_bin")
        return transformed

    monkeypatch.setattr(MarsNativeBinner, "transform", _drop_benchmark_bin)
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
    )

    with pytest.raises(ValueError, match="could not produce bins.*Fit failures"):
        selector.fit(
            df,
            target="target",
            features=["x"],
            benchmark_df=benchmark_df,
        )


@pytest.mark.parametrize(
    "benchmark_df",
    [
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0, 0, 0, 0]}),
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [None] * 4}),
    ],
)
def test_stats_selector_supervised_fine_scan_validates_benchmark_target(
    benchmark_df: pl.DataFrame,
) -> None:
    df = pl.DataFrame(
        {
            "x": [10.0, 11.0, 12.0, 13.0],
            "target": [0, 0, 1, 1],
        }
    )
    selector = MarsStatsSelector(
        skip_rough_scan=True,
        psi_thr=None,
        rc_thr=None,
    )

    with pytest.raises(ValueError, match="target column|Target column|requested target"):
        selector.fit(
            df,
            target="target",
            features=["x"],
            benchmark_df=benchmark_df,
        )


def test_stats_selector_benchmark_overrides_feature_start_reference_once(
    sample_credit_df: pl.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    benchmark_df = sample_credit_df.select(["income", "target"])
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        psi_thr=100.0,
        rc_thr=None,
        corr_thr=None,
        feature_start_aware_reference=True,
        rough_binning_params={"method": "quantile", "n_bins": 2},
    )

    with caplog.at_level("WARNING"):
        selector.fit(
            sample_credit_df,
            target="target",
            features=["income"],
            group_col="month",
            benchmark_df=benchmark_df,
        )

    messages = [
        message
        for message in caplog.messages
        if "ignored because `benchmark_df` was provided" in message
    ]
    assert len(messages) == 1


def test_stats_selector_explicit_benchmark_risk_corr_source() -> None:
    benchmark_df = pl.DataFrame(
        {
            "x": list(range(20)),
            "target": [0, 1] * 10,
        }
    )
    df = pl.DataFrame(
        {
            "month": ["A"] * 20 + ["B"] * 20,
            "x": list(range(40)),
            "target": [0, 1] * 20,
        }
    )
    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        psi_thr=100.0,
        rc_thr=-1.0,
        corr_thr=None,
        risk_corr_baseline="benchmark",
        rough_binning_params={"method": "quantile", "n_bins": 2},
    ).fit(
        df,
        target="target",
        features=["x"],
        group_col="month",
        benchmark_df=benchmark_df,
    )

    report = selector.get_binning_report(df, benchmark_df=benchmark_df)

    assert report.report_meta["risk_corr_reference_source"] == "benchmark_df"


def test_stats_selector_no_longer_exposes_max_samples() -> None:
    assert "max_samples" not in inspect.signature(MarsStatsSelector.fit).parameters
