from __future__ import annotations

import inspect
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.feature import MarsNativeBinner, MarsOptimalBinner
from mars.feature.binning.base import MarsBinnerBase


def test_feature_public_entry_exports_binner_classes():
    from mars.feature import MarsBinnerBase as PublicBase
    from mars.feature import MarsNativeBinner as PublicNative
    from mars.feature import MarsOptimalBinner as PublicOptimal

    assert PublicBase is MarsBinnerBase
    assert PublicNative is MarsNativeBinner
    assert PublicOptimal is MarsOptimalBinner


def test_native_binner_generates_bin_columns_and_handles_special_values(sample_credit_df):
    features = sample_credit_df.select(["age", "income", "utilization", "segment"])
    target = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=4,
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target, cat_features=["segment"])

    assert transformed.shape[0] == sample_credit_df.height
    assert {"age_bin", "income_bin", "utilization_bin", "segment_bin"}.issubset(transformed.columns)

    income_mapping = binner.get_bin_mapping("income")
    assert -1 in income_mapping
    assert any(idx < 0 for idx in income_mapping)


def test_native_binner_preserves_pandas_output_for_pandas_input(sample_credit_pd):
    features = sample_credit_pd[["age", "income", "utilization", "segment"]]
    target = sample_credit_pd["target"]

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=3,
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target, cat_features=["segment"])

    assert isinstance(transformed, pd.DataFrame)
    assert {"age_bin", "income_bin", "utilization_bin", "segment_bin"}.issubset(transformed.columns)


def test_binner_public_signatures_expose_user_visible_parameters():
    transform_sig = inspect.signature(MarsBinnerBase.transform)
    fit_transform_sig = inspect.signature(MarsBinnerBase.fit_transform)
    native_fit_sig = inspect.signature(MarsNativeBinner.fit)
    optimal_fit_sig = inspect.signature(MarsOptimalBinner.fit)

    assert list(transform_sig.parameters) == [
        "self",
        "X",
        "features",
        "on_missing",
        "return_type",
        "woe_batch_size",
        "lazy",
    ]
    assert transform_sig.parameters["return_type"].kind is inspect.Parameter.KEYWORD_ONLY
    assert transform_sig.parameters["woe_batch_size"].default == 200
    assert transform_sig.parameters["lazy"].default is False

    assert list(fit_transform_sig.parameters) == [
        "self",
        "X",
        "y",
        "features",
        "cat_features",
        "return_type",
        "woe_batch_size",
        "lazy",
    ]
    assert fit_transform_sig.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit_transform_sig.parameters["cat_features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit_transform_sig.parameters["return_type"].kind is inspect.Parameter.KEYWORD_ONLY

    assert list(native_fit_sig.parameters) == ["self", "X", "y", "features", "cat_features"]
    assert native_fit_sig.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert native_fit_sig.parameters["cat_features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "kwargs" not in native_fit_sig.parameters
    assert list(optimal_fit_sig.parameters) == ["self", "X", "y", "features", "cat_features"]
    assert optimal_fit_sig.parameters["y"].default is None
    assert optimal_fit_sig.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert optimal_fit_sig.parameters["cat_features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "kwargs" not in optimal_fit_sig.parameters


def test_native_binner_transform_can_explicitly_return_woe(sample_credit_df, recwarn):
    features = sample_credit_df.select(["age", "income", "utilization", "segment"])
    target = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=4,
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target, cat_features=["segment"], return_type="woe")

    assert {"age_woe", "income_woe", "utilization_woe", "segment_woe"}.issubset(transformed.columns)
    assert not any("replace" in str(warning.message) for warning in recwarn)


def test_materialize_woe_uses_columnwise_aggregation(sample_credit_df):
    source = inspect.getsource(MarsBinnerBase._materialize_woe)
    assert "unpivot" not in source

    features = sample_credit_df.select(["age", "income", "utilization", "segment"])
    target = sample_credit_df.get_column("target")
    binner = MarsNativeBinner(
        method="quantile",
        n_bins=4,
        special_values=[-999],
    )

    binner.fit(features, target, cat_features=["segment"])
    transformed = binner.transform(features, return_type="woe", woe_batch_size=2)

    assert {"age_woe", "income_woe", "utilization_woe", "segment_woe"}.issubset(transformed.columns)
    assert set(binner.bin_woes_) == {"age", "income", "utilization", "segment"}
    assert all(binner.bin_woes_[feature] for feature in binner.bin_woes_)


def test_native_cart_requires_y_and_quantile_allows_label_free(sample_credit_df):
    features = sample_credit_df.select(["age", "income"])

    with pytest.raises(ValueError, match="requires y"):
        MarsNativeBinner(method="cart").fit(features, features=["age"])

    binner = MarsNativeBinner(method="quantile", n_bins=3)
    binner.fit(features, features=["age"])

    assert binner._is_fitted
    assert "age_bin" in binner.transform(features).columns


def test_native_uniform_remove_empty_bins_handles_nan_and_special_values() -> None:
    values = np.array(
        [
            -999.0,
            np.nan,
            0.0,
            0.1,
            0.2,
            1.5,
            3.0,
            np.nan,
            8.0,
            13.0,
        ],
        dtype=np.float32,
    )
    features = pl.DataFrame({"score": values})

    binner = MarsNativeBinner(
        method="uniform",
        n_bins=6,
        special_values=[-999.0],
        remove_empty_bins=True,
    )

    transformed = binner.fit_transform(features, features=["score"])

    assert transformed.height == features.height
    assert "score_bin" in transformed.columns
    assert binner.bin_cuts_["score"][0] == float("-inf")
    assert binner.bin_cuts_["score"][-1] == float("inf")


def test_optimal_binner_requires_y(sample_credit_df):
    features = sample_credit_df.select(["age", "income"])
    target = sample_credit_df.get_column("target")

    with pytest.raises(ValueError, match="requires y"):
        MarsOptimalBinner(n_bins=3).fit(features, None, features=["age"])

    binner = MarsOptimalBinner(n_bins=3)
    binner.fit(features, target, features=["age"])

    assert binner._is_fitted


def test_native_binner_handles_notebook_numeric_edge_cases() -> None:
    values = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.1,
            0.2,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            -999.0,
            -998.0,
            np.nan,
            -777.0,
        ]
    )
    df = pl.DataFrame(
        {
            "repeat_num": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4],
            "all_null": [None] * len(values),
            "special_num": values,
            "nullable_int": [1, 2, None, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15],
        }
    )
    target = pl.Series("target", [0, 0, 0, 0, 0, 1, 1, 1, 1] * 2)

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=8,
        special_values=[-999.0, -998.0],
        missing_values=[-777.0],
        min_bin_size=0.1,
        merge_small_bins=True,
    )
    binner.fit(df, target)

    transformed = binner.transform(df, return_type="index")
    labels = binner.transform(
        df.select(["all_null", "special_num"]),
        features=["all_null", "special_num"],
        return_type="label",
    )

    assert len(binner.bin_cuts_["repeat_num"]) == len(set(binner.bin_cuts_["repeat_num"]))
    assert set(transformed["all_null_bin"].to_list()) == {-1}
    assert set(labels["all_null_bin"].to_list()) == {"Missing"}
    assert -1 in set(transformed["special_num_bin"].to_list())
    assert -3 in set(transformed["special_num_bin"].to_list())
    assert -4 in set(transformed["special_num_bin"].to_list())
    assert "Special_-999.0" in set(labels["special_num_bin"].to_list())
    assert "Special_-998.0" in set(labels["special_num_bin"].to_list())
    assert "nullable_int_bin" in transformed.columns


def test_native_cart_finds_obvious_supervised_split() -> None:
    x = np.linspace(0.0, 1.0, 120)
    df = pl.DataFrame({"score": x})
    target = pl.Series("target", (x >= 0.5).astype(int))

    binner = MarsNativeBinner(
        method="cart",
        n_bins=3,
        cart_params={"random_state": 7},
    )
    binner.fit(df, target, features=["score"])

    cuts = binner.bin_cuts_["score"][1:-1]
    assert any(0.45 <= cut <= 0.55 for cut in cuts)


def test_native_binner_maps_unknown_category_and_lazyframe() -> None:
    train = pl.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "C", "C", "S", None],
            "val": [0.1, 0.2, 0.4, 0.5, 0.8, 0.9, -999.0, None],
        }
    )
    target = pl.Series("target", [0, 0, 0, 1, 1, 1, 1, 0])

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=3,
        special_values=["S", -999.0],
    )
    binner.fit(train, target, cat_features=["cat"])

    test_df = pl.DataFrame({"cat": ["A", "Z", None, "S"], "val": [0.15, 0.7, None, -999.0]})
    transformed = binner.transform(test_df, return_type="index")
    lazy_result = binner.transform(test_df.lazy(), return_type="index", lazy=True)

    assert transformed["cat_bin"].to_list()[1] == -2
    assert transformed["cat_bin"].to_list()[2] == -1
    assert transformed["cat_bin"].to_list()[3] == -3
    assert isinstance(lazy_result, pl.LazyFrame)
    assert "val_bin" in lazy_result.collect().columns


def test_optimal_binner_handles_missing_special_join_and_direct_optbinning() -> None:
    pytest.importorskip("optbinning")
    from optbinning import OptimalBinning

    df = pl.DataFrame(
        {
            "num": [
                0.05,
                0.12,
                0.18,
                0.25,
                0.35,
                0.45,
                0.55,
                0.65,
                0.75,
                0.85,
                0.95,
                -999.0,
                -777.0,
                None,
            ],
            "cat": ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "SPECIAL", "A", None, "B"],
        }
    )
    target = pl.Series("target", [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])

    direct = OptimalBinning(
        name="num",
        dtype="numerical",
        max_n_bins=3,
        min_bin_n_event=1,
        min_bin_size=0.05,
        special_codes=[-999.0],
    )
    direct.fit(df["num"].to_numpy(), target.to_numpy())

    binner = MarsOptimalBinner(
        n_bins=3,
        min_bin_n_event=1,
        min_bin_size=0.05,
        min_prebin_size=0.05,
        n_prebins=6,
        special_values=[-999.0, "SPECIAL"],
        missing_values=[-777.0],
        join_threshold=1,
        max_cats_to_solver=5,
        min_cat_fraction=0.01,
        time_limit=5,
        n_jobs=1,
    )
    binner.fit(df, target, features=["num", "cat"], cat_features=["cat"])

    transformed = binner.transform(
        pl.DataFrame({"num": [-777.0, -999.0, 0.6], "cat": [None, "SPECIAL", "ZZZ"]}),
        return_type="index",
    )

    assert direct.status in {"OPTIMAL", "FEASIBLE"}
    assert transformed["num_bin"].to_list()[0] == -1
    assert transformed["num_bin"].to_list()[1] == -3
    assert transformed["cat_bin"].to_list()[0] == -1
    assert transformed["cat_bin"].to_list()[1] == -4
    assert transformed["cat_bin"].to_list()[2] is not None


def test_optimal_binner_uses_native_fallback_when_constraints_fail() -> None:
    values = np.linspace(0.0, 1.0, 120)
    df = pl.DataFrame({"score": values})
    target = pl.Series("target", (values >= 0.5).astype(int))

    binner = MarsOptimalBinner(
        n_bins=3,
        min_n_bins=4,
        min_bin_size=0.3,
        min_bin_n_event=1,
        n_prebins=20,
        n_jobs=1,
    )
    binner.fit(df, target, features=["score"])
    expected_binner = MarsNativeBinner(
        method="quantile",
        n_bins=binner.n_bins,
        min_bin_size=binner.min_bin_size,
        merge_small_bins=True,
        remove_empty_bins=False,
    )
    expected_binner.fit(df, target, features=["score"])

    cuts = binner.bin_cuts_["score"]
    assert np.allclose(cuts, expected_binner.bin_cuts_["score"])
    assert len(cuts) <= binner.n_bins + 1
    assert len(cuts) < binner.n_prebins
    assert "native fallback applied" in binner.fit_failures_["score"]


def test_optimal_binner_batches_native_fallback_for_all_failed_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2031)
    features = [f"x{index}" for index in range(4)]
    df = pl.DataFrame({feature: rng.normal(size=120) for feature in features})
    target = pl.Series("target", rng.integers(0, 2, size=120))
    native_fit_calls: list[tuple[str, ...]] = []
    original_fit = MarsNativeBinner.fit

    def _count_native_fit(
        self: MarsNativeBinner,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[object] | None = None,
        *,
        features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MarsNativeBinner:
        native_fit_calls.append(tuple(features or []))
        return original_fit(
            self,
            X,
            y,
            features=features,
            cat_features=cat_features,
        )

    monkeypatch.setattr(MarsNativeBinner, "fit", _count_native_fit)
    binner = MarsOptimalBinner(
        n_bins=3,
        min_n_bins=4,
        min_bin_size=0.3,
        min_bin_n_event=1,
        n_prebins=20,
        n_jobs=1,
    )
    expected_binner = MarsNativeBinner(
        method="quantile",
        n_bins=3,
        min_bin_size=0.3,
        merge_small_bins=True,
        remove_empty_bins=False,
        n_jobs=1,
    )

    binner.fit(df, target, features=features)
    original_fit(expected_binner, df, target, features=features)

    assert native_fit_calls == [tuple(features), tuple(features)]
    assert set(binner.fit_failures_) == set(features)
    for feature in features:
        assert np.allclose(binner.bin_cuts_[feature], expected_binner.bin_cuts_[feature])


def test_optimal_binner_batches_only_failed_features_after_mixed_solver_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.linspace(0.0, 1.0, 120)
    df = pl.DataFrame(
        {
            "stable": values,
            "sparse": [*values[:30], *([None] * 90)],
        }
    )
    target = pl.Series("target", np.tile([0, 1], 60))
    native_fit_calls: list[tuple[str, ...]] = []
    original_fit = MarsNativeBinner.fit

    def _count_native_fit(
        self: MarsNativeBinner,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[object] | None = None,
        *,
        features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MarsNativeBinner:
        native_fit_calls.append(tuple(features or []))
        return original_fit(
            self,
            X,
            y,
            features=features,
            cat_features=cat_features,
        )

    class _SuccessfulOptimalBinning:
        """为满足样本约束的特征返回稳定切点。"""

        def __init__(self, **_: object) -> None:
            self.status = "OPTIMAL"
            self.splits = np.array([0.5])

        def fit(self, _values: np.ndarray, _target: np.ndarray) -> None:
            return None

    monkeypatch.setattr(MarsNativeBinner, "fit", _count_native_fit)
    binner = MarsOptimalBinner(
        n_bins=3,
        min_n_bins=2,
        min_bin_size=0.3,
        min_bin_n_event=1,
        prebinning_method="quantile",
        n_prebins=10,
        n_jobs=1,
    )
    binner.OptimalBinning = _SuccessfulOptimalBinning

    binner.fit(df, target, features=["stable", "sparse"])

    assert native_fit_calls == [("stable", "sparse"), ("sparse",)]
    assert binner.bin_cuts_["stable"] == [float("-inf"), 0.5, float("inf")]
    assert set(binner.fit_failures_) == {"sparse"}
    assert binner.bin_cuts_["sparse"]


def test_optimal_binner_respects_explicit_native_fallback_params() -> None:
    values = np.linspace(0.0, 1.0, 120)
    df = pl.DataFrame({"score": values})
    target = pl.Series("target", (values >= 0.5).astype(int))

    fallback_params = {
        "method": "uniform",
        "n_bins": 3,
        "min_bin_size": 0.05,
        "merge_small_bins": False,
    }
    binner = MarsOptimalBinner(
        n_bins=8,
        min_n_bins=4,
        min_bin_size=0.3,
        min_bin_n_event=1,
        n_prebins=20,
        fallback_binner_params=fallback_params,
        n_jobs=1,
    )
    expected_binner = MarsNativeBinner(**fallback_params)

    binner.fit(df, target, features=["score"])
    expected_binner.fit(df, target, features=["score"])

    assert np.allclose(binner.bin_cuts_["score"], expected_binner.bin_cuts_["score"])
    assert "native fallback applied" in binner.fit_failures_["score"]


def test_optimal_binner_rejects_invalid_fallback_binner_params() -> None:
    with pytest.raises(ValueError, match="Allowed keys"):
        MarsOptimalBinner(fallback_binner_params={"n_jobs": 1})

    with pytest.raises(ValueError, match="prebinning_method"):
        MarsOptimalBinner(fallback_binner_params={"prebinning_method": "quantile"})


def test_optimal_binner_serializes_fallback_binner_params() -> None:
    binner = MarsOptimalBinner(
        fallback_binner_params={"method": "uniform", "n_bins": 3},
    )

    frame = pl.DataFrame({"x": list(range(20))})
    target = pl.Series("target", [0] * 10 + [1] * 10)
    binner.fit(frame, target)
    restored = MarsOptimalBinner.from_dict(binner.to_dict())

    assert isinstance(restored, MarsOptimalBinner)
    assert restored.fallback_binner_params == {"method": "uniform", "n_bins": 3}


def test_profile_bin_performance_preserves_feature_names_with_bin_text() -> None:
    values = np.linspace(0.0, 1.0, 80)
    df = pl.DataFrame(
        {
            "foo_bin": values,
            "foo_bin_score": values[::-1],
        }
    )
    target = pl.Series("target", (values >= 0.5).astype(int))
    binner = MarsNativeBinner(method="quantile", n_bins=4)

    binner.fit(df, target, features=["foo_bin", "foo_bin_score"])
    transformed = binner.transform(df, return_type="index")
    stats = binner.profile_bin_performance(df, target, include_bin_index=True)

    assert {"foo_bin_bin", "foo_bin_score_bin"}.issubset(transformed.columns)
    assert set(stats["feature"].unique().to_list()) == {"foo_bin", "foo_bin_score"}


def test_profile_bin_performance_excludes_null_and_nan_targets() -> None:
    frame = pl.DataFrame({"x": [0.0, 0.0, 1.0, 1.0]})
    target = pl.Series("target", [0.0, 1.0, None, float("nan")])
    binner = MarsNativeBinner(method="quantile", n_bins=2).fit(
        frame,
        features=["x"],
    )

    stats = binner.profile_bin_performance(frame, target)

    assert stats.get_column("count").sum() == 2
    assert stats.get_column("observed_count").sum() == 2
    assert stats.get_column("bad").sum() == pytest.approx(1.0)
    assert stats.get_column("good").sum() == pytest.approx(1.0)


def test_profile_bin_performance_rejects_target_without_observed_values() -> None:
    frame = pl.DataFrame({"x": [0.0, 1.0]})
    target = pl.Series("target", [None, float("nan")], dtype=pl.Float64)
    binner = MarsNativeBinner(method="quantile", n_bins=2).fit(
        frame,
        features=["x"],
    )

    with pytest.raises(ValueError, match="at least one observed value"):
        binner.profile_bin_performance(frame, target)


def test_profile_bin_performance_supports_bin_index_ordered_metrics() -> None:
    prob = [0.1] * 100 + [0.5] * 100 + [0.9] * 100 + [-999.0] * 20
    target = [1] * 10 + [0] * 90
    target += [1] * 80 + [0] * 20
    target += [1] * 20 + [0] * 80
    target += [1] * 20
    df = pl.DataFrame({"prob": prob})
    y = pl.Series("target", target)
    binner = MarsNativeBinner(
        method="uniform",
        n_bins=3,
        special_values=[-999.0],
    )
    binner.fit(df, y, features=["prob"])

    woe_stats = binner.profile_bin_performance(
        df,
        y,
        include_bin_index=True,
        ordered_metric_sort_by="woe",
    )
    index_stats = binner.profile_bin_performance(
        df,
        y,
        include_bin_index=True,
        ordered_metric_sort_by="bin_index",
    )

    special_row = index_stats.filter(pl.col("bin_index") < 0).row(0, named=True)
    assert special_row["bin_ks"] is None
    assert index_stats["KS"].max() != pytest.approx(woe_stats["KS"].max())


def test_binner_strict_subset_transform_and_fit_report() -> None:
    frame = pl.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    binner = MarsNativeBinner(n_bins=2).fit(frame)

    with pytest.raises(ValueError, match="missing required features"):
        binner.transform(frame.select("x"))

    subset = binner.transform(frame.select("x"), features=["x"])
    ignored = binner.transform(frame.select("x"), on_missing="ignore")
    report = binner.get_fit_report()

    assert subset.columns == ["x", "x_bin"]
    assert ignored.columns == ["x", "x_bin"]
    assert report.columns == [
        "feature",
        "dtype",
        "feature_type",
        "status",
        "usable",
        "n_bins",
        "reason",
    ]
    assert report["usable"].to_list() == [True, True]


def test_binner_json_artifact_roundtrip_and_validation(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": [1, 2, 3, 4], "cat": ["a", "b", "a", "c"]})
    binner = MarsNativeBinner(
        n_bins=2,
        special_values=[
            Decimal("-999.25"),
            date(2026, 1, 2),
            datetime(2026, 1, 2, 3, 4, 5),
            np.int64(-998),
        ],
        n_jobs=1,
    ).fit(frame, cat_features=["cat"])
    artifact_path = tmp_path / "binner.json"

    binner.save_json(artifact_path)
    restored = MarsBinnerBase.load_json(artifact_path)

    assert isinstance(restored, MarsNativeBinner)
    assert restored._requested_n_jobs == 1
    assert restored.special_values == binner.special_values
    assert restored.transform(frame).to_dicts() == binner.transform(frame).to_dicts()
    assert restored.get_fit_report().to_dicts() == binner.get_fit_report().to_dicts()
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["schema_version"] == 1

    with pytest.raises(ValueError, match="Legacy"):
        MarsBinnerBase.from_dict({"params": {}, "state": {}})
    payload = binner.to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        MarsBinnerBase.from_dict(payload)
    with pytest.raises(FileNotFoundError, match="parent directory"):
        binner.save_json(tmp_path / "missing" / "binner.json")


def test_binner_json_rejects_unknown_state_objects() -> None:
    binner = MarsNativeBinner(n_bins=2).fit(pl.DataFrame({"x": [1, 2, 3, 4]}))
    binner.special_values = [object()]

    with pytest.raises(TypeError, match="does not support"):
        binner.to_dict()


def test_binner_strict_rule_mutation_mapping_and_woe_contracts() -> None:
    frame = pl.DataFrame({"x": [1, 2, 3, 4], "cat": ["O'Reilly", "a", "b", "a"]})
    binner = MarsNativeBinner(n_bins=2).fit(frame, cat_features=["cat"])

    with pytest.raises(KeyError, match="unknown"):
        binner.get_bin_mapping("unknown")
    with pytest.raises(ValueError, match="unknown"):
        binner.prune(["unknown"])
    with pytest.raises(ValueError, match="at least one"):
        binner.prune([])
    with pytest.raises(ValueError, match="unknown"):
        binner.update_bins({"unknown": [1.5]})
    with pytest.raises(ValueError, match="WOE"):
        binner.transform(frame, return_type="woe")
    with pytest.raises(ValueError, match="WOE mapping"):
        binner.generate_sql(return_type="woe")

    sql = binner.generate_sql(features="cat", return_type="index")
    assert "O''Reilly" in sql
