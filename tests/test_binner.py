import inspect

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

    assert list(transform_sig.parameters) == ["self", "X", "return_type", "woe_batch_size", "lazy"]
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
    assert optimal_fit_sig.parameters["y"].default is inspect.Parameter.empty
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
    labels = binner.transform(df.select(["all_null", "special_num"]), return_type="label")

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
