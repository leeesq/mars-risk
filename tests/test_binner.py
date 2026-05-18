import inspect

import pandas as pd

from mars.feature import MarsNativeBinner, MarsOptimalBinner
from mars.feature.binner import MarsBinnerBase


def test_native_binner_generates_bin_columns_and_handles_special_values(sample_credit_df):
    features = sample_credit_df.select(["age", "income", "utilization", "segment"])
    target = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=4,
        cat_features=["segment"],
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target)

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
        cat_features=["segment"],
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target)

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

    assert list(fit_transform_sig.parameters) == ["self", "X", "y", "return_type", "woe_batch_size", "lazy"]
    assert fit_transform_sig.parameters["return_type"].kind is inspect.Parameter.KEYWORD_ONLY

    assert list(native_fit_sig.parameters) == ["self", "X", "y"]
    assert "kwargs" not in native_fit_sig.parameters
    assert list(optimal_fit_sig.parameters) == ["self", "X", "y"]
    assert "kwargs" not in optimal_fit_sig.parameters


def test_native_binner_transform_can_explicitly_return_woe(sample_credit_df):
    features = sample_credit_df.select(["age", "income", "utilization", "segment"])
    target = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        method="quantile",
        n_bins=4,
        cat_features=["segment"],
        special_values=[-999],
    )

    transformed = binner.fit_transform(features, target, return_type="woe")

    assert {"age_woe", "income_woe", "utilization_woe", "segment_woe"}.issubset(transformed.columns)
