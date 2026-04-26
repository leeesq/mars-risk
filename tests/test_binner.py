import pandas as pd

from mars.feature import MarsNativeBinner


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
