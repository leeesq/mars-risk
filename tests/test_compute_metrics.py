from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl
import pytest

from mars.compute import (
    actual_dist_expr,
    amount_distribution_exprs,
    amount_metric_exprs,
    bad_rate_expr,
    binary_distribution_exprs,
    binary_metric_exprs,
    expected_dist_expr,
    normalize_ordered_metric_sort_by,
    normalized_auc_expr,
    observed_auc_agg_expr,
    observed_iv_agg_expr,
    observed_ks_agg_expr,
    observed_lift_max_agg_expr,
    observed_lift_min_agg_expr,
    ordered_binary_metric_exprs,
    partition_distribution_expr,
    psi_exprs,
    psi_partition_prob_expr,
    psi_valid_condition,
    ratio_expr,
    risk_corr_expr,
)


def test_binary_metric_bundle_matches_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x"],
            "bin_index": [0, 1],
            "count": [50.0, 50.0],
            "observed_count": [50.0, 50.0],
            "bad": [5.0, 20.0],
        }
    )

    result = (
        df.with_columns(binary_distribution_exprs(["feature"]))
        .with_columns(binary_metric_exprs())
        .sort(["feature", "bin_index"])
        .with_columns(ordered_binary_metric_exprs(["feature"]))
    )
    row0 = result.row(0, named=True)
    row1 = result.row(1, named=True)

    assert row0["bad_rate"] == pytest.approx(0.1)
    assert row0["lift"] == pytest.approx(0.4)
    assert row0["bad_dist"] == pytest.approx(0.2)
    assert row0["good_dist"] == pytest.approx(0.6)
    assert row0["woe"] == pytest.approx(math.log(0.2 / 0.6), abs=1e-5)
    assert row1["ks_bin"] == pytest.approx(0.0, abs=1e-5)
    assert result["auc_bin"].sum() == pytest.approx(0.3, abs=1e-5)


def test_ordered_metric_sort_by_normalization() -> None:
    assert normalize_ordered_metric_sort_by(None) == "woe"
    assert normalize_ordered_metric_sort_by("bin_index") == "bin_index"

    with pytest.raises(ValueError, match="ordered_metric_sort_by"):
        normalize_ordered_metric_sort_by("auto")


def test_amount_metric_bundle_matches_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x"],
            "count": [2.0, 2.0],
            "tot_amt": [300.0, 700.0],
            "good_amt": [200.0, 300.0],
            "bad_amt": [100.0, 400.0],
        }
    )

    result = (
        df.with_columns(amount_distribution_exprs(["feature"]))
        .with_columns(amount_metric_exprs())
        .sort("tot_amt")
    )
    row0 = result.row(0, named=True)
    row1 = result.row(1, named=True)

    assert row0["avg_amt"] == 150.0
    assert row0["amt_bad_rate"] == pytest.approx(1 / 3)
    assert row0["lift_amt"] == pytest.approx((1 / 3) / 0.5)
    assert row1["lift_amt"] == pytest.approx((4 / 7) / 0.5)


def test_stability_exprs_match_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x"],
            "group": ["202401", "202401"],
            "bin_index": [0, 1],
            "count": [60.0, 40.0],
            "expected_dist": [0.5, 0.5],
            "bad_rate": [0.1, 0.2],
            "base_br": [0.1, 0.3],
            "observed_count": [60.0, 40.0],
        }
    )

    psi_result = df.with_columns(psi_exprs(["feature", "group"]))
    rc_result = df.group_by(["feature", "group"]).agg(risk_corr_expr())

    expected_psi = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
    assert psi_result["psi_bin"].sum() == pytest.approx(expected_psi)
    assert rc_result["risk_corr"][0] == pytest.approx(1.0)


def test_psi_partition_prob_expr_matches_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x", "y"],
            "bin_index": [0, 1, 0],
            "len": [30.0, 70.0, 50.0],
        }
    )

    result = df.with_columns(psi_partition_prob_expr(["feature"], output_col="prob")).sort(
        ["feature", "bin_index"],
    )

    assert result["prob"].to_list() == pytest.approx([0.3, 0.7, 1.0])


def test_psi_valid_condition_filters_index_bins() -> None:
    df = pl.DataFrame({"bin_index": [-3, -2, -1, 0, 1]})

    base = df.filter(
        psi_valid_condition(
            pl.col("bin_index"),
            include_missing=False,
            include_special=False,
        ),
    )
    include_all = df.filter(
        psi_valid_condition(
            pl.col("bin_index"),
            include_missing=True,
            include_special=True,
        ),
    )

    assert base["bin_index"].to_list() == [-2, 0, 1]
    assert include_all["bin_index"].to_list() == [-3, -2, -1, 0, 1]


def test_observed_metric_agg_exprs_match_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x", "y"],
            "observed_count": [10.0, 20.0, 0.0],
            "iv_bin": [0.1, 0.2, 0.5],
            "ks_bin": [10.0, 30.0, 99.0],
            "auc_bin": [0.2, 0.3, 0.9],
            "lift": [0.8, 1.5, 3.0],
        }
    )

    result = df.group_by("feature").agg(
        [
            observed_iv_agg_expr(),
            observed_ks_agg_expr(),
            observed_auc_agg_expr(),
            observed_lift_min_agg_expr(),
            observed_lift_max_agg_expr(output_col="lift_max"),
        ]
    ).sort("feature")

    x_row = result.row(0, named=True)
    y_row = result.row(1, named=True)
    assert x_row["iv"] == pytest.approx(0.3)
    assert x_row["ks"] == pytest.approx(30.0)
    assert x_row["auc"] == pytest.approx(0.5)
    assert x_row["lift_min"] == pytest.approx(0.8)
    assert x_row["lift_max"] == pytest.approx(1.5)
    assert y_row["iv"] is None
    assert y_row["ks"] is None


def test_partition_distribution_expr_matches_manual_values() -> None:
    df = pl.DataFrame(
        {
            "feature": ["x", "x", "y"],
            "expected_count": [20.0, 80.0, 10.0],
        }
    )

    result = df.with_columns(expected_dist_expr()).sort(["feature", "expected_count"])

    assert result["expected_dist"].to_list() == pytest.approx([0.2, 0.8, 1.0])


def test_compute_helpers_return_polars_expressions() -> None:
    factories: list[Callable[[], pl.Expr | list[pl.Expr]]] = [
        lambda: bad_rate_expr(),
        lambda: actual_dist_expr(),
        lambda: normalized_auc_expr(),
        lambda: partition_distribution_expr(["feature"], count_col="count", output_col="dist"),
        lambda: ratio_expr(
            numerator_col="observed_count",
            denominator_col="count",
            output_col="observed_rate",
        ),
        lambda: binary_metric_exprs(),
        lambda: amount_metric_exprs(),
        lambda: ordered_binary_metric_exprs(["feature"]),
        lambda: observed_iv_agg_expr(),
        lambda: observed_ks_agg_expr(),
        lambda: observed_auc_agg_expr(),
        lambda: psi_exprs(["feature"]),
        lambda: psi_partition_prob_expr(["feature"], output_col="prob"),
        lambda: psi_valid_condition(
            pl.col("bin_index"),
            include_missing=False,
            include_special=False,
        ),
        lambda: risk_corr_expr(),
    ]

    for factory in factories:
        value = factory()
        if isinstance(value, list):
            assert value
            assert all(isinstance(expr, pl.Expr) for expr in value)
        else:
            assert isinstance(value, pl.Expr)
