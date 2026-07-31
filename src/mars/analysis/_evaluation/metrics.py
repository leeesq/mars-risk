"""分箱评估指标计算。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from mars._compat import polars_is_in
from mars.compute import (
    OrderedMetricSortBy,
    amount_distribution_exprs,
    amount_metric_exprs,
    bad_dist_expr,
    binary_distribution_exprs,
    binary_metric_exprs,
    good_dist_expr,
    normalize_ordered_metric_sort_by,
    ordered_binary_metric_exprs,
    psi_exprs,
    woe_expr,
)
from mars.core.constants import FLOAT_TOLERANCE, METRIC_EPSILON
from mars.feature.binning.base import MarsBinnerBase
from mars.utils.logger import logger


def build_woe_table_from_mapping(woe_mapping: dict[str, dict[int, float]]) -> pl.DataFrame:
    """把分箱器缓存的 WOE 映射转换为长表。"""
    rows: list[dict[str, Any]] = []
    for feature, mapping in woe_mapping.items():
        for bin_index, woe in mapping.items():
            rows.append(
                {
                    "feature": str(feature),
                    "bin_index": int(bin_index),
                    "woe": float(woe),
                }
            )
    schema = {"feature": pl.String, "bin_index": pl.Int16, "woe": pl.Float32}
    if not rows:
        return pl.DataFrame([], schema=schema)
    return pl.DataFrame(rows, schema=schema)


def ensure_woe_info(binner: MarsBinnerBase, group_stats_raw: pl.DataFrame) -> None:
    """用已聚合统计表补齐分箱器缺失的 WOE 缓存。"""
    features = group_stats_raw["feature"].unique().to_list()
    missing_woe_features = [
        feature
        for feature in features
        if feature not in binner.bin_woes_ or not binner.bin_woes_[feature]
    ]
    if not missing_woe_features:
        return

    logger.debug("Calculating missing WOEs for %s features.", len(missing_woe_features))
    target_stats = group_stats_raw.filter(
        polars_is_in(pl.col("feature"), pl.Series(missing_woe_features))
    )
    woe_df = (
        target_stats
        .group_by(["feature", "bin_index"])
        .agg(
            [
                pl.col("bad").sum().alias("bin_bad"),
                pl.col("observed_count").sum().alias("bin_observed"),
            ]
        )
        .with_columns((pl.col("bin_observed") - pl.col("bin_bad")).alias("bin_good"))
        .with_columns(
            [
                pl.col("bin_bad").sum().over("feature").alias("feature_total_bad"),
                pl.col("bin_good").sum().over("feature").alias("feature_total_good"),
            ]
        )
        .with_columns(
            [
                bad_dist_expr(bad_col="bin_bad", total_bad_col="feature_total_bad"),
                good_dist_expr(
                    observed_count_col="bin_observed",
                    bad_col="bin_bad",
                    total_good_col="feature_total_good",
                ),
            ]
        )
        .with_columns(woe_expr())
    )
    woe_data = woe_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)
    temp_woe_map: dict[str, dict[int, float]] = defaultdict(dict)
    for feature, bin_index, woe in zip(
        woe_data["feature"],
        woe_data["bin_index"],
        woe_data["woe"],
    ):
        if bin_index is not None and not (isinstance(bin_index, float) and np.isnan(bin_index)):
            temp_woe_map[str(feature)][int(bin_index)] = float(woe)

    binner.bin_woes_.update(temp_woe_map)


def calculate_metrics_from_stats(
    *,
    binner: MarsBinnerBase,
    stats_df: pl.DataFrame,
    expected_dist: pl.DataFrame,
    group_col: str,
    include_missing: bool = True,
    include_special: bool = True,
    ordered_metric_sort_by: OrderedMetricSortBy = "woe",
) -> pl.DataFrame:
    """使用 compute 表达式 bundle 计算分箱指标。"""
    effective_sort_by = normalize_ordered_metric_sort_by(ordered_metric_sort_by)
    woe_df = build_woe_table_from_mapping(binner.bin_woes_)
    working_df = stats_df.with_columns(pl.col("bin_index").cast(pl.Int16))
    expected_dist = expected_dist.with_columns(pl.col("bin_index").cast(pl.Int16))
    if "observed_count" not in working_df.columns:
        working_df = working_df.with_columns(pl.col("count").alias("observed_count"))

    base_df = (
        working_df
        .join(expected_dist, on=["feature", "bin_index"], how="left")
        .join(woe_df, on=["feature", "bin_index"], how="left")
        .with_columns(
            [
                pl.col("expected_dist").fill_null(METRIC_EPSILON),
                pl.col("woe").fill_null(0.0),
            ]
        )
        .with_columns(binary_distribution_exprs([group_col, "feature"]))
    )
    if {"tot_amt", "good_amt", "bad_amt"}.issubset(base_df.columns):
        base_df = base_df.with_columns(amount_distribution_exprs([group_col, "feature"]))

    base_df = (
        base_df
        .with_columns(
            psi_exprs(
                [group_col, "feature"],
                include_missing=include_missing,
                include_special=include_special,
            )
        )
        .with_columns(binary_metric_exprs())
    )
    if {
        "tot_amt",
        "good_amt",
        "bad_amt",
        "observed_amt",
        "total_observed_amt",
        "total_bad_amt",
    }.issubset(base_df.columns):
        base_df = base_df.with_columns(amount_metric_exprs())

    if effective_sort_by == "woe":
        ordered_df = (
            base_df
            .sort([group_col, "feature", "woe"])
            .with_columns(ordered_binary_metric_exprs([group_col, "feature"]))
        )
    else:
        normal_bins = (
            base_df
            .filter(pl.col("bin_index") >= 0)
            .with_columns(binary_distribution_exprs([group_col, "feature"]))
            .with_columns(binary_metric_exprs())
            .sort([group_col, "feature", "bin_index"])
            .with_columns(ordered_binary_metric_exprs([group_col, "feature"]))
            .select(
                [
                    group_col,
                    "feature",
                    "bin_index",
                    "cum_bad_dist",
                    "cum_good_dist",
                    "ks_bin",
                    "auc_bin",
                ]
            )
        )
        ordered_df = (
            base_df
            .drop(["cum_bad_dist", "cum_good_dist", "ks_bin", "auc_bin"], strict=False)
            .join(normal_bins, on=[group_col, "feature", "bin_index"], how="left")
        )

    return (
        ordered_df
        .with_columns(
            pl.when(pl.col("psi_bin").abs() < FLOAT_TOLERANCE)
            .then(0.0)
            .otherwise(pl.col("psi_bin"))
            .alias("psi_bin")
        )
    )
