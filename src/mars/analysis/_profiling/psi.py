"""数据画像 PSI 趋势计算。"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from mars.analysis._profiling.metrics import feature_dtypes
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.compute import psi_contribution_expr, psi_partition_prob_expr, psi_valid_condition
from mars.core.constants import DIVISION_EPSILON, METRIC_EPSILON
from mars.feature import MarsNativeBinner
from mars.utils.logger import logger

MAX_PSI_GROUPS = 1000


def get_psi_trend(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    *,
    features: list[str] | None = None,
) -> pl.DataFrame:
    """计算特征在分组维度上的 PSI 趋势。"""
    if context.group_col is None:
        return pl.DataFrame()

    n_groups = context.working_df.select(pl.col(context.group_col).n_unique()).item()
    if n_groups > MAX_PSI_GROUPS:
        logger.warning(
            "PSI calculation aborted: column '%s' has %s unique values. "
            "Threshold is %s; check whether an ID column was passed as group_col.",
            context.group_col,
            n_groups,
            MAX_PSI_GROUPS,
        )
        return pl.DataFrame()

    candidates = [col for col in (features or context.features) if col != context.group_col]
    if not candidates:
        return pl.DataFrame()

    try:
        baseline_group = context.working_df.select(pl.col(context.group_col).min()).item()
    except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
        logger.warning("PSI baseline selection failed for group_col='%s': %s", context.group_col, exc)
        return pl.DataFrame()

    result_parts = _binned_psi_parts(context, options, candidates, baseline_group)
    if not result_parts:
        return pl.DataFrame()

    final_long = pl.concat(result_parts)
    pivot_df = final_long.pivot(on=context.group_col, index=["feature", "total"], values="psi")
    result = pivot_df.join(feature_dtypes(context), on="feature", how="left")
    psi_data_cols = sorted([col for col in result.columns if col not in ["feature", "dtype", "total"]])
    if not psi_data_cols:
        return result.sort("feature")

    result = (
        result.with_columns(pl.concat_list(psi_data_cols).alias("_tmp_psi_list"))
        .with_columns(
            [
                pl.col("_tmp_psi_list").list.mean().alias("group_mean"),
                pl.col("_tmp_psi_list").list.max().fill_null(0).alias("group_max"),
                pl.col("_tmp_psi_list").list.var().fill_null(0).alias("group_var"),
                pl.col("_tmp_psi_list").list.std().alias("_tmp_std"),
            ]
        )
        .with_columns(
            pl.when(pl.col("group_max") < options.psi_cv_ignore_threshold)
            .then(pl.lit(0.0))
            .otherwise(pl.col("_tmp_std") / (pl.col("group_mean") + DIVISION_EPSILON))
            .fill_null(0)
            .alias("group_cv")
        )
        .drop(["_tmp_psi_list", "_tmp_std", "group_max"])
    )
    return result.select(
        ["feature", "dtype", *psi_data_cols, "total", "group_mean", "group_var", "group_cv"]
    ).sort("feature")


def _binned_psi_parts(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    candidates: list[str],
    baseline_group: Any,
) -> list[pl.DataFrame]:
    """基于 NativeBinner 分箱结果分批计算画像 PSI。"""
    try:
        binner = MarsNativeBinner(
            method=options.psi_bin_method,
            n_bins=options.psi_n_bins,
            min_bin_size=options.psi_min_bin_size,
            special_values=options.special_values,
            missing_values=options.missing_values,
            merge_small_bins=options.psi_merge_small_bins,
            remove_empty_bins=options.psi_remove_empty_bins,
        )
        binner.fit(context.working_df, features=candidates)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        logger.warning("Profile PSI binning failed for %s features: %s", len(candidates), exc)
        return []

    bin_cuts = getattr(binner, "bin_cuts_", {})
    cat_cuts = getattr(binner, "cat_cuts_", {})
    fitted_cols = [col for col in candidates if col in bin_cuts or col in cat_cuts]
    if not fitted_cols:
        return []

    parts: list[pl.DataFrame] = []
    for start in range(0, len(fitted_cols), options.psi_batch_size):
        batch_cols = fitted_cols[start : start + options.psi_batch_size]
        try:
            part = _binned_psi_batch(context, options, binner, batch_cols, baseline_group)
        except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
            logger.warning(
                "Profile PSI batch failed for features %s-%s: %s",
                start,
                start + len(batch_cols) - 1,
                exc,
            )
            continue
        if not part.is_empty():
            parts.append(part)
    return parts


def _binned_psi_batch(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    binner: MarsNativeBinner,
    batch_cols: list[str],
    baseline_group: Any,
) -> pl.DataFrame:
    """计算一个分箱特征批次的 PSI。"""
    assert context.group_col is not None
    df_batch = context.working_df.select([*batch_cols, context.group_col])
    lf_binned = cast(pl.LazyFrame, binner.transform(df_batch, return_type="index", lazy=True))

    feat_map = {idx: name for idx, name in enumerate(batch_cols)}
    rename_map = {f"{name}_bin": str(idx) for idx, name in enumerate(batch_cols)}
    lf_stats = (
        lf_binned.rename(rename_map)
        .select([context.group_col, *rename_map.values()])
        .unpivot(
            index=[context.group_col],
            on=list(rename_map.values()),
            variable_name="feat_idx",
            value_name="bin_id",
        )
        .with_columns([pl.col("feat_idx").cast(pl.Int16), pl.col("bin_id").cast(pl.Int16)])
        .group_by([context.group_col, "feat_idx", "bin_id"])
        .len()
    )

    lf_skeleton = lf_stats.select(["feat_idx", "bin_id"]).unique()
    lf_psi = calc_psi_from_stats(
        stats_df=lf_stats,
        unique_bins_skel=lf_skeleton,
        group_col=context.group_col,
        baseline_group=baseline_group,
        include_missing=options.psi_include_missing,
        include_special=options.psi_include_special,
    )

    mapping_df = pl.LazyFrame(
        {"feat_idx": list(feat_map), "feature": list(feat_map.values())},
        schema={"feat_idx": pl.Int16, "feature": pl.String},
    )
    return (
        lf_psi.join(mapping_df, on="feat_idx", how="left")
        .select([context.group_col, "feature", "total", "psi"])
        .collect(engine="streaming")
    )


def calc_psi_from_stats(
    *,
    stats_df: pl.LazyFrame,
    unique_bins_skel: pl.LazyFrame,
    group_col: str,
    baseline_group: Any,
    include_missing: bool,
    include_special: bool,
) -> pl.LazyFrame:
    """基于聚合频次表计算 PSI。"""
    feat_col = "feat_idx" if "feat_idx" in stats_df.collect_schema().names() else "feature"
    filter_cond = psi_valid_condition(
        pl.col("bin_id"),
        include_missing=include_missing,
        include_special=include_special,
    )
    filtered_stats = stats_df.filter(filter_cond)
    filtered_skel = unique_bins_skel.filter(filter_cond)
    full_skeleton = filtered_skel.join(filtered_stats.select(group_col).unique(), how="cross")

    expected = (
        filtered_stats.filter(pl.col(group_col) == baseline_group)
        .with_columns(psi_partition_prob_expr([feat_col], output_col="E"))
        .select([feat_col, "bin_id", "E"])
    )
    actual = (
        filtered_stats.with_columns(
            psi_partition_prob_expr([group_col, feat_col], output_col="A"),
        )
        .select([group_col, feat_col, "bin_id", "A"])
    )
    global_actual = (
        filtered_stats.group_by([feat_col, "bin_id"])
        .agg(pl.col("len").sum().alias("total_len"))
        .with_columns(
            psi_partition_prob_expr(
                [feat_col],
                count_col="total_len",
                output_col="A_global",
            ),
        )
        .select([feat_col, "bin_id", "A_global"])
    )

    psi_group = (
        full_skeleton.join(actual, on=[group_col, feat_col, "bin_id"], how="left")
        .join(expected, on=[feat_col, "bin_id"], how="left")
        .with_columns([pl.col("A").fill_null(METRIC_EPSILON), pl.col("E").fill_null(METRIC_EPSILON)])
        .with_columns(psi_contribution_expr(pl.col("A"), pl.col("E"), epsilon=METRIC_EPSILON).alias("psi_contrib"))
        .group_by([group_col, feat_col])
        .agg(pl.col("psi_contrib").sum().alias("psi"))
    )
    psi_total = (
        filtered_skel.join(global_actual, on=[feat_col, "bin_id"], how="left")
        .join(expected, on=[feat_col, "bin_id"], how="left")
        .with_columns(
            [
                pl.col("A_global").fill_null(METRIC_EPSILON),
                pl.col("E").fill_null(METRIC_EPSILON),
            ]
        )
        .with_columns(
            psi_contribution_expr(pl.col("A_global"), pl.col("E"), epsilon=METRIC_EPSILON).alias("psi_contrib_total")
        )
        .group_by(feat_col)
        .agg(pl.col("psi_contrib_total").sum().alias("total"))
    )
    return psi_group.join(psi_total, on=feat_col, how="left")
