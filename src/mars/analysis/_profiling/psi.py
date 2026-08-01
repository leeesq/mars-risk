"""数据画像 PSI 趋势计算。"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from mars._compat import collect_streaming
from mars.analysis._profiling.metrics import feature_dtypes
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.compute import psi_contribution_expr, psi_partition_prob_expr, psi_valid_condition
from mars.core.constants import DIVISION_EPSILON, METRIC_EPSILON
from mars.feature import MarsNativeBinner

MAX_PSI_GROUPS = 1000
INTERNAL_PSI_GROUP_COL = "_mars_profile_psi_group"


def get_psi_trend(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    *,
    features: list[str] | None = None,
    benchmark_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """计算特征相对内部首组或外部 benchmark 的 PSI。"""
    if context.group_col is None and benchmark_df is None:
        raise ValueError("PSI requires `group_col`, `time_col`, or `benchmark_df`.")

    if context.group_col is not None:
        n_groups = context.working_df.select(pl.col(context.group_col).n_unique()).item()
        if n_groups > MAX_PSI_GROUPS:
            raise ValueError(
                f"PSI group column '{context.group_col}' has {n_groups} unique values; "
                f"the maximum is {MAX_PSI_GROUPS}."
            )

    selected_features = context.features if features is None else features
    candidates = [col for col in selected_features if col != context.group_col]
    if not candidates:
        raise ValueError("PSI requires at least one active feature.")

    baseline_group: Any = None
    if benchmark_df is not None:
        _validate_benchmark_frame(
            benchmark_df,
            current_df=context.working_df,
            candidates=candidates,
        )
    else:
        assert context.group_col is not None
        try:
            baseline_group = context.working_df.select(pl.col(context.group_col).min()).item()
        except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
            raise ValueError(
                f"PSI baseline selection failed for group_col='{context.group_col}'."
            ) from exc

    result_parts = _binned_psi_parts(
        context,
        options,
        candidates,
        baseline_group,
        benchmark_df=benchmark_df,
    )
    if not result_parts:
        raise ValueError("PSI did not produce any usable feature results.")

    final_long = pl.concat(result_parts)
    if context.group_col is None:
        total_df = final_long.select(["feature", "total"]).unique()
        return (
            total_df.join(feature_dtypes(context), on="feature", how="left")
            .select(["feature", "dtype", "total"])
            .sort("feature")
        )

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


def _validate_benchmark_frame(
    benchmark_df: pl.DataFrame,
    *,
    current_df: pl.DataFrame,
    candidates: list[str],
) -> None:
    """校验显式 PSI benchmark 的基础表结构。"""
    if benchmark_df.is_empty():
        raise ValueError("`benchmark_df` must contain at least one row for PSI calculation.")

    missing_features = [feature for feature in candidates if feature not in benchmark_df.columns]
    if missing_features:
        raise ValueError(
            "`benchmark_df` is missing active PSI features "
            f"{missing_features}. Group and time columns are not required."
        )

    incompatible_dtypes: dict[str, tuple[pl.DataType, pl.DataType]] = {}
    numeric_dtypes = MarsNativeBinner.NUMERIC_DTYPES
    categorical_dtypes = {pl.String, pl.Categorical}
    for feature in candidates:
        current_dtype = current_df.schema[feature]
        benchmark_dtype = benchmark_df.schema[feature]
        both_numeric = current_dtype in numeric_dtypes and benchmark_dtype in numeric_dtypes
        both_string_like = (
            current_dtype in categorical_dtypes and benchmark_dtype in categorical_dtypes
        )
        if current_dtype != benchmark_dtype and not both_numeric and not both_string_like:
            incompatible_dtypes[feature] = (current_dtype, benchmark_dtype)

    if incompatible_dtypes:
        raise ValueError(
            "`benchmark_df` has incompatible dtypes for active PSI features "
            f"{incompatible_dtypes}."
        )


def _aggregate_binned_stats(
    binned_df: pl.LazyFrame,
    *,
    group_col: str,
    rename_map: dict[str, str],
) -> pl.LazyFrame:
    """把一批分箱列聚合到 group、feature 和 bin 粒度。"""
    return (
        binned_df.rename(rename_map)
        .select([group_col, *rename_map.values()])
        .unpivot(
            index=[group_col],
            on=list(rename_map.values()),
            variable_name="feat_idx",
            value_name="bin_id",
        )
        .with_columns(
            [
                pl.col("feat_idx").cast(pl.Int16),
                pl.col("bin_id").cast(pl.Int16),
            ]
        )
        .group_by([group_col, "feat_idx", "bin_id"])
        .len()
    )


def _valid_feature_indices(
    stats_df: pl.LazyFrame,
    options: ProfileComputeOptions,
) -> set[int]:
    """返回在当前 PSI 箱口径下至少有一条观测的特征索引。"""
    valid_condition = psi_valid_condition(
        pl.col("bin_id"),
        include_missing=options.psi_include_missing,
        include_special=options.psi_include_special,
    )
    valid_features = collect_streaming(
        stats_df.filter(valid_condition).select("feat_idx").unique()
    )
    return set(valid_features.get_column("feat_idx").to_list())


def _validate_benchmark_observations(
    *,
    current_stats: pl.LazyFrame,
    benchmark_stats: pl.LazyFrame,
    feat_map: dict[int, str],
    options: ProfileComputeOptions,
) -> None:
    """确保显式基准两侧的每个特征都有可用于 PSI 的观测。"""
    current_valid = _valid_feature_indices(current_stats, options)
    benchmark_valid = _valid_feature_indices(benchmark_stats, options)
    expected_indices = set(feat_map)
    issues: dict[str, list[str]] = {}

    missing_current = sorted(expected_indices - current_valid)
    if missing_current:
        issues["current_df"] = [feat_map[index] for index in missing_current]

    missing_benchmark = sorted(expected_indices - benchmark_valid)
    if missing_benchmark:
        issues["benchmark_df"] = [feat_map[index] for index in missing_benchmark]

    if issues:
        raise ValueError(
            "PSI requires at least one included bin for every active feature on both sides; "
            f"invalid observations: {issues}."
        )


def _binned_psi_parts(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    candidates: list[str],
    baseline_group: Any,
    *,
    benchmark_df: pl.DataFrame | None,
) -> list[pl.DataFrame]:
    """基于 NativeBinner 分箱结果分批计算画像 PSI。"""
    strict_benchmark = benchmark_df is not None
    fit_df = benchmark_df if benchmark_df is not None else context.working_df
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
        categorical_candidates = [
            feature
            for feature in options.categorical_features
            if feature in candidates
        ]
        fit_kwargs: dict[str, Any] = {"features": candidates}
        if categorical_candidates:
            fit_kwargs["cat_features"] = categorical_candidates
        binner.fit(fit_df, **fit_kwargs)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        if strict_benchmark:
            raise ValueError(
                f"`benchmark_df` PSI binning failed for active features {candidates}: {exc}"
            ) from exc
        raise ValueError(
            f"Profile PSI binning failed for active features {candidates}: {exc}"
        ) from exc

    fit_failures = getattr(binner, "fit_failures_", {})
    if strict_benchmark and fit_failures:
        raise ValueError(f"`benchmark_df` PSI binning failed: {fit_failures}.")

    bin_cuts = getattr(binner, "bin_cuts_", {})
    cat_cuts = getattr(binner, "cat_cuts_", {})
    fitted_cols = [col for col in candidates if col in bin_cuts or col in cat_cuts]
    missing_fitted_cols = [col for col in candidates if col not in fitted_cols]
    if strict_benchmark and missing_fitted_cols:
        raise ValueError(
            "`benchmark_df` could not produce binning rules for active features "
            f"{missing_fitted_cols}."
        )
    if not fitted_cols:
        return []

    parts: list[pl.DataFrame] = []
    for start in range(0, len(fitted_cols), options.psi_batch_size):
        batch_cols = fitted_cols[start : start + options.psi_batch_size]
        try:
            part = _binned_psi_batch(
                context,
                options,
                binner,
                batch_cols,
                baseline_group,
                benchmark_df=benchmark_df,
            )
        except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
            if strict_benchmark:
                raise ValueError(
                    f"`benchmark_df` PSI calculation failed for features {batch_cols}: {exc}"
                ) from exc
            options.diagnostics.append(
                {
                    "component": "psi",
                    "features": list(batch_cols),
                    "reason": str(exc),
                }
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
    *,
    benchmark_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """计算一个分箱特征批次的 PSI。"""
    group_col = context.group_col or INTERNAL_PSI_GROUP_COL
    if context.group_col is None:
        df_batch = context.working_df.select(batch_cols).with_columns(
            pl.lit("Total").alias(group_col)
        )
    else:
        df_batch = context.working_df.select([*batch_cols, context.group_col])
    lf_binned = cast(
        pl.LazyFrame,
        binner.transform(
            df_batch,
            features=batch_cols,
            return_type="index",
            lazy=True,
        ),
    )

    feat_map = {idx: name for idx, name in enumerate(batch_cols)}
    rename_map = {f"{name}_bin": str(idx) for idx, name in enumerate(batch_cols)}
    lf_stats = _aggregate_binned_stats(lf_binned, group_col=group_col, rename_map=rename_map)

    lf_skeleton = lf_stats.select(["feat_idx", "bin_id"]).unique()
    if benchmark_df is None:
        lf_psi = calc_psi_from_stats(
            stats_df=lf_stats,
            unique_bins_skel=lf_skeleton,
            group_col=group_col,
            baseline_group=baseline_group,
            include_missing=options.psi_include_missing,
            include_special=options.psi_include_special,
        )
    else:
        benchmark_batch = benchmark_df.select(batch_cols).with_columns(
            pl.lit("Benchmark").alias(group_col)
        )
        benchmark_binned = cast(
            pl.LazyFrame,
            binner.transform(
                benchmark_batch,
                features=batch_cols,
                return_type="index",
                lazy=True,
            ),
        )
        benchmark_stats_grouped = _aggregate_binned_stats(
            benchmark_binned,
            group_col=group_col,
            rename_map=rename_map,
        )
        benchmark_stats = (
            benchmark_stats_grouped.group_by(["feat_idx", "bin_id"])
            .agg(pl.col("len").sum().alias("len"))
        )
        lf_skeleton = pl.concat(
            [
                lf_skeleton,
                benchmark_stats.select(["feat_idx", "bin_id"]),
            ]
        ).unique()
        _validate_benchmark_observations(
            current_stats=lf_stats,
            benchmark_stats=benchmark_stats,
            feat_map=feat_map,
            options=options,
        )
        lf_psi = calc_psi_from_stats(
            stats_df=lf_stats,
            unique_bins_skel=lf_skeleton,
            group_col=group_col,
            baseline_group=None,
            include_missing=options.psi_include_missing,
            include_special=options.psi_include_special,
            expected_stats_df=benchmark_stats,
        )

    mapping_df = pl.LazyFrame(
        {"feat_idx": list(feat_map), "feature": list(feat_map.values())},
        schema={"feat_idx": pl.Int16, "feature": pl.String},
    )
    result_query = (
        lf_psi.join(mapping_df, on="feat_idx", how="left")
        .select([group_col, "feature", "total", "psi"])
    )
    return collect_streaming(result_query)


def calc_psi_from_stats(
    *,
    stats_df: pl.LazyFrame,
    unique_bins_skel: pl.LazyFrame,
    group_col: str,
    baseline_group: Any,
    include_missing: bool,
    include_special: bool,
    expected_stats_df: pl.LazyFrame | None = None,
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

    if expected_stats_df is None:
        expected_source = filtered_stats.filter(pl.col(group_col) == baseline_group)
    else:
        expected_source = expected_stats_df.filter(filter_cond)

    expected = (
        expected_source
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
