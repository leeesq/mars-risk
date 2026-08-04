"""数据画像 PSI 趋势计算。"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from mars._compat import collect_streaming
from mars.analysis._profiling.metrics import feature_dtypes
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.compute import (
    missing_condition_expr,
    psi_contribution_expr,
    psi_partition_prob_expr,
    psi_valid_condition,
)
from mars.core.constants import METRIC_EPSILON
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
    return result.select(["feature", "dtype", *psi_data_cols, "total"]).sort("feature")


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


def _benchmark_degenerate_failures(
    benchmark_df: pl.DataFrame,
    candidates: list[str],
    options: ProfileComputeOptions,
) -> dict[str, str]:
    """识别显式 benchmark 中无法构造可解释 PSI 切分的特征。"""
    excluded_values = [*options.missing_values, *options.special_values]
    count_exprs: list[pl.Expr] = []
    for feature in candidates:
        invalid_condition = missing_condition_expr(
            feature,
            dtype=benchmark_df.schema[feature],
            missing_values=excluded_values,
        )
        count_exprs.append(
            pl.col(feature).filter(~invalid_condition).n_unique().alias(feature)
        )
    valid_counts = benchmark_df.select(count_exprs).row(0, named=True)

    failures: dict[str, str] = {}
    declared_categorical = set(options.categorical_features)
    for feature in candidates:
        valid_count = int(valid_counts[feature])
        if valid_count == 0:
            failures[feature] = "All values are missing or special."
            continue
        is_numeric = benchmark_df.schema[feature] in MarsNativeBinner.NUMERIC_DTYPES
        if valid_count == 1 and is_numeric and feature not in declared_categorical:
            failures[feature] = "Degenerate feature: single unique value."
    return failures


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


def _record_psi_failure(
    options: ProfileComputeOptions,
    feature: str,
    reason: str,
) -> None:
    """按特征记录 PSI 降级原因，并避免同一特征重复记录。"""
    already_recorded = any(
        diagnostic.get("component") == "psi"
        and feature in diagnostic.get("features", [])
        for diagnostic in options.diagnostics
    )
    if already_recorded:
        return
    options.diagnostics.append(
        {
            "component": "psi",
            "features": [feature],
            "reason": reason,
        }
    )


def _usable_benchmark_feature_indices(
    *,
    current_stats: pl.LazyFrame,
    benchmark_stats: pl.LazyFrame,
    feat_map: dict[int, str],
    options: ProfileComputeOptions,
) -> set[int]:
    """返回显式基准两侧都含有效 PSI 观测的特征索引。"""
    current_valid = _valid_feature_indices(current_stats, options)
    benchmark_valid = _valid_feature_indices(benchmark_stats, options)
    expected_indices = set(feat_map)
    usable_indices = current_valid & benchmark_valid

    for index in sorted(expected_indices - usable_indices):
        invalid_sides: list[str] = []
        if index not in current_valid:
            invalid_sides.append("current_df")
        if index not in benchmark_valid:
            invalid_sides.append("benchmark_df")
        _record_psi_failure(
            options,
            feat_map[index],
            "PSI has no included observations in " + " and ".join(invalid_sides) + ".",
        )
    return usable_indices


def _null_psi_part(
    context: ProfileRunContext,
    features: list[str],
) -> pl.DataFrame:
    """为不可计算特征构造保留输出行的空 PSI 长表。"""
    group_col = context.group_col or INTERNAL_PSI_GROUP_COL
    feature_df = pl.DataFrame({"feature": features}, schema={"feature": pl.String})
    if context.group_col is None:
        return feature_df.with_columns(
            [
                pl.lit("Total").alias(group_col),
                pl.lit(None, dtype=pl.Float64).alias("total"),
                pl.lit(None, dtype=pl.Float64).alias("psi"),
            ]
        ).select([group_col, "feature", "total", "psi"])

    group_values = context.working_df.select(group_col).unique().sort(group_col)
    return (
        feature_df.join(group_values, how="cross")
        .with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("total"),
                pl.lit(None, dtype=pl.Float64).alias("psi"),
            ]
        )
        .select([group_col, "feature", "total", "psi"])
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
    fit_df = benchmark_df if benchmark_df is not None else context.working_df
    benchmark_failures = (
        _benchmark_degenerate_failures(benchmark_df, candidates, options)
        if benchmark_df is not None
        else {}
    )
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
        if benchmark_df is not None:
            raise ValueError(
                f"`benchmark_df` PSI binning failed for active features {candidates}: {exc}"
            ) from exc
        raise ValueError(
            f"Profile PSI binning failed for active features {candidates}: {exc}"
        ) from exc

    fit_failures = getattr(binner, "fit_failures_", {})
    unavailable_fit_features: set[str] = set()
    if benchmark_df is not None:
        combined_failures = {
            **benchmark_failures,
            **{
                feature: str(reason)
                for feature, reason in fit_failures.items()
                if feature in candidates
            },
        }
        unavailable_fit_features = set(combined_failures)
        for feature in candidates:
            if feature in unavailable_fit_features:
                _record_psi_failure(options, feature, combined_failures[feature])

    bin_cuts = getattr(binner, "bin_cuts_", {})
    cat_cuts = getattr(binner, "cat_cuts_", {})
    fitted_cols = [
        col
        for col in candidates
        if (col in bin_cuts or col in cat_cuts) and col not in unavailable_fit_features
    ]
    missing_fitted_cols = [
        col
        for col in candidates
        if col not in fitted_cols and col not in unavailable_fit_features
    ]
    for feature in missing_fitted_cols:
        _record_psi_failure(options, feature, "PSI binning did not produce a usable rule.")

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
            for feature in batch_cols:
                _record_psi_failure(options, feature, str(exc))
            continue
        if not part.is_empty():
            parts.append(part)

    produced_features = {
        str(feature)
        for part in parts
        for feature in part.get_column("feature").unique().to_list()
    }
    unavailable_features = [
        feature for feature in candidates if feature not in produced_features
    ]
    for feature in unavailable_features:
        _record_psi_failure(options, feature, "PSI calculation produced no usable result.")
    if unavailable_features:
        parts.append(_null_psi_part(context, unavailable_features))
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
        usable_indices = _usable_benchmark_feature_indices(
            current_stats=lf_stats,
            benchmark_stats=benchmark_stats,
            feat_map=feat_map,
            options=options,
        )
        if not usable_indices:
            return pl.DataFrame()

        # 只将两侧都有有效观测的特征送入 PSI 计算，其余特征由上层补空行。
        usable_index_values = sorted(usable_indices)
        usable_filter = pl.col("feat_idx").is_in(usable_index_values)
        lf_stats = lf_stats.filter(usable_filter)
        benchmark_stats = benchmark_stats.filter(usable_filter)
        lf_skeleton = lf_skeleton.filter(usable_filter)
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
