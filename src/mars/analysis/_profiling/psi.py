"""数据画像 PSI 趋势计算。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import polars as pl

from mars._compat import collect_streaming
from mars.analysis._profiling.metrics import feature_dtypes
from mars.analysis._profiling.pivot import unavailable_pivot_report
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.compute import (
    missing_condition_expr,
    psi_contribution_expr,
    psi_partition_prob_expr,
    psi_valid_condition,
)
from mars.core.constants import METRIC_EPSILON
from mars.feature import MarsNativeBinner
from mars.reporting._metadata import _null_group_label

MAX_PSI_GROUPS = 1000
INTERNAL_PSI_GROUP_COL = "_mars_profile_psi_group"


def get_psi_trend(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    *,
    features: list[str] | None = None,
    benchmark_df: pl.DataFrame | None = None,
    binner: MarsNativeBinner | None = None,
    fit_df: pl.DataFrame | None = None,
    fit_population_failures: dict[str, str] | None = None,
) -> pl.DataFrame:
    """计算各分组及当前整体相对同一基准分布的 PSI。

    显式 benchmark 同时用于拟合分箱和构造参考分布；未提供时在当前数据拟合分箱，
    并以分组最小值对应的样本作为参考。无分组但有显式基准时只返回整体 PSI。
    """
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
    baseline_group: Any = None
    if benchmark_df is not None:
        _validate_benchmark_frame(
            benchmark_df,
            current_df=context.working_df,
            candidates=candidates,
        )

    # 空当前人口仍执行同一结构校验，但不构造分箱或虚拟分组值。
    if context.working_df.is_empty():
        return unavailable_pivot_report(replace(context, features=selected_features))
    if not candidates:
        raise ValueError("PSI requires at least one active feature.")

    if benchmark_df is None:
        assert context.group_col is not None
        try:
            baseline_group = context.working_df.select(pl.col(context.group_col).min()).item()
        except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
            raise ValueError(
                f"PSI baseline selection failed for group_col='{context.group_col}'."
            ) from exc

    # null 组须先获得不与真实文字冲突的身份，避免统计 join 将 null 当作未匹配行。
    # 基准组先按原始 dtype 选取，再转换标签，数值分组的最小值口径不变。
    if context.group_col is not None:
        null_label = _null_group_label(context.working_df[context.group_col].unique().to_list())
        context = replace(context, working_df=context.working_df.with_columns(
            pl.col(context.group_col).cast(pl.String).fill_null(null_label),
        ))
        if benchmark_df is None:
            baseline_group = null_label if baseline_group is None else str(baseline_group)

    result_parts = _binned_psi_parts(
        context,
        options,
        candidates,
        baseline_group,
        benchmark_df=benchmark_df,
        binner=binner,
        fit_df=fit_df,
        fit_population_failures=fit_population_failures,
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
    *,
    missing_values: list[Any],
    special_values: list[Any],
    categorical_features: list[str],
    batch_size: int,
) -> dict[str, str]:
    """分批检查实际规则拟合人口；局部参考分布不能冒充共享拟合人口。"""
    excluded_values = [*missing_values, *special_values]
    valid_counts: dict[str, Any] = {}
    for start in range(0, len(candidates), batch_size):
        count_exprs: list[pl.Expr] = []
        for feature in candidates[start : start + batch_size]:
            invalid_condition = missing_condition_expr(
                feature,
                dtype=benchmark_df.schema[feature],
                missing_values=excluded_values,
            )
            count_exprs.append(
                pl.col(feature).filter(~invalid_condition).n_unique().alias(feature)
            )
        valid_counts.update(benchmark_df.select(count_exprs).row(0, named=True))

    failures: dict[str, str] = {}
    declared_categorical = set(categorical_features)
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


def _record_psi_failure(
    options: ProfileComputeOptions,
    feature: str,
    reason: str,
    *,
    group: str = "*",
    status: str = "failed",
) -> None:
    """按特征和期间记录 PSI 例外；全局拟合失败仍沿用原诊断形状。"""
    already_recorded = any(
        diagnostic.get("component") == "psi"
        and feature in diagnostic.get("features", [])
        and diagnostic.get("group", "*") == group
        for diagnostic in options.diagnostics
    )
    if already_recorded:
        return
    diagnostic: dict[str, Any] = {
        "component": "psi", "features": [feature], "reason": reason,
    }
    if group != "*" or status != "failed":
        diagnostic.update(group=group, status=status)
    options.diagnostics.append(diagnostic)


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
    binner: MarsNativeBinner | None = None,
    fit_df: pl.DataFrame | None = None,
    fit_population_failures: dict[str, str] | None = None,
) -> list[pl.DataFrame]:
    """基于 NativeBinner 分箱结果分批计算画像 PSI。"""
    fit_population = fit_df if fit_df is not None else (
        benchmark_df if benchmark_df is not None else context.working_df
    )
    # 客群编排显式传入本次拟合诊断（空字典也有含义），内部首组参考不触发基准校验。
    if fit_population_failures is not None:
        benchmark_failures = fit_population_failures
    elif benchmark_df is not None and binner is None:
        benchmark_failures = _benchmark_degenerate_failures(
            fit_population, candidates,
            missing_values=options.missing_values, special_values=options.special_values,
            categorical_features=options.categorical_features, batch_size=options.psi_batch_size,
        )
    else:
        benchmark_failures = {}
    try:
        if binner is None:
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
            binner.fit(fit_population, **fit_kwargs)
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
    # 仅物化本批频数，避免人口判断和 PSI 分支重复执行行级转换。
    lf_stats = collect_streaming(
        _aggregate_binned_stats(lf_binned, group_col=group_col, rename_map=rename_map)
    ).lazy()

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
        benchmark_stats = collect_streaming(
            benchmark_stats_grouped.group_by(["feat_idx", "bin_id"])
            .agg(pl.col("len").sum().alias("len"))
        ).lazy()
        lf_skeleton = pl.concat(
            [
                lf_skeleton,
                benchmark_stats.select(["feat_idx", "bin_id"]),
            ]
        ).unique()
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
        .select([group_col, "feature", "total", "psi", "psi_reason", "total_reason"])
    )
    result: pl.DataFrame = collect_streaming(result_query)
    for reason_col, group_expr in (
        ("psi_reason", pl.col(group_col)), ("total_reason", pl.lit("Total")),
    ):
        # Total 在长表中逐期重复，收集原因时只保留一次实际业务身份。
        failures: pl.DataFrame = (
            result.select(
                "feature", group_expr.alias("group"), pl.col(reason_col).alias("reason"),
            )
            .filter(pl.col("reason").is_not_null()).unique().sort(["feature", "group"])
        )
        for row in failures.iter_rows(named=True):
            _record_psi_failure(
                options, row["feature"], row["reason"],
                group=row["group"], status="not_computed",
            )
    return result.select([group_col, "feature", "total", "psi"])


def _psi_population_reason(current_col: str) -> pl.Expr:
    """平滑前按固定参考及当前有效计数判断适用性，优先解释不可用参考。"""
    return (
        pl.when(pl.col("reference_population").fill_null(0) <= 0)
        .then(pl.lit("no_effective_reference_population"))
        .when(pl.col(current_col).fill_null(0) <= 0)
        .then(pl.lit("no_effective_current_population"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


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
    """按聚合频次计算各组和整体的 PSI。

    先排除未纳入计算的缺失值箱和特殊值箱，再分别归一化当前分布与参考分布。
    整体 PSI 使用各箱跨组加总后的频次，不平均各组 PSI。平滑前分别判断当前期间、
    固定参考和整体人口；零人口保留 typed null，并返回期间与整体的内部原因列。
    """
    feat_col = "feat_idx" if "feat_idx" in stats_df.collect_schema().names() else "feature"
    filter_cond = psi_valid_condition(
        pl.col("bin_id"),
        include_missing=include_missing,
        include_special=include_special,
    )
    filtered_stats = stats_df.filter(filter_cond)
    filtered_skel = unique_bins_skel.filter(filter_cond)

    if expected_stats_df is None:
        expected_source = filtered_stats.filter(pl.col(group_col) == baseline_group)
    else:
        expected_source = expected_stats_df.filter(filter_cond)

    # 业务身份取自过滤前频数；人口为零不能由其他特征或 epsilon 补造。
    identities: pl.LazyFrame = stats_df.select([group_col, feat_col]).unique()
    current_population: pl.LazyFrame = filtered_stats.group_by([group_col, feat_col]).agg(
        pl.col("len").sum().alias("current_population"),
    )
    reference_population: pl.LazyFrame = expected_source.group_by(feat_col).agg(
        pl.col("len").sum().alias("reference_population"),
    )
    total_population: pl.LazyFrame = filtered_stats.group_by(feat_col).agg(
        pl.col("len").sum().alias("total_population"),
    )
    applicability: pl.LazyFrame = (
        identities.join(current_population, on=[group_col, feat_col], how="left")
        .join(reference_population, on=feat_col, how="left")
        .join(total_population, on=feat_col, how="left")
        .with_columns(
            _psi_population_reason("current_population").alias("psi_reason"),
            _psi_population_reason("total_population").alias("total_reason"),
        )
    )
    full_skeleton: pl.LazyFrame = filtered_skel.join(
        applicability.filter(pl.col("psi_reason").is_null()).select([group_col, feat_col]),
        on=feat_col, how="inner",
    )
    total_skeleton: pl.LazyFrame = filtered_skel.join(
        applicability.filter(pl.col("total_reason").is_null()).select(feat_col).unique(),
        on=feat_col, how="inner",
    )

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
        total_skeleton.join(global_actual, on=[feat_col, "bin_id"], how="left")
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
    return (
        applicability.join(psi_group, on=[group_col, feat_col], how="left")
        .join(psi_total, on=feat_col, how="left")
        .select([group_col, feat_col, "psi", "total", "psi_reason", "total_reason"])
    )
