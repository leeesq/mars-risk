"""数据画像客群编排，固定分箱人口与 PSI 参考人口分别传递。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from mars.analysis._profiling.context import (
    build_run_context,
    prepare_benchmark_data,
    prepare_profile_data,
)
from mars.analysis._profiling.metrics import normalize_profile_metrics
from mars.analysis._profiling.psi import (
    _benchmark_degenerate_failures,
    _validate_benchmark_frame,
)
from mars.feature import MarsNativeBinner
from mars.feature._scopes import build_scopes, scope_filter, validate_segment_roles
from mars.reporting import MarsProfileReport

if TYPE_CHECKING:
    from mars.analysis.profiler import MarsDataProfiler


def _frame(value: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """只转换已聚合的输出，不缓存原始人口。"""
    return value if isinstance(value, pl.DataFrame) else pl.from_pandas(value)


def _project(
    frame: pl.DataFrame | pd.DataFrame, columns: list[str],
) -> pl.DataFrame | pd.DataFrame:
    """在转换前投影实际存在列，缺列错误仍交给原画像校验。"""
    available = [column for column in dict.fromkeys(columns) if column in frame.columns]
    return frame.select(available) if isinstance(frame, pl.DataFrame) else frame.loc[:, available]


def _fit_binner(
    profiler: MarsDataProfiler,
    population: pl.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> MarsNativeBinner:
    """仅在给定拟合人口学习规则，不以参考分布替换该人口。"""
    binner = MarsNativeBinner(
        method=profiler.psi_bin_method,
        n_bins=profiler.psi_n_bins,
        min_bin_size=profiler.psi_min_bin_size,
        special_values=profiler.special_values,
        missing_values=profiler.missing_values,
        merge_small_bins=profiler.psi_merge_small_bins,
        remove_empty_bins=profiler.psi_remove_empty_bins,
    )
    binner.fit(population, features=features, cat_features=categorical_features)
    binner._cache_X = None
    binner._cache_y = None
    return binner


def generate_segmented_profile(
    profiler: MarsDataProfiler,
    df: pl.DataFrame | pd.DataFrame,
    segment_col: str,
    kwargs: dict[str, Any],
) -> MarsProfileReport:
    """逐客群复用画像计算，保留整体首组与原始 dtype；不常驻各客群宽表。"""
    features: list[str] | None = kwargs["features"]
    validate_segment_roles(segment_col, features, {
        "group_col": kwargs["group_col"], "time_col": kwargs["time_col"],
        "categorical_features": kwargs["categorical_features"],
    })
    selection = normalize_profile_metrics(kwargs["metrics"], require_metrics=False)
    comparisons = bool(selection.comparison_metrics)
    excluded = [*list(kwargs["exclude_features"] or []), segment_col]

    # 显式特征时先投影，再做 Pandas 转换；对比缺列仍由原画像入口统一校验。
    columns = features if features is not None else [c for c in df.columns if c not in excluded]
    df = _project(df, [*columns, segment_col,
                      *[c for c in (kwargs["group_col"], kwargs["time_col"]) if c is not None]])
    current, active = prepare_profile_data(
        df, ensure_polars=profiler._ensure_polars_dataframe,
        features=features, exclude_features=excluded, include_dtypes=kwargs["include_dtypes"],
        allow_missing_features=comparisons and not selection.dq_metrics and not selection.stat_metrics,
    )
    input_is_pandas = isinstance(df, pd.DataFrame)
    benchmark_input: pl.DataFrame | pd.DataFrame | None = kwargs["benchmark_df"]
    if benchmark_input is not None:
        benchmark_columns = features if features is not None else [
            c for c in benchmark_input.columns if c not in excluded
        ]
        benchmark_input = _project(benchmark_input, [*benchmark_columns, segment_col])
    benchmark = prepare_benchmark_data(benchmark_input) if benchmark_input is not None else None
    scopes = build_scopes(current, segment_col, benchmark)
    context = build_run_context(
        current, features=active, group_col=kwargs["group_col"],
        time_col=kwargs["time_col"], time_grain=kwargs["time_grain"],
    )
    candidates = [feature for feature in active if feature not in {context.group_col, kwargs["time_col"]}]
    psi_requested = "psi" in selection.stat_metrics
    if benchmark is not None and benchmark.is_empty():
        raise ValueError("`benchmark_df` must contain at least one row.")
    if psi_requested and benchmark is not None:
        _validate_benchmark_frame(benchmark, current_df=context.working_df, candidates=candidates)
    if comparisons and benchmark is None:
        raise ValueError("Metrics 'schema' and 'unseen' require `benchmark_df`.")
    declared = list(kwargs["categorical_features"] or [])
    comparison_candidates = set(candidates)
    if comparisons and benchmark is not None:
        comparison_candidates.update(
            feature for feature in benchmark.columns
            if feature not in {*excluded, kwargs["group_col"], kwargs["time_col"]}
        )
    declared_scope = set(candidates) if psi_requested else comparison_candidates
    if (psi_requested or comparisons) and any(feature not in declared_scope for feature in declared):
        raise ValueError("`categorical_features` are outside the active comparison or PSI scope.")

    # 所有 scope 固定使用整体最小组；缺该组的客群绝不跳到自身下一期。
    baseline_group: Any = None
    first_group: pl.DataFrame | None = None
    if benchmark is None and context.group_col is not None:
        baseline_group = context.working_df.select(pl.col(context.group_col).min()).item()
        condition = pl.col(context.group_col).is_null() if baseline_group is None else (
            pl.col(context.group_col) == pl.lit(baseline_group)
        )
        first_group = context.working_df.filter(condition)
    fit_population = benchmark if benchmark is not None else current
    shared_binner: MarsNativeBinner | None = None
    shared_fit_failures: dict[str, str] = {}
    if psi_requested and not current.is_empty() and (benchmark is not None or first_group is not None):
        if profiler.binning_scope == "shared":
            shared_binner = _fit_binner(profiler, fit_population, candidates, declared)
            if benchmark is not None:
                shared_fit_failures = _benchmark_degenerate_failures(
                    fit_population, candidates,
                    missing_values=profiler.missing_values, special_values=profiler.special_values,
                    categorical_features=declared, batch_size=profiler.psi_batch_size,
                )

    overview_parts: list[pl.DataFrame] = []
    dq_parts: dict[str, list[pl.DataFrame]] = {metric: [] for metric in selection.dq_metrics}
    stats_parts: dict[str, list[pl.DataFrame]] = {metric: [] for metric in selection.stat_metrics}
    comparison_parts: dict[str, list[pl.DataFrame]] = {metric: [] for metric in selection.comparison_metrics}
    status_parts: list[pl.DataFrame] = []
    scope_diagnostics: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    report_meta: dict[str, Any] = {}
    for scope_id in scopes["scope_id"].to_list():
        population = scope_filter(current, scopes, scope_id)
        local_fit = scope_filter(fit_population, scopes, scope_id)
        local_benchmark = scope_filter(benchmark, scopes, scope_id) if benchmark is not None else None
        reference = local_benchmark if benchmark is not None else (
            scope_filter(first_group, scopes, scope_id) if first_group is not None else None
        )
        reference_reason: str | None = None
        if psi_requested and reference is not None and reference.is_empty():
            reference_reason = "missing_segment_benchmark" if benchmark is not None else "missing_first_group"
        fixed_binner = shared_binner
        fit_failures = shared_fit_failures
        if psi_requested and profiler.binning_scope == "per_segment" and reference_reason is None:
            if not population.is_empty() and reference is not None:
                try:
                    fixed_binner = _fit_binner(profiler, local_fit, candidates, declared)
                    if benchmark is not None:
                        fit_failures = _benchmark_degenerate_failures(
                            local_fit, candidates,
                            missing_values=profiler.missing_values, special_values=profiler.special_values,
                            categorical_features=declared, batch_size=profiler.psi_batch_size,
                        )
                except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
                    reference_reason = f"psi_binning_failed: {exc}"

        local_kwargs = {**kwargs, "exclude_features": excluded,
                        "benchmark_df": local_benchmark if comparisons else reference,
                        "psi_binner": fixed_binner,
                        "psi_fit_df": fit_population if profiler.binning_scope == "shared" else local_fit,
                        "psi_fit_population_failures": fit_failures,
                        "psi_reference_reason": reference_reason, "allow_empty_benchmark": True}
        # 对比检查始终读取对应客群的显式基准；无 benchmark 的首组只用于 PSI。
        local_report = profiler._generate_profile(population, **local_kwargs)
        if scope_id == "all":
            report_meta = dict(local_report.report_meta)
        scope_diagnostics.extend(
            {**diagnostic, "scope_id": scope_id}
            for diagnostic in local_report.report_meta.get("diagnostics", [])
        )
        overview_parts.append(_frame(local_report.overview_table).with_columns(pl.lit(scope_id).alias("scope_id")))
        for destination, tables in (
            (dq_parts, local_report.dq_tables), (stats_parts, local_report.stats_tables),
            (comparison_parts, local_report.comparison_tables),
        ):
            for metric, table in tables.items():
                destination[metric].append(_frame(table).with_columns(pl.lit(scope_id).alias("scope_id")))
        if local_report._calculation_status is not None:
            status_parts.append(_frame(local_report._calculation_status).with_columns(pl.lit(scope_id).alias("scope_id")))
        references.append({
            "scope_id": scope_id, "binning_scope": profiler.binning_scope,
            "fit_row_count": fit_population.height if profiler.binning_scope == "shared" else local_fit.height,
            "expected_row_count": reference.height if reference is not None else None,
            "reference_kind": "benchmark" if benchmark is not None else "first_group",
            "reference_group": baseline_group, "reason": reference_reason,
        })
        del local_report, population, local_fit, local_benchmark, reference

    report_meta.update({
        "input_type": "pandas" if input_is_pandas else "polars", "segment_col": segment_col,
        "binning_scope": profiler.binning_scope, "scope_semantics_version": 1,
        "benchmark_row_count": benchmark.height if benchmark is not None else None,
        "scope_references": references,
        "diagnostics": scope_diagnostics,
    })
    profiler._determine_output_format(input_is_pandas)
    report = MarsProfileReport(
        overview=profiler._format_output(pl.concat(overview_parts, how="diagonal_relaxed")),
        dq_tables={metric: profiler._format_output(pl.concat(parts, how="diagonal_relaxed"))
                   for metric, parts in dq_parts.items()},
        stats_tables={metric: (
            profiler._format_psi_output(pl.concat(parts, how="diagonal_relaxed"))
            if metric == "psi" else profiler._format_output(pl.concat(parts, how="diagonal_relaxed"))
        )
                      for metric, parts in stats_parts.items()},
        comparison_tables={metric: profiler._format_output(pl.concat(parts, how="diagonal_relaxed"))
                           for metric, parts in comparison_parts.items()},
        report_meta=report_meta, feature_metadata=kwargs["feature_metadata"],
        business_context=kwargs["business_context"],
    )
    report._scopes = scopes
    if status_parts:
        report._calculation_status = profiler._format_output(pl.concat(status_parts, how="diagonal_relaxed"))
    report._normalize_feature_tables()
    report.source = {"kind": "analysis", "producer": "mars", "input_provenance": "caller_provided_dataframe"}
    return report
