"""统计筛选的客群编排，复用原漏斗与正式分箱规则。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from mars.compute.binning import normalize_binary_target_column
from mars.feature._scopes import build_scopes, resolve_scopes, scope_filter, validate_segment_roles
from mars.feature.binning.base import MarsBinnerBase
from mars.feature.binning.optimal import MarsOptimalBinner

if TYPE_CHECKING:
    from mars.feature.selection.stats import MarsStatsSelector
    from mars.reporting import MarsBinningReport


def tag_binning_report(
    report: MarsBinningReport, scopes: pl.DataFrame, identity: str,
) -> MarsBinningReport:
    """新生成的专用报告携带身份；固定模板由公共 scope adapter 消费。"""
    def tag(frame: Any) -> Any:
        """仅给已计算统计增加稳定身份，不改动业务字段。"""
        if frame is None:
            return None
        if isinstance(frame, pd.DataFrame):
            return frame.assign(scope_id=identity)
        return frame.with_columns(pl.lit(identity).alias("scope_id"))

    report._summary = tag(report._summary)
    report._detail = tag(report._detail)
    report._trend_dict = {name: tag(frame) for name, frame in report._trend_dict.items()}
    report._missing_by_day = tag(report._missing_by_day)
    report._risk_corr_reference = tag(report._risk_corr_reference)
    report._calculation_status = tag(report._calculation_status)
    report._scopes = scopes
    report.report_meta["default_scope_id"] = identity
    return report


def _clone_rule(binner: MarsBinnerBase | None) -> MarsBinnerBase | None:
    """规则副本不包含原始样本，WOE 更新与剪枝只影响当前客群。"""
    return None if binner is None else MarsBinnerBase.from_dict(binner.to_dict())


def select_scope(
    selector: MarsStatsSelector, segments: list[Any] | None, include_all: bool,
) -> tuple[MarsStatsSelector, str | None]:
    """消费入口严格选择一个 scope，失败客群不能借用整体名单。"""
    selector._check_is_fitted()
    if selector.scopes_ is None:
        if segments is not None or not include_all:
            raise ValueError("This selector has no independently fitted customer segments.")
        return selector, None
    identities = resolve_scopes(selector.scopes_, segments, include_all)
    if len(identities) != 1:
        raise ValueError("Select exactly one scope; use include_all=False for a customer segment.")
    identity = identities[0]
    if identity not in selector.selectors_by_scope_:
        row = selector.scopes_.filter(pl.col("scope_id") == identity).row(0, named=True)
        raise ValueError(f"Scope {identity!r} is unavailable: {row['status']}: {row['reason']}.")
    return selector.selectors_by_scope_[identity], identity


def _common_rules(
    selector: MarsStatsSelector, population: pl.DataFrame, candidates: list[str],
    *, fine: bool, mapping_population: pl.DataFrame | None = None,
) -> MarsBinnerBase | None:
    """在整体既定拟合人口学习阶段候选并集，转换映射也来自同一人口。"""
    if not candidates:
        return None
    if selector.target is None:
        raise ValueError("Shared binning requires a target column.")
    if fine:
        parameters: dict[str, Any] = {
            "n_jobs": selector.n_jobs, **selector.binning_params,
            "missing_values": selector.missing_values,
            "special_values": selector.special_values,
        }
        binner: MarsBinnerBase = MarsOptimalBinner(**parameters)
        observed = population.select([*candidates, selector.target]).filter(
            pl.col(selector.target).is_not_null()
        )
        binner.fit(observed.select(candidates), observed[selector.target], features=candidates)
        binner.profile_bin_performance(
            observed.select(candidates), observed[selector.target], features=candidates,
            update_woe=True, batch_size=selector.batch_size,
        )
    else:
        # 独立策略副本只借用粗筛规则学习，阈值结果不会进入任何 scope 的漏斗。
        learner = type(selector)(**selector.get_params(deep=False))
        learner.target = selector.target
        learner._feature_source_map = selector._feature_source_map
        mapping_frame = population if selector.target in population.columns else mapping_population
        if mapping_frame is None:
            raise ValueError("Shared WOE mappings require an observed target population.")
        learner._filter_rough(mapping_frame, candidates, benchmark_df=population)
        if learner._rough_binner is None:
            raise ValueError("Shared rough binning produced no fitted rules.")
        binner = learner._rough_binner
    binner.clear_cache()
    return binner


def fit_selector(
    selector: MarsStatsSelector, df: pl.DataFrame | pd.DataFrame, *, target: str,
    benchmark_df: pl.DataFrame | pd.DataFrame | None, features: list[str] | None,
    segment_col: str | None, **context: Any,
) -> MarsStatsSelector:
    """校验全部输入，再顺序执行客群漏斗，不保存原始宽表或基准表。"""
    # 入口角色或数据校验也可能失败，必须先清除上一次运行的规则与公开决策。
    selector._reset_correlation()
    selector.selected_features_ = []
    selector.report_records_ = []
    selector._rough_binner = None
    selector._stage3_binner = None
    selector._stability_report = None
    selector._feature_iv_dict = {}
    selector._funnel_stats = []
    selector.scopes_ = None
    selector.selectors_by_scope_ = {}
    selector.selected_features_by_segment_ = {}
    selector.segment_col = segment_col
    selector._fixed_rough_binner = None
    selector._fixed_fine_binner = None
    selector._pause_after_quality = False
    if segment_col is None:
        if selector.selection_scope != "global" or selector.binning_scope != "shared":
            raise ValueError("Non-default selection/binning scope requires segment_col.")
        return selector._fit_single(
            df, target=target, features=features, benchmark_df=benchmark_df, **context,
        )
    validate_segment_roles(segment_col, features, {
        "target": target, "group_col": context.get("group_col"),
        "time_col": context.get("time_col"),
    })
    columns = list(df.columns)
    if segment_col not in columns:
        raise ValueError(f"Segment column {segment_col!r} was not found in df.")
    excluded = {target, segment_col, context.get("group_col"), context.get("time_col")}
    candidates = list(features) if features is not None else [c for c in columns if c not in excluded]
    projection = list(dict.fromkeys([
        *candidates, target, segment_col,
        *[c for c in (context.get("group_col"), context.get("time_col")) if c is not None],
    ]))
    projected = df.select(projection) if isinstance(df, pl.DataFrame) else df.loc[:, projection]
    frame = normalize_binary_target_column(selector._ensure_polars_dataframe(projected), target)
    benchmark: pl.DataFrame | None = None
    if benchmark_df is not None:
        benchmark_columns = list(dict.fromkeys([
            *candidates, segment_col, *([target] if target in benchmark_df.columns else []),
        ]))
        projected_benchmark = (
            benchmark_df.select(benchmark_columns) if isinstance(benchmark_df, pl.DataFrame)
            else benchmark_df.loc[:, benchmark_columns]
        )
        benchmark = selector._ensure_polars_dataframe(projected_benchmark)
        if target in benchmark.columns:
            benchmark = normalize_binary_target_column(benchmark, target)
    scopes = build_scopes(frame, segment_col, benchmark)
    if selector.selection_scope == "global":
        result = selector._fit_single(
            df, target=target, features=candidates, benchmark_df=benchmark_df, **context,
        )
        result.segment_col = segment_col
        result._corr_parameters["segment_col"] = segment_col
        return result

    # 完整表先验证契约错误；单客群统计不足才作为局部状态发布。
    selector.target = target
    selector._prepare_evaluation_frame(frame)
    selector._prepare_selector_benchmark(benchmark, features=candidates)
    from mars.analysis._evaluation.context import prepare_group_context

    grouped, group_name = prepare_group_context(
        frame, group_col=context.get("group_col"), time_col=context.get("time_col"),
        time_grain=context.get("time_grain"), mars_group_col="__mars_selector_reference",
    )
    group_values = grouped[group_name].drop_nulls().unique().sort().to_list()
    reference_group = group_values[0] if group_values else None
    children: dict[str, MarsStatsSelector] = {}
    failures: dict[str, tuple[str, str]] = {}
    # benchmark 的内部转换不能改变由当前输入与 set_output 决定的输出格式。
    selector._determine_output_format(input_is_pandas=isinstance(df, pd.DataFrame))
    output_config = selector._output_config
    output_pandas = selector._return_pandas
    parameters: dict[str, Any] = {
        **selector.get_params(deep=False), "selection_scope": "global", "binning_scope": "shared",
    }
    binning_scope = selector.binning_scope
    for identity in scopes["scope_id"].to_list():
        child = type(selector)(**parameters)
        child._pause_after_quality = True
        child._scope_context = identity != "all"
        child._reference_group = reference_group
        current = scope_filter(frame, scopes, identity)
        baseline = scope_filter(benchmark, scopes, identity) if benchmark is not None else None
        fit_population = baseline if baseline is not None else current
        supervised = not child.skip_fine_scan or child.rough_binning_params.get("method") == "cart"
        if selector.binning_scope == "per_segment" and supervised and (
            target not in fit_population.columns or fit_population[target].drop_nulls().n_unique() < 2
        ):
            failures[identity] = ("fit_unavailable", "single_class_fit")
            continue
        observed_classes = current[target].drop_nulls().n_unique()
        if identity != "all" and observed_classes < 2:
            failures[identity] = (
                "statistics_undefined", "no_observed_labels" if not observed_classes else "single_class_statistics",
            )
            continue
        if selector.binning_scope == "per_segment" and baseline is not None:
            count = baseline[target].drop_nulls().n_unique() if target in baseline.columns else 0
            supervised = not child.skip_fine_scan or child.rough_binning_params.get("method") == "cart"
            if baseline.is_empty() or (supervised and count < 2):
                failures[identity] = ("fit_unavailable", "missing_reference" if baseline.is_empty() else "single_class_fit")
                continue
        try:
            child._fit_single(
                current, target=target, features=candidates, benchmark_df=baseline, **context,
            )
        except (ValueError, RuntimeError) as exc:
            if identity == "all":
                raise
            failures[identity] = ("failed", str(exc))
            continue
        children[identity] = child
    if "all" not in children:
        raise ValueError("The overall selection could not be fitted.")

    common_rough: MarsBinnerBase | None = None
    common_fine: MarsBinnerBase | None = None
    if selector.binning_scope == "shared":
        all_child = children["all"]
        population = benchmark if benchmark is not None else frame
        rough_union = [f for f in candidates if any(
            f in child._pending_features and (child.skip_fine_scan or not child._should_bypass_filter(f))
            for child in children.values()
        )]
        if not selector.skip_rough_scan:
            common_rough = _common_rules(
                all_child, population, rough_union, fine=False, mapping_population=frame,
            )
        fine_candidates: set[str] = set()
        for identity, child in children.items():
            active = list(child._pending_features)
            child._fixed_rough_binner = _clone_rule(common_rough)
            if not child.skip_fine_scan and not child.skip_rough_scan:
                scan = [f for f in active if not child._should_bypass_filter(f)]
                protected = [f for f in active if child._should_bypass_filter(f)]
                current = scope_filter(frame, scopes, identity)
                baseline = scope_filter(benchmark, scopes, identity) if benchmark is not None else None
                # 只预探查阶段并集；真实决策随后由同一固定规则执行，避免重复记入漏斗。
                saved_records, saved_iv = child.report_records_, child._feature_iv_dict
                child.report_records_, child._feature_iv_dict = [], {}
                active = child._filter_rough(current, scan, benchmark_df=baseline) + protected
                child.report_records_, child._feature_iv_dict = saved_records, saved_iv
            fine_candidates.update(active)
        if not selector.skip_fine_scan:
            fine_union = [f for f in candidates if f in fine_candidates]
            common_fine = _common_rules(all_child, population, fine_union, fine=True)

    completed: dict[str, MarsStatsSelector] = {}
    for identity, child in children.items():
        current = scope_filter(frame, scopes, identity)
        baseline = scope_filter(benchmark, scopes, identity) if benchmark is not None else None
        child._pause_after_quality = False
        child._fixed_rough_binner = _clone_rule(common_rough)
        child._fixed_fine_binner = _clone_rule(common_fine)
        if selector.binning_scope == "shared":
            fit_source = "benchmark_df" if benchmark is not None else "df"
            child._binning_fit_source = f"shared overall {fit_source}"
            child._corr_parameters["woe_mapping_population"] = (
                f"overall {fit_source} observed labels"
                if target in (benchmark if benchmark is not None else frame).columns
                else "overall df observed labels; benchmark contains no target"
            )
        try:
            child._resume_selection(
                current, child._pending_candidates, child._pending_features,
                child._pending_white, baseline,
            )
        except (ValueError, RuntimeError) as exc:
            if identity == "all":
                raise
            child.clear_cache()
            failures[identity] = ("failed", str(exc))
            continue
        child._fixed_rough_binner = None
        child._fixed_fine_binner = None
        child._output_config = output_config
        child._return_pandas = output_pandas
        completed[identity] = child
    overall = completed["all"]
    selector.__dict__.update(overall.__dict__)
    selector.selection_scope = "per_segment"
    selector.binning_scope = binning_scope
    selector.segment_col = segment_col
    selector.selectors_by_scope_ = completed
    selector.selected_features_by_segment_ = {
        identity: list(completed[identity].selected_features_) if identity in completed else None
        for identity in scopes["scope_id"].to_list()
    }
    rows = scopes.to_dicts()
    for row in rows:
        identity = row["scope_id"]
        row["status"], row["reason"] = failures.get(identity, ("computed", None))
    selector.scopes_ = pl.DataFrame(rows, schema=scopes.schema)
    selector._is_fitted = True
    return selector
