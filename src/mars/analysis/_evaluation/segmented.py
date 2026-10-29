"""风险画像客群编排；规则与参考人口分别解析，不缓存原始样本。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from mars.analysis._evaluation.aggregation import aggregate_basic_stats
from mars.analysis._evaluation.context import (
    binning_requires_target,
    build_binner,
    count_observed_target_classes,
    normalize_binary_target_column,
    prepare_group_context,
    resolve_date_bounds,
)
from mars.analysis._evaluation.metrics import ensure_woe_info
from mars.compute import OrderedMetricSortBy, RiskCorrBaseline, to_polars_frame
from mars.feature._scopes import build_scopes, scope_filter, validate_segment_roles
from mars.feature.binning._json_codec import read_json_artifact, write_json_artifact
from mars.feature.binning.base import MarsBinnerBase
from mars.reporting import MarsBinningReport
from mars.reporting._metadata import FeatureMetadata, is_metadata_column

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsBinEvaluator, MarsRiskProfile


def _validate_population(frame: pl.DataFrame, target: str | None, weights: str | None) -> pl.DataFrame:
    """先验证全部输入契约，避免非法标签或权重被客群局部失败掩盖。"""
    if target is not None and target in frame.columns:
        frame = normalize_binary_target_column(frame, target)
    if weights is not None:
        if weights not in frame.columns:
            raise ValueError(f"weights_col {weights!r} is missing from input.")
        values = pl.col(weights).cast(pl.Float64, strict=False)
        if frame.select((values.is_null() | ~values.is_finite() | (values < 0)).any()).item():
            raise ValueError(f"weights_col {weights!r} must contain finite non-negative weights.")
    return frame


def _rule_identity(binner: MarsBinnerBase) -> str:
    """以正式规则内容形成可追溯身份，不读取或散列原始宽表。"""
    payload = json.dumps(binner.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_scope_rules(
    evaluator: MarsBinEvaluator, population: pl.DataFrame, features: list[str],
    target: str | None, weights_col: str | None, batch_size: int,
) -> MarsBinnerBase:
    """在明确拟合人口中学习边界和转换映射，再释放训练缓存。"""
    has_labels = target is not None and target in population.columns
    observed = count_observed_target_classes(population, target) if has_labels and target else 0
    supervised = binning_requires_target(
        binning_type=evaluator.binning_type, binner_params=evaluator.binner_params,
    )
    if population.is_empty():
        raise ValueError("missing_fit_population")
    if supervised and observed < 2:
        raise ValueError("single_class_fit" if observed else "no_observed_fit_labels")
    binner = build_binner(
        binning_type=evaluator.binning_type, binner_params=dict(evaluator.binner_params),
        fit_has_target=observed > 0, fit_df=population,
        target=target or "dummy_target", features=features,
    )
    # 无监督边界仍在完整拟合人口学习；转换用 WOE 只取该人口已表现标签。
    if target is not None and observed:
        usable = binner._usable_rule_features()
        for start in range(0, len(usable), batch_size):
            batch = usable[start:start + batch_size]
            projection = list(dict.fromkeys(batch + [target] + ([weights_col] if weights_col else [])))
            binned = binner.transform(population.select(projection), features=batch)
            fit_group = "__mars_scope_fit"
            while fit_group in binned.columns:
                fit_group += "_"
            binned = binned.with_columns(pl.lit("Fit").alias(fit_group))
            stats = aggregate_basic_stats(
                binned, features=batch, group_col=fit_group, target_col=target,
                weights_col=weights_col, batch_size=batch_size,
            )
            ensure_woe_info(binner, stats)
    binner._cache_X = None
    binner._cache_y = None
    return binner


def _null_reference_metrics(
    report: MarsBinningReport, reason: str, target: str | None, *,
    metrics: tuple[str, ...] = ("psi",), features: list[str] | None = None,
) -> None:
    """按独立指标的参考人口擦除不可计算值，并替换同指标已有失败状态。"""
    affected = features if features is not None else to_polars_frame(report.summary_table)["feature"].to_list()
    condition = pl.col("feature").is_in(affected)
    columns = (["psi_max"] if "psi" in metrics else []) + (["rc_min"] if "risk_corr" in metrics else [])
    report._summary = to_polars_frame(report.summary_table).with_columns([
        pl.when(condition).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col(c)).alias(c)
        for c in columns
    ])
    if "psi" in metrics:
        detail = to_polars_frame(report.detail_table)
        report._detail = detail.with_columns([
            pl.when(condition).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col(c)).alias(c)
            for c in ["psi_bin", "psi", "expected_dist"] if c in detail.columns
        ])
    for metric in metrics:
        if metric not in report._trend_dict:
            continue
        trend = to_polars_frame(report._trend_dict[metric])
        report._trend_dict[metric] = trend.with_columns([
            pl.when(condition).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col(c)).alias(c)
            for c in trend.columns if not is_metadata_column(c)
        ])
    status = to_polars_frame(report.get_table("calculation_status"))
    status = status.filter(~(condition & pl.col("metric").is_in(list(metrics))))
    extra = pl.DataFrame([
        {"feature": f, "target": target, "group": None, "metric": metric, "status": "not_computed", "reason": reason}
        for f in affected for metric in metrics
    ], schema=status.schema)
    report._calculation_status = pl.concat([status, extra], how="vertical_relaxed")


def evaluate_segmented(
    evaluator: MarsBinEvaluator, df: pl.DataFrame | pd.DataFrame, *, segment_col: str,
    target: str | None, features: list[str] | None, group_col: str | None,
    time_col: str | None, time_grain: str | None,
    feature_data_source: dict[str, list[str]] | None, feature_metadata: FeatureMetadata | None,
    business_context: dict[str, Any] | None, weights_col: str | None, amount_col: str | None,
    binner: MarsBinnerBase | None, segment_binners: dict[str, MarsBinnerBase] | None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None, psi_include_missing: bool,
    psi_include_special: bool, feature_start_aware_reference: bool | None,
    risk_corr_baseline: RiskCorrBaseline | None, ordered_metric_sort_by: OrderedMetricSortBy | None,
    batch_size: int,
) -> MarsRiskProfile:
    """顺序执行整体和各客群，所有连接在单一人口内完成后再加身份合表。"""
    from mars.analysis.evaluator import MarsBinEvaluator, MarsRiskProfile

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    validate_segment_roles(segment_col, features, {
        "target": target, "group_col": group_col, "time_col": time_col,
        "weights_col": weights_col, "amount_col": amount_col,
    })
    if evaluator.binning_scope == "per_segment" and binner is not None:
        raise ValueError("An overall binner conflicts with binning_scope='per_segment'; use segment_binners.")
    if segment_binners is not None and (binner is not None or evaluator.binner_params):
        raise ValueError("segment_binners conflicts with binner and evaluator binner_params.")
    if binner is not None and evaluator.binner_params:
        raise ValueError("binner conflicts with evaluator binner_params.")
    if segment_binners is not None and evaluator.binning_scope != "per_segment":
        raise ValueError("segment_binners requires binning_scope='per_segment'.")
    if features is None:
        excluded = {segment_col, target, group_col, time_col, weights_col, amount_col, evaluator.MARS_GROUP_COL}
        features = [c for c in df.columns if c not in excluded]
    if not features:
        raise ValueError("No feature columns remain after excluding declared roles.")
    # 先在原后端投影，再转换；无关宽列不会进入 Polars 或客群切片。
    selected = list(dict.fromkeys(features + [c for c in [segment_col, target, group_col, time_col, weights_col, amount_col] if c in df.columns]))
    projected = df.loc[:, selected] if isinstance(df, pd.DataFrame) else df.select(selected)
    frame = _validate_population(to_polars_frame(projected), target, weights_col)
    benchmark_projected = (
        benchmark_df.loc[:, [c for c in selected if c in benchmark_df.columns]]
        if isinstance(benchmark_df, pd.DataFrame)
        else benchmark_df.select([c for c in selected if c in benchmark_df.columns])
        if benchmark_df is not None else None
    )
    benchmark = (
        _validate_population(to_polars_frame(benchmark_projected), target, weights_col)
        if benchmark_projected is not None else None
    )
    scopes = build_scopes(frame, segment_col, benchmark)
    roles = features + [c for c in [group_col, time_col, amount_col] if c]
    missing = sorted(set(roles) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}.")
    if benchmark is not None:
        missing = sorted(set(features) - set(benchmark.columns))
        if missing:
            raise ValueError(f"benchmark_df is missing active feature columns: {missing}.")
    projection = list(dict.fromkeys(features + [c for c in [segment_col, target, group_col, time_col, weights_col, amount_col] if c in frame.columns]))
    frame = frame.select(projection)
    if benchmark is not None:
        benchmark = benchmark.select([c for c in projection if c in benchmark.columns])
    grouped, internal_group = prepare_group_context(
        frame, group_col=group_col, time_col=time_col, time_grain=time_grain,
        mars_group_col=evaluator.MARS_GROUP_COL,
    )
    first_group = grouped.select(pl.col(internal_group).min()).item()
    if segment_binners is not None:
        absent = set(scopes["scope_id"].to_list()) - set(segment_binners) - set(evaluator._scope_fit_failures)
        if absent:
            raise ValueError(f"Fixed segment_binners is missing current scope rules: {sorted(absent)}.")

    rules: dict[str, MarsBinnerBase] = {}
    reports: dict[str, MarsBinningReport] = {}
    failed_status: list[dict[str, Any]] = []
    scope_rows = scopes.to_dicts()
    for row in scope_rows:
        scope_id = row["scope_id"]
        if scope_id in evaluator._scope_fit_failures:
            reason = evaluator._scope_fit_failures[scope_id]
            row.update(status="failed", reason=reason)
            failed_status.extend({
                "scope_id": scope_id, "feature": f, "target": target, "group": None,
                "metric": "binning", "status": "failed", "reason": reason,
            } for f in features)
            continue
        current = scope_filter(frame, scopes, scope_id)
        reference = scope_filter(benchmark, scopes, scope_id) if benchmark is not None else None
        fit_population = reference if reference is not None else current
        fixed = segment_binners.get(scope_id) if segment_binners is not None else binner
        if evaluator.binning_scope == "shared" and scope_id != "all":
            fixed = rules["all"]
        if fixed is None:
            try:
                fixed = _fit_scope_rules(evaluator, fit_population, features, target, weights_col, batch_size)
            except (ValueError, RuntimeError) as exc:
                if scope_id == "all":
                    raise
                row.update(status="failed", reason=str(exc))
                failed_status.extend({
                    "scope_id": scope_id, "feature": f, "target": target, "group": None,
                    "metric": "binning", "status": "failed", "reason": str(exc),
                } for f in features)
                continue
        rules[scope_id] = fixed
        active_features = [f for f in features if f in fixed._usable_rule_features()]
        unavailable = [f for f in features if f not in active_features]
        if unavailable:
            row.update(status="partial" if active_features else "failed", reason="features_without_usable_rules")
            failed_status.extend({
                "scope_id": scope_id, "feature": f, "target": target, "group": None,
                "metric": "binning", "status": "failed",
                "reason": str(fixed.fit_failures_.get(f, "missing_rules")),
            } for f in unavailable)
        if not active_features:
            if scope_id == "all":
                raise ValueError("No usable overall binning rules are available.")
            continue
        # 只复制正式规则，不复制训练表；报告补齐统计 WOE 不能改写已学转换。
        report_rules = MarsBinnerBase.from_dict(fixed.to_dict())
        child = MarsBinEvaluator(
            binning_type=evaluator.binning_type,
            feature_start_aware_reference=evaluator.feature_start_aware_reference,
            risk_corr_baseline=evaluator.risk_corr_baseline,
            ordered_metric_sort_by=evaluator.ordered_metric_sort_by,
        )
        child._scope_context = True
        child._reference_group = first_group
        run = child.evaluate(
            current, target=target, features=active_features, group_col=group_col, time_col=time_col,
            time_grain=time_grain, feature_data_source=feature_data_source,
            feature_metadata=feature_metadata, business_context=business_context,
            weights_col=weights_col, amount_col=amount_col, binner=report_rules,
            benchmark_df=reference, psi_include_missing=psi_include_missing,
            psi_include_special=psi_include_special,
            feature_start_aware_reference=feature_start_aware_reference,
            risk_corr_baseline=risk_corr_baseline, ordered_metric_sort_by=ordered_metric_sort_by,
            batch_size=batch_size,
        )
        report = run.report
        report._report_meta.update({
            "rule_id": _rule_identity(fixed), "binning_scope": evaluator.binning_scope,
            "binning_fit_source": "explicit_rules" if binner is not None or segment_binners is not None else "benchmark_df" if benchmark is not None else "df",
            "binning_fit_scope": "all" if evaluator.binning_scope == "shared" else scope_id,
            "psi_reference_scope": scope_id, "reference_group": first_group if benchmark is None else None,
            "woe_mapping_population": "fixed_rule_fit_population", "table_woe": "current_scope_statistics",
            "binning_fit_row_count": (benchmark.height if benchmark is not None else frame.height) if evaluator.binning_scope == "shared" else fit_population.height,
        })
        expected_population = reference
        if expected_population is None:
            expected_population = scope_filter(grouped, scopes, scope_id).filter(
                pl.col(internal_group).is_null() if first_group is None else pl.col(internal_group) == first_group
            )
        report._report_meta.update({
            "psi_reference_row_count": expected_population.height,
            "psi_reference_date_range": resolve_date_bounds(expected_population, time_col),
        })
        reports[scope_id] = report
        del current, reference, fit_population, report_rules

    primary = reports["all"]

    def combine(attribute: str) -> pl.DataFrame | None:
        """只合并已聚合统计表，保留单个 scope 的行序与稳定身份。"""
        parts = []
        for identity, report in reports.items():
            table = getattr(report, attribute)
            if table is not None:
                parts.append(to_polars_frame(table).with_columns(pl.lit(identity).alias("scope_id")))
        return pl.concat(parts, how="diagonal_relaxed") if parts else None

    summary = combine("summary_table")
    detail = combine("detail_table")
    assert summary is not None and detail is not None
    trends: dict[str, pl.DataFrame | pd.DataFrame] = {}
    for metric in dict.fromkeys(m for report in reports.values() for m in report.trend_tables):
        trends[metric] = pl.concat([
            to_polars_frame(report.trend_tables[metric]).with_columns(pl.lit(identity).alias("scope_id"))
            for identity, report in reports.items() if metric in report.trend_tables
        ], how="diagonal_relaxed")
    meta = dict(primary.report_meta)
    meta.update(segment_col=segment_col, binning_scope=evaluator.binning_scope,
                scope_metadata={identity: dict(report.report_meta) for identity, report in reports.items()})
    merged = MarsBinningReport(
        summary, trends, detail, group_col=primary.group_col, detail_group_col=primary.detail_group_col,
        feature_data_source=primary.feature_data_source, dt_col=primary.dt_col,
        missing_by_day_table=combine("missing_by_day_table"),
        risk_corr_reference_table=combine("risk_corr_reference_table"), report_meta=meta,
        feature_metadata=primary.feature_metadata, business_context=business_context,
    )
    statuses = [to_polars_frame(r.get_table("calculation_status")).with_columns(pl.lit(identity).alias("scope_id")) for identity, r in reports.items()]
    if failed_status:
        statuses.append(pl.DataFrame(failed_status, schema=statuses[0].schema))
    merged._calculation_status = pl.concat(statuses, how="diagonal_relaxed")
    merged._scopes = pl.DataFrame(scope_rows, schema=scopes.schema)
    merged.source = dict(primary.source)
    if isinstance(df, pd.DataFrame):
        merged._summary = to_polars_frame(merged._summary).to_pandas()
        merged._detail = to_polars_frame(merged._detail).to_pandas()
        merged._trend_dict = {k: to_polars_frame(v).to_pandas() for k, v in merged._trend_dict.items()}
        merged._calculation_status = to_polars_frame(merged._calculation_status).to_pandas()
    return MarsRiskProfile(merged, rules["all"], primary.report_meta["targets"], meta, rules, merged._scopes)


def save_binning_rules(profile: MarsRiskProfile, path: str | Path) -> None:
    """将 scope 目录和各正式分箱规则写入单一受限 JSON 产物。"""
    write_json_artifact(path, {
        "artifact_type": "mars_segmented_binners", "schema_version": 1,
        "scopes": profile.scopes.to_dicts() if profile.scopes is not None else None,
        "binning_scope": profile.metadata.get("binning_scope", "shared"),
        "rules": {identity: rules.to_dict() for identity, rules in (profile.binners_by_scope or {"all": profile.binner}).items()},
    })


def load_binning_rules(path: str | Path) -> dict[str, MarsBinnerBase]:
    """仅恢复固定规则，不恢复原始数据或分析器。"""
    payload = read_json_artifact(path)
    if payload.get("artifact_type") != "mars_segmented_binners" or payload.get("schema_version") != 1:
        raise ValueError("Unsupported segmented binner artifact type or version.")
    return {identity: MarsBinnerBase.from_dict(value) for identity, value in payload["rules"].items()}
