"""高层风险画像的原始值 KS 预检与报告装配，不扩展评估器公开接口。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from mars._compat import _left_join_nulls
from mars.analysis._evaluation.context import (
    _resolve_evaluation_features,
    normalize_binary_target_column,
    prepare_group_context,
)
from mars.compute import is_numeric_dtype, to_polars_frame
from mars.compute._raw_ks import _calculate_raw_ks
from mars.feature._scopes import build_scopes, scope_filter
from mars.reporting import MarsBinningReport
from mars.reporting._metadata import _null_group_label, is_metadata_column


def _prepare_raw_ks(
    df: pl.DataFrame | pd.DataFrame,
    *,
    targets: list[str],
    features: list[str] | None,
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
    internal_group_col: str,
    weights_col: str | None,
    amount_col: str | None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None,
    max_features: int,
    missing_values: list[Any] | None,
    special_values: list[Any] | None,
    segment_col: str | None = None,
) -> pl.DataFrame:
    """复用评估上下文，先检查数值特征上限，再计算原始值 KS。

    数值类型遵循拟合数据 schema；benchmark 只参与类型识别，不进入统计样本。
    无标签模式不向原始样本注入伪标签，计算层会返回明确的空值诊断。
    """
    if features is not None:
        selected = list(dict.fromkeys(features + [c for c in [*targets, group_col, time_col, weights_col, amount_col, segment_col] if c in df.columns]))
        df = df.loc[:, selected] if isinstance(df, pd.DataFrame) else df.select(selected)
        if benchmark_df is not None:
            columns = [c for c in selected if c in benchmark_df.columns]
            benchmark_df = benchmark_df.loc[:, columns] if isinstance(benchmark_df, pd.DataFrame) else benchmark_df.select(columns)
    if segment_col is not None:
        original = to_polars_frame(df)
        scopes = build_scopes(original, segment_col)
        return pl.concat([
            _prepare_raw_ks(
                scope_filter(original, scopes, identity), targets=targets, features=features,
                group_col=group_col, time_col=time_col, time_grain=time_grain,
                internal_group_col=internal_group_col, weights_col=weights_col,
                amount_col=amount_col, benchmark_df=benchmark_df, max_features=max_features,
                missing_values=missing_values, special_values=special_values,
            ).with_columns(pl.lit(identity).alias("scope_id"))
            for identity in scopes["scope_id"].to_list()
        ], how="vertical_relaxed")
    frame, internal_group_col = prepare_group_context(
        to_polars_frame(df),
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        mars_group_col=internal_group_col,
    )
    fit_schema = (
        to_polars_frame(benchmark_df).schema
        if benchmark_df is not None
        else frame.schema
    )
    features_by_target: dict[str, list[str]] = {}
    for target in targets or ["dummy_target"]:
        candidates = _resolve_evaluation_features(
            frame,
            target=target,
            group_col=internal_group_col,
            features=features,
            weights_col=weights_col,
            amount_col=amount_col,
        )
        features_by_target[target] = [
            f for f in candidates if is_numeric_dtype(fit_schema.get(f))
        ]
    numeric_features = set(f for fs in features_by_target.values() for f in fs)
    if len(numeric_features) > max_features:
        raise ValueError(
            f"Raw KS selected {len(numeric_features)} numeric features, exceeding "
            f"max_raw_ks_features={max_features}. Reduce features or increase the limit."
        )
    for feature in numeric_features:
        if not is_numeric_dtype(frame.schema.get(feature)):
            raise ValueError(
                f"Raw KS requires numeric evaluation data for feature={feature!r}."
            )
    for target in targets:
        if target in frame.columns:
            frame = normalize_binary_target_column(frame, target)
    if not targets:
        # 用户数据恰好含 dummy_target 时也不能将其误认作显式标签。
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int8).alias("dummy_target"))
    return _calculate_raw_ks(
        frame,
        features_by_target=features_by_target,
        group_col=internal_group_col,
        weights_col=weights_col,
        missing_values=missing_values,
        special_values=special_values,
    )


def _replace_ks_columns(
    frame: pl.DataFrame,
    replacements: pl.DataFrame,
    *,
    keys: list[str],
    columns: list[str],
) -> pl.DataFrame:
    """按匹配键替换 KS，原始值 KS 无法计算时保留空值，不回退到分箱 KS。"""
    renamed = {col: f"__ks_{index}" for index, col in enumerate(columns)}
    joined = _left_join_nulls(
        frame,
        replacements.select(keys + columns)
        .rename(renamed)
        .with_columns(pl.lit(True).alias("__raw_ks")),
        on=keys,
    )
    return joined.with_columns(
        [
            pl.when(pl.col("__raw_ks").fill_null(False))
            .then(pl.col(renamed[col]))
            .otherwise(pl.col(col))
            .alias(col)
            for col in columns
        ]
    ).select(frame.columns)


def _apply_raw_ks(
    report: MarsBinningReport,
    values: pl.DataFrame,
    *,
    primary_target: str,
) -> MarsBinningReport:
    """将最终 KS 一次性写入各报告粒度，并保留每项原始值计算的诊断。"""
    summary = to_polars_frame(report.summary_table)
    detail = to_polars_frame(report.detail_table)
    group_col = report.detail_group_col
    assert group_col is not None
    scope_keys = ["scope_id"] if "scope_id" in detail.columns else []
    keys = scope_keys + ["feature", "y", group_col]
    detail = detail.with_columns(pl.col("ks_bin").max().over(keys).alias("ks"))
    detail = _replace_ks_columns(
        detail,
        values.rename({"group": group_col}),
        keys=keys,
        columns=["ks"],
    )
    totals = values.filter(pl.col("group") == "Total")
    summary_keys = scope_keys + ["feature"]
    if "target" in summary.columns:
        totals = totals.rename({"y": "target"})
        summary_keys.append("target")
    else:
        totals = totals.filter(pl.col("y") == primary_target)
    summary = _replace_ks_columns(summary, totals, keys=summary_keys, columns=["ks"])

    trends = dict(report.trend_tables)
    primary_values = values.filter(pl.col("y") == primary_target)
    if "ks" in trends and not primary_values.is_empty():
        trend = to_polars_frame(trends["ks"])
        # 只映射宽表展示列名，真实空组键留在原明细中，并与业务文字 null 区分。
        null_label = _null_group_label(detail[group_col].unique().to_list())
        raw_trend = primary_values.with_columns(
            pl.col("group").fill_null(null_label)
        ).pivot(index=scope_keys + ["feature"], on="group", values="ks")
        columns = [col for col in trend.columns if not is_metadata_column(col)]
        raw_trend = raw_trend.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(col)
                for col in columns
                if col not in raw_trend.columns
            ]
        )
        updated = _replace_ks_columns(
            trend, raw_trend, keys=scope_keys + ["feature"], columns=[c for c in columns if c != "scope_id"]
        )
        trends["ks"] = (
            updated.to_pandas() if isinstance(trends["ks"], pd.DataFrame) else updated
        )

    raw_features = set(values["feature"].to_list())
    meta = dict(report.report_meta)
    meta["ks_method"] = "raw"
    meta["ks_source_by_feature"] = {
        f: "raw" if f in raw_features else "binned"
        for f in summary["feature"].unique().to_list()
    }
    meta["raw_ks_diagnostics"] = values.to_dicts()
    updated_report: MarsBinningReport = MarsBinningReport(
        summary_table=(
            summary.to_pandas()
            if isinstance(report.summary_table, pd.DataFrame)
            else summary
        ),
        detail_table=(
            detail.to_pandas()
            if isinstance(report.detail_table, pd.DataFrame)
            else detail
        ),
        trend_tables=trends,
        group_col=report.group_col,
        detail_group_col=group_col,
        feature_data_source=report.feature_data_source,
        dt_col=report.dt_col,
        missing_by_day_table=report.missing_by_day_table,
        risk_corr_reference_table=report.risk_corr_reference_table,
        report_meta=meta,
        feature_metadata=report.feature_metadata,
        business_context=report.business_context,
    )
    updated_report.report_id = report.report_id
    updated_report.source = dict(report.source)
    updated_report._calculation_status = report._calculation_status
    if hasattr(report, "_scopes"):
        updated_report._scopes = report._scopes
    return updated_report
