"""分箱评估 RC 参考表构造。"""

from __future__ import annotations

from typing import Any

import polars as pl

from mars.analysis._evaluation.aggregation import aggregate_basic_stats
from mars.compute import RiskCorrBaseline, bad_rate_expr, risk_corr_expr


def empty_risk_corr_reference_table(target_name: str | None) -> pl.DataFrame:
    """构造空的 RC 参考表。

    Parameters
    ----------
    target_name : str | None
        当前目标列名称；仅用于保持调用处语义一致。

    Returns
    -------
    pl.DataFrame
        固定 schema 的空 RC 参考表。
    """
    _ = target_name
    return pl.DataFrame(
        schema={
            "y": pl.String,
            "feature": pl.String,
            "bin_index": pl.Int16,
            "base_br": pl.Float64,
            "source": pl.String,
        }
    )


def attach_risk_corr_reference_context(
    reference_df: pl.DataFrame,
    *,
    target_name: str | None,
    source: str,
) -> pl.DataFrame:
    """补齐 RC 参考表的统一上下文列。

    Parameters
    ----------
    reference_df : pl.DataFrame
        至少包含 `feature`、`bin_index` 和 `base_br` 的参考坏率表。
    target_name : str | None
        当前目标列名称。
    source : str
        参考来源标识。

    Returns
    -------
    pl.DataFrame
        schema 统一后的 RC 参考表。
    """
    if reference_df.is_empty():
        return empty_risk_corr_reference_table(target_name)

    return (
        reference_df
        .filter(pl.col("bin_index") >= 0)
        .select(
            [
                pl.lit(str(target_name or "dummy_target")).alias("y"),
                pl.col("feature").cast(pl.String).alias("feature"),
                pl.col("bin_index").cast(pl.Int16).alias("bin_index"),
                pl.col("base_br").cast(pl.Float64).alias("base_br"),
                pl.lit(source).alias("source"),
            ]
        )
    )


def build_benchmark_risk_corr_reference(
    benchmark_binned: pl.DataFrame,
    *,
    has_target: bool,
    features: list[str],
    weights_col: str | None,
    target_name: str | None,
    mars_group_col: str,
) -> pl.DataFrame:
    """基于 benchmark 样本构造 RC 参考表。

    Parameters
    ----------
    benchmark_binned : pl.DataFrame
        已按当前分箱规则转换的 benchmark 样本。
    has_target : bool
        当前评估是否存在目标列。
    features : list[str]
        需要构造参考的特征列表。
    weights_col : str | None
        样本权重列。
    target_name : str | None
        当前目标列名称。
    mars_group_col : str
        内部分组列名称。

    Returns
    -------
    pl.DataFrame
        schema 统一后的 benchmark RC 参考表。
    """
    if not has_target or target_name is None:
        return empty_risk_corr_reference_table(target_name)

    benchmark_binned = benchmark_binned.with_columns(pl.lit("Benchmark").alias(mars_group_col))
    benchmark_stats: pl.DataFrame = aggregate_basic_stats(
        benchmark_binned,
        group_col=mars_group_col,
        features=features,
        target_col=target_name,
        weights_col=weights_col,
    )
    reference_df: pl.DataFrame = (
        benchmark_stats
        .group_by(["feature", "bin_index"])
        .agg(
            [
                pl.col("count").sum().alias("count"),
                pl.col("observed_count").sum().alias("observed_count"),
                pl.col("bad").sum().alias("bad"),
            ]
        )
        .with_columns(bad_rate_expr(output_col="base_br"))
    )
    return attach_risk_corr_reference_context(
        reference_df,
        target_name=target_name,
        source="benchmark_df",
    )


def build_risk_corr_reference_table(
    *,
    target_name: str | None,
    metrics_groups: pl.DataFrame,
    metrics_total: pl.DataFrame,
    group_col: str,
    risk_corr_baseline: RiskCorrBaseline,
    benchmark_binned: pl.DataFrame | None,
    benchmark_features: list[str],
    benchmark_weights_col: str | None,
    feature_start_reference: dict[str, Any] | None,
    has_target: bool,
    mars_group_col: str,
) -> tuple[pl.DataFrame, str]:
    """按统一策略选择 RC 参考表。

    Parameters
    ----------
    target_name : str | None
        当前目标列名称。
    metrics_groups : pl.DataFrame
        分组维度指标长表。
    metrics_total : pl.DataFrame
        Total 维度指标长表。
    group_col : str
        分组列名称。
    risk_corr_baseline : RiskCorrBaseline
        RC 参考策略。
    benchmark_binned : pl.DataFrame | None
        已按当前分箱规则转换的 benchmark 样本。
    benchmark_features : list[str]
        benchmark 需要使用的特征。
    benchmark_weights_col : str | None
        benchmark 权重列。
    feature_start_reference : dict[str, Any] | None
        feature-start 参考上下文。
    has_target : bool
        当前评估是否存在目标列。
    mars_group_col : str
        evaluator 内部分组列名称。

    Returns
    -------
    tuple[pl.DataFrame, str]
        RC 参考表和参考来源。

    Raises
    ------
    ValueError
        当 benchmark 策略没有可用参考来源时抛出。
    """
    if not has_target:
        return empty_risk_corr_reference_table(target_name), "total"

    if risk_corr_baseline == "total":
        reference_df: pl.DataFrame = metrics_total.select(
            [
                "feature",
                "bin_index",
                pl.col("bad_rate").alias("base_br"),
            ]
        )
        return (
            attach_risk_corr_reference_context(
                reference_df,
                target_name=target_name,
                source="total",
            ),
            "total",
        )

    if risk_corr_baseline == "first_group":
        first_group: Any = metrics_groups.select(pl.col(group_col).min()).item()
        reference_df = (
            metrics_groups
            .filter(pl.col(group_col) == first_group)
            .select(
                [
                    "feature",
                    "bin_index",
                    pl.col("bad_rate").alias("base_br"),
                ]
            )
        )
        return (
            attach_risk_corr_reference_context(
                reference_df,
                target_name=target_name,
                source="first_group",
            ),
            "first_group",
        )

    if benchmark_binned is not None:
        reference_df = build_benchmark_risk_corr_reference(
            benchmark_binned,
            has_target=has_target,
            features=benchmark_features,
            weights_col=benchmark_weights_col,
            target_name=target_name,
            mars_group_col=mars_group_col,
        )
        return reference_df, "benchmark_df"

    if feature_start_reference is not None:
        baseline_df = feature_start_reference.get("baseline_bad_rate")
        if isinstance(baseline_df, pl.DataFrame) and not baseline_df.is_empty():
            return (
                attach_risk_corr_reference_context(
                    baseline_df,
                    target_name=target_name,
                    source="feature_start_reference",
                ),
                "feature_start_reference",
            )

    raise ValueError(
        "`risk_corr_baseline='benchmark'` requires `benchmark_df` or "
        "`feature_start_aware_reference=True` with a valid feature-start reference.",
    )


def build_risk_corr_long(
    metrics_df: pl.DataFrame,
    baseline_df: pl.DataFrame,
    *,
    group_col: str,
) -> pl.DataFrame:
    """基于参考坏率表计算分组级 RC 长表。

    Parameters
    ----------
    metrics_df : pl.DataFrame
        包含 `feature`、分组列、`bin_index` 和 `bad_rate` 的指标表。
    baseline_df : pl.DataFrame
        包含 `feature`、`bin_index` 和 `base_br` 的参考坏率表。
    group_col : str
        分组列名称。

    Returns
    -------
    pl.DataFrame
        包含 `feature`、分组列和 `risk_corr` 的长表。
    """
    return (
        metrics_df
        .filter(pl.col("bin_index") >= 0)
        .join(baseline_df, on=["feature", "bin_index"], how="left")
        .group_by(["feature", group_col])
        .agg(risk_corr_expr())
    )
