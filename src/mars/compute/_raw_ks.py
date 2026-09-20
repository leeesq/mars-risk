"""原始数值的加权双样本 KS；仅处理已归一化标签，不依赖分箱或报告层。"""

from __future__ import annotations

from typing import Any

import polars as pl

from mars._compat import _left_join_nulls
from mars.compute.missing import missing_condition_expr, values_to_exclude


def _aggregate_raw_ks(frame: pl.DataFrame, *, grouped: bool) -> pl.DataFrame:
    """在已排序的有效样本上合并并列值，再计算各组经验分布的最大绝对差。"""
    if not grouped:
        frame = frame.with_columns(pl.lit("Total").alias("group"))
    class_max = pl.col("weight").max().over("group", "label")
    frame = frame.with_columns(
        pl.when(class_max > 0)
        .then(pl.col("weight") / class_max)
        .otherwise(0)
        .alias("weight")
    )
    points = frame.group_by("group", "value", maintain_order=True).agg(
        pl.when(pl.col("label") == 0)
        .then(pl.col("weight"))
        .otherwise(0)
        .sum()
        .alias("good"),
        pl.when(pl.col("label") == 1)
        .then(pl.col("weight"))
        .otherwise(0)
        .sum()
        .alias("bad"),
        pl.len().alias("valid_count"),
    )
    points = points.with_columns(
        pl.col("good").sum().over("group").alias("total_good"),
        pl.col("bad").sum().over("group").alias("total_bad"),
    ).with_columns(
        (
            pl.col("good").cum_sum().over("group") / pl.col("total_good")
            - pl.col("bad").cum_sum().over("group") / pl.col("total_bad")
        )
        .abs()
        .mul(100)
        .alias("ks"),
    )
    return points.group_by("group", maintain_order=True).agg(
        pl.when((pl.col("total_good").first() > 0) & (pl.col("total_bad").first() > 0))
        .then(pl.col("ks").max())
        .otherwise(None)
        .alias("ks"),
        pl.col("valid_count").sum(),
    )


def _calculate_raw_ks(
    df: pl.DataFrame,
    *,
    features_by_target: dict[str, list[str]],
    group_col: str,
    weights_col: str | None,
    missing_values: list[Any] | None,
    special_values: list[Any] | None,
) -> pl.DataFrame:
    """逐特征排序并复用于各标签，返回 Total 与分组 KS 及有效样本诊断。

    标签列必须已归一化为 0/1/null；不存在的标签视为全未观测。原始特征保留
    整数精度，缺失与特殊值共用底座规则。权重按各组各类的最大值缩放，防止累计溢出；
    缩放不改变经验分布。内存只随单个特征的样本量增长，不展开整张宽表。
    """
    schema = {
        "feature": pl.String,
        "y": pl.String,
        "group": pl.String,
        "ks": pl.Float64,
        "valid_count": pl.UInt32,
        "reason": pl.String,
    }
    features = list(dict.fromkeys(f for fs in features_by_target.values() for f in fs))
    groups = df.select(pl.col(group_col).alias("group")).unique(maintain_order=True)
    groups = pl.concat([groups, pl.DataFrame({"group": ["Total"]})]).unique(
        maintain_order=True
    )
    outputs: list[pl.DataFrame] = []
    for feature in features:
        dtype = df.schema[feature]
        excluded = values_to_exclude(
            dtype, missing_values=missing_values, special_values=special_values
        )
        valid = ~missing_condition_expr(feature, dtype=dtype, missing_values=excluded)
        # 对整数也保持原 dtype 排序，避免超过 2**53 的不同值被浮点转换合并。
        active_targets = [
            t
            for t, fs in features_by_target.items()
            if feature in fs and t in df.columns
        ]
        selected = list(
            dict.fromkeys(
                [feature, group_col]
                + active_targets
                + ([weights_col] if weights_col else [])
            )
        )
        ordered = (
            df.select(selected)
            .filter(valid & pl.col(feature).is_finite())
            .sort(feature)
        )
        for target, active_features in features_by_target.items():
            if feature not in active_features:
                continue
            label = (
                pl.col(target) if target in df.columns else pl.lit(None, dtype=pl.Int8)
            )
            weight = (
                pl.col(weights_col).cast(pl.Float64, strict=False)
                if weights_col
                else pl.lit(1.0)
            )
            frame = ordered.select(
                pl.col(feature).alias("value"),
                label.alias("label"),
                pl.col(group_col).alias("group"),
                weight.alias("weight"),
            ).filter(pl.col("label").is_not_null())
            invalid_weight = (
                pl.col("weight").is_null()
                | ~pl.col("weight").is_finite()
                | (pl.col("weight") < 0)
            )
            if frame.select(invalid_weight.any()).item():
                raise ValueError(
                    f"Raw KS requires finite non-negative weights: weights_col={weights_col!r}, "
                    f"feature={feature!r}, target={target!r}."
                )
            total = _aggregate_raw_ks(frame, grouped=False)
            grouped = _aggregate_raw_ks(frame, grouped=True).filter(
                (pl.col("group") != "Total") | pl.col("group").is_null()
            )
            values = pl.concat([total, grouped])
            result = (
                _left_join_nulls(groups, values, on=["group"])
                .with_columns(
                    pl.col("valid_count").fill_null(0),
                    pl.lit(feature).alias("feature"),
                    pl.lit(target).alias("y"),
                )
                .with_columns(
                    pl.when(pl.col("valid_count") == 0)
                    .then(pl.lit("no_valid_samples"))
                    .when(pl.col("ks").is_null())
                    .then(pl.lit("insufficient_class_weight"))
                    .otherwise(None)
                    .alias("reason"),
                )
            )
            outputs.append(result.select(list(schema)))
    return pl.concat(outputs) if outputs else pl.DataFrame(schema=schema)
