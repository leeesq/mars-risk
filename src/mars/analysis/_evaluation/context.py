"""分箱评估输入上下文构造。"""

from __future__ import annotations

import inspect
from typing import Any

import polars as pl

from mars.feature.binning.base import MarsBinnerBase
from mars.feature.binning.lite_opt import MarsLiteOptBinner
from mars.feature.binning.native import MarsNativeBinner
from mars.feature.binning.optimal import MarsOptimalBinner
from mars.utils.date import MarsDate
from mars.utils.logger import logger


def normalize_binary_target_column(df: pl.DataFrame, target: str) -> pl.DataFrame:
    """校验并归一化二分类 target 列。"""
    dtype = df.schema[target]
    if dtype == pl.Boolean:
        return df.with_columns(pl.col(target).cast(pl.Int8).alias(target))

    if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
        invalid_values = (
            df
            .filter(pl.col(target).is_not_null() & ~pl.col(target).is_in([0, 1]))
            .select(pl.col(target).unique().head(5))
            .to_series()
            .to_list()
        )
        if invalid_values:
            raise ValueError(
                f"Target column '{target}' contains invalid values {invalid_values}. "
                "Please clean it to 0/1/True/False/null before evaluation.",
            )
        return df.with_columns(pl.col(target).cast(pl.Int8).alias(target))

    if dtype in {pl.Float32, pl.Float64}:
        valid_expr = (
            pl.col(target).is_null()
            | pl.col(target).is_nan()
            | pl.col(target).is_in([0.0, 1.0])
        )
        invalid_values = (
            df
            .filter(~valid_expr)
            .select(pl.col(target).unique().head(5))
            .to_series()
            .to_list()
        )
        if invalid_values:
            raise ValueError(
                f"Target column '{target}' contains invalid values {invalid_values}. "
                "Please clean it to 0/1/True/False/null before evaluation.",
            )
        return df.with_columns(pl.col(target).fill_nan(None).cast(pl.Int8).alias(target))

    if dtype == pl.String:
        valid_strings = ["0", "1", "true", "false", "True", "False", ""]
        invalid_values = (
            df
            .filter(pl.col(target).is_not_null() & ~pl.col(target).is_in(valid_strings))
            .select(pl.col(target).unique().head(5))
            .to_series()
            .to_list()
        )
        if invalid_values:
            raise ValueError(
                f"Target column '{target}' contains invalid values {invalid_values}. "
                "Please clean it to 0/1/True/False/null before evaluation.",
            )
        normalized = (
            pl.when(pl.col(target).is_null() | (pl.col(target) == ""))
            .then(None)
            .when(pl.col(target).str.to_lowercase() == "true")
            .then(1)
            .when(pl.col(target).str.to_lowercase() == "false")
            .then(0)
            .otherwise(pl.col(target).cast(pl.Int8))
            .alias(target)
        )
        return df.with_columns(normalized)

    invalid_values = (
        df
        .filter(pl.col(target).is_not_null())
        .select(pl.col(target).unique().head(5))
        .to_series()
        .to_list()
    )
    if invalid_values:
        raise ValueError(
            f"Target column '{target}' contains invalid values {invalid_values}. "
            "Please clean it to 0/1/True/False/null before evaluation.",
        )
    return df.with_columns(pl.lit(None).cast(pl.Int8).alias(target))


def resolve_profile_by(
    *,
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
) -> str | None:
    """把公开分组参数解析为内部趋势维度。"""
    if group_col:
        return group_col
    if time_col:
        return time_grain or "month"
    return None


def prepare_group_context(
    df: pl.DataFrame,
    *,
    profile_by: str | None,
    dt_col: str | None,
    mars_group_col: str,
) -> tuple[pl.DataFrame, str]:
    """构造内部趋势分组上下文。"""
    if dt_col and not profile_by:
        profile_by = "month"

    if dt_col and profile_by is not None and MarsDate.is_time_grain(profile_by):
        date_expr = MarsDate.from_grain(dt_col, profile_by).alias(mars_group_col)
        return df.with_columns(date_expr), mars_group_col

    if profile_by:
        if profile_by in df.columns:
            return df.with_columns(pl.col(profile_by).cast(pl.String).alias(mars_group_col)), mars_group_col
        logger.warning("Column '%s' was not found. Falling back to snapshot mode.", profile_by)

    return df.with_columns(pl.lit("Total").alias(mars_group_col)), mars_group_col


def normalize_feature_data_source(
    feature_data_source: dict[str, list[str]] | None,
    features: list[str],
) -> dict[str, str]:
    """将数据源到特征列表的映射标准化为特征到数据源的字典。"""
    feature_set = set(features)
    if not feature_data_source:
        return {feature: "UNMAPPED" for feature in features}

    normalized: dict[str, str] = {}
    mapped_features: set[str] = set()
    for data_source, source_features in feature_data_source.items():
        for feature in source_features or []:
            if feature not in feature_set:
                raise ValueError(
                    "feature_data_source contains features outside the active evaluation feature set: "
                    f"{feature}",
                )
            normalized[feature] = str(data_source)
            mapped_features.add(feature)

    for feature in feature_set - mapped_features:
        normalized[feature] = "UNMAPPED"
    return normalized


def build_binner(
    *,
    binning_type: str,
    binner_params: dict[str, Any],
    has_target: bool,
    working_df: pl.DataFrame,
    target: str,
    features: list[str],
) -> MarsBinnerBase:
    """按 evaluator 默认策略构造并拟合分箱器。"""
    binner_factory: dict[str, type[MarsBinnerBase]] = {
        "native": MarsNativeBinner,
        "optimal": MarsOptimalBinner,
        "lite_opt": MarsLiteOptBinner,
    }
    binner_cls = binner_factory[binning_type]

    sig = inspect.signature(binner_cls)
    valid_keys = set(sig.parameters)
    valid_keys.difference_update({"self", "features", "cat_features"})
    clean_kwargs = {key: value for key, value in binner_params.items() if key in valid_keys}
    ignored_keys = set(binner_params) - set(clean_kwargs)
    if ignored_keys:
        logger.debug("Auto-cleaned kwargs for %s. Ignored: %s", binner_cls.__name__, ignored_keys)

    if not has_target:
        if binner_cls in {MarsOptimalBinner, MarsLiteOptBinner}:
            logger.warning("No target provided. Falling back to native quantile binning.")
            binner_cls = MarsNativeBinner
            clean_kwargs["method"] = "quantile"
        elif clean_kwargs.get("method") == "cart":
            logger.warning("No target provided. Forcing native method='quantile'.")
            clean_kwargs["method"] = "quantile"

    binner = binner_cls(**clean_kwargs)
    fit_df = working_df
    y_series = None
    if has_target:
        is_supervised_binner = (
            binner_cls is MarsOptimalBinner
            or binner_cls is MarsLiteOptBinner
            or clean_kwargs.get("method") == "cart"
        )
        if is_supervised_binner:
            fit_df = working_df.filter(pl.col(target).is_not_null())
            y_series = fit_df.get_column(target)
    binner.fit(fit_df, y_series, features=features)
    return binner
