"""分箱评估输入上下文构造。"""

from __future__ import annotations

import inspect
from datetime import date
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


def count_observed_target_classes(df: pl.DataFrame, target: str) -> int:
    """统计二分类 target 的有效类别数。"""
    return int(
        df
        .filter(pl.col(target).is_not_null())
        .select(pl.col(target).n_unique())
        .item()
    )


def binning_requires_target(
    *,
    binning_type: str,
    binner_params: dict[str, Any],
) -> bool:
    """判断自动构建的分箱器是否需要监督标签。"""
    if binning_type in {"optimal", "lite_opt"}:
        return True
    return binning_type == "native" and binner_params.get("method") == "cart"


def prepare_benchmark_frame(
    benchmark_df: pl.DataFrame,
    *,
    features: list[str],
    weights_col: str | None,
    target: str | None,
    require_binary_target: bool,
) -> pl.DataFrame:
    """校验 benchmark schema，并按需归一化监督标签。"""
    if benchmark_df.is_empty():
        raise ValueError("`benchmark_df` must contain at least one row.")

    missing_features = sorted(set(features) - set(benchmark_df.columns))
    if missing_features:
        raise ValueError(
            "`benchmark_df` is missing active feature columns: "
            f"{missing_features}. All evaluated features must be fitted on the same benchmark."
        )

    if weights_col and weights_col not in benchmark_df.columns:
        raise ValueError(
            f"`benchmark_df` is missing weights_col={weights_col!r}; "
            "benchmark and evaluation distributions must use the same weighting scope."
        )

    if not require_binary_target:
        return benchmark_df
    if target is None or target not in benchmark_df.columns:
        raise ValueError(
            "`benchmark_df` must contain the requested target column when it is used "
            "for supervised binning or benchmark risk correlation."
        )

    normalized_df = normalize_binary_target_column(benchmark_df, target)
    observed_classes = count_observed_target_classes(normalized_df, target)
    if observed_classes < 2:
        raise ValueError(
            f"Target column {target!r} in `benchmark_df` must have at least 2 observed "
            "classes after excluding null / NaN values."
        )
    return normalized_df


def resolve_date_bounds(
    df: pl.DataFrame,
    time_col: str | None,
) -> tuple[str | None, str | None]:
    """解析有效日期并返回精确到日的最小值和最大值。"""
    if not time_col or time_col not in df.columns:
        return None, None

    parsed_col = "__mars_report_date"
    try:
        bounds = (
            df
            .select(MarsDate.smart_parse_expr(time_col).alias(parsed_col))
            .select(
                pl.col(parsed_col).min().alias("start_dt"),
                pl.col(parsed_col).max().alias("end_dt"),
            )
            .row(0, named=True)
        )
    except (pl.exceptions.PolarsError, ValueError, TypeError):
        return None, None

    start_dt = bounds["start_dt"]
    end_dt = bounds["end_dt"]
    return (
        start_dt.isoformat() if isinstance(start_dt, date) else None,
        end_dt.isoformat() if isinstance(end_dt, date) else None,
    )


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
    fit_has_target: bool,
    fit_df: pl.DataFrame,
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

    if not fit_has_target:
        if binner_cls in {MarsOptimalBinner, MarsLiteOptBinner}:
            logger.warning("No target provided. Falling back to native quantile binning.")
            binner_cls = MarsNativeBinner
            clean_kwargs["method"] = "quantile"
        elif clean_kwargs.get("method") == "cart":
            logger.warning("No target provided. Forcing native method='quantile'.")
            clean_kwargs["method"] = "quantile"

    binner = binner_cls(**clean_kwargs)
    effective_fit_df = fit_df
    y_series = None
    if fit_has_target:
        is_supervised_binner = (
            binner_cls is MarsOptimalBinner
            or binner_cls is MarsLiteOptBinner
            or clean_kwargs.get("method") == "cart"
        )
        if is_supervised_binner:
            effective_fit_df = fit_df.filter(pl.col(target).is_not_null())
            y_series = effective_fit_df.get_column(target)
    binner.fit(effective_fit_df, y_series, features=features)
    return binner
