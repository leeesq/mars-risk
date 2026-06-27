"""高层风险画像工作流入口。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import polars as pl

from mars.compute import RiskCorrBaseline, to_polars_frame
from mars.feature.binning.base import MarsBinnerBase
from mars.reporting import MarsBinningReport
from mars.utils.logger import logger

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsRiskProfile


_ProfileRiskBinningType = Literal["native", "optimal", "lite_opt"]
_ProfileRiskMonotonicTrend = Literal[
    "ascending",
    "descending",
    "peak",
    "valley",
    "auto",
    "auto_asc_desc",
]

_FORBIDDEN_ADVANCED_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "method",
        "prebinning_method",
        "n_bins",
        "min_bin_size",
        "monotonic_trend",
        "missing_values",
        "special_values",
        "n_jobs",
    }
)

_ALLOWED_ADVANCED_PARAM_KEYS: dict[_ProfileRiskBinningType, tuple[str, ...]] = {
    "native": (
        "merge_small_bins",
        "cart_params",
        "remove_empty_bins",
    ),
    "optimal": (
        "min_n_bins",
        "min_bin_n_event",
        "n_prebins",
        "min_prebin_size",
        "solver",
        "time_limit",
        "max_cats_to_solver",
        "min_cat_fraction",
        "cart_params",
        "join_threshold",
    ),
    "lite_opt": (
        "n_prebins",
        "join_threshold",
    ),
}

_ALLOWED_MONOTONIC_TREND_VALUES: dict[_ProfileRiskBinningType, tuple[str, ...]] = {
    "native": (),
    "optimal": (
        "ascending",
        "descending",
        "auto",
        "auto_asc_desc",
    ),
    "lite_opt": (
        "ascending",
        "descending",
        "peak",
        "valley",
        "auto",
        "auto_asc_desc",
    ),
}

_PUBLIC_BINNER_ARG_NAMES: tuple[str, ...] = (
    "method",
    "n_bins",
    "min_bin_size",
    "monotonic_trend",
    "missing_values",
    "special_values",
    "n_jobs",
)

_PROFILE_RISK_DEFAULT_MONOTONIC_TREND = "auto_asc_desc"


def _normalize_profile_risk_binning_type(binning_type: str) -> _ProfileRiskBinningType:
    """校验并规范化 `profile_risk` 使用的 `binning_type`。"""
    normalized = binning_type.strip().lower()
    valid_binning_types = tuple(_ALLOWED_ADVANCED_PARAM_KEYS)
    if normalized not in valid_binning_types:
        valid_text = ", ".join(f"`{item}`" for item in valid_binning_types)
        raise ValueError(
            f"`binning_type` must be one of {valid_text}. Got {binning_type!r}."
        )
    return normalized


def _resolve_method_param_key(binning_type: _ProfileRiskBinningType) -> str:
    """根据 `binning_type` 返回底层分箱方法参数名。"""
    if binning_type == "native":
        return "method"
    return "prebinning_method"


def _format_allowed_advanced_param_text(binning_type: _ProfileRiskBinningType) -> str:
    """生成当前 `binning_type` 可接受的高级分箱参数文本。"""
    allowed_keys = _ALLOWED_ADVANCED_PARAM_KEYS[binning_type]
    return ", ".join(f"`{key}`" for key in allowed_keys)


def _normalize_advanced_binning_params(
    advanced_binning_params: dict[str, dict[str, Any]] | None,
) -> dict[_ProfileRiskBinningType, dict[str, Any]]:
    """标准化 `advanced_binning_params` 的顶层结构。"""
    if advanced_binning_params is None:
        return {}

    valid_bucket_names = set(_ALLOWED_ADVANCED_PARAM_KEYS)
    invalid_bucket_names = sorted(
        bucket_name
        for bucket_name in advanced_binning_params
        if bucket_name not in valid_bucket_names
    )
    if invalid_bucket_names:
        invalid_text = ", ".join(f"`{bucket_name}`" for bucket_name in invalid_bucket_names)
        valid_text = ", ".join(f"`{bucket_name}`" for bucket_name in _ALLOWED_ADVANCED_PARAM_KEYS)
        raise ValueError(
            "`advanced_binning_params` only supports these buckets: "
            f"{valid_text}. Received: {invalid_text}."
        )

    normalized: dict[_ProfileRiskBinningType, dict[str, Any]] = {}
    for bucket_name, bucket_params in advanced_binning_params.items():
        if not isinstance(bucket_params, dict):
            raise ValueError(
                f"`advanced_binning_params[{bucket_name!r}]` must be a dict, "
                f"got {type(bucket_params).__name__}."
            )
        normalized[cast(_ProfileRiskBinningType, bucket_name)] = dict(bucket_params)

    return normalized


def _resolve_monotonic_trend(
    *,
    binning_type: _ProfileRiskBinningType,
    monotonic_trend: _ProfileRiskMonotonicTrend | None,
) -> str | None:
    """解析 `monotonic_trend` 的生效值。"""
    if monotonic_trend is None:
        if binning_type in {"optimal", "lite_opt"}:
            return _PROFILE_RISK_DEFAULT_MONOTONIC_TREND
        return None

    if binning_type == "native":
        warnings.warn(
            "`monotonic_trend` is ignored when `binning_type='native'`.",
            UserWarning,
            stacklevel=3,
        )
        return None

    allowed_values = _ALLOWED_MONOTONIC_TREND_VALUES[binning_type]
    if monotonic_trend not in allowed_values:
        allowed_text = ", ".join(f"`{value}`" for value in allowed_values)
        raise ValueError(
            f"`monotonic_trend={monotonic_trend!r}` is not supported for "
            f"`binning_type={binning_type!r}`. Allowed values are: {allowed_text}."
        )

    return monotonic_trend


def _build_effective_binner_params(
    *,
    binning_type: _ProfileRiskBinningType,
    advanced_binning_params: dict[str, dict[str, Any]] | None,
    method: Literal["quantile", "uniform", "cart"] | None,
    n_bins: int | None,
    min_bin_size: float | int | None,
    monotonic_trend: _ProfileRiskMonotonicTrend | None,
    missing_values: list[Any] | None,
    special_values: list[Any] | None,
    n_jobs: int | None,
) -> dict[str, Any]:
    """合并 `profile_risk` 公开分箱参数并校验高级参数契约。"""
    normalized_advanced_params = _normalize_advanced_binning_params(advanced_binning_params)
    effective_binner_params: dict[str, Any] = dict(
        normalized_advanced_params.get(binning_type, {})
    )
    allowed_keys = set(_ALLOWED_ADVANCED_PARAM_KEYS[binning_type])
    forbidden_keys = sorted(
        key for key in effective_binner_params if key in _FORBIDDEN_ADVANCED_PARAM_KEYS
    )
    unknown_keys = sorted(
        key
        for key in effective_binner_params
        if key not in allowed_keys and key not in _FORBIDDEN_ADVANCED_PARAM_KEYS
    )

    # 这里明确区分“应该走公开参数”的键和“当前 binner 根本不支持”的键，
    # 统一报错并提示当前 `binning_type` 还能传哪些高级参数。
    if forbidden_keys or unknown_keys:
        invalid_keys = forbidden_keys + unknown_keys
        invalid_text = ", ".join(f"`{key}`" for key in invalid_keys)
        allowed_text = _format_allowed_advanced_param_text(binning_type)
        public_text = ", ".join(f"`{name}`" for name in _PUBLIC_BINNER_ARG_NAMES)
        raise ValueError(
            f"`advanced_binning_params[{binning_type!r}]` received unsupported keys: "
            f"{invalid_text}. Allowed keys are: {allowed_text}. Public binning "
            f"arguments must be passed explicitly via {public_text}. The internal "
            "`prebinning_method` alias is also not allowed here."
        )

    if method is not None:
        effective_binner_params[_resolve_method_param_key(binning_type)] = method
    if n_bins is not None:
        effective_binner_params["n_bins"] = n_bins
    if min_bin_size is not None:
        effective_binner_params["min_bin_size"] = min_bin_size
    resolved_monotonic_trend = _resolve_monotonic_trend(
        binning_type=binning_type,
        monotonic_trend=monotonic_trend,
    )
    if resolved_monotonic_trend is not None:
        effective_binner_params["monotonic_trend"] = resolved_monotonic_trend
    if missing_values is not None:
        effective_binner_params["missing_values"] = missing_values
    if special_values is not None:
        effective_binner_params["special_values"] = special_values
    if n_jobs is not None:
        effective_binner_params["n_jobs"] = n_jobs
    return effective_binner_params


def profile_risk(
    df: pl.DataFrame | pd.DataFrame,
    *,
    target: str | list[str] | None = None,
    features: list[str] | None = None,
    feature_data_source: dict[str, list[str]] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    weights_col: str | None = None,
    amount_col: str | None = None,
    feature_start_aware_reference: bool = False,
    risk_corr_baseline: RiskCorrBaseline = "total",
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    binning_type: Literal["native", "optimal", "lite_opt"] = "native",
    method: Literal["quantile", "uniform", "cart"] | None = None,
    n_bins: int | None = None,
    min_bin_size: float | int | None = None,
    monotonic_trend: Literal[
        "ascending",
        "descending",
        "peak",
        "valley",
        "auto",
        "auto_asc_desc",
    ] | None = None,
    missing_values: list[Any] | None = None,
    special_values: list[Any] | None = None,
    n_jobs: int | None = None,
    binner: MarsBinnerBase | None = None,
    advanced_binning_params: dict[str, dict[str, Any]] | None = None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    batch_size: int = 100,
) -> MarsRiskProfile:
    """
    运行高层分箱风险评估工作流。

    `profile_risk` 统一编排分箱器构建、分箱评估、多目标合并和结果对象组装，
    最终返回 `MarsRiskProfile`。函数本身不承担绘图副作用；如需查看趋势图，
    请直接调用 `report.plot_risk_trends(...)`。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        待评估样本表。
    target : str | list[str] | None
        目标列名或目标列列表。传入 `None` 时进入无标签模式，只计算分布指标与 PSI。
    features : list[str] | None
        本次参与评估的特征列。传入 `None` 时自动从输入表推断。
    feature_data_source : dict[str, list[str]] | None
        特征来源映射，用于在报告中保留数据源维度。
    group_col : str | None
        已存在的分组列名。
    time_col : str | None
        原始日期列名。
    time_grain : str | None
        时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
    weights_col : str | None
        样本权重列名。
    amount_col : str | None
        金额列名。
    feature_start_aware_reference : bool
        是否启用 feature-start aware reference，用于 PSI 基准重锚。
    risk_corr_baseline : RiskCorrBaseline
        RC 的基准选择方式。
    psi_include_missing : bool
        计算 PSI 时是否纳入缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否纳入特殊值箱。
    binning_type : Literal["native", "optimal", "lite_opt"]
        未显式传入 `binner` 时使用的分箱器类型。
    method : Literal["quantile", "uniform", "cart"] | None
        高层公开的分箱方法参数。`native` 下映射到底层 `method`，
        `optimal` 和 `lite_opt` 下映射到底层 `prebinning_method`。
        传入 `None` 时沿用高层默认解析逻辑。
    n_bins : int | None
        高层公开的目标分箱数。传入后映射到底层 `n_bins`。
    min_bin_size : float | int | None
        高层公开的最小分箱样本约束。传入后映射到底层 `min_bin_size`。
    monotonic_trend : Literal["ascending", "descending", "peak", "valley", "auto", "auto_asc_desc"] | None
        高层公开的趋势约束参数。仅对 `optimal` 和 `lite_opt` 生效；当
        `binning_type="native"` 时会发出 `warnings.warn(...)` 提示并忽略。
        当通过 `profile_risk` 使用 `optimal` 或 `lite_opt` 且未显式传值时，
        高层默认补成 `"auto_asc_desc"`。这与直接构造 `MarsLiteOptBinner()`
        时底层默认的 `"auto"` 不同。
    missing_values : list[Any] | None
        显式透传给分箱器的 `missing_values`。
    special_values : list[Any] | None
        显式透传给分箱器的 `special_values`。
    n_jobs : int | None
        显式透传给分箱器的 `n_jobs`。
    binner : MarsBinnerBase | None
        显式复用的分箱器。传入后不可再同时传 `advanced_binning_params`，
        也不可再传 `method`、`n_bins`、`min_bin_size`、`monotonic_trend`、
        `missing_values`、`special_values` 或 `n_jobs`。
    advanced_binning_params : dict[str, dict[str, Any]] | None
        按分箱器类型分仓的高级分箱参数入口。顶层只允许 `native`、`optimal`
        和 `lite_opt` 三个仓；运行时只读取并校验当前 `binning_type` 对应仓。
        当前激活仓不得包含任何已公开的高频参数。

        当前允许的高级参数如下：

        - `native`: `merge_small_bins`, `cart_params`, `remove_empty_bins`
        - `optimal`: `min_n_bins`, `min_bin_n_event`, `n_prebins`,
          `min_prebin_size`, `solver`, `time_limit`, `max_cats_to_solver`,
          `min_cat_fraction`, `cart_params`, `join_threshold`
        - `lite_opt`: `n_prebins`, `join_threshold`
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        外部 benchmark 样本。
    batch_size : int
        批量评估时的特征批大小。

    Returns
    -------
    MarsRiskProfile
        单次风险评估结果，包含 `MarsBinningReport`、分箱器、目标列列表和元数据。

    Raises
    ------
    ValueError
        当 `binner` 与 `advanced_binning_params` 同时传入、当前激活高级参数仓包含
        非法键，或显式复用 `binner` 时继续传入公开分箱参数时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import profile_risk
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> profile = profile_risk(
    ...     df,
    ...     target="y",
    ...     features=["age"],
    ...     method="quantile",
    ...     n_bins=3,
    ... )
    >>> profile.targets
    ['y']
    """
    from mars.analysis.evaluator import MarsBinEvaluator, MarsRiskProfile

    input_is_pandas = isinstance(df, pd.DataFrame)
    if binner is not None and advanced_binning_params is not None:
        raise ValueError(
            "`binner` and `advanced_binning_params` cannot be provided together."
        )

    if binner is not None and any(
        value is not None
        for value in (
            method,
            n_bins,
            min_bin_size,
            monotonic_trend,
            missing_values,
            special_values,
            n_jobs,
        )
    ):
        raise ValueError(
            "`method`, `n_bins`, `min_bin_size`, `monotonic_trend`, "
            "`missing_values`, `special_values`, and `n_jobs` cannot be "
            "provided together with `binner`."
        )

    if target is None or target == []:
        target_list: list[str] = []
        primary_target: str | None = None
        is_multi_target = False
    else:
        target_list = [target] if isinstance(target, str) else list(target)
        primary_target = target_list[0]
        is_multi_target = len(target_list) > 1

    normalized_binning_type = _normalize_profile_risk_binning_type(binning_type)
    effective_binning_type = normalized_binning_type
    effective_method = method

    # 无标签场景无法运行监督式分箱，所以这里统一回退到 `native + quantile`，
    # 并且同步切换到 `advanced_binning_params["native"]` 这一仓。
    if not target_list and (
        normalized_binning_type in {"optimal", "lite_opt"} or method == "cart"
    ):
        logger.warning(
            "No target provided. Forcing `binning_type='native'` and `method='quantile'`."
        )
        effective_binning_type = "native"
        effective_method = "quantile"

    effective_binner_params = _build_effective_binner_params(
        binning_type=effective_binning_type,
        advanced_binning_params=advanced_binning_params,
        method=effective_method,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        monotonic_trend=monotonic_trend,
        missing_values=missing_values,
        special_values=special_values,
        n_jobs=n_jobs,
    )

    primary_evaluator = MarsBinEvaluator(
        binning_type=effective_binning_type,
        binner_params=effective_binner_params,
        feature_start_aware_reference=feature_start_aware_reference,
        risk_corr_baseline=risk_corr_baseline,
    )
    primary_run = primary_evaluator.evaluate(
        df=df,
        target=primary_target,
        features=features,
        binner=binner,
        feature_data_source=feature_data_source,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        feature_start_aware_reference=feature_start_aware_reference,
        risk_corr_baseline=risk_corr_baseline,
        psi_include_missing=psi_include_missing,
        psi_include_special=psi_include_special,
        benchmark_df=benchmark_df,
        weights_col=weights_col,
        amount_col=amount_col,
        batch_size=batch_size,
    )
    primary_report = primary_run.report
    trained_binner = primary_run.binner

    if not is_multi_target:
        final_report = primary_report
        final_targets = primary_run.targets
    else:
        p_summary = to_polars_frame(primary_report.summary_table).with_columns(
            pl.lit(primary_target).alias("target")
        )
        p_detail = to_polars_frame(primary_report.detail_table)
        p_reference = to_polars_frame(primary_report.risk_corr_reference_table)
        all_details: list[pl.DataFrame] = [p_detail]
        all_summaries: list[pl.DataFrame] = [p_summary]
        all_references: list[pl.DataFrame] = [p_reference]

        for sec_target in target_list[1:]:
            sec_run = MarsBinEvaluator(
                binning_type=effective_binning_type,
                feature_start_aware_reference=feature_start_aware_reference,
                risk_corr_baseline=risk_corr_baseline,
            ).evaluate(
                df=df,
                target=sec_target,
                features=features,
                binner=trained_binner,
                feature_data_source=feature_data_source,
                group_col=group_col,
                time_col=time_col,
                time_grain=time_grain,
                feature_start_aware_reference=feature_start_aware_reference,
                risk_corr_baseline=risk_corr_baseline,
                psi_include_missing=psi_include_missing,
                psi_include_special=psi_include_special,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                amount_col=amount_col,
                batch_size=batch_size,
            )
            all_details.append(to_polars_frame(sec_run.report.detail_table))
            all_summaries.append(
                to_polars_frame(sec_run.report.summary_table).with_columns(
                    pl.lit(sec_target).alias("target")
                )
            )
            all_references.append(to_polars_frame(sec_run.report.risk_corr_reference_table))

        final_detail: pl.DataFrame | pd.DataFrame = pl.concat(all_details, how="vertical_relaxed")
        final_summary: pl.DataFrame | pd.DataFrame = pl.concat(
            all_summaries,
            how="vertical_relaxed",
        )
        final_reference: pl.DataFrame | pd.DataFrame = pl.concat(
            all_references,
            how="vertical_relaxed",
        )
        if input_is_pandas:
            final_detail = final_detail.to_pandas()
            final_summary = final_summary.to_pandas()
            final_reference = final_reference.to_pandas()

        logger.info("`trend_tables` in the merged report contains primary-target data only.")
        merged_meta = dict(primary_report.report_meta or {})
        meta_df = cast(pl.DataFrame, primary_evaluator._ensure_polars_dataframe(df))
        merged_meta["targets"] = [str(t) for t in target_list]
        event_rate_map: dict[str, float | None] = {}
        for target_name in target_list:
            if target_name in meta_df.columns:
                try:
                    event_rate_map[str(target_name)] = float(
                        meta_df.select(pl.col(target_name).cast(pl.Float64).mean()).item()
                    )
                except Exception:
                    event_rate_map[str(target_name)] = None
        merged_meta["event_rate_by_target"] = event_rate_map

        final_report = MarsBinningReport(
            summary_table=final_summary,
            trend_tables=primary_report.trend_tables,
            detail_table=final_detail,
            group_col=primary_report.group_col,
            detail_group_col=primary_report.detail_group_col,
            feature_data_source=primary_report.feature_data_source,
            dt_col=primary_report.dt_col,
            missing_by_day_table=primary_report.missing_by_day_table,
            risk_corr_reference_table=final_reference,
            report_meta=merged_meta,
        )
        final_targets = [str(t) for t in target_list]

    return MarsRiskProfile(
        report=final_report,
        binner=trained_binner,
        targets=final_targets,
        metadata=dict(final_report.report_meta or {}),
    )
