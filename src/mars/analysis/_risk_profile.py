"""高层风险画像工作流入口。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import polars as pl

from mars.compute import OrderedMetricSortBy, RiskCorrBaseline, to_polars_frame
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

_FORBIDDEN_BINNER_PARAM_KEYS: frozenset[str] = frozenset(
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

_ALLOWED_BINNER_PARAM_KEYS: dict[_ProfileRiskBinningType, tuple[str, ...]] = {
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
        "peak",
        "valley",
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
_ALL_RECOGNIZED_BINNER_PARAM_KEYS: frozenset[str] = frozenset(
    key
    for allowed_keys in _ALLOWED_BINNER_PARAM_KEYS.values()
    for key in allowed_keys
)


def _normalize_profile_risk_binning_type(binning_type: str) -> _ProfileRiskBinningType:
    """校验并规范化 `profile_risk` 使用的 `binning_type`。"""
    normalized = binning_type.strip().lower()
    valid_binning_types = tuple(_ALLOWED_BINNER_PARAM_KEYS)
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


def _format_param_names(param_names: tuple[str, ...] | frozenset[str]) -> str:
    """生成参数名列表文本。"""
    return ", ".join(f"`{key}`" for key in sorted(param_names))


def _format_allowed_binner_param_text(binning_type: _ProfileRiskBinningType) -> str:
    """生成当前 `binning_type` 可接受的高级分箱器参数文本。"""
    allowed_keys = _ALLOWED_BINNER_PARAM_KEYS[binning_type]
    return ", ".join(f"`{key}`" for key in allowed_keys)


def _normalize_binner_params(
    binner_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """标准化 `profile_risk` 的单层分箱器参数。"""
    if binner_params is None:
        return {}

    if not isinstance(binner_params, dict):
        raise ValueError(
            f"`binner_params` must be a dict, got {type(binner_params).__name__}."
        )
    return dict(binner_params)


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
    binner_params: dict[str, Any] | None,
    method: Literal["quantile", "uniform", "cart"] | None,
    n_bins: int | None,
    min_bin_size: float | int | None,
    monotonic_trend: _ProfileRiskMonotonicTrend | None,
    missing_values: list[Any] | None,
    special_values: list[Any] | None,
    n_jobs: int | None,
) -> dict[str, Any]:
    """合并 `profile_risk` 公开分箱参数并校验分箱器参数契约。"""
    normalized_binner_params = _normalize_binner_params(binner_params)
    allowed_keys = set(_ALLOWED_BINNER_PARAM_KEYS[binning_type])
    effective_binner_params: dict[str, Any] = {
        key: value
        for key, value in normalized_binner_params.items()
        if key in allowed_keys
    }
    forbidden_keys = sorted(
        key for key in normalized_binner_params if key in _FORBIDDEN_BINNER_PARAM_KEYS
    )
    unknown_keys = sorted(
        key
        for key in normalized_binner_params
        if key not in _ALL_RECOGNIZED_BINNER_PARAM_KEYS
        and key not in _FORBIDDEN_BINNER_PARAM_KEYS
    )

    if forbidden_keys or unknown_keys:
        invalid_keys = forbidden_keys + unknown_keys
        invalid_text = ", ".join(f"`{key}`" for key in invalid_keys)
        allowed_text = _format_allowed_binner_param_text(binning_type)
        recognized_text = _format_param_names(_ALL_RECOGNIZED_BINNER_PARAM_KEYS)
        public_text = _format_param_names(_PUBLIC_BINNER_ARG_NAMES)
        raise ValueError(
            f"`binner_params` received unsupported keys for "
            f"`binning_type={binning_type!r}`: {invalid_text}. Current "
            f"`binning_type` accepts: {allowed_text}. All recognized binner-only "
            f"keys are: {recognized_text}. Public binning arguments must be passed "
            f"explicitly via {public_text}. The internal `prebinning_method` alias "
            "is also not allowed here."
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
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    binner_params: dict[str, Any] | None = None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    feature_start_aware_reference: bool = False,
    risk_corr_baseline: RiskCorrBaseline = "total",
    ordered_metric_sort_by: OrderedMetricSortBy = "woe",
    batch_size: int = 100,
    n_jobs: int | None = None,
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
    binning_type : Literal["native", "optimal", "lite_opt"]
        自动构建分箱器时使用的分箱器类型。
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
    psi_include_missing : bool
        计算 PSI 时是否纳入缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否纳入特殊值箱。
    binner_params : dict[str, Any] | None
        单层高级分箱器参数入口。同一份字典可以同时包含多种分箱器参数；
        运行时只读取当前 `binning_type` 适用的键，其他已识别但不适用的键会被忽略。
        该参数不得包含任何已公开的高频参数，也不得包含底层别名 `prebinning_method`。

        当前允许的高级参数如下：

        - `native`: `merge_small_bins`, `cart_params`, `remove_empty_bins`
        - `optimal`: `min_n_bins`, `min_bin_n_event`, `n_prebins`,
          `min_prebin_size`, `solver`, `time_limit`, `max_cats_to_solver`,
          `min_cat_fraction`, `cart_params`, `join_threshold`
        - `lite_opt`: `n_prebins`, `join_threshold`
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        基准期样本。未传显式 ``binner`` 时用于拟合分箱规则和构造 PSI 基准；
        监督分箱时必须包含当前首个 target 的至少两个有效类别。多 target 场景只用
        首个 target 拟合一次，后续 target 复用同一分箱规则。
    feature_start_aware_reference : bool
        是否启用 feature-start aware reference，用于 PSI 基准重锚。
    risk_corr_baseline : RiskCorrBaseline
        RC 的基准选择方式。
    ordered_metric_sort_by : OrderedMetricSortBy
        KS/AUC 的排序口径。默认 `"woe"` 适合普通特征预测力评估；
        评估概率、分数或强有序变量时建议传 `"bin_index"`。
    batch_size : int
        批量评估时的特征批大小。
    n_jobs : int | None
        显式透传给分箱器的 `n_jobs`。

    Returns
    -------
    MarsRiskProfile
        单次风险评估结果，包含 `MarsBinningReport`、分箱器、目标列列表和元数据。

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

    # 无标签场景无法运行监督式分箱，所以这里统一回退到 `native + quantile`。
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
        binner_params=binner_params,
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
        ordered_metric_sort_by=ordered_metric_sort_by,
    )
    primary_run = primary_evaluator.evaluate(
        df=df,
        target=primary_target,
        features=features,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        feature_data_source=feature_data_source,
        weights_col=weights_col,
        amount_col=amount_col,
        benchmark_df=benchmark_df,
        psi_include_missing=psi_include_missing,
        psi_include_special=psi_include_special,
        feature_start_aware_reference=feature_start_aware_reference,
        risk_corr_baseline=risk_corr_baseline,
        ordered_metric_sort_by=ordered_metric_sort_by,
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
                ordered_metric_sort_by=ordered_metric_sort_by,
            ).evaluate(
                df=df,
                target=sec_target,
                features=features,
                group_col=group_col,
                time_col=time_col,
                time_grain=time_grain,
                feature_data_source=feature_data_source,
                weights_col=weights_col,
                amount_col=amount_col,
                binner=trained_binner,
                benchmark_df=benchmark_df,
                psi_include_missing=psi_include_missing,
                psi_include_special=psi_include_special,
                feature_start_aware_reference=feature_start_aware_reference,
                risk_corr_baseline=risk_corr_baseline,
                ordered_metric_sort_by=ordered_metric_sort_by,
                batch_size=batch_size,
            )
            all_details.append(to_polars_frame(sec_run.report.detail_table))
            all_summaries.append(
                to_polars_frame(sec_run.report.summary_table).with_columns(
                    pl.lit(sec_target).alias("target")
                )
            )
            all_references.append(to_polars_frame(sec_run.report.risk_corr_reference_table))

        final_detail: pl.DataFrame = pl.concat(all_details, how="vertical_relaxed")
        final_summary: pl.DataFrame = pl.concat(
            all_summaries,
            how="vertical_relaxed",
        )
        final_reference: pl.DataFrame = pl.concat(
            all_references,
            how="vertical_relaxed",
        )
        if input_is_pandas:
            final_detail = final_detail.to_pandas()
            final_summary = final_summary.to_pandas()
            final_reference = final_reference.to_pandas()

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
