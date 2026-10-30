"""MARS 数据画像与稳定性分析入口。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import polars as pl

from mars.analysis._profiling.comparison import (
    build_schema_comparison,
    build_unseen_comparison,
)
from mars.analysis._profiling.context import (
    build_run_context,
    prepare_benchmark_data,
    prepare_profile_data,
)
from mars.analysis._profiling.metrics import (
    build_profile_status,
    calculate_overview,
    normalize_profile_metrics,
)
from mars.analysis._profiling.pivot import generate_pivot_reports, unavailable_pivot_report
from mars.analysis._profiling.psi import get_psi_trend
from mars.analysis._profiling.sparkline import compute_sparklines
from mars.analysis._profiling.types import ProfileBinMethod, ProfileComputeOptions
from mars.core.base import MarsBaseEstimator
from mars.reporting import MarsProfileReport
from mars.reporting._metadata import FeatureMetadata, capture_feature_dtypes, validate_group_labels
from mars.utils.decorators import time_it


def profile_stats(
    df: pl.DataFrame | pd.DataFrame,
    *,
    metrics: list[str],
    features: list[str] | None = None,
    feature_metadata: FeatureMetadata | None = None,
    business_context: dict[str, Any] | None = None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    categorical_features: list[str] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    missing_values: list[int | float | str] | None = None,
    special_values: list[Any] | None = None,
    exclude_features: list[str] | None = None,
    include_dtypes: type | pl.DataType | list[type | pl.DataType] | None = None,
    enable_sparkline: bool = False,
    sparkline_bins: int = 8,
    sparkline_sample_size: int = 200_000,
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    psi_remove_empty_bins: bool = True,
    psi_merge_small_bins: bool = True,
    psi_min_bin_size: float = 0.02,
    segment_col: str | None = None,
    binning_scope: str = "shared",
) -> MarsProfileReport:
    """
    按指定指标生成数据画像，默认不生成字符分布图。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        待画像样本表。
    metrics : list[str]
        本次计算的数据质量、统计或对比指标，必须显式传入。
        ``schema`` 和 ``unseen`` 需要 ``benchmark_df``，且不在默认指标中。
    features : list[str] | None
        本次画像的特征列。默认从当前表中选择符合排除列和类型条件的列；
        对比指标默认检查当前表与基准表的列名并集。
    feature_metadata : FeatureMetadata | None
        完整特征字典或含 feature 列的 Pandas/Polars 表；按本次特征裁剪。
    business_context : dict[str, Any] | None
        用户提供的 JSON 业务上下文，与实际计算配置分开保存。
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        PSI 和对比指标的基准样本。计算 PSI 时，用全部基准样本拟合分箱，
        并以基准整体分布为参考；未分组时仍可计算当前整体相对基准的 PSI。
        基准样本不计入 overview、数据质量或其他当前数据统计。
        未提供时，PSI 在当前数据上拟合分箱，并以最小分组为参考。
    categorical_features : list[str] | None
        额外声明为类别的特征，供 unseen 和 PSI 识别整数编码类别。
        默认按字段类型判断；声明的列必须属于本次分析范围。
    group_col : str | None
        已存在的趋势分组列。与时间列同时提供时，优先使用此列分组。
    time_col : str | None
        原始日期列。未指定 ``group_col`` 时，按 ``time_grain`` 生成趋势分组。
    time_grain : str | None
        时间聚合粒度，例如 ``"day"``、``"week"``、``"month"`` 或 ``"7d"``。
        未指定时按月聚合；仅按日期生成分组时生效。
    missing_values : list[int | float | str] | None
        除 null 和 NaN 外，额外视为缺失的取值。默认不增加业务缺失码。
    special_values : list[Any] | None
        从连续统计中排除、在 PSI 中单独分箱的取值。默认无额外特殊值；
        是否将特殊值箱计入 PSI 由 ``psi_include_special`` 控制。
    exclude_features : list[str] | None
        本次画像需要排除的列名。
    include_dtypes : type | pl.DataType | list[type | pl.DataType] | None
        本次画像允许保留的数据类型。
    enable_sparkline : bool
        是否在 overview 中生成数值特征的字符分布图。默认关闭，不影响统计指标。
    sparkline_bins : int
        字符分布图的直方图分箱数，默认 8；仅在启用字符图时生效。
    sparkline_sample_size : int
        字符分布图使用的最多行数，默认 200,000。超出时仅为字符图随机抽样，
        不改变画像统计指标使用的样本。
    psi_include_missing : bool
        是否将缺失值箱计入 PSI。默认排除，缺失率仍单独计算。
    psi_include_special : bool
        是否将特殊值箱计入 PSI。默认排除，不影响特殊值识别。
    psi_remove_empty_bins : bool
        是否移除 PSI 数值分箱中没有拟合样本的箱，默认开启。
    psi_merge_small_bins : bool
        是否合并 PSI 数值分箱中的小样本箱，默认开启。
    psi_min_bin_size : float
        PSI 数值分箱合并小箱时的最小样本占比，默认 0.02；
        仅在 ``psi_merge_small_bins=True`` 时生效。
    segment_col : str | None
        本次客群列。默认不切分；指定后同时保留完整人口和各客群统计。
    binning_scope : str
        PSI 规则拟合范围，``"shared"`` 共享整体切点，``"per_segment"`` 独立拟合。
        仅影响 PSI，独立拟合必须同时指定客群并请求 PSI。

    Returns
    -------
    MarsProfileReport
        包含 overview、所请求的趋势表、对比表和运行元数据的报告。

    Notes
    -----
    缺失率、零值率、唯一值率和众数占比均以当前样本总数为分母。
    均值、分位数等连续统计排除 null、NaN、自定义缺失值和特殊值。
    PSI 的缺失值箱、特殊值箱是否参与计算由对应开关控制。
    多 chunk 输入无需先 rechunk；不适用的连续统计保留 typed null。
    无分组、时间和 benchmark 的 PSI 保留表与特征身份，并在 calculation_status
    中标记 ``not_computed/no_group_or_benchmark``。
    有参考但当前样本为空时，PSI 仍校验 benchmark 结构，并标记
    ``not_computed/empty_current_population``，不拟合分箱。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1, 2, None], "month": ["202601", "202601", "202602"]})
    >>> report = profile_stats(df, metrics=["missing", "mean"], group_col="month")
    >>> "missing" in report.dq_tables
    True
    >>> benchmark = pl.DataFrame({"x": [1, 1, 2, 2]})
    >>> current = pl.DataFrame({"x": [1, 2, 2, 2]})
    >>> psi_report = profile_stats(current, metrics=["psi"], benchmark_df=benchmark)
    >>> psi_report.stats_tables["psi"].columns
    ['feature', 'dtype', 'total']
    """
    profiler = MarsDataProfiler(
        missing_values=missing_values,
        special_values=special_values,
        psi_remove_empty_bins=psi_remove_empty_bins,
        psi_merge_small_bins=psi_merge_small_bins,
        psi_min_bin_size=psi_min_bin_size,
        binning_scope=binning_scope,
    )
    return profiler.generate_profile(
        df,
        metrics=metrics,
        features=features,
        feature_metadata=feature_metadata,
        business_context=business_context,
        benchmark_df=benchmark_df,
        categorical_features=categorical_features,
        exclude_features=exclude_features,
        include_dtypes=include_dtypes,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        enable_sparkline=enable_sparkline,
        sparkline_bins=sparkline_bins,
        sparkline_sample_size=sparkline_sample_size,
        psi_include_missing=psi_include_missing,
        psi_include_special=psi_include_special,
        segment_col=segment_col,
    )


class MarsDataProfiler(MarsBaseEstimator):
    """
    数据质量、统计分布和稳定性画像器。

    构造函数保存缺失处理和 PSI 分箱策略。每次调用 :meth:`generate_profile`
    都会接收本次数据、特征和分组配置，同一实例可以连续处理不同 DataFrame。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"age": [20, 30, None], "month": ["202601", "202601", "202602"]})
    >>> profiler = MarsDataProfiler(missing_values=[-999])
    >>> report = profiler.generate_profile(df, group_col="month")
    >>> report.get_profile_data().overview.height > 0
    True
    """

    def __init__(
        self,
        *,
        missing_values: list[int | float | str] | None = None,
        special_values: list[Any] | None = None,
        psi_n_bins: int = 10,
        psi_bin_method: ProfileBinMethod = "quantile",
        psi_remove_empty_bins: bool = True,
        psi_merge_small_bins: bool = True,
        psi_min_bin_size: float = 0.02,
        psi_cv_ignore_threshold: float = 0.05,
        psi_batch_size: int = 50,
        overview_batch_size: int = 500,
        binning_scope: str = "shared",
    ) -> None:
        """
        初始化画像策略。

        Parameters
        ----------
        missing_values : list[int | float | str] | None
            除 null 和 NaN 外，额外视为缺失的取值。默认不增加业务缺失码。
        special_values : list[Any] | None
            从连续统计中排除、在 PSI 中单独分箱的取值。默认无额外特殊值。
        psi_n_bins : int
            PSI 数值分箱数上限及类别 Top-K 数量，默认 10。
            类别特征只保留频数最高的 K 个类别，其他类别进入 Other 箱。
        psi_bin_method : ProfileBinMethod
            PSI 数值分箱方法，默认等频分箱 ``"quantile"``，也可使用 ``"uniform"``。
        psi_remove_empty_bins : bool
            是否移除 PSI 数值分箱中没有拟合样本的箱，默认开启。
        psi_merge_small_bins : bool
            是否合并 PSI 数值分箱中的小样本箱，默认开启。
        psi_min_bin_size : float
            PSI 数值分箱合并小箱时的最小样本占比，默认 0.02；
            仅在 ``psi_merge_small_bins=True`` 时生效。
        psi_cv_ignore_threshold : float
            分组 PSI 最大值低于此阈值时，将变异系数置零，默认 0.05。
        psi_batch_size : int
            每批计算 PSI 的特征数，默认 50；控制中间表规模，不改变指标口径。
        overview_batch_size : int
            每批计算 overview 的特征数，默认 500；控制中间表规模，不改变指标口径。
        binning_scope : str
            PSI 分箱规则范围，``"shared"`` 或 ``"per_segment"``；不改变其他统计。

        Raises
        ------
        ValueError
            分箱范围不是 ``"shared"`` 或 ``"per_segment"`` 时抛出。
        """
        super().__init__()
        self.missing_values: list[Any] = list(missing_values or [])
        self.special_values: list[Any] = list(special_values or [])
        self.psi_batch_size = psi_batch_size
        self.psi_n_bins = psi_n_bins
        self.psi_bin_method = psi_bin_method
        self.psi_remove_empty_bins = psi_remove_empty_bins
        self.psi_merge_small_bins = psi_merge_small_bins
        self.psi_min_bin_size = psi_min_bin_size
        self.psi_cv_ignore_threshold = psi_cv_ignore_threshold
        self.overview_batch_size = overview_batch_size
        if binning_scope not in {"shared", "per_segment"}:
            raise ValueError("`binning_scope` must be 'shared' or 'per_segment'.")
        self.binning_scope = binning_scope

    @time_it
    def generate_profile(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        metrics: list[str] | None = None,
        features: list[str] | None = None,
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
        categorical_features: list[str] | None = None,
        exclude_features: list[str] | None = None,
        include_dtypes: type | pl.DataType | list[type | pl.DataType] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        enable_sparkline: bool = True,
        sparkline_bins: int = 8,
        sparkline_sample_size: int = 200_000,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        segment_col: str | None = None,
    ) -> MarsProfileReport:
        """
        生成一次数据画像报告。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待画像样本表。
        metrics : list[str] | None
            本次计算的指标。默认计算数据质量和统计指标；
            ``schema`` 和 ``unseen`` 需要显式请求并提供 ``benchmark_df``。
        features : list[str] | None
            本次画像的特征列；不传时使用过滤后的全部候选列。
            对比指标默认检查当前表与基准表的列名并集，排除分组、日期和显式排除列。
        feature_metadata : FeatureMetadata | None
            完整特征字典或含 feature 列的 Pandas/Polars 表。
        business_context : dict[str, Any] | None
            用户提供的 JSON 业务上下文，未知信息不推断。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            PSI 和对比指标的基准样本。计算 PSI 时，用全部基准样本拟合分箱，
            并以基准整体分布为参考；未分组时仍可比较两表的整体分布。
            基准样本不计入 overview、数据质量或其他当前数据统计。
            未提供时，PSI 在当前数据上拟合分箱，并以最小分组为参考。
        categorical_features : list[str] | None
            额外声明为类别的特征，供 unseen 和 PSI 识别整数编码类别。
            默认按字段类型判断；声明的列必须属于本次分析范围。
        exclude_features : list[str] | None
            本次画像需要排除的列名。
        include_dtypes : type | pl.DataType | list[type | pl.DataType] | None
            只保留指定数据类型的特征列。
        group_col : str | None
            已存在的趋势分组列。与时间列同时提供时，优先使用此列分组。
        time_col : str | None
            原始日期列。未指定 ``group_col`` 时，按 ``time_grain`` 生成趋势分组。
        time_grain : str | None
            时间聚合粒度，例如 ``"day"``、``"week"``、``"month"`` 或 ``"7d"``。
            未指定时按月聚合；仅按日期生成分组时生效。
        enable_sparkline : bool
            是否在 overview 中生成数值特征的字符分布图，默认开启。
        sparkline_bins : int
            字符分布图的直方图分箱数，默认 8；仅在启用字符图时生效。
        sparkline_sample_size : int
            字符分布图使用的最多行数，默认 200,000。超出时仅为字符图随机抽样，
            不改变画像统计指标使用的样本。
        psi_include_missing : bool
            是否将缺失值箱计入 PSI。默认排除，缺失率仍单独计算。
        psi_include_special : bool
            是否将特殊值箱计入 PSI。默认排除，不影响特殊值识别。
        segment_col : str | None
            本次运行的客群列，自动从默认特征中排除。与显式特征、分组和日期角色冲突时报错。

        Returns
        -------
        MarsProfileReport
            包含 overview、数据质量趋势表、统计趋势表、对比表和运行元数据的报告。

        Raises
        ------
        ValueError
            指标、特征、benchmark 或对比配置无效时抛出。

        Notes
        -----
        当前数据的统计指标使用全部样本，字符分布图可按 ``sparkline_sample_size`` 抽样。
        缺失率等数据质量指标以样本总数为分母；均值、分位数等连续统计排除
        null、NaN、自定义缺失值和特殊值。
        多 chunk 输入保持原值和 chunk 结构，无需预先 rechunk；不适用的统计
        保留 typed null。PSI 独立计算，无参考或当前人口为空时保留表和特征身份，
        并记录未计算原因；空当前人口仍校验 benchmark 结构，不拟合分箱。

        Examples
        --------
        >>> benchmark = pl.DataFrame({"x": [1, 1, 2, 2]})
        >>> current = pl.DataFrame({"x": [1, 2, 2, 2]})
        >>> report = MarsDataProfiler().generate_profile(
        ...     current,
        ...     metrics=["psi"],
        ...     benchmark_df=benchmark,
        ...     enable_sparkline=False,
        ... )
        >>> report.stats_tables["psi"].height
        1

        """
        selection = normalize_profile_metrics(metrics, require_metrics=False)
        if self.binning_scope == "per_segment" and (
            segment_col is None or "psi" not in selection.stat_metrics
        ):
            raise ValueError("`binning_scope='per_segment'` requires `segment_col` and metric 'psi'.")
        kwargs: dict[str, Any] = {
            "metrics": metrics, "features": features, "feature_metadata": feature_metadata,
            "business_context": business_context, "benchmark_df": benchmark_df,
            "categorical_features": categorical_features, "exclude_features": exclude_features,
            "include_dtypes": include_dtypes, "group_col": group_col, "time_col": time_col,
            "time_grain": time_grain, "enable_sparkline": enable_sparkline,
            "sparkline_bins": sparkline_bins, "sparkline_sample_size": sparkline_sample_size,
            "psi_include_missing": psi_include_missing, "psi_include_special": psi_include_special,
        }
        if segment_col is not None:
            from mars.analysis._profiling.segmented import generate_segmented_profile

            return generate_segmented_profile(self, df, segment_col, kwargs)
        return self._generate_profile(df, **kwargs)

    def _generate_profile(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        metrics: list[str] | None = None,
        features: list[str] | None = None,
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
        categorical_features: list[str] | None = None,
        exclude_features: list[str] | None = None,
        include_dtypes: type | pl.DataType | list[type | pl.DataType] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        enable_sparkline: bool = True,
        sparkline_bins: int = 8,
        sparkline_sample_size: int = 200_000,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        psi_binner: Any = None,
        psi_fit_df: pl.DataFrame | None = None,
        psi_fit_population_failures: dict[str, str] | None = None,
        psi_reference_reason: str | None = None,
        psi_precomputed_counts: dict[tuple[str, ...], tuple[pl.DataFrame, pl.DataFrame]] | None = None,
        allow_empty_benchmark: bool = False,
    ) -> MarsProfileReport:
        """按明确人口计算一份画像，客群编排只传规则和参考，不留存输入宽表。"""
        selection = normalize_profile_metrics(metrics, require_metrics=False)
        comparison_requested = bool(selection.comparison_metrics)
        prepared_df, prepared_features = prepare_profile_data(
            df,
            ensure_polars=self._ensure_polars_dataframe,
            features=features,
            exclude_features=exclude_features,
            include_dtypes=include_dtypes,
            allow_missing_features=comparison_requested
            and not selection.dq_metrics
            and not selection.stat_metrics,
        )
        context = build_run_context(
            prepared_df,
            features=prepared_features,
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
        )
        validate_group_labels(context.working_df, context.group_col)
        options = ProfileComputeOptions(
            missing_values=self.missing_values,
            special_values=self.special_values,
            psi_n_bins=self.psi_n_bins,
            psi_bin_method=self.psi_bin_method,
            psi_remove_empty_bins=self.psi_remove_empty_bins,
            psi_merge_small_bins=self.psi_merge_small_bins,
            psi_min_bin_size=self.psi_min_bin_size,
            psi_cv_ignore_threshold=self.psi_cv_ignore_threshold,
            psi_batch_size=self.psi_batch_size,
            overview_batch_size=self.overview_batch_size,
            sparkline_bins=sparkline_bins,
            sparkline_sample_size=sparkline_sample_size,
            psi_include_missing=psi_include_missing,
            psi_include_special=psi_include_special,
            categorical_features=list(categorical_features or []),
            diagnostics=[],
        )

        # 基准样本只用于 PSI 和对比指标，不能混入当前数据的质量或统计汇总。
        benchmark_pl: pl.DataFrame | None = None
        psi_features: list[str] | None = None
        if benchmark_df is not None and (
            "psi" in selection.stat_metrics or comparison_requested
        ):
            benchmark_pl = prepare_benchmark_data(benchmark_df)
            if benchmark_pl.is_empty() and not allow_empty_benchmark:
                raise ValueError("`benchmark_df` must contain at least one row.")
            excluded_psi_cols = {column for column in [context.group_col, time_col] if column}
            psi_features = [
                feature
                for feature in context.features
                if feature not in excluded_psi_cols
            ]
        if comparison_requested and benchmark_pl is None:
            raise ValueError("Metrics 'schema' and 'unseen' require `benchmark_df`.")

        comparison_features: list[str] = []
        if comparison_requested:
            assert benchmark_pl is not None
            excluded = set(exclude_features or [])
            excluded.update(column for column in [group_col, time_col, context.group_col] if column)
            if features is not None:
                comparison_features = [feature for feature in features if feature not in excluded]
            else:
                comparison_features = [
                    feature
                    for feature in dict.fromkeys([*prepared_df.columns, *benchmark_pl.columns])
                    if feature not in excluded
                ]
            undeclared = [
                feature
                for feature in (categorical_features or [])
                if feature not in comparison_features
            ]
            if undeclared:
                raise ValueError(
                    "`categorical_features` are outside the active comparison scope: "
                    f"{undeclared}."
                )
        elif "psi" in selection.stat_metrics:
            active_psi_scope = {
                feature
                for feature in context.features
                if feature not in {context.group_col, time_col}
            }
            undeclared = [
                feature
                for feature in (categorical_features or [])
                if feature not in active_psi_scope
            ]
            if undeclared:
                raise ValueError(
                    "`categorical_features` are outside the active PSI scope: "
                    f"{undeclared}."
                )

        sparkline_df = compute_sparklines(context, options) if enable_sparkline else pl.DataFrame()
        overview_df = calculate_overview(context, selection, options, sparkline_df)

        pivots = generate_pivot_reports(
            context, options,
            [*selection.dq_metrics, *[m for m in selection.stat_metrics if m != "psi"]],
            overview_df,
        )
        dq_tables = {metric: pivots[metric] for metric in selection.dq_metrics}
        stat_tables = {metric: pivots[metric] for metric in selection.stat_metrics if metric != "psi"}

        if "psi" in selection.stat_metrics:
            # 无参考或无当前样本时显式保留 typed null；PSI 不经过通用聚合占位表。
            if psi_reference_reason is not None:
                stat_tables["psi"] = unavailable_pivot_report(context)
            elif context.group_col or benchmark_pl is not None:
                stat_tables["psi"] = get_psi_trend(
                    context,
                    options,
                    features=psi_features,
                    benchmark_df=benchmark_pl,
                    binner=psi_binner,
                    fit_df=psi_fit_df,
                    fit_population_failures=psi_fit_population_failures,
                    precomputed_counts=psi_precomputed_counts,
                )
            else:
                stat_tables["psi"] = unavailable_pivot_report(context)
            stat_tables = {metric: stat_tables[metric] for metric in selection.stat_metrics}

        comparison_tables: dict[str, pl.DataFrame] = {}
        if "schema" in selection.comparison_metrics:
            assert benchmark_pl is not None
            comparison_tables["schema"] = build_schema_comparison(
                prepared_df,
                benchmark_pl,
                comparison_features,
            )
        if "unseen" in selection.comparison_metrics:
            assert benchmark_pl is not None
            comparison_tables["unseen"] = build_unseen_comparison(
                context,
                benchmark_pl,
                comparison_features,
                options,
            )

        from mars import __version__ as mars_version

        report_meta: dict[str, Any] = {
            "mars_version": mars_version,
            "polars_version": pl.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_type": "pandas" if isinstance(df, pd.DataFrame) else "polars",
            "row_count": prepared_df.height,
            "column_count": prepared_df.width,
            "features": list(prepared_features),
            "feature_dtypes": capture_feature_dtypes(prepared_df, prepared_features),
            "metrics": [
                *selection.dq_metrics,
                *selection.stat_metrics,
                *selection.comparison_metrics,
            ],
            "group_col": group_col,
            "time_col": time_col,
            "time_grain": time_grain,
            "benchmark_row_count": benchmark_pl.height if benchmark_pl is not None else None,
            "categorical_features": list(categorical_features or []),
            "profile_config": {
                "missing_values": list(self.missing_values),
                "special_values": list(self.special_values),
                "overview_batch_size": self.overview_batch_size,
                "psi_batch_size": self.psi_batch_size,
                "psi_remove_empty_bins": self.psi_remove_empty_bins,
                "psi_merge_small_bins": self.psi_merge_small_bins,
                "psi_min_bin_size": self.psi_min_bin_size,
                "psi_cv_ignore_threshold": self.psi_cv_ignore_threshold,
                "enable_sparkline": enable_sparkline,
                "sparkline_sample_size": sparkline_sample_size,
                "sparkline_bins": sparkline_bins,
                "psi_n_bins": self.psi_n_bins,
                "psi_bin_method": self.psi_bin_method,
                "psi_include_missing": psi_include_missing,
                "psi_include_special": psi_include_special,
            },
            "diagnostics": list(options.diagnostics),
        }

        formatted_stats: dict[str, pl.DataFrame | pd.DataFrame] = {
            metric: self._format_psi_output(table) if metric == "psi" else self._format_output(table)
            for metric, table in stat_tables.items()
        }
        report = MarsProfileReport(
            overview=self._format_output(overview_df),
            dq_tables=self._format_output(dq_tables),
            stats_tables=formatted_stats,
            comparison_tables=self._format_output(comparison_tables),
            report_meta=report_meta,
            feature_metadata=feature_metadata,
            business_context=business_context,
        )
        status = build_profile_status(
            context, overview_df, selection.stat_metrics, options.diagnostics,
            has_psi_reference=context.group_col is not None or benchmark_pl is not None,
            psi_features=psi_features,
        )
        if psi_reference_reason is not None and "psi" in selection.stat_metrics:
            missing_status = pl.DataFrame(
                {
                    "feature": psi_features if psi_features is not None else context.features,
                },
                schema={"feature": pl.String},
            ).with_columns(
                pl.lit("*").alias("group"), pl.lit("psi").alias("metric"),
                pl.lit("not_computed").alias("status"), pl.lit(psi_reference_reason).alias("reason"),
            )
            status = pl.concat([status.filter(pl.col("metric") != "psi"), missing_status])
        if status.height:
            report._calculation_status = self._format_output(status)
        report._normalize_feature_tables()
        report.source = {"kind": "analysis", "producer": "mars", "input_provenance": "caller_provided_dataframe"}
        return report

    def _format_psi_output(self, table: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
        """新报告的 Pandas PSI 保留可空浮点列，使未计算人口序列化为正式 null。"""
        result: pl.DataFrame | pd.DataFrame = self._format_output(table)
        if isinstance(result, pd.DataFrame):
            for column, dtype in table.schema.items():
                if dtype in (pl.Float32, pl.Float64) and table[column].null_count():
                    result[column] = result[column].astype("Float64")
        return result
