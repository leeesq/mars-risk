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
from mars.analysis._profiling.metrics import calculate_overview, normalize_profile_metrics
from mars.analysis._profiling.pivot import generate_pivot_report
from mars.analysis._profiling.psi import get_psi_trend
from mars.analysis._profiling.sparkline import compute_sparklines
from mars.analysis._profiling.types import ProfileBinMethod, ProfileComputeOptions
from mars.core.base import MarsBaseEstimator
from mars.reporting import MarsProfileReport
from mars.utils.decorators import time_it


def profile_stats(
    df: pl.DataFrame | pd.DataFrame,
    *,
    metrics: list[str],
    features: list[str] | None = None,
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
) -> MarsProfileReport:
    """
    为指定指标生成轻量画像报告。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        待画像样本表。
    metrics : list[str]
        本次需要计算的数据质量或统计指标，必须显式传入。
    features : list[str] | None
        本次画像的特征列。
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        PSI 基准样本。传入后用其拟合分箱并构造 expected distribution；不会参与
        overview、数据质量或其他统计指标。未提供分组时仍会计算当前全量相对基准的 PSI。
    categorical_features : list[str] | None
        显式声明的类别特征，用于整数编码类别的 unseen 和 PSI 分箱。
    group_col : str | None
        已存在的分组列名。
    time_col : str | None
        原始日期列名，用于生成趋势分组。
    time_grain : str | None
        时间聚合粒度，例如 ``"day"``、``"week"``、``"month"`` 或 ``"7d"``。
    missing_values : list[int | float | str] | None
        额外视为缺失的取值。
    special_values : list[Any] | None
        连续统计和 PSI 计算中需要识别的特殊值。
    exclude_features : list[str] | None
        本次画像需要排除的列名。
    include_dtypes : type | pl.DataType | list[type | pl.DataType] | None
        本次画像允许保留的数据类型。
    enable_sparkline : bool
        是否生成 overview 中的字符分布图。
    sparkline_bins : int
        字符分布图的直方图分箱数量。
    sparkline_sample_size : int
        字符分布图最多使用的采样行数。
    psi_include_missing : bool
        计算 PSI 时是否纳入缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否纳入特殊值箱。
    psi_remove_empty_bins : bool
        数值特征 PSI 分箱时是否移除全局空箱。
    psi_merge_small_bins : bool
        数值特征 PSI 分箱时是否合并小样本箱。
    psi_min_bin_size : float
        数值特征 PSI 合并小样本箱时使用的最小样本占比。

    Returns
    -------
    MarsProfileReport
        包含所请求指标表的画像报告。

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
    )
    return profiler.generate_profile(
        df,
        metrics=metrics,
        features=features,
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
    )


class MarsDataProfiler(MarsBaseEstimator):
    """
    数据质量、统计分布和稳定性画像器。

    `MarsDataProfiler` 只保存稳定策略，不保存本次运行数据。每次调用
    :meth:`generate_profile` 都会构造独立的运行上下文，因此同一个实例可以安全复用到
    不同 DataFrame。

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
    ) -> None:
        """
        初始化画像策略。

        Parameters
        ----------
        missing_values : list[int | float | str] | None
            额外视为缺失的取值。
        special_values : list[Any] | None
            连续统计和 PSI 计算中需要识别的特殊值。
        psi_n_bins : int
            PSI 计算使用的最大分箱数。
        psi_bin_method : ProfileBinMethod
            PSI 数值特征分箱策略。
        psi_remove_empty_bins : bool
            数值特征 PSI 分箱时是否移除全局空箱。
        psi_merge_small_bins : bool
            数值特征 PSI 分箱时是否合并小样本箱。
        psi_min_bin_size : float
            数值特征 PSI 合并小样本箱时使用的最小样本占比。
        psi_cv_ignore_threshold : float
            分组 PSI 最大值低于该阈值时，变异系数强制置零。
        psi_batch_size : int
            PSI 趋势计算的特征批大小。
        overview_batch_size : int
            overview 计算的特征批大小。
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

    @time_it
    def generate_profile(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        metrics: list[str] | None = None,
        features: list[str] | None = None,
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
    ) -> MarsProfileReport:
        """
        生成一次数据画像报告。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待画像样本表。
        metrics : list[str] | None
            本次计算的指标；不传时使用默认全量画像指标。
        features : list[str] | None
            本次画像的特征列；不传时使用过滤后的全部候选列。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            PSI 基准样本。传入后用其拟合分箱并构造 expected distribution；不会参与
            overview、数据质量或其他统计指标。未提供分组时仍会计算当前全量相对基准的 PSI。
        categorical_features : list[str] | None
            显式声明的类别特征，用于整数编码类别的 unseen 和 PSI 分箱。
        exclude_features : list[str] | None
            本次画像需要排除的列名。
        include_dtypes : type | pl.DataType | list[type | pl.DataType] | None
            只保留指定数据类型的特征列。
        group_col : str | None
            已存在的分组列名。
        time_col : str | None
            原始日期列名；与 `time_grain` 配合时会生成临时时间分组列。
        time_grain : str | None
            时间聚合粒度，例如 ``"day"``、``"week"``、``"month"`` 或 ``"7d"``。
        enable_sparkline : bool
            是否生成 overview 中的字符分布图。
        sparkline_bins : int
            字符分布图的直方图分箱数量。
        sparkline_sample_size : int
            字符分布图最多使用的采样行数。
        psi_include_missing : bool
            计算 PSI 时是否纳入缺失值箱。
        psi_include_special : bool
            计算 PSI 时是否纳入特殊值箱。

        Returns
        -------
        MarsProfileReport
            包含 overview、数据质量趋势表和统计趋势表的数据画像报告。

        Raises
        ------
        ValueError
            指标、特征、benchmark 或对比配置无效时抛出。

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

        # benchmark 只在 PSI 或 comparison 请求中解析。
        benchmark_pl: pl.DataFrame | None = None
        psi_features: list[str] | None = None
        if benchmark_df is not None and (
            "psi" in selection.stat_metrics or comparison_requested
        ):
            benchmark_pl = prepare_benchmark_data(benchmark_df)
            if benchmark_pl.is_empty():
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

        dq_tables = {
            metric: generate_pivot_report(context, options, metric)
            for metric in selection.dq_metrics
        }

        stat_tables: dict[str, pl.DataFrame] = {}
        for metric in selection.stat_metrics:
            pivot = generate_pivot_report(context, options, metric)
            stat_tables[metric] = pivot

        if (context.group_col or benchmark_pl is not None) and "psi" in selection.stat_metrics:
            psi_df = get_psi_trend(
                context,
                options,
                features=psi_features,
                benchmark_df=benchmark_pl,
            )
            if not psi_df.is_empty():
                stat_tables["psi"] = psi_df

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
                "psi_n_bins": self.psi_n_bins,
                "psi_bin_method": self.psi_bin_method,
                "psi_include_missing": psi_include_missing,
                "psi_include_special": psi_include_special,
            },
            "diagnostics": list(options.diagnostics),
        }

        return MarsProfileReport(
            overview=self._format_output(overview_df),
            dq_tables=self._format_output(dq_tables),
            stats_tables=self._format_output(stat_tables),
            comparison_tables=self._format_output(comparison_tables),
            report_meta=report_meta,
        )
