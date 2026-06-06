"""MARS 特征与模型监控对象。"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, NamedTuple, Union

import pandas as pd
import polars as pl

from mars.analysis import MarsBinEvaluator
from mars.core.base import MarsBaseEstimator
from mars.feature.binner import MarsBinnerBase

FrameLike = Union[pl.DataFrame, pd.DataFrame]
TrendColumnOrder = Literal["asc", "desc"]


class MarsMonitoringData(NamedTuple):
    """
    监控报告底层数据对象集合。

    Attributes
    ----------
    summary : DataFrame
        特征级监控汇总表。
    detail : DataFrame
        分箱明细表。
    trends : dict of str to DataFrame
        指标趋势表字典。
    missing_by_day : DataFrame or None
        按日缺失率趋势表。
    bin_stat : DataFrame
        全量分箱统计表。
    bin_stat_trends : dict of str to DataFrame
        分箱统计指标趋势表字典。
    target_observation : DataFrame or None
        target 表现覆盖情况表。
    """

    summary: FrameLike
    detail: FrameLike
    trends: Dict[str, FrameLike]
    missing_by_day: FrameLike | None
    bin_stat: FrameLike
    bin_stat_trends: Dict[str, FrameLike]
    target_observation: FrameLike | None


@dataclass
class MarsMonitoringReport:
    """
    单次特征或模型监控结果。

    该对象不承担文件导出职责，只保存可二次加工的结构化数据。模型分、概率列
    或评分列会作为特殊特征进入同一套监控链路。

    Attributes
    ----------
    summary_table : DataFrame
        特征级监控汇总表。
    detail_table : DataFrame
        分箱明细表。
    trend_tables : dict of str to DataFrame
        PSI、缺失率、坏账率等指标趋势表。
    missing_by_day_table : DataFrame or None
        按日缺失率趋势表。
    bin_stat_table : DataFrame
        全量分箱统计表。
    bin_stat_trend_tables : dict of str to DataFrame
        每个分箱随时间或分组变化的统计趋势。
    target_observation_table : DataFrame or None
        target 表现覆盖情况表。
    binner : MarsBinnerBase
        本次监控使用的分箱器。
    features : list of str
        本次监控覆盖的特征列表。
    target : str or None
        本次监控使用的目标变量列名。
    metadata : dict
        本次运行的上下文信息。
    """

    summary_table: FrameLike
    detail_table: FrameLike
    trend_tables: Dict[str, FrameLike]
    missing_by_day_table: FrameLike | None
    bin_stat_table: FrameLike
    bin_stat_trend_tables: Dict[str, FrameLike]
    target_observation_table: FrameLike | None
    binner: MarsBinnerBase
    features: list[str]
    target: str | None
    metadata: dict[str, Any]

    def get_monitoring_data(self) -> MarsMonitoringData:
        """
        返回监控报告的底层结构化数据。

        Returns
        -------
        MarsMonitoringData
            汇总、明细、趋势、分箱统计和 target 表现覆盖表集合。

        Examples
        --------
        >>> import polars as pl
        >>> from mars.feature import MarsNativeBinner
        >>> report = MarsMonitoringReport(
        ...     summary_table=pl.DataFrame(),
        ...     detail_table=pl.DataFrame(),
        ...     trend_tables={},
        ...     missing_by_day_table=None,
        ...     bin_stat_table=pl.DataFrame(),
        ...     bin_stat_trend_tables={},
        ...     target_observation_table=None,
        ...     binner=MarsNativeBinner(),
        ...     features=[],
        ...     target=None,
        ...     metadata={},
        ... )
        >>> report.get_monitoring_data().trends
        {}
        """
        return MarsMonitoringData(
            summary=self.summary_table,
            detail=self.detail_table,
            trends=self.trend_tables,
            missing_by_day=self.missing_by_day_table,
            bin_stat=self.bin_stat_table,
            bin_stat_trends=self.bin_stat_trend_tables,
            target_observation=self.target_observation_table,
        )


class MarsMonitor(MarsBaseEstimator):
    """
    特征与模型监控器。

    监控器把模型分、概率列和普通特征都视为待监控字段，复用分箱评估链路
    生成 PSI、缺失率、分箱占比和已表现样本上的风险指标。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.monitoring import MarsMonitor
    >>> df = pl.DataFrame(
    ...     {
    ...         "month": ["2026-01", "2026-01", "2026-02", "2026-02"],
    ...         "score": [0.1, 0.8, 0.2, 0.9],
    ...         "target": [0, 1, 0, 1],
    ...     }
    ... )
    >>> report = MarsMonitor(binner_params={"n_bins": 2}).monitor(
    ...     df,
    ...     features=["score"],
    ...     target="target",
    ...     group_col="month",
    ... )
    >>> report.features
    ['score']
    """

    DEFAULT_BIN_STAT_METRICS = [
        "count",
        "observed_count",
        "pct",
        "bad_rate",
        "mean",
        "min",
        "max",
        "median",
    ]

    def __init__(
        self,
        *,
        binning_type: Literal["native", "opt"] = "native",
        binner_params: Dict[str, Any] | None = None,
        bin_stat_metrics: List[str] | None = None,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
    ) -> None:
        """
        初始化监控器。

        Parameters
        ----------
        binning_type : Literal['native', 'opt']
            未显式传入分箱器时使用的分箱策略。
        binner_params : Dict[str, Any] | None
            构造默认分箱器时使用的参数。
        bin_stat_metrics : List[str] | None
            需要展开为分箱趋势表的统计指标。
        psi_include_missing : bool
            计算 PSI 时是否包含缺失值箱。
        psi_include_special : bool
            计算 PSI 时是否包含特殊值箱。
        """
        super().__init__()
        self.binning_type = binning_type
        self.binner_params = dict(binner_params or {})
        self.bin_stat_metrics = list(bin_stat_metrics or self.DEFAULT_BIN_STAT_METRICS)
        self.psi_include_missing = psi_include_missing
        self.psi_include_special = psi_include_special

    def monitor(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        *,
        features: List[str],
        target: str | None,
        binner: MarsBinnerBase | None = None,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        feature_data_source: Dict[str, List[str]] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        weights_col: str | None = None,
        batch_size: int = 100,
        trend_column_order: TrendColumnOrder = "asc",
    ) -> MarsMonitoringReport:
        """
        执行一次特征或模型分监控。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待监控样本表。
        features : List[str]
            本次监控的特征、模型分或概率列。
        target : str | None
            目标变量列名。传入时非空值必须为 ``0``、``1``、``True`` 或 ``False``；
            空值表示尚未到表现期。为 ``None`` 时只输出无标签分布监控。
        binner : MarsBinnerBase | None
            显式复用的分箱器。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame, None]
            外部 benchmark 样本。
        feature_data_source : Dict[str, List[str]] | None
            特征来源映射。
        group_col : str | None
            已存在的分组列名。
        time_col : str | None
            原始日期列名。
        time_grain : str | None
            时间聚合粒度。
        weights_col : str | None
            样本权重列名。
        batch_size : int
            批量评估时的特征批大小。
        trend_column_order : TrendColumnOrder
            趋势宽表中时间或分组取值列的展示顺序；``"asc"`` 保持从早到晚，
            ``"desc"`` 则从晚到早。``Total`` 列如存在会固定在最后。

        Returns
        -------
        MarsMonitoringReport
            单次监控结果对象。

        Raises
        ------
        ValueError
            当特征列、target 列或 target 取值不满足监控要求时抛出。
        """
        self._validate_trend_column_order(trend_column_order)
        working_df = self._ensure_polars_dataframe(df)
        if isinstance(working_df, pl.LazyFrame):
            working_df = working_df.collect()
        output_as_pandas = self._return_pandas

        missing_features = [feature for feature in features if feature not in working_df.columns]
        if missing_features:
            raise ValueError(f"features contain columns not found in df: {missing_features}")
        if target is not None and target not in working_df.columns:
            raise ValueError(f"target column '{target}' was not found in df. Use target=None for label-free monitoring.")

        evaluator = MarsBinEvaluator(
            binning_type=self.binning_type,
            binner_params=self.binner_params,
        )
        risk_profile = evaluator.evaluate(
            df,
            target=target,
            features=features,
            binner=binner,
            feature_data_source=feature_data_source,
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
            benchmark_df=benchmark_df,
            weights_col=weights_col,
            batch_size=batch_size,
        )

        prepared_df, resolved_group_col = self._prepare_monitoring_context(
            evaluator=evaluator,
            df=working_df,
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
        )
        if target is not None:
            prepared_df = MarsBinEvaluator._normalize_binary_target_column(prepared_df, target)

        detail_result = self._ensure_polars_dataframe(risk_profile.report.detail_table)
        detail_table = detail_result.collect() if isinstance(detail_result, pl.LazyFrame) else detail_result
        value_stats = self._build_bin_value_stats(
            df=prepared_df,
            binner=risk_profile.binner,
            features=features,
            group_col=resolved_group_col,
        )
        enriched_detail = self._enrich_detail_with_value_stats(
            detail_table=detail_table,
            value_stats=value_stats,
            group_col=resolved_group_col,
        )
        target_observation_table = self._build_target_observation_table(
            df=prepared_df,
            target=target,
            group_col=resolved_group_col,
        )
        summary_table = self._build_summary_table(
            summary_table_result=self._ensure_polars_dataframe(risk_profile.report.summary_table),
            target_observation_table=target_observation_table,
            target=target,
            group_col=resolved_group_col,
        )
        bin_stat_table = self._build_bin_stat_table(
            detail_table=enriched_detail,
            group_col=resolved_group_col,
        )
        bin_stat_trend_tables = self._build_bin_stat_trend_tables(
            detail_table=enriched_detail,
            group_col=resolved_group_col,
            trend_column_order=trend_column_order,
        )

        metadata = dict(risk_profile.metadata)
        metadata.update(
            {
                "monitoring_feature_count": len(features),
                "target": target,
                "psi_include_missing": self.psi_include_missing,
                "psi_include_special": self.psi_include_special,
                "trend_column_order": trend_column_order,
            }
        )
        if target_observation_table is not None:
            total_observation = target_observation_table.filter(
                pl.col(resolved_group_col) == "Total"
            )
            if not total_observation.is_empty():
                metadata["target_observed_rate"] = total_observation.select(
                    pl.col("target_observed_rate").first()
                ).item()

        trend_tables: dict[str, pl.DataFrame] = {}
        for name, table in risk_profile.report.trend_tables.items():
            trend_result = self._ensure_polars_dataframe(table)
            trend_table = trend_result.collect() if isinstance(trend_result, pl.LazyFrame) else trend_result
            trend_tables[name] = self._reorder_trend_table(
                trend_table,
                trend_column_order=trend_column_order,
            )

        missing_by_day_table = None
        if risk_profile.report.missing_by_day_table is not None:
            missing_result = self._ensure_polars_dataframe(risk_profile.report.missing_by_day_table)
            missing_by_day_table = (
                missing_result.collect() if isinstance(missing_result, pl.LazyFrame) else missing_result
            )
            missing_by_day_table = self._reorder_trend_table(
                missing_by_day_table,
                trend_column_order=trend_column_order,
            )

        metadata["trend_value_columns"] = self._collect_trend_value_columns(
            trend_tables=trend_tables,
            bin_stat_trend_tables=bin_stat_trend_tables,
            missing_by_day_table=missing_by_day_table,
            trend_column_order=trend_column_order,
        )

        self._return_pandas = output_as_pandas
        return MarsMonitoringReport(
            summary_table=self._format_output(summary_table),
            detail_table=self._format_output(enriched_detail),
            trend_tables=self._format_output(trend_tables),
            missing_by_day_table=self._format_output(missing_by_day_table),
            bin_stat_table=self._format_output(bin_stat_table),
            bin_stat_trend_tables=self._format_output(bin_stat_trend_tables),
            target_observation_table=self._format_output(target_observation_table),
            binner=risk_profile.binner,
            features=list(features),
            target=target,
            metadata=metadata,
        )

    @staticmethod
    def _prepare_monitoring_context(
        *,
        evaluator: MarsBinEvaluator,
        df: pl.DataFrame,
        group_col: str | None,
        time_col: str | None,
        time_grain: str | None,
    ) -> tuple[pl.DataFrame, str]:
        """复用评估器的分组解析逻辑，保证监控附表和风险评估口径一致。"""
        profile_by = evaluator._resolve_profile_by(
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
        )
        return evaluator._prepare_context(df, profile_by, time_col)

    def _build_bin_value_stats(
        self,
        *,
        df: pl.DataFrame,
        binner: MarsBinnerBase,
        features: List[str],
        group_col: str,
    ) -> pl.DataFrame:
        """按分箱和分组计算原始特征值的统计量。"""
        binned_result = self._ensure_polars_dataframe(binner.transform(df, return_type="index"))
        binned_df = binned_result.collect() if isinstance(binned_result, pl.LazyFrame) else binned_result
        frames: list[pl.DataFrame] = []
        for feature in features:
            bin_col = f"{feature}_bin"
            if feature not in df.columns or bin_col not in binned_df.columns:
                continue

            source = df.select([group_col, feature]).hstack(binned_df.select(bin_col))
            group_stats = self._aggregate_feature_value_stats(
                source=source,
                feature=feature,
                bin_col=bin_col,
                group_col=group_col,
            )
            total_stats = (
                self._aggregate_feature_value_stats(
                    source=source.with_columns(pl.lit("Total").alias(group_col)),
                    feature=feature,
                    bin_col=bin_col,
                    group_col=group_col,
                )
            )
            frames.extend([group_stats, total_stats])

        if not frames:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    group_col: pl.String,
                    "bin_index": pl.Int16,
                    "mean": pl.Float64,
                    "min": pl.Float64,
                    "max": pl.Float64,
                    "median": pl.Float64,
                }
            )
        return pl.concat(frames, how="vertical_relaxed")

    @staticmethod
    def _aggregate_feature_value_stats(
        *,
        source: pl.DataFrame,
        feature: str,
        bin_col: str,
        group_col: str,
    ) -> pl.DataFrame:
        """计算单个特征在一个分组层级下的分箱原始值统计。"""
        dtype = source.schema[feature]
        if dtype.is_numeric():
            value_expr = pl.col(feature).cast(pl.Float64)
            agg_exprs = [
                value_expr.mean().alias("mean"),
                value_expr.min().alias("min"),
                value_expr.max().alias("max"),
                value_expr.median().alias("median"),
            ]
        else:
            agg_exprs = [
                pl.lit(None).cast(pl.Float64).alias("mean"),
                pl.lit(None).cast(pl.Float64).alias("min"),
                pl.lit(None).cast(pl.Float64).alias("max"),
                pl.lit(None).cast(pl.Float64).alias("median"),
            ]

        return (
            source
            .group_by([group_col, bin_col])
            .agg(agg_exprs)
            .rename({bin_col: "bin_index"})
            .with_columns([
                pl.lit(feature).alias("feature"),
                pl.col(group_col).cast(pl.String),
                pl.col("bin_index").cast(pl.Int16),
            ])
            .select(["feature", group_col, "bin_index", "mean", "min", "max", "median"])
        )

    @staticmethod
    def _enrich_detail_with_value_stats(
        *,
        detail_table: pl.DataFrame,
        value_stats: pl.DataFrame,
        group_col: str,
    ) -> pl.DataFrame:
        """把原始值统计量并入分箱评估明细表。"""
        if value_stats.is_empty():
            return detail_table
        return detail_table.join(
            value_stats,
            on=["feature", group_col, "bin_index"],
            how="left",
        )

    @staticmethod
    def _build_target_observation_table(
        *,
        df: pl.DataFrame,
        target: str | None,
        group_col: str,
    ) -> pl.DataFrame | None:
        """构建 target 表现覆盖情况表。"""
        if target is None:
            return None

        agg_exprs = [
            pl.len().alias("sample_count"),
            pl.col(target).is_not_null().sum().alias("target_observed_count"),
            pl.col(target).is_null().sum().alias("target_unobserved_count"),
            pl.col(target).fill_null(0).cast(pl.Float64).sum().alias("bad"),
        ]
        group_table = df.group_by(group_col).agg(agg_exprs).with_columns(pl.col(group_col).cast(pl.String))
        total_table = (
            df
            .with_columns(pl.lit("Total").alias(group_col))
            .group_by(group_col)
            .agg(agg_exprs)
        )
        return (
            pl.concat([total_table, group_table], how="vertical_relaxed")
            .with_columns([
                pl.when(pl.col("sample_count") > 0)
                .then(pl.col("target_observed_count") / pl.col("sample_count"))
                .otherwise(None)
                .alias("target_observed_rate"),
                pl.when(pl.col("target_observed_count") > 0)
                .then(pl.col("bad") / pl.col("target_observed_count"))
                .otherwise(None)
                .alias("bad_rate_observed"),
            ])
            .sort(group_col)
        )

    @staticmethod
    def _build_summary_table(
        *,
        summary_table_result: pl.DataFrame | pl.LazyFrame,
        target_observation_table: pl.DataFrame | None,
        target: str | None,
        group_col: str,
    ) -> pl.DataFrame:
        """在特征汇总表中补充全局 target 表现覆盖率。"""
        summary_table = (
            summary_table_result.collect()
            if isinstance(summary_table_result, pl.LazyFrame)
            else summary_table_result
        )
        if target is None or target_observation_table is None or target_observation_table.is_empty():
            return summary_table

        total_observation = target_observation_table.filter(pl.col(group_col) == "Total")
        if total_observation.is_empty():
            return summary_table

        observed_rate = total_observation.select(pl.col("target_observed_rate").first()).item()
        return summary_table.with_columns(pl.lit(observed_rate).alias("target_observed_rate"))

    @staticmethod
    def _build_bin_stat_table(
        *,
        detail_table: pl.DataFrame,
        group_col: str,
    ) -> pl.DataFrame:
        """提取全量层级的分箱统计表。"""
        return (
            detail_table
            .filter((pl.col(group_col) == "Total") & (pl.col("bin_index") != 9999))
            .sort(["feature", "bin_index"])
        )

    def _build_bin_stat_trend_tables(
        self,
        *,
        detail_table: pl.DataFrame,
        group_col: str,
        trend_column_order: TrendColumnOrder,
    ) -> dict[str, pl.DataFrame]:
        """把分箱统计明细展开为按时间或分组的趋势宽表。"""
        trend_source = detail_table.filter(
            (pl.col(group_col) != "Total") & (pl.col("bin_index") != 9999)
        )
        if trend_source.is_empty():
            return {}

        index_cols = ["feature", "bin_index", "bin_label", "bin_type"]
        trend_tables: dict[str, pl.DataFrame] = {}
        for metric in self.bin_stat_metrics:
            if metric not in trend_source.columns:
                continue
            pivot_df = (
                trend_source
                .pivot(index=index_cols, on=group_col, values=metric)
                .sort(["feature", "bin_index"])
                .with_columns(pl.lit("Float64").alias("dtype"))
            )
            value_cols = [col for col in pivot_df.columns if col not in {*index_cols, "dtype"}]
            sorted_value_cols = self._order_trend_value_columns(
                value_cols,
                trend_column_order=trend_column_order,
            )
            trend_tables[metric] = pivot_df.select(index_cols + ["dtype"] + sorted_value_cols)

        return trend_tables

    @staticmethod
    def _validate_trend_column_order(trend_column_order: str) -> None:
        """校验趋势列排序方向。"""
        if trend_column_order not in {"asc", "desc"}:
            raise ValueError("trend_column_order must be one of {'asc', 'desc'}.")

    @classmethod
    def _order_trend_value_columns(
        cls,
        columns: list[str],
        *,
        trend_column_order: TrendColumnOrder,
    ) -> list[str]:
        """按自然顺序排列趋势取值列，并把 Total 固定在最后。"""
        value_cols = [col for col in columns if col != "Total"]
        ordered_cols = sorted(
            value_cols,
            key=cls._natural_sort_key,
            reverse=trend_column_order == "desc",
        )
        if "Total" in columns:
            ordered_cols.append("Total")
        return ordered_cols

    @classmethod
    def _reorder_trend_table(
        cls,
        table: pl.DataFrame,
        *,
        trend_column_order: TrendColumnOrder,
    ) -> pl.DataFrame:
        """重排趋势宽表列，保持元数据列在前、取值列按配置排序。"""
        metadata_cols = [
            col
            for col in ["feature", "bin_index", "bin_label", "bin_type", "dtype"]
            if col in table.columns
        ]
        metadata_set = set(metadata_cols)
        value_cols = [col for col in table.columns if col not in metadata_set]
        ordered_value_cols = cls._order_trend_value_columns(
            value_cols,
            trend_column_order=trend_column_order,
        )
        return table.select(metadata_cols + ordered_value_cols)

    @classmethod
    def _collect_trend_value_columns(
        cls,
        *,
        trend_tables: dict[str, pl.DataFrame],
        bin_stat_trend_tables: dict[str, pl.DataFrame],
        missing_by_day_table: pl.DataFrame | None,
        trend_column_order: TrendColumnOrder,
    ) -> list[str]:
        """收集本次 report 中可用于报警顺序识别的趋势取值列。"""
        value_cols: set[str] = set()
        for table in [*trend_tables.values(), *bin_stat_trend_tables.values()]:
            value_cols.update(cls._trend_value_columns(table))
        if missing_by_day_table is not None:
            value_cols.update(cls._trend_value_columns(missing_by_day_table))
        return cls._order_trend_value_columns(
            list(value_cols),
            trend_column_order=trend_column_order,
        )

    @staticmethod
    def _trend_value_columns(table: pl.DataFrame) -> list[str]:
        """识别趋势宽表中的时间或分组取值列，不包含 Total。"""
        metadata_cols = {"feature", "dtype", "bin_index", "bin_label", "bin_type", "Total"}
        return [col for col in table.columns if col not in metadata_cols]

    @staticmethod
    def _natural_sort_key(value: str) -> list[tuple[int, int | str]]:
        """生成适合时间标签和普通分组标签的自然排序键。"""
        parts = re.split(r"(\d+)", str(value))
        key: list[tuple[int, int | str]] = []
        for part in parts:
            if part.isdigit():
                key.append((0, int(part)))
            elif part:
                key.append((1, part))
        return key
