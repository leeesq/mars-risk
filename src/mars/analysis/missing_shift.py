"""静态训练数据的缺失率异常扫描与展示。"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from math import erfc, log, sqrt
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame
from mars.compute.missing import is_numeric_dtype, missing_condition_expr
from mars.core.base import MarsBaseEstimator
from mars.core.constants import FLOAT_TOLERANCE
from mars.utils.date import MarsDate

AnomalyType = Literal["segment_shift", "boundary", "point", "high_level"]
Direction = Literal["increase", "decrease"]

_SUPPORTED_DETECTORS: tuple[AnomalyType, ...] = (
    "segment_shift",
    "boundary",
    "point",
    "high_level",
)
_DETECTOR_PRIORITY: dict[AnomalyType, int] = {
    "high_level": 0,
    "boundary": 1,
    "point": 2,
    "segment_shift": 3,
}


@dataclass(frozen=True)
class MarsMissingShiftConfig:
    """
    配置缺失率异常扫描的检测器、统计门槛与业务红线。

    Parameters
    ----------
    enabled_detectors : tuple[AnomalyType, ...]
        启用的检测器。默认同时检测分段变化、边界异常、单日异常和持续高缺失。
    min_period_samples : int
        单个时间切片参与检测所需的最小样本数，默认 ``30``。
    min_segment_size : int
        分段变点两侧各自所需的最少有效时间切片数，默认 ``3``。
    reference_window : int
        边界和单日事件使用的相邻参考切片数，默认 ``3``。
    max_boundary_periods : int
        开头或结尾边界事件最多覆盖的切片数，默认 ``3``。
    max_segment_candidates : int
        每个连续有效区间保留的 Binseg 候选上限，默认 ``5``。
    min_abs_delta : float
        缺失率绝对变化门槛，默认 ``0.03``。
    min_effect_delta : float
        进入统计确认的最小有效变化，默认 ``0.005``。
    min_relative_delta : float
        相对缺失率变化门槛，默认 ``0.30``。
    fdr_q_threshold : float
        Benjamini-Hochberg 全局 FDR 的 q 值门槛，默认 ``0.05``。
    high_missing_rate_threshold : float
        持续高缺失检测的全局缺失率红线，默认 ``0.90``。
    feature_high_missing_rate_thresholds : Mapping[str, float]
        按特征覆盖持续高缺失红线的映射，默认空映射。
    min_high_periods : int
        合并为持续高缺失事件所需的连续切片数，默认 ``2``。

    Examples
    --------
    >>> config = MarsMissingShiftConfig(min_period_samples=50)
    >>> config.high_missing_rate_threshold
    0.9
    """

    enabled_detectors: tuple[AnomalyType, ...] = _SUPPORTED_DETECTORS
    min_period_samples: int = 30
    min_segment_size: int = 3
    reference_window: int = 3
    max_boundary_periods: int = 3
    max_segment_candidates: int = 5
    min_abs_delta: float = 0.03
    min_effect_delta: float = 0.005
    min_relative_delta: float = 0.30
    fdr_q_threshold: float = 0.05
    high_missing_rate_threshold: float = 0.90
    feature_high_missing_rate_thresholds: Mapping[str, float] = field(default_factory=dict)
    min_high_periods: int = 2


@dataclass(frozen=True)
class _WindowStats:
    """保存事件窗口或参考窗口的缺失计数。"""

    missing_count: int
    total_count: int

    @property
    def rate(self) -> float:
        """返回窗口缺失率。"""
        return self.missing_count / self.total_count


@dataclass(frozen=True)
class _Candidate:
    """保存一个检测器生成的内部异常候选。"""

    feature: str
    data_source: str
    anomaly_type: AnomalyType
    event_start_idx: int
    event_end_idx: int
    event_stats: _WindowStats
    reference_type: str
    reference_start_period: str | None
    reference_end_period: str | None
    reference_stats: _WindowStats | None
    threshold: float | None
    delta: float
    abs_delta: float
    relative_delta: float
    p_value: float | None
    q_value: float | None
    test_method: str | None


def _empty_detail_table() -> pl.DataFrame:
    """构造固定 schema 的空异常明细表。"""
    return pl.DataFrame(
        schema={
            "feature": pl.String,
            "data_source": pl.String,
            "anomaly_type": pl.String,
            "detected_by": pl.String,
            "event_start_period": pl.String,
            "event_end_period": pl.String,
            "reference_type": pl.String,
            "reference_start_period": pl.String,
            "reference_end_period": pl.String,
            "event_missing_count": pl.Int64,
            "event_total_count": pl.Int64,
            "event_missing_rate": pl.Float64,
            "reference_missing_count": pl.Int64,
            "reference_total_count": pl.Int64,
            "reference_missing_rate": pl.Float64,
            "threshold": pl.Float64,
            "delta": pl.Float64,
            "abs_delta": pl.Float64,
            "relative_delta": pl.Float64,
            "p_value": pl.Float64,
            "q_value": pl.Float64,
            "direction": pl.String,
            "reason": pl.String,
        }
    )


def _empty_summary_table() -> pl.DataFrame:
    """构造固定 schema 的空特征汇总表。"""
    return pl.DataFrame(
        schema={
            "feature": pl.String,
            "data_source": pl.String,
            "anomaly_count": pl.Int64,
            "anomaly_types": pl.String,
            "max_abs_delta": pl.Float64,
            "max_event_missing_rate": pl.Float64,
            "first_anomaly_period": pl.String,
            "last_anomaly_period": pl.String,
            "primary_direction": pl.String,
        }
    )


def _empty_source_table() -> pl.DataFrame:
    """构造固定 schema 的空数据源汇总表。"""
    return pl.DataFrame(
        schema={
            "data_source": pl.String,
            "feature_count": pl.Int64,
            "anomaly_feature_count": pl.Int64,
            "anomaly_feature_rate": pl.Float64,
            "anomaly_count": pl.Int64,
            "max_abs_delta": pl.Float64,
        }
    )


@dataclass(frozen=True)
class MarsMissingShiftResult:
    """
    保存缺失率异常扫描的结构化结果与 Notebook 展示能力。

    Parameters
    ----------
    summary_table : pl.DataFrame
        特征级异常汇总表。
    detail_table : pl.DataFrame
        统一事件 schema 的异常明细表。
    source_table : pl.DataFrame
        数据源级异常汇总表；未提供数据源映射时为空表。
    trend_table : pl.DataFrame
        ``feature × period`` 粒度的缺失计数、缺失率和检测资格长表。
    config : MarsMissingShiftConfig
        生成当前结果所使用的扫描配置。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {"dt": ["2026-01-01"] * 30, "x": [None] + list(range(29))}
    ... )
    >>> result = MarsMissingShiftScanner().scan(df, date_col="dt", features=["x"])
    >>> result.trend_table.height
    1
    """

    summary_table: pl.DataFrame
    detail_table: pl.DataFrame
    source_table: pl.DataFrame
    trend_table: pl.DataFrame
    config: MarsMissingShiftConfig

    def show_summary(
        self,
        features: str | Sequence[str] | None = None,
        *,
        sort_by: str = "max_abs_delta",
        sort_ascending: bool = False,
    ) -> pd.io.formats.style.Styler:
        """
        返回适合 Notebook 展示的特征级异常汇总表。

        Parameters
        ----------
        features : str | Sequence[str] | None
            需要展示的特征；默认展示全部异常特征。
        sort_by : str
            排序列，默认 ``"max_abs_delta"``。
        sort_ascending : bool
            是否升序排序，默认 ``False``。

        Returns
        -------
        pd.io.formats.style.Styler
            带缺失率和变化幅度格式的汇总表。

        Raises
        ------
        ValueError
            当指定不存在的排序列时抛出。

        Examples
        --------
        >>> hasattr(result.show_summary(), "to_html")
        True
        """
        frame = _filter_pandas_features(to_pandas_frame(self.summary_table), features)
        if sort_by not in frame.columns:
            raise ValueError(f"Summary sort column '{sort_by}' was not found.")
        frame = frame.sort_values(sort_by, ascending=sort_ascending)
        return _style_missing_shift_table(
            frame,
            caption="Missing Shift Summary",
            gradient_columns=["max_abs_delta", "max_event_missing_rate"],
        )

    def show_detail(
        self,
        features: str | Sequence[str] | None = None,
        *,
        anomaly_types: str | Sequence[str] | None = None,
    ) -> pd.io.formats.style.Styler:
        """
        返回适合 Notebook 展示的异常事件明细表。

        Parameters
        ----------
        features : str | Sequence[str] | None
            需要展示的特征；默认展示全部异常特征。
        anomaly_types : str | Sequence[str] | None
            需要展示的事件类型；默认展示全部类型。

        Returns
        -------
        pd.io.formats.style.Styler
            带变化幅度和显著性格式的明细表。

        Examples
        --------
        >>> hasattr(result.show_detail(), "to_html")
        True
        """
        frame = _filter_pandas_features(to_pandas_frame(self.detail_table), features)
        requested_types = _normalize_names(anomaly_types)
        if requested_types is not None:
            frame = frame[frame["anomaly_type"].isin(requested_types)]
        if not frame.empty:
            frame = frame.sort_values(
                ["abs_delta", "event_start_period"],
                ascending=[False, True],
            )
        return _style_missing_shift_table(
            frame,
            caption="Missing Shift Detail",
            gradient_columns=["abs_delta", "event_missing_rate"],
        )

    def show_trend(
        self,
        features: str | Sequence[str] | None = None,
        *,
        max_features: int = 20,
        group_ascending: bool = True,
    ) -> pd.io.formats.style.Styler:
        """
        将长版日缺失率按需透视为 Notebook 趋势热力表。

        Parameters
        ----------
        features : str | Sequence[str] | None
            需要展示的特征。默认按最大异常幅度选择异常特征。
        max_features : int
            未显式指定特征时最多展示的特征数，默认 ``20``。
        group_ascending : bool
            日期列是否按升序排列，默认 ``True``。

        Returns
        -------
        pd.io.formats.style.Styler
            异常单元格标红、低样本单元格标灰的缺失率趋势表。

        Examples
        --------
        >>> hasattr(result.show_trend(features="x"), "to_html")
        True
        """
        selected = self._resolve_display_features(features, max_features=max_features)
        trend = to_pandas_frame(self.trend_table)
        filtered = trend[trend["feature"].isin(selected)].copy()
        period_order = sorted(filtered["period"].unique(), reverse=not group_ascending)
        if filtered.empty:
            empty = pd.DataFrame(columns=["feature", "dtype", "data_source"])
            return _style_missing_shift_table(empty, caption="Missing Rate Trend")

        pivot = (
            filtered
            .pivot(
                index=["feature", "dtype", "data_source"],
                columns="period",
                values="missing_rate",
            )
            .reindex(columns=period_order)
            .reset_index()
        )
        status = filtered.set_index(["feature", "period"])[
            ["is_detection_eligible", "is_anomaly"]
        ].to_dict("index")
        date_columns = [column for column in pivot.columns if column not in _TREND_META_COLUMNS]

        def style_cells(frame: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
            for row_idx, row in frame.iterrows():
                feature = str(row["feature"])
                for period in date_columns:
                    cell_status = status.get((feature, period), {})
                    if cell_status.get("is_anomaly"):
                        styles.at[row_idx, period] = "background-color: #f8696b; color: #7f0000"
                    elif not cell_status.get("is_detection_eligible", True):
                        styles.at[row_idx, period] = "background-color: #d9d9d9; color: #666666"
            return styles

        styler = _style_missing_shift_table(
            pivot,
            caption="Missing Rate Trend",
            percentage_columns=date_columns,
        )
        return styler.apply(style_cells, axis=None)

    def plot_trends(
        self,
        features: str | Sequence[str] | None = None,
        *,
        max_features: int = 12,
        columns: int = 2,
        figsize_per_plot: tuple[float, float] = (6.0, 3.2),
    ) -> Any:
        """
        绘制缺失率趋势、业务红线、异常区间和低样本日期。

        Parameters
        ----------
        features : str | Sequence[str] | None
            需要绘制的特征。默认按异常幅度选择异常特征。
        max_features : int
            未显式指定特征时最多绘制的特征数，默认 ``12``。
        columns : int
            子图列数，默认 ``2``。
        figsize_per_plot : tuple[float, float]
            每个子图的宽度和高度，默认 ``(6.0, 3.2)``。

        Returns
        -------
        matplotlib.figure.Figure
            包含全部选中特征趋势的 Matplotlib Figure。

        Raises
        ------
        ValueError
            当没有可绘制特征或图形参数非法时抛出。

        Examples
        --------
        >>> figure = result.plot_trends(features="x")
        >>> len(figure.axes)
        1
        """
        if columns < 1:
            raise ValueError("`columns` must be >= 1.")
        selected = self._resolve_display_features(features, max_features=max_features)
        if not selected:
            raise ValueError("No anomaly features are available; pass `features` explicitly.")

        import matplotlib.pyplot as plt

        trend = to_pandas_frame(self.trend_table)
        detail = to_pandas_frame(self.detail_table)
        row_count = int(np.ceil(len(selected) / columns))
        figure, axes = plt.subplots(
            row_count,
            columns,
            figsize=(figsize_per_plot[0] * columns, figsize_per_plot[1] * row_count),
            squeeze=False,
        )
        colors = {
            "segment_shift": "#f4a261",
            "boundary": "#e76f51",
            "point": "#9b5de5",
            "high_level": "#d00000",
        }
        for plot_idx, feature in enumerate(selected):
            axis = axes.flat[plot_idx]
            feature_trend = trend[trend["feature"] == feature].sort_values("period")
            periods = feature_trend["period"].astype(str).tolist()
            positions = np.arange(len(periods))
            rates = feature_trend["missing_rate"].astype(float).to_numpy()
            axis.plot(positions, rates, marker="o", linewidth=1.5, color="#2a6fbb")

            excluded = ~feature_trend["is_detection_eligible"].astype(bool).to_numpy()
            if excluded.any():
                axis.scatter(positions[excluded], rates[excluded], color="#8c8c8c", zorder=4)

            threshold = self.config.feature_high_missing_rate_thresholds.get(
                feature,
                self.config.high_missing_rate_threshold,
            )
            axis.axhline(threshold, color="#d00000", linestyle="--", linewidth=1.0)

            period_positions = {period: idx for idx, period in enumerate(periods)}
            events = detail[detail["feature"] == feature]
            for _, event in events.iterrows():
                start = period_positions.get(str(event["event_start_period"]))
                end = period_positions.get(str(event["event_end_period"]))
                if start is None or end is None:
                    continue
                event_type = cast(AnomalyType, str(event["anomaly_type"]))
                axis.axvspan(
                    start - 0.35,
                    end + 0.35,
                    color=colors[event_type],
                    alpha=0.16,
                )

            axis.set_title(feature)
            axis.set_ylim(-0.02, 1.02)
            axis.set_ylabel("Missing rate")
            tick_step = max(1, len(periods) // 8)
            axis.set_xticks(positions[::tick_step], periods[::tick_step], rotation=45, ha="right")
            axis.grid(axis="y", alpha=0.2)

        for axis in axes.flat[len(selected):]:
            figure.delaxes(axis)
        figure.tight_layout()
        return figure

    def write_excel(self, path: str | Path) -> None:
        """
        将四张缺失率异常表导出为带格式的 Excel 工作簿。

        Parameters
        ----------
        path : str | Path
            输出文件路径；父目录必须已经存在。

        Returns
        -------
        None
            方法只写出文件，不返回对象。

        Raises
        ------
        FileNotFoundError
            当输出父目录不存在时抛出。
        RuntimeError
            当 Excel 渲染或写入失败时抛出。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     result.write_excel(Path(tmp) / "missing_shift.xlsx")
        """
        output_path = Path(path)
        if output_path.parent and not output_path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {output_path.parent}")

        tables = {
            "summary": self.summary_table,
            "detail": self.detail_table,
            "source": self.source_table,
            "trend": self.trend_table,
        }
        try:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                workbook = writer.book
                header_format = workbook.add_format(
                    {"bold": True, "bg_color": "#D9EAF7", "border": 1}
                )
                percent_format = workbook.add_format({"num_format": "0.00%"})
                for sheet_name, table in tables.items():
                    frame = to_pandas_frame(table)
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    worksheet.freeze_panes(1, 0)
                    if len(frame.columns) > 0:
                        worksheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
                    for column_idx, column in enumerate(frame.columns):
                        max_value_width = max(
                            (len(str(value)) for value in frame[column].tolist()),
                            default=0,
                        )
                        width = min(max(max_value_width, len(str(column))) + 2, 42)
                        cell_format = (
                            percent_format
                            if "rate" in str(column) or "relative_delta" == column
                            else None
                        )
                        worksheet.set_column(column_idx, column_idx, width, cell_format)
                        worksheet.write(0, column_idx, column, header_format)
                    if "abs_delta" in frame.columns and not frame.empty:
                        delta_col = frame.columns.get_loc("abs_delta")
                        worksheet.conditional_format(
                            1,
                            delta_col,
                            len(frame),
                            delta_col,
                            {"type": "3_color_scale"},
                        )
                    if sheet_name == "trend" and not frame.empty:
                        eligible_col = frame.columns.get_loc("is_detection_eligible")
                        worksheet.conditional_format(
                            1,
                            0,
                            len(frame),
                            len(frame.columns) - 1,
                            {
                                "type": "formula",
                                "criteria": f"=${_excel_column_name(eligible_col)}2=FALSE",
                                "format": workbook.add_format({"bg_color": "#D9D9D9"}),
                            },
                        )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Failed to export missing shift Excel to '{output_path}'.") from exc

    def _resolve_display_features(
        self,
        features: str | Sequence[str] | None,
        *,
        max_features: int,
    ) -> list[str]:
        """解析展示特征，并对显式未知特征执行严格校验。"""
        if max_features < 1:
            raise ValueError("`max_features` must be >= 1.")
        known_features = self.trend_table.get_column("feature").unique().to_list()
        requested = _normalize_names(features)
        if requested is not None:
            unknown = [feature for feature in requested if feature not in known_features]
            if unknown:
                raise ValueError(f"Trend features not found: {unknown}")
            return requested
        if self.summary_table.is_empty():
            return []
        selected = (
            self.summary_table
            .sort("max_abs_delta", descending=True)
            .head(max_features)
            .get_column("feature")
            .to_list()
        )
        return [str(feature) for feature in selected]


class MarsMissingShiftScanner(MarsBaseEstimator):
    """
    扫描静态训练宽表中的日级或其他时间粒度缺失率异常。

    该实验 API 同时检测持续分段变化、首尾边界异常、内部单日异常和持续高缺失。
    统计型候选统一通过业务效果门槛与全局 FDR 确认；结果只用于数据质量复核，
    不会自动删除特征或阻断后续建模。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis.missing_shift import MarsMissingShiftScanner
    >>> df = pl.DataFrame(
    ...     {"dt": ["2026-01-01"] * 30, "x": [None] + list(range(29))}
    ... )
    >>> result = MarsMissingShiftScanner().scan(df, date_col="dt", features=["x"])
    >>> result.trend_table.get_column("feature").to_list()
    ['x']
    """

    def scan(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        date_col: str,
        features: list[str] | None = None,
        time_grain: str = "1d",
        missing_values: list[Any] | None = None,
        feature_data_source: dict[str, str] | None = None,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
        config: MarsMissingShiftConfig | None = None,
    ) -> MarsMissingShiftResult:
        """
        按时间切片扫描缺失率异常并生成结构化报告。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待扫描的当前数据宽表。
        date_col : str
            日期列名，按 ``time_grain`` 聚合。
        features : list[str] | None
            待扫描特征；默认扫描除 ``date_col`` 外的全部列。
        time_grain : str
            时间粒度，默认 ``"1d"``，复用 ``MarsDate.from_grain`` 的格式。
        missing_values : list[Any] | None
            除 null/NaN 外需要按缺失处理的业务占位值。
        feature_data_source : dict[str, str] | None
            特征到数据源的映射，用于趋势和数据源汇总。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            可选历史基准。提供时首尾边界优先与其聚合缺失率比较。
        config : MarsMissingShiftConfig | None
            检测配置；默认使用 ``MarsMissingShiftConfig()``。

        Returns
        -------
        MarsMissingShiftResult
            包含汇总、事件明细、数据源汇总、长版趋势和配置的报告对象。

        Raises
        ------
        ValueError
            当日期、特征、benchmark 或配置不满足输入契约时抛出。

        Examples
        --------
        >>> config = MarsMissingShiftConfig(min_period_samples=2)
        >>> df = pl.DataFrame(
        ...     {"dt": ["2026-01-01", "2026-01-01"], "x": [None, 1]}
        ... )
        >>> result = MarsMissingShiftScanner().scan(
        ...     df, date_col="dt", features=["x"], config=config
        ... )
        >>> result.trend_table.get_column("missing_rate").to_list()
        [0.5]
        """
        scan_config = config or MarsMissingShiftConfig()
        working_df = cast(pl.DataFrame, self._ensure_polars_dataframe(df))
        target_features = self._resolve_features(working_df, date_col, features)
        self._validate_config(scan_config, target_features)
        source_map = feature_data_source or {}

        period_col = "__mars_missing_shift_period"
        period_df = (
            working_df
            .with_columns(MarsDate.from_grain(date_col, time_grain).cast(pl.String).alias(period_col))
            .filter(pl.col(period_col).is_not_null())
        )
        if period_df.is_empty():
            raise ValueError(f"Date column '{date_col}' cannot be parsed into valid periods.")

        trend_table = self._build_trend_table(
            period_df,
            features=target_features,
            period_col=period_col,
            missing_values=missing_values,
            feature_data_source=source_map,
            min_period_samples=scan_config.min_period_samples,
        )
        benchmark_stats = self._build_benchmark_stats(
            benchmark_df,
            current_df=working_df,
            features=target_features,
            missing_values=missing_values,
            min_samples=scan_config.min_period_samples,
        )
        detail_table = self._scan_anomalies(
            trend_table,
            benchmark_stats=benchmark_stats,
            config=scan_config,
        )
        annotated_trend = self._annotate_trend(trend_table, detail_table)
        summary_table = self._build_summary_table(detail_table)
        source_table = self._build_source_table(
            detail_table,
            target_features=target_features,
            feature_data_source=source_map,
        )
        return MarsMissingShiftResult(
            summary_table=summary_table,
            detail_table=detail_table,
            source_table=source_table,
            trend_table=annotated_trend,
            config=scan_config,
        )

    @staticmethod
    def _validate_config(config: MarsMissingShiftConfig, features: list[str]) -> None:
        """校验配置值、检测器名称和特征级覆盖。"""
        unknown_detectors = [
            detector for detector in config.enabled_detectors if detector not in _SUPPORTED_DETECTORS
        ]
        if unknown_detectors:
            raise ValueError(f"Unsupported missing shift detectors: {unknown_detectors}")
        if not config.enabled_detectors:
            raise ValueError("`enabled_detectors` must contain at least one detector.")

        positive_integer_params = {
            "min_period_samples": config.min_period_samples,
            "min_segment_size": config.min_segment_size,
            "reference_window": config.reference_window,
            "max_boundary_periods": config.max_boundary_periods,
            "max_segment_candidates": config.max_segment_candidates,
            "min_high_periods": config.min_high_periods,
        }
        for name, value in positive_integer_params.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"`{name}` must be an integer >= 1.")
        if config.min_segment_size < 2:
            raise ValueError("`min_segment_size` must be >= 2.")

        probability_params = {
            "min_abs_delta": config.min_abs_delta,
            "min_effect_delta": config.min_effect_delta,
            "fdr_q_threshold": config.fdr_q_threshold,
            "high_missing_rate_threshold": config.high_missing_rate_threshold,
        }
        for name, value in probability_params.items():
            if not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be between 0 and 1.")
        if config.min_relative_delta < 0:
            raise ValueError("`min_relative_delta` must be >= 0.")

        unknown_overrides = sorted(
            set(config.feature_high_missing_rate_thresholds).difference(features)
        )
        if unknown_overrides:
            raise ValueError(f"High-missing threshold features not found: {unknown_overrides}")
        for feature, threshold in config.feature_high_missing_rate_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(
                    f"High-missing threshold for feature '{feature}' must be between 0 and 1."
                )

    @staticmethod
    def _resolve_features(
        df: pl.DataFrame,
        date_col: str,
        features: list[str] | None,
    ) -> list[str]:
        """解析并严格校验待扫描特征。"""
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' was not found.")
        resolved = (
            [column for column in df.columns if column != date_col]
            if features is None
            else [feature for feature in features if feature != date_col]
        )
        missing_features = [feature for feature in resolved if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        if not resolved:
            raise ValueError("No features are available for missing shift scanning.")
        if len(resolved) != len(set(resolved)):
            raise ValueError("`features` must not contain duplicates.")
        return resolved

    @staticmethod
    def _build_trend_table(
        df: pl.DataFrame,
        *,
        features: list[str],
        period_col: str,
        missing_values: list[Any] | None,
        feature_data_source: dict[str, str],
        min_period_samples: int,
    ) -> pl.DataFrame:
        """单次聚合生成特征日级缺失计数与检测资格长表。"""
        schema = df.schema
        missing_exprs = [
            missing_condition_expr(
                feature,
                dtype=schema.get(feature),
                missing_values=missing_values,
            ).sum().alias(feature)
            for feature in features
        ]
        grouped = (
            df
            .group_by(period_col)
            .agg([pl.len().alias("__total_count"), *missing_exprs])
            .sort(period_col)
        )
        feature_frames: list[pl.DataFrame] = []
        for feature in features:
            feature_frame = (
                grouped
                .select(
                    [
                        pl.lit(feature).alias("feature"),
                        pl.lit(str(schema[feature])).alias("dtype"),
                        pl.lit(feature_data_source.get(feature, "UNMAPPED")).alias(
                            "data_source"
                        ),
                        pl.col(period_col).cast(pl.String).alias("period"),
                        pl.col(feature).cast(pl.Int64).alias("missing_count"),
                        pl.col("__total_count").cast(pl.Int64).alias("total_count"),
                    ]
                )
                .with_columns(
                    [
                        (pl.col("missing_count") / pl.col("total_count")).alias("missing_rate"),
                        (pl.col("total_count") >= min_period_samples).alias(
                            "is_detection_eligible"
                        ),
                        pl.when(pl.col("total_count") < min_period_samples)
                        .then(pl.lit("period_sample_count_below_minimum"))
                        .otherwise(None)
                        .cast(pl.String)
                        .alias("exclusion_reason"),
                    ]
                )
            )
            feature_frames.append(feature_frame)
        return pl.concat(feature_frames, how="vertical_relaxed").sort(["feature", "period"])

    def _build_benchmark_stats(
        self,
        benchmark_df: pl.DataFrame | pd.DataFrame | None,
        *,
        current_df: pl.DataFrame,
        features: list[str],
        missing_values: list[Any] | None,
        min_samples: int,
    ) -> dict[str, _WindowStats] | None:
        """校验显式 benchmark 并聚合每个特征的缺失计数。"""
        if benchmark_df is None:
            return None
        benchmark = cast(pl.DataFrame, self._ensure_polars_dataframe(benchmark_df))
        if benchmark.is_empty():
            raise ValueError("`benchmark_df` must not be empty.")
        if benchmark.height < min_samples:
            raise ValueError(
                "`benchmark_df` must contain at least "
                f"{min_samples} rows; got {benchmark.height}."
            )
        missing_features = [feature for feature in features if feature not in benchmark.columns]
        if missing_features:
            raise ValueError(f"`benchmark_df` features not found: {missing_features}")

        incompatible = [
            feature
            for feature in features
            if not _dtypes_compatible(current_df.schema[feature], benchmark.schema[feature])
        ]
        if incompatible:
            raise ValueError(f"`benchmark_df` feature dtypes are incompatible: {incompatible}")

        benchmark_exprs = [
            missing_condition_expr(
                feature,
                dtype=benchmark.schema[feature],
                missing_values=missing_values,
            ).sum().alias(feature)
            for feature in features
        ]
        counts = benchmark.select(benchmark_exprs).row(0, named=True)
        return {
            feature: _WindowStats(
                missing_count=int(counts[feature]),
                total_count=benchmark.height,
            )
            for feature in features
        }

    def _scan_anomalies(
        self,
        trend_table: pl.DataFrame,
        *,
        benchmark_stats: dict[str, _WindowStats] | None,
        config: MarsMissingShiftConfig,
    ) -> pl.DataFrame:
        """运行全部启用检测器，并完成 FDR、效果门槛和事件合并。"""
        statistical_candidates: list[_Candidate] = []
        high_level_candidates: list[_Candidate] = []
        for partition_key, feature_df in trend_table.partition_by("feature", as_dict=True).items():
            feature = str(partition_key[0] if isinstance(partition_key, tuple) else partition_key)
            ordered = feature_df.sort("period").with_row_index("__idx")
            eligible_blocks = _eligible_blocks(ordered)
            source = str(ordered.get_column("data_source")[0])

            if "segment_shift" in config.enabled_detectors:
                for block in eligible_blocks:
                    statistical_candidates.extend(
                        self._segment_candidates(
                            ordered,
                            block=block,
                            feature=feature,
                            data_source=source,
                            config=config,
                        )
                    )
            if "boundary" in config.enabled_detectors:
                statistical_candidates.extend(
                    self._boundary_candidates(
                        ordered,
                        eligible_blocks=eligible_blocks,
                        feature=feature,
                        data_source=source,
                        benchmark_stats=(
                            benchmark_stats.get(feature) if benchmark_stats is not None else None
                        ),
                        config=config,
                    )
                )
            if "point" in config.enabled_detectors:
                for block in eligible_blocks:
                    statistical_candidates.extend(
                        self._point_candidates(
                            ordered,
                            block=block,
                            feature=feature,
                            data_source=source,
                            config=config,
                        )
                    )
            if "high_level" in config.enabled_detectors:
                high_level_candidates.extend(
                    self._high_level_candidates(
                        ordered,
                        eligible_blocks=eligible_blocks,
                        feature=feature,
                        data_source=source,
                        config=config,
                    )
                )

        adjusted = _apply_benjamini_hochberg(statistical_candidates)
        confirmed = [
            candidate
            for candidate in adjusted
            if candidate.q_value is not None
            and candidate.q_value <= config.fdr_q_threshold
            and candidate.abs_delta >= config.min_effect_delta
            and (
                candidate.abs_delta >= config.min_abs_delta
                or candidate.relative_delta >= config.min_relative_delta
            )
        ]
        all_candidates = [*confirmed, *high_level_candidates]
        if not all_candidates:
            return _empty_detail_table()

        rows: list[dict[str, Any]] = []
        for partition_key, feature_df in trend_table.partition_by("feature", as_dict=True).items():
            feature = str(partition_key[0] if isinstance(partition_key, tuple) else partition_key)
            feature_candidates = [
                candidate for candidate in all_candidates if candidate.feature == feature
            ]
            rows.extend(_merge_candidate_rows(feature_df.sort("period"), feature_candidates, config))
        if not rows:
            return _empty_detail_table()
        return pl.DataFrame(rows, schema=_empty_detail_table().schema).sort(
            ["feature", "event_start_period"]
        )

    @staticmethod
    def _segment_candidates(
        feature_df: pl.DataFrame,
        *,
        block: list[int],
        feature: str,
        data_source: str,
        config: MarsMissingShiftConfig,
    ) -> list[_Candidate]:
        """使用惩罚式 Binseg 为一个连续有效区间生成分段变化候选。"""
        if len(block) < config.min_segment_size * 2:
            return []
        try:
            import ruptures as rpt
        except ImportError as exc:
            raise ImportError("`MarsMissingShiftScanner` requires dependency `ruptures`.") from exc

        values = (
            feature_df
            .filter(pl.col("__idx").is_in(block))
            .get_column("missing_rate")
            .to_numpy()
            .astype(float)
        )
        variance = float(np.var(values))
        penalty = max(variance * log(max(len(values), 2)), FLOAT_TOLERANCE)
        model = rpt.Binseg(model="l2", min_size=config.min_segment_size, jump=1)
        fitted = model.fit(values.reshape(-1, 1))
        try:
            local_points = [point for point in fitted.predict(pen=penalty) if point < len(values)]
        except rpt.exceptions.BadSegmentationParameters:
            return []

        scored_points: list[tuple[float, int]] = []
        for local_point in local_points:
            before_local = list(range(local_point - config.min_segment_size, local_point))
            after_local = list(range(local_point, local_point + config.min_segment_size))
            if before_local[0] < 0 or after_local[-1] >= len(block):
                continue
            before_stats = _window_stats(feature_df, [block[idx] for idx in before_local])
            after_stats = _window_stats(feature_df, [block[idx] for idx in after_local])
            scored_points.append((abs(after_stats.rate - before_stats.rate), local_point))
        scored_points.sort(reverse=True)

        candidates: list[_Candidate] = []
        periods = feature_df.get_column("period").to_list()
        for _, local_point in scored_points[: config.max_segment_candidates]:
            before_indices = [
                block[idx]
                for idx in range(local_point - config.min_segment_size, local_point)
            ]
            event_indices = [
                block[idx]
                for idx in range(local_point, local_point + config.min_segment_size)
            ]
            candidates.append(
                _statistical_candidate(
                    feature=feature,
                    data_source=data_source,
                    anomaly_type="segment_shift",
                    event_indices=event_indices,
                    event_stats=_window_stats(feature_df, event_indices),
                    reference_type="previous_window",
                    reference_start_period=str(periods[before_indices[0]]),
                    reference_end_period=str(periods[before_indices[-1]]),
                    reference_stats=_window_stats(feature_df, before_indices),
                    min_effect_delta=config.min_effect_delta,
                )
            )
        return candidates

    @staticmethod
    def _boundary_candidates(
        feature_df: pl.DataFrame,
        *,
        eligible_blocks: list[list[int]],
        feature: str,
        data_source: str,
        benchmark_stats: _WindowStats | None,
        config: MarsMissingShiftConfig,
    ) -> list[_Candidate]:
        """为全局开头和结尾选择最能解释边界异常的候选窗口。"""
        if not eligible_blocks:
            return []
        periods = feature_df.get_column("period").to_list()
        candidates: list[_Candidate] = []
        boundary_specs = [
            ("start", eligible_blocks[0]),
            ("end", eligible_blocks[-1]),
        ]
        for side, block in boundary_specs:
            side_candidates: list[_Candidate] = []
            max_event_size = min(config.max_boundary_periods, len(block))
            for event_size in range(1, max_event_size + 1):
                event_indices = block[:event_size] if side == "start" else block[-event_size:]
                if benchmark_stats is not None:
                    reference_stats = benchmark_stats
                    reference_type = "benchmark"
                    reference_start = None
                    reference_end = None
                else:
                    if side == "start":
                        reference_indices = block[event_size : event_size + config.reference_window]
                        reference_type = "next_window"
                    else:
                        reference_indices = block[
                            max(0, len(block) - event_size - config.reference_window) : -event_size
                        ]
                        reference_type = "previous_window"
                    if len(reference_indices) < config.reference_window:
                        continue
                    reference_stats = _window_stats(feature_df, reference_indices)
                    reference_start = str(periods[reference_indices[0]])
                    reference_end = str(periods[reference_indices[-1]])

                side_candidates.append(
                    _statistical_candidate(
                        feature=feature,
                        data_source=data_source,
                        anomaly_type="boundary",
                        event_indices=event_indices,
                        event_stats=_window_stats(feature_df, event_indices),
                        reference_type=reference_type,
                        reference_start_period=reference_start,
                        reference_end_period=reference_end,
                        reference_stats=reference_stats,
                        min_effect_delta=config.min_effect_delta,
                    )
                )
            if side_candidates:
                side_candidates.sort(
                    key=lambda candidate: (
                        candidate.abs_delta,
                        candidate.event_end_idx - candidate.event_start_idx + 1,
                    ),
                    reverse=True,
                )
                candidates.append(side_candidates[0])
        return candidates

    @staticmethod
    def _point_candidates(
        feature_df: pl.DataFrame,
        *,
        block: list[int],
        feature: str,
        data_source: str,
        config: MarsMissingShiftConfig,
    ) -> list[_Candidate]:
        """生成同时偏离左右参考窗口的内部单日候选。"""
        window = config.reference_window
        if len(block) < window * 2 + 1:
            return []
        periods = feature_df.get_column("period").to_list()
        candidates: list[_Candidate] = []
        for local_idx in range(window, len(block) - window):
            event_idx = block[local_idx]
            left_indices = block[local_idx - window : local_idx]
            right_indices = block[local_idx + 1 : local_idx + 1 + window]
            event_stats = _window_stats(feature_df, [event_idx])
            left_stats = _window_stats(feature_df, left_indices)
            right_stats = _window_stats(feature_df, right_indices)
            left_delta = event_stats.rate - left_stats.rate
            right_delta = event_stats.rate - right_stats.rate
            same_direction = left_delta * right_delta > 0
            if not same_direction or min(abs(left_delta), abs(right_delta)) < config.min_effect_delta:
                continue
            reference_indices = [*left_indices, *right_indices]
            candidates.append(
                _statistical_candidate(
                    feature=feature,
                    data_source=data_source,
                    anomaly_type="point",
                    event_indices=[event_idx],
                    event_stats=event_stats,
                    reference_type="surrounding_window",
                    reference_start_period=str(periods[left_indices[0]]),
                    reference_end_period=str(periods[right_indices[-1]]),
                    reference_stats=_window_stats(feature_df, reference_indices),
                    min_effect_delta=config.min_effect_delta,
                )
            )
        return candidates

    @staticmethod
    def _high_level_candidates(
        feature_df: pl.DataFrame,
        *,
        eligible_blocks: list[list[int]],
        feature: str,
        data_source: str,
        config: MarsMissingShiftConfig,
    ) -> list[_Candidate]:
        """将连续越过特征红线的有效日期合并为高缺失事件。"""
        threshold = config.feature_high_missing_rate_thresholds.get(
            feature,
            config.high_missing_rate_threshold,
        )
        rates = feature_df.get_column("missing_rate").to_list()
        candidates: list[_Candidate] = []
        for block in eligible_blocks:
            run: list[int] = []
            for idx in [*block, -1]:
                if idx >= 0 and float(rates[idx]) >= threshold:
                    run.append(idx)
                    continue
                if len(run) >= config.min_high_periods:
                    event_stats = _window_stats(feature_df, run)
                    delta = event_stats.rate - threshold
                    candidates.append(
                        _Candidate(
                            feature=feature,
                            data_source=data_source,
                            anomaly_type="high_level",
                            event_start_idx=run[0],
                            event_end_idx=run[-1],
                            event_stats=event_stats,
                            reference_type="business_threshold",
                            reference_start_period=None,
                            reference_end_period=None,
                            reference_stats=None,
                            threshold=threshold,
                            delta=delta,
                            abs_delta=abs(delta),
                            relative_delta=abs(delta) / max(threshold, FLOAT_TOLERANCE),
                            p_value=None,
                            q_value=None,
                            test_method=None,
                        )
                    )
                run = []
        return candidates

    @staticmethod
    def _annotate_trend(
        trend_table: pl.DataFrame,
        detail_table: pl.DataFrame,
    ) -> pl.DataFrame:
        """根据最终事件为长版趋势表补充异常标记和事件类型。"""
        if detail_table.is_empty():
            return trend_table.with_columns(
                [
                    pl.lit(False).alias("is_anomaly"),
                    pl.lit(None).cast(pl.String).alias("anomaly_types"),
                ]
            )
        event_rows = detail_table.select(
            ["feature", "event_start_period", "event_end_period", "detected_by"]
        ).to_dicts()
        annotations: list[dict[str, Any]] = []
        for row in trend_table.select(["feature", "period"]).to_dicts():
            event_types = sorted(
                {
                    event_type
                    for event in event_rows
                    if event["feature"] == row["feature"]
                    and event["event_start_period"] <= row["period"] <= event["event_end_period"]
                    for event_type in str(event["detected_by"]).split(",")
                },
                key=lambda value: _DETECTOR_PRIORITY[cast(AnomalyType, value)],
            )
            annotations.append(
                {
                    "feature": row["feature"],
                    "period": row["period"],
                    "is_anomaly": bool(event_types),
                    "anomaly_types": ",".join(event_types) if event_types else None,
                }
            )
        annotation_df = pl.DataFrame(
            annotations,
            schema={
                "feature": pl.String,
                "period": pl.String,
                "is_anomaly": pl.Boolean,
                "anomaly_types": pl.String,
            },
        )
        return trend_table.join(annotation_df, on=["feature", "period"], how="left")

    @staticmethod
    def _build_summary_table(detail_table: pl.DataFrame) -> pl.DataFrame:
        """由事件明细构造特征级异常汇总。"""
        if detail_table.is_empty():
            return _empty_summary_table()
        return (
            detail_table
            .group_by(["feature", "data_source"])
            .agg(
                [
                    pl.len().alias("anomaly_count"),
                    pl.col("anomaly_type").unique().sort().str.join(",").alias("anomaly_types"),
                    pl.col("abs_delta").max().alias("max_abs_delta"),
                    pl.col("event_missing_rate").max().alias("max_event_missing_rate"),
                    pl.col("event_start_period").min().alias("first_anomaly_period"),
                    pl.col("event_end_period").max().alias("last_anomaly_period"),
                    pl.col("direction").mode().first().alias("primary_direction"),
                ]
            )
            .sort(["max_abs_delta", "anomaly_count"], descending=[True, True])
        )

    @staticmethod
    def _build_source_table(
        detail_table: pl.DataFrame,
        *,
        target_features: list[str],
        feature_data_source: dict[str, str],
    ) -> pl.DataFrame:
        """由事件明细构造数据源级异常汇总。"""
        if not feature_data_source:
            return _empty_source_table()
        source_features = pl.DataFrame(
            {
                "feature": target_features,
                "data_source": [
                    feature_data_source.get(feature, "UNMAPPED") for feature in target_features
                ],
            }
        )
        feature_counts = source_features.group_by("data_source").agg(
            pl.len().alias("feature_count")
        )
        if detail_table.is_empty():
            return (
                feature_counts
                .with_columns(
                    [
                        pl.lit(0).cast(pl.Int64).alias("anomaly_feature_count"),
                        pl.lit(0.0).alias("anomaly_feature_rate"),
                        pl.lit(0).cast(pl.Int64).alias("anomaly_count"),
                        pl.lit(None).cast(pl.Float64).alias("max_abs_delta"),
                    ]
                )
                .select(_empty_source_table().columns)
                .sort("data_source")
            )
        anomaly_summary = detail_table.group_by("data_source").agg(
            [
                pl.col("feature").n_unique().alias("anomaly_feature_count"),
                pl.len().alias("anomaly_count"),
                pl.col("abs_delta").max().alias("max_abs_delta"),
            ]
        )
        return (
            feature_counts
            .join(anomaly_summary, on="data_source", how="left")
            .with_columns(
                [
                    pl.col("anomaly_feature_count").fill_null(0),
                    pl.col("anomaly_count").fill_null(0),
                    (
                        pl.col("anomaly_feature_count").fill_null(0)
                        / pl.col("feature_count")
                    ).alias("anomaly_feature_rate"),
                ]
            )
            .select(_empty_source_table().columns)
            .sort(["anomaly_feature_count", "max_abs_delta"], descending=[True, True])
        )


_TREND_META_COLUMNS = {"feature", "dtype", "data_source"}


def _eligible_blocks(feature_df: pl.DataFrame) -> list[list[int]]:
    """将有效日期拆成不会跨越低样本日期的连续索引块。"""
    eligible = feature_df.get_column("is_detection_eligible").to_list()
    blocks: list[list[int]] = []
    current: list[int] = []
    for idx, is_eligible in enumerate(eligible):
        if is_eligible:
            current.append(idx)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _window_stats(feature_df: pl.DataFrame, indices: Sequence[int]) -> _WindowStats:
    """聚合指定行索引对应窗口的缺失计数。"""
    selected = feature_df.filter(pl.col("__idx").is_in(indices))
    return _WindowStats(
        missing_count=int(selected.get_column("missing_count").sum()),
        total_count=int(selected.get_column("total_count").sum()),
    )


def _statistical_candidate(
    *,
    feature: str,
    data_source: str,
    anomaly_type: AnomalyType,
    event_indices: Sequence[int],
    event_stats: _WindowStats,
    reference_type: str,
    reference_start_period: str | None,
    reference_end_period: str | None,
    reference_stats: _WindowStats,
    min_effect_delta: float,
) -> _Candidate:
    """计算业务效果量与两比例显著性，构造统计型候选。"""
    delta = event_stats.rate - reference_stats.rate
    abs_delta = abs(delta)
    relative_delta = abs_delta / max(abs(reference_stats.rate), min_effect_delta)
    p_value, test_method = _two_proportion_test(event_stats, reference_stats)
    return _Candidate(
        feature=feature,
        data_source=data_source,
        anomaly_type=anomaly_type,
        event_start_idx=min(event_indices),
        event_end_idx=max(event_indices),
        event_stats=event_stats,
        reference_type=reference_type,
        reference_start_period=reference_start_period,
        reference_end_period=reference_end_period,
        reference_stats=reference_stats,
        threshold=None,
        delta=delta,
        abs_delta=abs_delta,
        relative_delta=relative_delta,
        p_value=p_value,
        q_value=None,
        test_method=test_method,
    )


def _two_proportion_test(
    event_stats: _WindowStats,
    reference_stats: _WindowStats,
) -> tuple[float, str]:
    """按期望频数选择 Fisher 精确检验或两比例 z 检验。"""
    table = np.array(
        [
            [
                event_stats.missing_count,
                event_stats.total_count - event_stats.missing_count,
            ],
            [
                reference_stats.missing_count,
                reference_stats.total_count - reference_stats.missing_count,
            ],
        ],
        dtype=float,
    )
    grand_total = float(table.sum())
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / grand_total
    if bool((expected < 5).any()):
        scipy_stats: Any = importlib.import_module("scipy.stats")
        _odds_ratio, p_value = scipy_stats.fisher_exact(
            table.astype(int),
            alternative="two-sided",
        )
        return float(p_value), "fisher_exact"
    return (
        _two_proportion_pvalue(
            float(table[0, 0]),
            float(table[0].sum()),
            float(table[1, 0]),
            float(table[1].sum()),
        ),
        "z_test",
    )


def _two_proportion_pvalue(
    missing_a: float,
    total_a: float,
    missing_b: float,
    total_b: float,
) -> float:
    """计算两比例 z 检验的双侧 p 值。"""
    rate_a = missing_a / total_a
    rate_b = missing_b / total_b
    pooled = (missing_a + missing_b) / (total_a + total_b)
    variance = pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b)
    if variance <= 0:
        return 1.0 if abs(rate_a - rate_b) <= FLOAT_TOLERANCE else 0.0
    z_score = abs(rate_b - rate_a) / sqrt(variance)
    return float(erfc(z_score / sqrt(2.0)))


def _apply_benjamini_hochberg(candidates: list[_Candidate]) -> list[_Candidate]:
    """对全部特征和统计检测器候选执行全局 Benjamini-Hochberg 校正。"""
    if not candidates:
        return []
    ordered = sorted(enumerate(candidates), key=lambda item: cast(float, item[1].p_value))
    adjusted: list[float] = [1.0] * len(candidates)
    running_min = 1.0
    total = len(candidates)
    for reverse_idx in range(total - 1, -1, -1):
        original_idx, candidate = ordered[reverse_idx]
        rank = reverse_idx + 1
        raw_adjusted = cast(float, candidate.p_value) * total / rank
        running_min = min(running_min, raw_adjusted, 1.0)
        adjusted[original_idx] = running_min
    return [
        replace(candidate, q_value=adjusted[idx]) for idx, candidate in enumerate(candidates)
    ]


def _merge_candidate_rows(
    feature_df: pl.DataFrame,
    candidates: list[_Candidate],
    config: MarsMissingShiftConfig,
) -> list[dict[str, Any]]:
    """合并同特征重叠候选，并保留所有检测证据。"""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.event_start_idx, item.event_end_idx))
    clusters: list[list[_Candidate]] = []
    current = [ordered[0]]
    current_end = ordered[0].event_end_idx
    for candidate in ordered[1:]:
        if candidate.event_start_idx <= current_end:
            current.append(candidate)
            current_end = max(current_end, candidate.event_end_idx)
        else:
            clusters.append(current)
            current = [candidate]
            current_end = candidate.event_end_idx
    clusters.append(current)

    periods = feature_df.get_column("period").to_list()
    rows: list[dict[str, Any]] = []
    indexed_df = feature_df.with_row_index("__idx")
    for cluster in clusters:
        primary = min(
            cluster,
            key=lambda item: (
                _DETECTOR_PRIORITY[item.anomaly_type],
                -item.abs_delta,
            ),
        )
        start_idx = min(candidate.event_start_idx for candidate in cluster)
        end_idx = max(candidate.event_end_idx for candidate in cluster)
        event_stats = _window_stats(indexed_df, list(range(start_idx, end_idx + 1)))
        statistical = [candidate for candidate in cluster if candidate.q_value is not None]
        evidence = min(statistical, key=lambda item: cast(float, item.q_value)) if statistical else None
        reference_candidate = evidence or primary
        reference_stats = reference_candidate.reference_stats
        threshold = next(
            (
                candidate.threshold
                for candidate in cluster
                if candidate.anomaly_type == "high_level"
            ),
            primary.threshold,
        )
        if reference_stats is not None:
            delta = event_stats.rate - reference_stats.rate
            relative_base = max(abs(reference_stats.rate), config.min_effect_delta)
        else:
            effective_threshold = cast(float, threshold)
            delta = event_stats.rate - effective_threshold
            relative_base = max(effective_threshold, FLOAT_TOLERANCE)
        direction: Direction = "increase" if delta >= 0 else "decrease"
        detected_types = sorted(
            {candidate.anomaly_type for candidate in cluster},
            key=lambda item: _DETECTOR_PRIORITY[item],
        )
        reasons: list[str] = list(detected_types)
        if evidence is not None:
            reasons.extend([cast(str, evidence.test_method), "fdr"])
            if abs(delta) >= config.min_abs_delta:
                reasons.append("abs_delta")
            if abs(delta) / relative_base >= config.min_relative_delta:
                reasons.append("relative_delta")
        rows.append(
            {
                "feature": primary.feature,
                "data_source": primary.data_source,
                "anomaly_type": primary.anomaly_type,
                "detected_by": ",".join(detected_types),
                "event_start_period": str(periods[start_idx]),
                "event_end_period": str(periods[end_idx]),
                "reference_type": reference_candidate.reference_type,
                "reference_start_period": reference_candidate.reference_start_period,
                "reference_end_period": reference_candidate.reference_end_period,
                "event_missing_count": event_stats.missing_count,
                "event_total_count": event_stats.total_count,
                "event_missing_rate": event_stats.rate,
                "reference_missing_count": (
                    reference_stats.missing_count if reference_stats is not None else None
                ),
                "reference_total_count": (
                    reference_stats.total_count if reference_stats is not None else None
                ),
                "reference_missing_rate": reference_stats.rate if reference_stats else None,
                "threshold": threshold,
                "delta": delta,
                "abs_delta": abs(delta),
                "relative_delta": abs(delta) / relative_base,
                "p_value": evidence.p_value if evidence is not None else None,
                "q_value": evidence.q_value if evidence is not None else None,
                "direction": direction,
                "reason": ",".join(dict.fromkeys(reasons)),
            }
        )
    return rows


def _dtypes_compatible(current: pl.DataType, benchmark: pl.DataType) -> bool:
    """判断当前数据与 benchmark 特征 dtype 是否可安全共享缺失语义。"""
    return current == benchmark or (
        is_numeric_dtype(current) and is_numeric_dtype(benchmark)
    )


def _normalize_names(values: str | Sequence[str] | None) -> list[str] | None:
    """将可选单值或序列统一为字符串列表。"""
    if values is None:
        return None
    return [values] if isinstance(values, str) else list(values)


def _filter_pandas_features(
    frame: pd.DataFrame,
    features: str | Sequence[str] | None,
) -> pd.DataFrame:
    """按可选特征列表过滤 Pandas 展示副本。"""
    requested = _normalize_names(features)
    if requested is None or "feature" not in frame.columns:
        return frame.copy()
    return frame[frame["feature"].isin(requested)].copy()


def _style_missing_shift_table(
    frame: pd.DataFrame,
    *,
    caption: str,
    gradient_columns: Sequence[str] | None = None,
    percentage_columns: Sequence[str] | None = None,
) -> pd.io.formats.style.Styler:
    """为缺失率结果构造统一 Pandas Styler。"""
    styler = frame.style.hide(axis="index").set_caption(caption)
    gradients = [
        column for column in (gradient_columns or []) if column in frame.columns and not frame.empty
    ]
    if gradients:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=gradients, axis=None)
    percentage_candidates = list(percentage_columns or []) + [
        column
        for column in frame.columns
        if "rate" in str(column) or column in {"delta", "abs_delta", "relative_delta"}
    ]
    percentage = list(dict.fromkeys(column for column in percentage_candidates if column in frame.columns))
    if percentage:
        styler = styler.format("{:.2%}", subset=percentage, na_rep="-")
    numeric = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if column not in percentage
    ]
    if numeric:
        styler = styler.format("{:.4f}", subset=numeric, na_rep="-")
    return styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("background-color", "#edf2f7"), ("text-align", "left")],
            },
            {
                "selector": "caption",
                "props": [("font-size", "1.15em"), ("font-weight", "bold")],
            },
        ]
    )


def _excel_column_name(zero_based_index: int) -> str:
    """将零起始列索引转换为 Excel 列名。"""
    value = zero_based_index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result
