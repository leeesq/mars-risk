"""静态训练数据缺失率异常扫描。"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.compute.missing import build_missing_by_period_stats, missing_condition_expr
from mars.core.base import MarsBaseEstimator
from mars.core.constants import FLOAT_TOLERANCE, PROBABILITY_EPSILON
from mars.utils.date import MarsDate

Direction = Literal["increase", "decrease"]


def _empty_detail_table() -> pl.DataFrame:
    """构造空异常明细表。"""
    return pl.DataFrame(
        schema={
            "feature": pl.String,
            "data_source": pl.String,
            "start_period": pl.String,
            "end_period": pl.String,
            "change_period": pl.String,
            "before_missing_rate": pl.Float64,
            "after_missing_rate": pl.Float64,
            "abs_delta": pl.Float64,
            "relative_delta": pl.Float64,
            "p_value": pl.Float64,
            "direction": pl.String,
            "anomaly_score": pl.Float64,
            "reason": pl.String,
        }
    )


def _empty_summary_table() -> pl.DataFrame:
    """构造空特征汇总表。"""
    return pl.DataFrame(
        schema={
            "feature": pl.String,
            "data_source": pl.String,
            "anomaly_count": pl.Int64,
            "max_abs_delta": pl.Float64,
            "first_anomaly_period": pl.String,
            "last_anomaly_period": pl.String,
            "primary_direction": pl.String,
        }
    )


def _empty_source_table() -> pl.DataFrame:
    """构造空数据源汇总表。"""
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
class _MissingSegmentStats:
    """保存变点两侧缺失率统计。"""

    before_rate: float
    after_rate: float
    before_missing: float
    after_missing: float
    before_total: float
    after_total: float


@dataclass(frozen=True)
class MarsMissingShiftResult:
    """
    缺失率异常扫描结果。

    `MarsMissingShiftResult` 保存静态训练数据缺失率异常扫描的结构化输出。该结果只用于
    数据质量复核，不会自动删除特征，也不会阻断后续建模或筛选流程。

    Attributes
    ----------
    summary_table : pl.DataFrame
        特征级异常汇总表。
    detail_table : pl.DataFrame
        异常窗口明细表。
    source_table : pl.DataFrame
        数据源级异常汇总表；未传 `feature_data_source` 时为空表。
    missing_rate_table : pl.DataFrame
        按时间粒度展开的缺失率宽表。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import MarsMissingShiftScanner
    >>> df = pl.DataFrame({"dt": ["2026-01-01"], "x": [None]})
    >>> result = MarsMissingShiftScanner().scan(df, date_col="dt", features=["x"])
    >>> result.detail_table.is_empty()
    True
    """

    summary_table: pl.DataFrame
    detail_table: pl.DataFrame
    source_table: pl.DataFrame
    missing_rate_table: pl.DataFrame

    def write_excel(self, path: str | Path) -> None:
        """
        将缺失率异常扫描结果导出到 Excel。

        Parameters
        ----------
        path : str | Path
            输出 Excel 文件路径。父目录必须已经存在。

        Returns
        -------
        None
            方法只写出文件，不返回对象。

        Raises
        ------
        FileNotFoundError
            当父目录不存在时抛出。

        Examples
        --------
        >>> result = MarsMissingShiftResult(
        ...     summary_table=pl.DataFrame(),
        ...     detail_table=pl.DataFrame(),
        ...     source_table=pl.DataFrame(),
        ...     missing_rate_table=pl.DataFrame(),
        ... )
        >>> isinstance(result.summary_table, pl.DataFrame)
        True
        """
        output_path = Path(path)
        if output_path.parent and not output_path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {output_path.parent}")

        sheets: dict[str, pl.DataFrame] = {
            "summary": self.summary_table,
            "detail": self.detail_table,
            "source": self.source_table,
            "missing_rate": self.missing_rate_table,
        }
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_pandas().to_excel(writer, sheet_name=sheet_name, index=False)


class MarsMissingShiftScanner(MarsBaseEstimator):
    """
    静态训练数据缺失率异常扫描器。

    该扫描器面向建模前的宽表数据质量复核。它先按时间粒度计算每个特征的缺失率序列，
    再使用 `ruptures` 识别候选变点，并通过幅度、相对变化和两比例 z 检验确认异常。
    输出结果只作为人工复核和数据源排查依据，不自动裁决特征生死。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import MarsMissingShiftScanner
    >>> df = pl.DataFrame({"dt": ["2026-01-01", "2026-01-02"], "x": [1, None]})
    >>> result = MarsMissingShiftScanner().scan(df, date_col="dt", features=["x"])
    >>> "x" in result.missing_rate_table.get_column("feature").to_list()
    True
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
        min_segment_size: int = 3,
        min_abs_delta: float = 0.03,
        min_effect_delta: float = 0.005,
        min_relative_delta: float = 0.30,
        pvalue_threshold: float = 0.01,
        max_change_points: int = 5,
    ) -> MarsMissingShiftResult:
        """
        扫描静态训练数据中的缺失率异常跳变。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待扫描的训练宽表。
        date_col : str
            日期列名称，会按 `time_grain` 聚合。
        features : list[str] | None
            待扫描特征。默认 ``None``，扫描除 `date_col` 外的全部列。
        time_grain : str
            时间粒度，默认 ``"1d"``，复用 `MarsDate.from_grain` 支持的格式。
        missing_values : list[Any] | None
            业务自定义缺失值，默认 ``None``。
        feature_data_source : dict[str, str] | None
            特征到数据源的映射，默认 ``None``，用于生成数据源级异常汇总。
        min_segment_size : int
            变点两侧最小连续时间段数量，默认 ``3``。
        min_abs_delta : float
            直接确认异常的缺失率绝对变化阈值，默认 ``0.03``。
        min_effect_delta : float
            进入相对变化或显著性判断的最小有效变化，默认 ``0.005``。
        min_relative_delta : float
            相对变化阈值，默认 ``0.30``。
        pvalue_threshold : float
            两比例 z 检验 p 值阈值，默认 ``0.01``。
        max_change_points : int
            每个特征最多评估的候选变点数量，默认 ``5``。

        Returns
        -------
        MarsMissingShiftResult
            缺失率异常扫描结果。

        Raises
        ------
        ValueError
            当日期列、特征列或阈值参数非法时抛出。

        Notes
        -----
        当前实现依赖核心依赖 `ruptures` 执行变点检测。
        """
        self._validate_scan_params(
            min_segment_size=min_segment_size,
            min_abs_delta=min_abs_delta,
            min_effect_delta=min_effect_delta,
            min_relative_delta=min_relative_delta,
            pvalue_threshold=pvalue_threshold,
            max_change_points=max_change_points,
        )
        working_df: pl.DataFrame = cast(pl.DataFrame, self._ensure_polars_dataframe(df))
        target_features: list[str] = self._resolve_features(working_df, date_col, features)
        period_col = "__mars_missing_shift_period"
        period_df: pl.DataFrame = working_df.with_columns(
            MarsDate.from_grain(date_col, time_grain).cast(pl.String).alias(period_col)
        ).filter(pl.col(period_col).is_not_null())
        if period_df.is_empty():
            raise ValueError(f"Date column '{date_col}' cannot be parsed into valid periods.")

        missing_rate_table: pl.DataFrame = build_missing_by_period_stats(
            period_df,
            features=target_features,
            period_col=period_col,
            missing_values=missing_values,
        )
        long_stats: pl.DataFrame = self._build_long_missing_stats(
            period_df,
            features=target_features,
            period_col=period_col,
            missing_values=missing_values,
        )
        detail_table: pl.DataFrame = self._scan_anomalies(
            long_stats,
            feature_data_source=feature_data_source or {},
            min_segment_size=min_segment_size,
            min_abs_delta=min_abs_delta,
            min_effect_delta=min_effect_delta,
            min_relative_delta=min_relative_delta,
            pvalue_threshold=pvalue_threshold,
            max_change_points=max_change_points,
        )
        summary_table: pl.DataFrame = self._build_summary_table(detail_table)
        source_table: pl.DataFrame = self._build_source_table(
            detail_table,
            target_features=target_features,
            feature_data_source=feature_data_source or {},
        )
        return MarsMissingShiftResult(
            summary_table=summary_table,
            detail_table=detail_table,
            source_table=source_table,
            missing_rate_table=missing_rate_table,
        )

    @staticmethod
    def _validate_scan_params(
        *,
        min_segment_size: int,
        min_abs_delta: float,
        min_effect_delta: float,
        min_relative_delta: float,
        pvalue_threshold: float,
        max_change_points: int,
    ) -> None:
        """校验扫描阈值参数。"""
        if min_segment_size < 2:
            raise ValueError("`min_segment_size` must be >= 2.")
        if max_change_points < 1:
            raise ValueError("`max_change_points` must be >= 1.")
        bounded_params = {
            "min_abs_delta": min_abs_delta,
            "min_effect_delta": min_effect_delta,
            "min_relative_delta": min_relative_delta,
            "pvalue_threshold": pvalue_threshold,
        }
        for name, value in bounded_params.items():
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0.")
        if pvalue_threshold > 1:
            raise ValueError("`pvalue_threshold` must be <= 1.")

    @staticmethod
    def _resolve_features(
        df: pl.DataFrame,
        date_col: str,
        features: list[str] | None,
    ) -> list[str]:
        """解析并校验待扫描特征。"""
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' was not found.")
        if features is None:
            resolved = [col for col in df.columns if col != date_col]
        else:
            missing_features = [feature for feature in features if feature not in df.columns]
            if missing_features:
                raise ValueError(f"Features not found: {missing_features}")
            resolved = [feature for feature in features if feature != date_col]
        if not resolved:
            raise ValueError("No features are available for missing shift scanning.")
        return resolved

    @staticmethod
    def _build_long_missing_stats(
        df: pl.DataFrame,
        *,
        features: list[str],
        period_col: str,
        missing_values: list[Any] | None,
    ) -> pl.DataFrame:
        """构造长表缺失计数和缺失率。"""
        schema = df.schema
        exprs: list[pl.Expr] = []
        for feature in features:
            missing_expr = missing_condition_expr(
                feature,
                dtype=schema.get(feature),
                missing_values=missing_values,
            )
            exprs.extend(
                [
                    missing_expr.sum().alias(f"{feature}__missing_count"),
                    pl.len().alias(f"{feature}__total_count"),
                ]
            )
        grouped = df.group_by(period_col).agg(exprs).sort(period_col)
        feature_frames: list[pl.DataFrame] = []
        for feature in features:
            feature_frame = grouped.select(
                [
                    pl.col(period_col).alias("period"),
                    pl.lit(feature).alias("feature"),
                    pl.col(f"{feature}__missing_count").cast(pl.Float64).alias("missing_count"),
                    pl.col(f"{feature}__total_count").cast(pl.Float64).alias("total_count"),
                ]
            ).with_columns(
                pl.when(pl.col("total_count") > 0)
                .then(pl.col("missing_count") / pl.col("total_count"))
                .otherwise(None)
                .alias("missing_rate")
            )
            feature_frames.append(feature_frame)
        return pl.concat(feature_frames, how="vertical_relaxed")

    def _scan_anomalies(
        self,
        long_stats: pl.DataFrame,
        *,
        feature_data_source: dict[str, str],
        min_segment_size: int,
        min_abs_delta: float,
        min_effect_delta: float,
        min_relative_delta: float,
        pvalue_threshold: float,
        max_change_points: int,
    ) -> pl.DataFrame:
        """对每个特征执行变点检测并输出异常明细。"""
        try:
            import ruptures as rpt
        except ImportError as exc:
            raise ImportError("`MarsMissingShiftScanner` requires dependency `ruptures`.") from exc

        rows: list[dict[str, Any]] = []
        for feature, feature_df in long_stats.partition_by("feature", as_dict=True).items():
            feature_name = str(feature[0] if isinstance(feature, tuple) else feature)
            ordered_df = feature_df.sort("period")
            values = ordered_df.get_column("missing_rate").fill_null(0.0).to_numpy().astype(float)
            if len(values) < min_segment_size * 2:
                continue

            candidates = self._detect_change_points(
                values,
                ruptures_module=rpt,
                min_segment_size=min_segment_size,
                max_change_points=max_change_points,
            )
            for change_idx in candidates:
                segment_stats = self._segment_stats(
                    ordered_df,
                    change_idx=change_idx,
                    min_segment_size=min_segment_size,
                )
                if segment_stats is None:
                    continue
                anomaly = self._build_anomaly_row(
                    ordered_df,
                    feature=feature_name,
                    data_source=feature_data_source.get(feature_name, "UNMAPPED"),
                    change_idx=change_idx,
                    stats=segment_stats,
                    min_abs_delta=min_abs_delta,
                    min_effect_delta=min_effect_delta,
                    min_relative_delta=min_relative_delta,
                    pvalue_threshold=pvalue_threshold,
                )
                if anomaly is not None:
                    rows.append(anomaly)

        if not rows:
            return _empty_detail_table()
        return pl.DataFrame(rows).sort(["feature", "change_period"])

    @staticmethod
    def _detect_change_points(
        values: np.ndarray,
        *,
        ruptures_module: Any,
        min_segment_size: int,
        max_change_points: int,
    ) -> list[int]:
        """调用 ruptures 生成候选变点。"""
        effective_points = min(max_change_points, max(1, len(values) // min_segment_size - 1))
        if effective_points <= 0:
            return []
        model = ruptures_module.Binseg(model="l2", min_size=min_segment_size, jump=1)
        fitted_model = model.fit(values.reshape(-1, 1))
        for n_bkps in range(effective_points, 0, -1):
            try:
                change_points = fitted_model.predict(n_bkps=n_bkps)
            except ruptures_module.exceptions.BadSegmentationParameters:
                continue
            return sorted({point for point in change_points if 0 < point < len(values)})
        return []

    @staticmethod
    def _segment_stats(
        feature_df: pl.DataFrame,
        *,
        change_idx: int,
        min_segment_size: int,
    ) -> _MissingSegmentStats | None:
        """计算候选变点两侧的缺失统计。"""
        before_start = max(0, change_idx - min_segment_size)
        after_end = min(feature_df.height, change_idx + min_segment_size)
        before_df = feature_df.slice(before_start, change_idx - before_start)
        after_df = feature_df.slice(change_idx, after_end - change_idx)
        if before_df.height < min_segment_size or after_df.height < min_segment_size:
            return None

        before_missing = float(before_df.get_column("missing_count").sum())
        after_missing = float(after_df.get_column("missing_count").sum())
        before_total = float(before_df.get_column("total_count").sum())
        after_total = float(after_df.get_column("total_count").sum())
        if before_total <= 0 or after_total <= 0:
            return None
        return _MissingSegmentStats(
            before_rate=before_missing / before_total,
            after_rate=after_missing / after_total,
            before_missing=before_missing,
            after_missing=after_missing,
            before_total=before_total,
            after_total=after_total,
        )

    @staticmethod
    def _build_anomaly_row(
        feature_df: pl.DataFrame,
        *,
        feature: str,
        data_source: str,
        change_idx: int,
        stats: _MissingSegmentStats,
        min_abs_delta: float,
        min_effect_delta: float,
        min_relative_delta: float,
        pvalue_threshold: float,
    ) -> dict[str, Any] | None:
        """按混合阈值确认候选变点是否为异常。"""
        delta = stats.after_rate - stats.before_rate
        abs_delta = abs(delta)
        relative_base = max(abs(stats.before_rate), min_effect_delta)
        relative_delta = abs_delta / relative_base
        p_value = _two_proportion_pvalue(
            stats.before_missing,
            stats.before_total,
            stats.after_missing,
            stats.after_total,
        )
        reasons: list[str] = []
        if abs_delta >= min_abs_delta:
            reasons.append("abs_delta")
        if abs_delta >= min_effect_delta and relative_delta >= min_relative_delta:
            reasons.append("relative_delta")
        if abs_delta >= min_effect_delta and p_value <= pvalue_threshold:
            reasons.append("z_test")
        if not reasons:
            return None

        direction: Direction = "increase" if delta > 0 else "decrease"
        periods = feature_df.get_column("period").to_list()
        anomaly_score = max(
            abs_delta / max(min_abs_delta, FLOAT_TOLERANCE),
            relative_delta / max(min_relative_delta, FLOAT_TOLERANCE),
            -np.log10(max(p_value, PROBABILITY_EPSILON))
            / max(-np.log10(pvalue_threshold), FLOAT_TOLERANCE),
        )
        return {
            "feature": feature,
            "data_source": data_source,
            "start_period": str(periods[max(0, change_idx - 1)]),
            "end_period": str(periods[min(len(periods) - 1, change_idx)]),
            "change_period": str(periods[change_idx]),
            "before_missing_rate": stats.before_rate,
            "after_missing_rate": stats.after_rate,
            "abs_delta": abs_delta,
            "relative_delta": relative_delta,
            "p_value": p_value,
            "direction": direction,
            "anomaly_score": float(anomaly_score),
            "reason": ",".join(reasons),
        }

    @staticmethod
    def _build_summary_table(detail_table: pl.DataFrame) -> pl.DataFrame:
        """由异常明细构造特征级汇总。"""
        if detail_table.is_empty():
            return _empty_summary_table()
        return (
            detail_table
            .group_by(["feature", "data_source"])
            .agg(
                [
                    pl.len().alias("anomaly_count"),
                    pl.col("abs_delta").max().alias("max_abs_delta"),
                    pl.col("change_period").min().alias("first_anomaly_period"),
                    pl.col("change_period").max().alias("last_anomaly_period"),
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
        """由异常明细构造数据源级汇总。"""
        if not feature_data_source:
            return _empty_source_table()

        source_feature_df = pl.DataFrame(
            {
                "feature": target_features,
                "data_source": [
                    feature_data_source.get(feature, "UNMAPPED") for feature in target_features
                ],
            }
        )
        feature_counts = source_feature_df.group_by("data_source").agg(
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
                .sort("data_source")
            )

        source_anomaly = (
            detail_table
            .group_by("data_source")
            .agg(
                [
                    pl.col("feature").n_unique().alias("anomaly_feature_count"),
                    pl.len().alias("anomaly_count"),
                    pl.col("abs_delta").max().alias("max_abs_delta"),
                ]
            )
        )
        return (
            feature_counts
            .join(source_anomaly, on="data_source", how="left")
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
            .sort(["anomaly_feature_count", "max_abs_delta"], descending=[True, True])
        )


def _two_proportion_pvalue(
    missing_a: float,
    total_a: float,
    missing_b: float,
    total_b: float,
) -> float:
    """计算两比例 z 检验的双侧 p 值。"""
    if total_a <= 0 or total_b <= 0:
        return 1.0
    rate_a = missing_a / total_a
    rate_b = missing_b / total_b
    pooled = (missing_a + missing_b) / (total_a + total_b)
    variance = pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b)
    if variance <= 0:
        return 1.0 if abs(rate_a - rate_b) == 0 else 0.0
    z_score = abs(rate_b - rate_a) / sqrt(variance)
    return float(erfc(z_score / sqrt(2.0)))
