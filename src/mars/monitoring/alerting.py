"""MARS 监控报警文本生成器。"""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Literal

import pandas as pd
import polars as pl

from mars.core.base import MarsBaseEstimator
from mars.monitoring.monitor import MarsMonitoringReport

AlertSeverity = Literal["严重", "警告", "关注"]


@dataclass(frozen=True)
class MarsMonitoringAlertConfig:
    """
    监控报警阈值配置。

    Attributes
    ----------
    psi_warn : float
        PSI 警告阈值。
    psi_critical : float
        PSI 严重阈值。
    missing_delta_warn : float
        缺失率跨期变化警告阈值。
    missing_delta_critical : float
        缺失率跨期变化严重阈值。
    bin_pct_delta_warn : float
        分箱占比跨期变化警告阈值。
    bin_pct_delta_critical : float
        分箱占比跨期变化严重阈值。
    bad_rate_delta_warn : float
        坏账率跨期变化警告阈值。
    bad_rate_delta_critical : float
        坏账率跨期变化严重阈值。
    risk_corr_warn : float
        风险趋势相关性警告阈值。
    risk_corr_critical : float
        风险趋势相关性严重阈值。
    target_observed_rate_warn : float
        target 表现覆盖率警告阈值。
    target_observed_rate_critical : float
        target 表现覆盖率严重阈值。
    score_mean_relative_delta_warn : float
        模型分均值相对变化警告阈值。
    score_mean_relative_delta_critical : float
        模型分均值相对变化严重阈值。
    max_items_per_priority : int
        每个优先级最多输出的报警条数。
    """

    psi_warn: float = 0.10
    psi_critical: float = 0.25
    missing_delta_warn: float = 0.03
    missing_delta_critical: float = 0.08
    bin_pct_delta_warn: float = 0.05
    bin_pct_delta_critical: float = 0.10
    bad_rate_delta_warn: float = 0.03
    bad_rate_delta_critical: float = 0.08
    risk_corr_warn: float = 0.80
    risk_corr_critical: float = 0.60
    target_observed_rate_warn: float = 0.80
    target_observed_rate_critical: float = 0.50
    score_mean_relative_delta_warn: float = 0.05
    score_mean_relative_delta_critical: float = 0.10
    max_items_per_priority: int = 8


@dataclass(frozen=True)
class _AlertItem:
    """内部报警条目。"""

    severity: AlertSeverity
    text: str


class MarsMonitoringAlerter(MarsBaseEstimator):
    """
    监控报警文本生成器。

    该组件只消费 `MarsMonitoringReport` 中已有的结构化数据，不重新计算分箱。
    若报告缺少某类表或字段，对应检查会跳过并写入文本末尾的数据跳过说明。
    报警器会读取报告元数据中的趋势列顺序，用于识别模型分均值趋势和 target 覆盖率的基准期、
    最新期。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.feature import MarsNativeBinner
    >>> from mars.monitoring import MarsMonitoringReport, generate_monitoring_alert
    >>> report = MarsMonitoringReport(
    ...     summary_table=pl.DataFrame({"feature": ["score"], "psi_max": [0.0]}),
    ...     detail_table=pl.DataFrame(),
    ...     trend_tables={},
    ...     missing_by_day_table=None,
    ...     bin_stat_table=pl.DataFrame(),
    ...     bin_stat_trend_tables={},
    ...     target_observation_table=None,
    ...     binner=MarsNativeBinner(),
    ...     features=["score"],
    ...     target=None,
    ...     metadata={},
    ... )
    >>> "MARS 监控报警摘要" in generate_monitoring_alert(report, score_key="score", model_features=[])
    True
    """

    def __init__(self, config: MarsMonitoringAlertConfig | None = None) -> None:
        """
        初始化监控报警器。

        Parameters
        ----------
        config : MarsMonitoringAlertConfig | None
            报警阈值配置；不传时使用默认阈值。
        """
        super().__init__()
        self.config = config or MarsMonitoringAlertConfig()

    def generate(
        self,
        report: MarsMonitoringReport,
        *,
        score_key: str,
        model_features: List[str],
    ) -> str:
        """
        基于监控报告生成中文报警文本。

        Parameters
        ----------
        report : MarsMonitoringReport
            监控模块生成的结构化报告。
        score_key : str
            模型分、概率或评分字段名。
        model_features : List[str]
            模型使用的特征列表。

        Returns
        -------
        str
            按优先级排序的中文报警文本。
        """
        features = self._dedupe_features(score_key, model_features)
        report_metadata = dict(report.metadata or {})
        skipped: list[str] = []
        alerts: list[_AlertItem] = []

        summary_table = self._ensure_optional_frame(report.summary_table)
        if summary_table is None or summary_table.is_empty() or "feature" not in summary_table.columns:
            self._add_skip(skipped, "summary_table 缺少 feature 维度，跳过特征级汇总检查")
        else:
            self._check_summary_table(
                summary_table=summary_table,
                score_key=score_key,
                features=features,
                alerts=alerts,
                skipped=skipped,
            )

        target_table = self._ensure_optional_frame(report.target_observation_table)
        self._check_target_observation(
            target_table=target_table,
            report_metadata=report_metadata,
            alerts=alerts,
            skipped=skipped,
        )

        trend_tables = {
            name: frame
            for name, table in report.trend_tables.items()
            if (frame := self._ensure_optional_frame(table)) is not None
        }
        self._check_trend_tables(
            trend_tables=trend_tables,
            report_metadata=report_metadata,
            score_key=score_key,
            features=features,
            alerts=alerts,
            skipped=skipped,
        )

        bin_stat_trend_tables = {
            name: frame
            for name, table in report.bin_stat_trend_tables.items()
            if (frame := self._ensure_optional_frame(table)) is not None
        }
        self._check_bin_stat_trends(
            bin_stat_trend_tables=bin_stat_trend_tables,
            report_metadata=report_metadata,
            score_key=score_key,
            features=features,
            alerts=alerts,
            skipped=skipped,
        )

        return self._format_alert_text(
            alerts=alerts,
            skipped=skipped,
            score_key=score_key,
            model_features=model_features,
        )

    def _ensure_optional_frame(
        self,
        table: pl.DataFrame | pd.DataFrame | None,
    ) -> pl.DataFrame | None:
        """把可选表转换为 Polars DataFrame，缺表时返回 None。"""
        if table is None:
            return None
        result = self._ensure_polars_dataframe(table)
        return result.collect() if isinstance(result, pl.LazyFrame) else result

    @staticmethod
    def _dedupe_features(score_key: str, model_features: List[str]) -> list[str]:
        """按输入顺序去重，保证模型分排在检查范围首位。"""
        result: list[str] = []
        for feature in [score_key, *model_features]:
            if feature not in result:
                result.append(feature)
        return result

    def _check_summary_table(
        self,
        *,
        summary_table: pl.DataFrame,
        score_key: str,
        features: list[str],
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查特征级汇总指标。"""
        available_features = set(summary_table.get_column("feature").cast(pl.String).to_list())
        for feature in features:
            if feature not in available_features:
                self._add_skip(skipped, f"summary_table 缺少字段 `{feature}`，跳过该字段汇总检查")
                continue

            feature_row = summary_table.filter(pl.col("feature") == feature)
            label = self._feature_label(feature, score_key)
            self._add_high_value_alert(
                alerts,
                value=self._first_float(feature_row, "psi_max"),
                warn=self.config.psi_warn,
                critical=self.config.psi_critical,
                message=f"{label} PSI 最大值达到 {{value:.4f}}",
            )

            missing_min = self._first_float(feature_row, "missing_min")
            missing_max = self._first_float(feature_row, "missing_max")
            if missing_min is not None and missing_max is not None:
                self._add_high_value_alert(
                    alerts,
                    value=abs(missing_max - missing_min),
                    warn=self.config.missing_delta_warn,
                    critical=self.config.missing_delta_critical,
                    message=f"{label} 缺失率区间变化达到 {{value:.4f}}",
                )

            self._add_low_value_alert(
                alerts,
                value=self._first_float(feature_row, "rc_min"),
                warn=self.config.risk_corr_warn,
                critical=self.config.risk_corr_critical,
                message=f"{label} 风险趋势相关性最低为 {{value:.4f}}",
            )

    def _check_trend_tables(
        self,
        *,
        trend_tables: dict[str, pl.DataFrame],
        report_metadata: dict[str, Any],
        score_key: str,
        features: list[str],
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查 report 趋势表中的 PSI、缺失率和坏账率波动。"""
        if "psi" not in trend_tables:
            self._add_skip(skipped, "trend_tables 缺少 psi，跳过 PSI 趋势检查")
        else:
            self._check_feature_trend_max(
                table=trend_tables["psi"],
                report_metadata=report_metadata,
                metric_name="PSI 趋势最大值",
                score_key=score_key,
                features=features,
                warn=self.config.psi_warn,
                critical=self.config.psi_critical,
                alerts=alerts,
                skipped=skipped,
            )

        if "missing" not in trend_tables:
            self._add_skip(skipped, "trend_tables 缺少 missing，跳过缺失率趋势检查")
        else:
            self._check_feature_trend_delta(
                table=trend_tables["missing"],
                report_metadata=report_metadata,
                metric_name="缺失率趋势变化",
                score_key=score_key,
                features=features,
                warn=self.config.missing_delta_warn,
                critical=self.config.missing_delta_critical,
                alerts=alerts,
                skipped=skipped,
            )

        if "bad_rate" not in trend_tables:
            self._add_skip(skipped, "trend_tables 缺少 bad_rate，跳过坏账率趋势检查")
        else:
            self._check_feature_trend_delta(
                table=trend_tables["bad_rate"],
                report_metadata=report_metadata,
                metric_name="坏账率趋势变化",
                score_key=score_key,
                features=features,
                warn=self.config.bad_rate_delta_warn,
                critical=self.config.bad_rate_delta_critical,
                alerts=alerts,
                skipped=skipped,
            )

    def _check_bin_stat_trends(
        self,
        *,
        bin_stat_trend_tables: dict[str, pl.DataFrame],
        report_metadata: dict[str, Any],
        score_key: str,
        features: list[str],
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查分箱层级的占比、坏账率和模型分均值趋势。"""
        if "pct" not in bin_stat_trend_tables:
            self._add_skip(skipped, "bin_stat_trend_tables 缺少 pct，跳过分箱占比漂移检查")
        else:
            self._check_bin_trend_delta(
                table=bin_stat_trend_tables["pct"],
                report_metadata=report_metadata,
                metric_name="分箱占比变化",
                score_key=score_key,
                features=features,
                warn=self.config.bin_pct_delta_warn,
                critical=self.config.bin_pct_delta_critical,
                alerts=alerts,
                skipped=skipped,
            )

        if "bad_rate" not in bin_stat_trend_tables:
            self._add_skip(skipped, "bin_stat_trend_tables 缺少 bad_rate，跳过分箱坏账率波动检查")
        else:
            self._check_bin_trend_delta(
                table=bin_stat_trend_tables["bad_rate"],
                report_metadata=report_metadata,
                metric_name="分箱坏账率变化",
                score_key=score_key,
                features=features,
                warn=self.config.bad_rate_delta_warn,
                critical=self.config.bad_rate_delta_critical,
                alerts=alerts,
                skipped=skipped,
            )

        if "pct" not in bin_stat_trend_tables or "mean" not in bin_stat_trend_tables:
            self._add_skip(skipped, "bin_stat_trend_tables 缺少 pct 或 mean，跳过模型分均值趋势估算")
            return

        self._check_score_mean_trend(
            pct_table=bin_stat_trend_tables["pct"],
            mean_table=bin_stat_trend_tables["mean"],
            report_metadata=report_metadata,
            score_key=score_key,
            alerts=alerts,
            skipped=skipped,
        )

    def _check_target_observation(
        self,
        *,
        target_table: pl.DataFrame | None,
        report_metadata: dict[str, Any],
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查 target 表现覆盖率。"""
        if target_table is None or target_table.is_empty():
            self._add_skip(skipped, "target_observation_table 缺失，跳过 target 表现覆盖率检查")
            return
        if "target_observed_rate" not in target_table.columns:
            self._add_skip(skipped, "target_observation_table 缺少 target_observed_rate")
            return

        group_col = self._infer_group_col(target_table)
        observed_table = target_table.filter(pl.col(group_col) != "Total")
        if observed_table.is_empty():
            observed_table = target_table

        rates = self._numeric_values_from_columns(observed_table, ["target_observed_rate"])
        if not rates:
            self._add_skip(skipped, "target_observation_table 无有效表现覆盖率")
            return

        min_rate = min(rates)
        self._add_low_value_alert(
            alerts,
            value=min_rate,
            warn=self.config.target_observed_rate_warn,
            critical=self.config.target_observed_rate_critical,
            message="target 表现覆盖率最低为 {value:.4f}，风险类指标解释需谨慎",
        )

        latest_label = self._latest_group_label(report_metadata)
        if latest_label is not None:
            latest_group = observed_table.filter(pl.col(group_col).cast(pl.String) == latest_label)
            if latest_group.is_empty():
                latest_group = observed_table.sort(group_col).tail(1)
        else:
            latest_group = observed_table.sort(group_col).tail(1)
        latest_rate = self._first_float(latest_group, "target_observed_rate")
        if latest_rate is not None and latest_rate != min_rate:
            latest_label = latest_group.select(pl.col(group_col).cast(pl.String).first()).item()
            self._add_low_value_alert(
                alerts,
                value=latest_rate,
                warn=self.config.target_observed_rate_warn,
                critical=self.config.target_observed_rate_critical,
                message=f"最新分组 `{latest_label}` target 表现覆盖率为 {{value:.4f}}",
            )

    def _check_feature_trend_max(
        self,
        *,
        table: pl.DataFrame,
        report_metadata: dict[str, Any],
        metric_name: str,
        score_key: str,
        features: list[str],
        warn: float,
        critical: float,
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查特征趋势表中的最大值。"""
        for feature in features:
            values = self._feature_trend_values(table, feature, report_metadata, skipped)
            if not values:
                continue
            self._add_high_value_alert(
                alerts,
                value=max(values),
                warn=warn,
                critical=critical,
                message=f"{self._feature_label(feature, score_key)} {metric_name} 达到 {{value:.4f}}",
            )

    def _check_feature_trend_delta(
        self,
        *,
        table: pl.DataFrame,
        report_metadata: dict[str, Any],
        metric_name: str,
        score_key: str,
        features: list[str],
        warn: float,
        critical: float,
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查特征趋势表中的跨期极差。"""
        for feature in features:
            values = self._feature_trend_values(table, feature, report_metadata, skipped)
            if len(values) < 2:
                continue
            self._add_high_value_alert(
                alerts,
                value=max(values) - min(values),
                warn=warn,
                critical=critical,
                message=f"{self._feature_label(feature, score_key)} {metric_name} 达到 {{value:.4f}}",
            )

    def _check_bin_trend_delta(
        self,
        *,
        table: pl.DataFrame,
        report_metadata: dict[str, Any],
        metric_name: str,
        score_key: str,
        features: list[str],
        warn: float,
        critical: float,
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """检查分箱趋势表中的最大跨期极差。"""
        if "feature" not in table.columns:
            self._add_skip(skipped, f"{metric_name} 表缺少 feature")
            return

        value_cols = self._ordered_trend_value_columns(table, report_metadata)
        if not value_cols:
            self._add_skip(skipped, f"{metric_name} 表缺少趋势取值列")
            return

        for feature in features:
            feature_rows = table.filter(pl.col("feature") == feature)
            if feature_rows.is_empty():
                self._add_skip(skipped, f"{metric_name} 表缺少字段 `{feature}`")
                continue

            max_delta = 0.0
            target_bin = None
            for row in feature_rows.iter_rows(named=True):
                values = self._row_numeric_values(row, value_cols)
                if len(values) < 2:
                    continue
                delta = max(values) - min(values)
                if delta > max_delta:
                    max_delta = delta
                    target_bin = row.get("bin_label") or row.get("bin_index")

            if target_bin is None:
                continue

            self._add_high_value_alert(
                alerts,
                value=max_delta,
                warn=warn,
                critical=critical,
                message=(
                    f"{self._feature_label(feature, score_key)} {metric_name} 最大变化为 "
                    f"{{value:.4f}}，分箱={target_bin}"
                ),
            )

    def _check_score_mean_trend(
        self,
        *,
        pct_table: pl.DataFrame,
        mean_table: pl.DataFrame,
        report_metadata: dict[str, Any],
        score_key: str,
        alerts: list[_AlertItem],
        skipped: list[str],
    ) -> None:
        """基于分箱占比和分箱均值估算模型分整体均值趋势。"""
        if "feature" not in pct_table.columns or "feature" not in mean_table.columns:
            self._add_skip(skipped, "模型分均值趋势表缺少 feature")
            return

        pct_rows = pct_table.filter(pl.col("feature") == score_key)
        mean_rows = mean_table.filter(pl.col("feature") == score_key)
        if pct_rows.is_empty() or mean_rows.is_empty():
            self._add_skip(skipped, f"模型分 `{score_key}` 缺少 pct 或 mean 分箱趋势")
            return

        mean_value_cols = set(self._ordered_trend_value_columns(mean_rows, report_metadata))
        value_cols = [
            col
            for col in self._ordered_trend_value_columns(pct_rows, report_metadata)
            if col in mean_value_cols
        ]
        if len(value_cols) < 2:
            self._add_skip(skipped, f"模型分 `{score_key}` 均值趋势可用分组不足")
            return

        mean_by_group: dict[str, float] = {}
        index_cols = ["feature", "bin_index", "bin_label", "bin_type"]
        joined = pct_rows.join(
            mean_rows,
            on=[col for col in index_cols if col in pct_rows.columns and col in mean_rows.columns],
            how="inner",
            suffix="_mean",
        )
        for group in value_cols:
            pct_col = group
            mean_col = f"{group}_mean"
            if pct_col not in joined.columns or mean_col not in joined.columns:
                continue
            group_value = (
                joined
                .select((pl.col(pct_col).cast(pl.Float64) * pl.col(mean_col).cast(pl.Float64)).sum())
                .item()
            )
            if group_value is not None and isfinite(float(group_value)):
                mean_by_group[group] = float(group_value)

        if len(mean_by_group) < 2:
            self._add_skip(skipped, f"模型分 `{score_key}` 均值趋势有效分组不足")
            return

        ordered_groups = [group for group in value_cols if group in mean_by_group]
        if len(ordered_groups) < 2:
            self._add_skip(skipped, f"模型分 `{score_key}` 均值趋势有效分组不足")
            return

        if self._trend_column_order(report_metadata) == "desc":
            baseline_group = ordered_groups[-1]
            latest_group = ordered_groups[0]
        else:
            baseline_group = ordered_groups[0]
            latest_group = ordered_groups[-1]

        first_value = mean_by_group[baseline_group]
        latest_value = mean_by_group[latest_group]
        denominator = max(abs(first_value), 1e-9)
        relative_delta = abs(latest_value - first_value) / denominator
        self._add_high_value_alert(
            alerts,
            value=relative_delta,
            warn=self.config.score_mean_relative_delta_warn,
            critical=self.config.score_mean_relative_delta_critical,
            message=(
                f"模型分 `{score_key}` 估算均值相对变化达到 {{value:.4f}}，"
                f"基准={first_value:.4f}，最新={latest_value:.4f}"
            ),
        )

    def _feature_trend_values(
        self,
        table: pl.DataFrame,
        feature: str,
        report_metadata: dict[str, Any],
        skipped: list[str],
    ) -> list[float]:
        """提取单个特征在趋势表中的数值序列。"""
        if "feature" not in table.columns:
            self._add_skip(skipped, "趋势表缺少 feature")
            return []

        feature_rows = table.filter(pl.col("feature") == feature)
        if feature_rows.is_empty():
            self._add_skip(skipped, f"趋势表缺少字段 `{feature}`")
            return []

        value_cols = self._ordered_trend_value_columns(feature_rows, report_metadata)
        if not value_cols:
            return []
        return self._numeric_values_from_columns(feature_rows, value_cols)

    @staticmethod
    def _trend_value_columns(table: pl.DataFrame) -> list[str]:
        """识别趋势宽表中的分组取值列。"""
        index_cols = {"feature", "dtype", "bin_index", "bin_label", "bin_type", "Total"}
        return [col for col in table.columns if col not in index_cols]

    @classmethod
    def _ordered_trend_value_columns(
        cls,
        table: pl.DataFrame,
        report_metadata: dict[str, Any],
    ) -> list[str]:
        """按 report 元数据中的趋势顺序提取当前表可用的取值列。"""
        table_value_cols = cls._trend_value_columns(table)
        metadata_cols = [
            str(col)
            for col in report_metadata.get("trend_value_columns", [])
            if str(col) in table_value_cols
        ]
        remaining_cols = [col for col in table_value_cols if col not in metadata_cols]
        return metadata_cols + remaining_cols

    @staticmethod
    def _trend_column_order(report_metadata: dict[str, Any]) -> Literal["asc", "desc"]:
        """读取 report 记录的趋势列排序方向，缺省按升序处理。"""
        value = report_metadata.get("trend_column_order")
        return "desc" if value == "desc" else "asc"

    @classmethod
    def _latest_group_label(cls, report_metadata: dict[str, Any]) -> str | None:
        """根据趋势列顺序返回最新分组标签。"""
        value_cols = [str(col) for col in report_metadata.get("trend_value_columns", [])]
        if not value_cols:
            return None
        if cls._trend_column_order(report_metadata) == "desc":
            return value_cols[0]
        return value_cols[-1]

    @staticmethod
    def _numeric_values_from_columns(table: pl.DataFrame, columns: list[str]) -> list[float]:
        """从指定列中提取有限数值。"""
        values: list[float] = []
        for col in columns:
            if col not in table.columns:
                continue
            for value in table.get_column(col).to_list():
                parsed = MarsMonitoringAlerter._as_float(value)
                if parsed is not None:
                    values.append(parsed)
        return values

    @staticmethod
    def _row_numeric_values(row: dict[str, Any], columns: list[str]) -> list[float]:
        """从行字典中提取有限数值。"""
        values: list[float] = []
        for col in columns:
            parsed = MarsMonitoringAlerter._as_float(row.get(col))
            if parsed is not None:
                values.append(parsed)
        return values

    @staticmethod
    def _first_float(table: pl.DataFrame, column: str) -> float | None:
        """读取单列表的首个有限浮点数。"""
        if column not in table.columns or table.is_empty():
            return None
        return MarsMonitoringAlerter._as_float(table.select(pl.col(column).first()).item())

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """把标量转为有限浮点数，失败或空值返回 None。"""
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(result):
            return None
        return result

    @staticmethod
    def _feature_label(feature: str, score_key: str) -> str:
        """返回报警文本中使用的字段称谓。"""
        if feature == score_key:
            return f"模型分 `{feature}`"
        return f"特征 `{feature}`"

    @staticmethod
    def _infer_group_col(table: pl.DataFrame) -> str:
        """从 target 表现覆盖表中推断分组列名。"""
        metric_cols = {
            "sample_count",
            "target_observed_count",
            "target_unobserved_count",
            "target_observed_rate",
            "bad",
            "bad_rate_observed",
        }
        for col in table.columns:
            if col not in metric_cols:
                return col
        return table.columns[0]

    def _add_high_value_alert(
        self,
        alerts: list[_AlertItem],
        *,
        value: float | None,
        warn: float,
        critical: float,
        message: str,
    ) -> None:
        """按越高越差的指标阈值添加报警。"""
        if value is None:
            return
        severity = self._high_value_severity(value, warn=warn, critical=critical)
        if severity is not None:
            alerts.append(_AlertItem(severity=severity, text=message.format(value=value)))

    def _add_low_value_alert(
        self,
        alerts: list[_AlertItem],
        *,
        value: float | None,
        warn: float,
        critical: float,
        message: str,
    ) -> None:
        """按越低越差的指标阈值添加报警。"""
        if value is None:
            return
        severity = self._low_value_severity(value, warn=warn, critical=critical)
        if severity is not None:
            alerts.append(_AlertItem(severity=severity, text=message.format(value=value)))

    @staticmethod
    def _high_value_severity(
        value: float,
        *,
        warn: float,
        critical: float,
    ) -> AlertSeverity | None:
        """判断越高越差指标的报警等级。"""
        if value >= critical:
            return "严重"
        if value >= warn:
            return "警告"
        if value >= warn / 2:
            return "关注"
        return None

    @staticmethod
    def _low_value_severity(
        value: float,
        *,
        warn: float,
        critical: float,
    ) -> AlertSeverity | None:
        """判断越低越差指标的报警等级。"""
        attention = warn + (1.0 - warn) / 2.0
        if value <= critical:
            return "严重"
        if value <= warn:
            return "警告"
        if value <= attention:
            return "关注"
        return None

    @staticmethod
    def _add_skip(skipped: list[str], text: str) -> None:
        """追加去重后的跳过说明。"""
        if text not in skipped:
            skipped.append(text)

    def _format_alert_text(
        self,
        *,
        alerts: list[_AlertItem],
        skipped: list[str],
        score_key: str,
        model_features: list[str],
    ) -> str:
        """组装最终报警文本。"""
        severity_order: Dict[AlertSeverity, int] = {"严重": 0, "警告": 1, "关注": 2}
        sorted_alerts = sorted(alerts, key=lambda item: severity_order[item.severity])
        counts = {
            severity: sum(item.severity == severity for item in sorted_alerts)
            for severity in severity_order
        }

        lines = [
            "MARS 监控报警摘要",
            "",
            f"总体结论：严重 {counts['严重']} 项，警告 {counts['警告']} 项，关注 {counts['关注']} 项。",
            f"检查范围：模型分 `{score_key}`；模型特征 {len(model_features)} 个。",
        ]
        if not sorted_alerts:
            lines.append("未发现超过默认阈值的监控报警项。")

        for severity in severity_order:
            items = [item for item in sorted_alerts if item.severity == severity]
            if not items:
                continue
            lines.append("")
            lines.append(f"{severity}：")
            for idx, item in enumerate(items[: self.config.max_items_per_priority], start=1):
                lines.append(f"{idx}. {item.text}")
            remaining = len(items) - self.config.max_items_per_priority
            if remaining > 0:
                lines.append(f"... 还有 {remaining} 项{severity}报警未展开。")

        if skipped:
            lines.append("")
            lines.append("数据跳过：")
            for idx, item in enumerate(skipped, start=1):
                lines.append(f"{idx}. {item}")

        return "\n".join(lines)


def generate_monitoring_alert(
    report: MarsMonitoringReport,
    *,
    score_key: str,
    model_features: List[str],
    config: MarsMonitoringAlertConfig | None = None,
) -> str:
    """
    生成监控报警文本的轻量函数入口。

    Parameters
    ----------
    report : MarsMonitoringReport
        监控模块生成的结构化报告。
    score_key : str
        模型分、概率或评分字段名。
    model_features : List[str]
        模型使用的特征列表。
    config : MarsMonitoringAlertConfig | None
        报警阈值配置；不传时使用默认阈值。

    Returns
    -------
    str
        按优先级排序的中文报警文本。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.feature import MarsNativeBinner
    >>> from mars.monitoring import MarsMonitoringReport, generate_monitoring_alert
    >>> report = MarsMonitoringReport(
    ...     summary_table=pl.DataFrame({"feature": ["score"], "psi_max": [0.2]}),
    ...     detail_table=pl.DataFrame(),
    ...     trend_tables={},
    ...     missing_by_day_table=None,
    ...     bin_stat_table=pl.DataFrame(),
    ...     bin_stat_trend_tables={},
    ...     target_observation_table=None,
    ...     binner=MarsNativeBinner(),
    ...     features=["score"],
    ...     target=None,
    ...     metadata={},
    ... )
    >>> "警告" in generate_monitoring_alert(report, score_key="score", model_features=[])
    True
    """
    return MarsMonitoringAlerter(config=config).generate(
        report,
        score_key=score_key,
        model_features=model_features,
    )
