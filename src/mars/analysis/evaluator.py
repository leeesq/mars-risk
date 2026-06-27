"""MARS 特征分箱评估模块。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.analysis._evaluation.aggregation import (
    aggregate_basic_stats,
    build_missing_by_day_table,
    get_benchmark_dist,
    rollup_total_stats,
)
from mars.analysis._evaluation.context import (
    build_binner,
    normalize_binary_target_column,
    normalize_feature_data_source,
    prepare_group_context,
    resolve_profile_by,
)
from mars.analysis._evaluation.metrics import (
    calculate_metrics_from_stats,
    ensure_woe_info,
)
from mars.analysis._evaluation.references import (
    build_risk_corr_long,
    build_risk_corr_reference_table,
    empty_risk_corr_reference_table,
)
from mars.analysis._evaluation.report_parts import build_binning_trend_tables
from mars.compute import (
    RiskCorrBaseline,
    amount_distribution_exprs,
    amount_metric_exprs,
    bad_rate_expr,
    bin_missing_rate_expr,
    distribution_rate_expr,
    global_distribution_expr,
    normalize_risk_corr_baseline,
    normalized_auc_expr,
    observed_auc_agg_expr,
    observed_iv_agg_expr,
    observed_ks_agg_expr,
    observed_lift_max_agg_expr,
    observed_lift_min_agg_expr,
    ordered_count_metric_exprs,
    risk_corr_expr,
)
from mars.core.base import MarsBaseEstimator
from mars.feature.binning.base import MarsBinnerBase
from mars.reporting import MarsBinningReport
from mars.utils.date import MarsDate
from mars.utils.decorators import time_it
from mars.utils.logger import logger


@dataclass
class MarsRiskProfile:
    """
    单次风险画像结果。

    `MarsRiskProfile` 是 `MarsBinEvaluator.evaluate` 和 `profile_risk` 的统一返回对象，
    用来把评估报告、拟合后的分箱器、本次目标列和元数据放在同一个结果容器中。
    调用方可以直接使用 `report` 查看或导出报表，也可以复用 `binner` 对新样本做转换。

    Attributes
    ----------
    report : MarsBinningReport
        风险评估报告，包含汇总表、明细表、趋势表和导出方法。
    binner : MarsBinnerBase
        本次评估拟合或显式传入的分箱器。
    targets : list of str
        本次画像覆盖的目标列列表；无标签画像时为空列表。
    metadata : dict
        本次运行的列名、特征范围、分箱配置和其他上下文信息。
    """

    report: MarsBinningReport
    binner: MarsBinnerBase
    targets: list[str]
    metadata: dict[str, Any]


class MarsBinEvaluator(MarsBaseEstimator):
    """
    分箱效果评估器。

    该评估器用于对一组特征的分箱结果计算 IV、KS、AUC、PSI、Lift、单调性、
    分组趋势和明细分布。构造函数只保存稳定评估策略和默认分箱器配置；数据、
    目标列、特征范围、分组列和时间聚合粒度都由每次 `evaluate` 调用传入。

    每次 `evaluate` 都会返回新的 `MarsRiskProfile`，其中包含报告对象和本次使用的
    分箱器。评估器实例不会把上一次拟合出的分箱器作为下一次调用的隐式状态，
    因此同一个实例可以安全地连续评估不同特征集合或不同数据集。

    Attributes
    ----------
    binning_type : str
        默认分箱器类型。
    binner_params : dict
        默认分箱器参数副本。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import MarsBinEvaluator
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> evaluator = MarsBinEvaluator(
    ...     binning_type="native",
    ...     binner_params={"method": "quantile", "n_bins": 2},
    ... )
    >>> profile = evaluator.evaluate(df, target="y", features=["age"])
    >>> "age" in profile.report.summary_table.get_column("feature").to_list()
    True
    """

    MARS_GROUP_COL = "mars_group"

    def __init__(
        self,
        *,
        binning_type: Literal["native", "optimal", "lite_opt"] = "native",
        binner_params: Dict[str, Any] | None = None,
        feature_start_aware_reference: bool = False,
        risk_corr_baseline: RiskCorrBaseline = "total",
    ) -> None:
        """
        初始化分箱评估器。

        Parameters
        ----------
        binning_type : Literal['native', 'optimal', 'lite_opt']
            未显式传入分箱器时使用的默认分箱策略。
        binner_params : Dict[str, Any] | None
            构造默认分箱器时使用的参数。
        feature_start_aware_reference : bool
            是否默认启用 feature-start aware reference，用于 PSI 基准重锚。
        risk_corr_baseline : RiskCorrBaseline
            RC 的默认基准选择方式。

        Raises
        ------
        ValueError
            当 `binning_type` 不是 `native`、`optimal` 或 `lite_opt` 时抛出。
        """
        super().__init__()
        normalized_binning_type = str(binning_type).lower()
        valid_binning_types = {"native", "optimal", "lite_opt"}
        if normalized_binning_type not in valid_binning_types:
            raise ValueError(
                "binning_type must be one of {'native', 'optimal', 'lite_opt'}, "
                f"got {binning_type!r}."
            )
        self.target: str | None = None
        self.binner: MarsBinnerBase | None = None
        self.has_target_: bool = False
        self.binning_type = normalized_binning_type
        self.binner_params = dict(binner_params or {})
        self.feature_start_aware_reference = bool(feature_start_aware_reference)
        self.risk_corr_baseline = normalize_risk_corr_baseline(risk_corr_baseline)

    @time_it
    def evaluate(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        *,
        target: str | None = None,
        features: List[str] | None = None,
        binner: MarsBinnerBase | None = None,
        feature_data_source: Dict[str, List[str]] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        feature_start_aware_reference: bool | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        weights_col: str | None = None,
        amount_col: str | None = None,
        batch_size: int = 100,
    ) -> MarsRiskProfile:
        """
        对一次数据上下文执行分箱评估。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待评估样本表。
        target : str | None
            二分类目标列名。为 `None` 或列不存在时会进入无标签模式，只计算分布类指标
            和 PSI，不计算 IV、KS、AUC 等依赖标签的指标。
        features : List[str] | None
            本次评估的特征列；不传时自动排除目标列和分组列后选择候选特征。
        binner : MarsBinnerBase | None
            显式复用的分箱器；传入后不会再根据 `binning_type` 和 `binner_params`
            构造新分箱器。
        feature_data_source : Dict[str, List[str]] | None
            特征来源映射，只对本次 active features 生效，用于报告中保留来源分层。
        group_col : str | None
            已存在的分组列名，例如月份、客群或样本切片。
        time_col : str | None
            原始日期列名；与 `time_grain` 配合时会生成临时时间分组列。
        time_grain : str | None
            时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
            仅在传入 `time_col` 时生效，默认按 `"month"` 聚合。
        feature_start_aware_reference : bool | None
            是否按特征首次出现的分组选择 PSI 基准。传入 `None` 时沿用实例初始化时保存的默认值。
        risk_corr_baseline : RiskCorrBaseline | None
            本次评估使用的 RC 基准；传入 `None` 时沿用实例初始化时保存的默认值。
        psi_include_missing : bool
            计算 PSI 时是否单独保留缺失值分布。
        psi_include_special : bool
            计算 PSI 时是否单独保留特殊值分布。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame, None]
            外部 benchmark 样本；传入后分布稳定性可与该样本进行对比。
        weights_col : str | None
            样本权重列名。
        amount_col : str | None
            金额列名；传入后会在 `detail_table` 中额外产出金额视角指标。
        batch_size : int
            批量评估时的特征批大小。

        Returns
        -------
        MarsRiskProfile
            单次风险画像结果，包含评估报告、分箱器、目标列列表和运行元数据。

        Raises
        ------
        ValueError
            当必要列缺失、分箱器配置冲突或输入数据无法评估时抛出。
        """
        # 上下文准备
        working_df = self._ensure_polars_dataframe(df)
        if benchmark_df is not None:
            benchmark_df = self._ensure_polars_dataframe(benchmark_df)
        original_target = target
        effective_target = target if target else "dummy_target"
        profile_by = resolve_profile_by(
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
        )
        dt_col = time_col
        effective_feature_start_reference = (
            self.feature_start_aware_reference
            if feature_start_aware_reference is None
            else bool(feature_start_aware_reference)
        )
        effective_risk_corr_baseline = normalize_risk_corr_baseline(
            risk_corr_baseline or self.risk_corr_baseline,
        )

        # 允许 target 为空，或数据集中本就不存在目标列。
        has_target = target is not None and target in working_df.columns

        if not has_target:
            logger.info(
                f"Label-free mode enabled: target '{target}' was not found. "
                "A dummy target will be injected and only distribution metrics plus PSI will be evaluated."
            )
            # 临时注入常量标签，保持下游统计链路可复用。
            working_df = working_df.with_columns(pl.lit(0).cast(pl.Int32).alias(effective_target))

        # 检查 Target 有效性，并把未到表现期的空值保留为 null。
        if has_target:
            working_df = normalize_binary_target_column(working_df, effective_target)
            n_unique = (
                working_df
                .filter(pl.col(effective_target).is_not_null())
                .select(pl.col(effective_target).n_unique())
                .item()
            )
            if n_unique < 2:
                raise ValueError(
                    f"Target column '{effective_target}' must have at least 2 observed classes "
                    "after excluding null / NaN values."
                )

        working_df, group_col = prepare_group_context(
            working_df,
            profile_by=profile_by,
            dt_col=dt_col,
            mars_group_col=self.MARS_GROUP_COL,
        )
        if amount_col is not None and amount_col not in working_df.columns:
            raise ValueError(f"Amount column '{amount_col}' was not found in dataframe.")

        # 自动识别特征列
        # 排除 target, weights, 和刚刚生成的统一 mars_group 列
        exclude_cols = {effective_target, group_col}
        if weights_col:
            exclude_cols.add(weights_col)
        if amount_col:
            exclude_cols.add(amount_col)

        if features:
            target_features = [col for col in features if col != amount_col]
        else:
            target_features = [col for col in working_df.columns if col not in exclude_cols]

        effective_feature_data_source = feature_data_source if feature_data_source is not None else {}
        feature_source_map = normalize_feature_data_source(effective_feature_data_source, target_features)

        if binner is not None and self.binner_params:
            raise ValueError("`binner` and evaluator-level `binner_params` cannot be provided together.")

        active_binner = binner
        if active_binner is None:
            active_binner = build_binner(
                binning_type=self.binning_type,
                binner_params=dict(self.binner_params),
                has_target=has_target,
                working_df=working_df,
                target=effective_target,
                features=target_features,
            )

        # 将原始特征映射为分箱索引列 `{feature}_bin`。
        logger.debug("Transforming features to bin indices.")
        df_binned = active_binner.transform(working_df, return_type="index")
        missing_values = getattr(active_binner, "missing_values", None)
        if missing_values is None:
            missing_values = self.binner_params.get("missing_values")
        missing_by_day_table = build_missing_by_day_table(
            df=working_df,
            features=target_features,
            dt_col=dt_col,
            output_kind="pandas" if isinstance(df, pd.DataFrame) else "polars",
            missing_values=missing_values,
        )

        # 流式扫描并聚合到最细粒度统计表。
        logger.debug("Step 1: scanning grouped bin statistics.")
        group_stats_raw = aggregate_basic_stats(
            df_binned,
            group_col=group_col,
            features=target_features,
            target_col=effective_target,
            weights_col=weights_col,
            amount_col=amount_col,
            batch_size=batch_size,
        )

        ensure_woe_info(active_binner, group_stats_raw)

        # 获取 PSI 基准分布。若无外部基准，默认取最早一组。
        expected_dist = get_benchmark_dist(
            binner=active_binner,
            group_stats_raw=group_stats_raw,
            benchmark_df=benchmark_df,
            group_col=group_col,
            features=target_features,
            weights_col=weights_col,
        )
        feature_start_reference = None
        if effective_feature_start_reference:
            if benchmark_df is not None:
                logger.warning(
                    "`feature_start_aware_reference=True` was ignored because `benchmark_df` was provided."
                )
            elif not dt_col or dt_col not in working_df.columns:
                logger.warning(
                    "`feature_start_aware_reference=True` requires a valid `dt_col`; "
                    "falling back to the default reference logic."
                )
            else:
                feature_start_reference = self._build_feature_start_reference(
                    df_binned=df_binned,
                    missing_by_day_table=missing_by_day_table,
                    features=target_features,
                    dt_col=dt_col,
                    profile_by=profile_by,
                    group_col=group_col,
                    weights_col=weights_col,
                    target=effective_target,
                    has_target=has_target,
                )
                if feature_start_reference is not None and not feature_start_reference["expected_dist"].is_empty():
                    expected_dist = self._merge_feature_expected_dist(
                        default_expected_dist=expected_dist,
                        feature_expected_dist=feature_start_reference["expected_dist"],
                    )
        monitor_metrics_groups = None
        monitor_metrics_total = None

        # 汇总 total 统计量，得到全量样本的分布表现。
        logger.debug("Step 2: rolling up total statistics.")
        total_stats_raw = rollup_total_stats(
            group_stats_raw,
            group_col=group_col,
        )
        logger.debug("Step 3: calculating metrics.")

        # 计算 Trend 数据
        metrics_groups = (
            calculate_metrics_from_stats(
                binner=active_binner,
                stats_df=group_stats_raw,
                expected_dist=expected_dist,
                group_col=group_col,
                include_missing=psi_include_missing,
                include_special=psi_include_special,
            )
            .with_columns(pl.col(group_col).cast(pl.String))
        )

        # 计算 total 数据
        metrics_total = calculate_metrics_from_stats(
            binner=active_binner,
            stats_df=total_stats_raw,
            expected_dist=expected_dist,
            group_col=group_col,
            include_missing=psi_include_missing,
            include_special=psi_include_special,
        )

        # 合并分组与总体结果
        if feature_start_reference is not None:
            monitor_group_stats_raw = feature_start_reference.get("monitor_group_stats_raw")
            if monitor_group_stats_raw is not None and not monitor_group_stats_raw.is_empty():
                monitor_total_stats_raw = rollup_total_stats(
                    monitor_group_stats_raw,
                    group_col=group_col,
                )
                monitor_metrics_groups = (
                    calculate_metrics_from_stats(
                        binner=active_binner,
                        stats_df=monitor_group_stats_raw,
                        expected_dist=expected_dist,
                        group_col=group_col,
                        include_missing=psi_include_missing,
                        include_special=psi_include_special,
                    )
                    .with_columns(pl.col(group_col).cast(pl.String))
                )
                monitor_metrics_total = calculate_metrics_from_stats(
                    binner=active_binner,
                    stats_df=monitor_total_stats_raw,
                    expected_dist=expected_dist,
                    group_col=group_col,
                    include_missing=psi_include_missing,
                    include_special=psi_include_special,
                ).select(monitor_metrics_groups.columns)

        metrics_total = metrics_total.select(metrics_groups.columns)
        risk_corr_reference_table, risk_corr_reference_source = (
            build_risk_corr_reference_table(
                target_name=effective_target,
                metrics_groups=metrics_groups,
                metrics_total=metrics_total,
                group_col=group_col,
                risk_corr_baseline=effective_risk_corr_baseline,
                benchmark_df=benchmark_df,
                benchmark_features=target_features,
                benchmark_weights_col=weights_col,
                feature_start_reference=feature_start_reference,
                binner=active_binner,
                has_target=has_target,
                mars_group_col=self.MARS_GROUP_COL,
            )
        )

        # 单点评估时避免重复拼接一份语义相同的 Total 结果。
        is_single_snapshot = (
            metrics_groups.select(pl.col(group_col).n_unique()).item() == 1 and
            metrics_groups.select(pl.col(group_col).first()).item() == "Total"
        )

        if is_single_snapshot:
            logger.debug("Single snapshot detected. Skipping total concatenation.")
            stats_long = metrics_groups
        else:
            stats_long = pl.concat([metrics_total, metrics_groups])

        # 单调性检查 (Monotonicity Check)
        logger.debug("Step 4: checking monotonicity.")
        if has_target:
            monotonicity_df = (
                stats_long
                .filter((pl.col("bin_index") >= 0) & (pl.col(group_col) == "Total"))
                .group_by("feature")
                .agg(pl.corr("bin_index", "bad_rate", method="spearman").fill_nan(1.0).alias("mono"))
            )
        else:
            # 无标签时直接赋予默认单调性
            monotonicity_df = pl.DataFrame({"feature": target_features, "mono": [1.0] * len(target_features)})

        report = self._format_report(
            stats_long,
            metrics_groups,
            metrics_total,
            group_col,
            monotonicity_df,
            binner=active_binner,
            target_name=effective_target,
            feature_source_map=feature_source_map,
            display_group_col=profile_by or ("month" if dt_col else None),
            dt_col=dt_col,
            missing_by_day_table=missing_by_day_table,
            risk_corr_reference_table=risk_corr_reference_table,
            feature_valid_groups_df=feature_start_reference["valid_groups"] if feature_start_reference else None,
            monitor_metrics_groups=monitor_metrics_groups,
            monitor_metrics_total=monitor_metrics_total,
        )
        profile_label = profile_by
        if dt_col and not profile_by:
            profile_label = "month (auto)"
        report._report_meta = {
            "row_count": int(working_df.height),
            "feature_count": len(target_features),
            "profile_by_input": profile_label,
            "group_col": group_col,
            "group_count": max(int(stats_long[group_col].n_unique()) - 1, 0),
            "dt_col": dt_col,
            "start_dt": None,
            "end_dt": None,
            "targets": [original_target] if has_target and original_target else [],
            "event_rate_by_target": {},
            "feature_start_aware_reference": bool(feature_start_reference),
            "psi_include_missing": psi_include_missing,
            "psi_include_special": psi_include_special,
            "risk_corr_baseline": effective_risk_corr_baseline,
            "risk_corr_reference_source": risk_corr_reference_source,
            "amount_col": amount_col,
            "feature_start_reference_features": sorted((feature_start_reference or {}).get("feature_start_dates", {}).keys()),
            "feature_start_reference_dates": dict((feature_start_reference or {}).get("feature_start_dates", {})),
        }
        if dt_col and dt_col in working_df.columns:
            try:
                report._report_meta["start_dt"] = str(working_df.select(pl.col(dt_col).min()).item())
                report._report_meta["end_dt"] = str(working_df.select(pl.col(dt_col).max()).item())
            except Exception:
                report._report_meta["start_dt"] = None
                report._report_meta["end_dt"] = None
        if has_target and original_target and original_target in working_df.columns:
            try:
                event_rate = float(working_df.select(pl.col(original_target).cast(pl.Float64).mean()).item())
            except Exception:
                event_rate = None
            report._report_meta["event_rate_by_target"] = {str(original_target): event_rate}
        targets = [str(original_target)] if has_target and original_target else []

        # 无标签模式下擦除依赖真实坏样本标签的指标，保留分布类结果。
        if not has_target:
            null_cols = [
                "observed_count",
                "bad",
                "good",
                "bad_rate",
                "lift",
                "trend",
                "cum_observed_count",
                "cum_bad",
                "cum_good",
                "cum_bad_rate",
                "ks_bin",
                "auc_bin",
                "iv_bin",
                "mono",
                "good_amt",
                "bad_amt",
                "amt_bad_rate",
                "lift_amt",
            ]

            # detail_table 擦除
            dt_cols = [c for c in null_cols if c in report._detail.columns]
            if isinstance(report._detail, pd.DataFrame):
                for c in dt_cols:
                    report._detail[c] = np.nan
            else:
                report._detail = report._detail.with_columns([
                    pl.lit(None).cast(pl.Float64).alias(c) for c in dt_cols
                ])

            # summary_table 擦除
            sum_cols = ["iv", "ks", "auc", "rc_min", "lift_min", "lift_max", "mono"]
            sum_cols = [c for c in sum_cols if c in report._summary.columns]
            if isinstance(report._summary, pd.DataFrame):
                for c in sum_cols:
                    report._summary[c] = np.nan
            else:
                report._summary = report._summary.with_columns([
                    pl.lit(None).cast(pl.Float64).alias(c) for c in sum_cols
                ])

            # 无标签模式仅保留 PSI 趋势表。
            if "psi" in report._trend_dict:
                report._trend_dict = {"psi": report._trend_dict["psi"]}
            else:
                report._trend_dict = {}

        logger.info(f"Evaluation complete. [Features: {len(target_features)} | Groups: {stats_long[group_col].n_unique() - 1}]")
        run = MarsRiskProfile(
            report=report,
            binner=active_binner,
            targets=targets,
            metadata=dict(report.report_meta or {}),
        )
        return run

    def _build_feature_start_reference(self, **kwargs: Any) -> dict[str, Any] | None:
        """复用既有 feature-start 参考构造逻辑，并对外统一新命名。"""
        return self._build_feature_start_baseline_reference(**kwargs)

    @staticmethod
    def _empty_risk_corr_reference_table(target_name: str | None) -> pl.DataFrame:
        """构造空的 RC 参考表。"""
        _ = target_name
        return pl.DataFrame(
            schema={
                "y": pl.String,
                "feature": pl.String,
                "bin_index": pl.Int16,
                "base_br": pl.Float64,
                "source": pl.String,
            }
        )

    def _attach_risk_corr_reference_context(
        self,
        reference_df: pl.DataFrame,
        *,
        target_name: str | None,
        source: str,
    ) -> pl.DataFrame:
        """把参考坏率表补齐为统一 schema。"""
        if reference_df.is_empty():
            return self._empty_risk_corr_reference_table(target_name)
        return (
            reference_df
            .filter(pl.col("bin_index") >= 0)
            .select(
                [
                    pl.lit(str(target_name or "dummy_target")).alias("y"),
                    pl.col("feature").cast(pl.String).alias("feature"),
                    pl.col("bin_index").cast(pl.Int16).alias("bin_index"),
                    pl.col("base_br").cast(pl.Float64).alias("base_br"),
                    pl.lit(source).alias("source"),
                ]
            )
        )

    def _build_benchmark_risk_corr_reference(
        self,
        benchmark_df: pl.DataFrame,
        *,
        binner: MarsBinnerBase,
        has_target: bool,
        features: List[str],
        weights_col: str | None,
        target_name: str | None,
    ) -> pl.DataFrame:
        """基于 benchmark 样本构造 RC 参考表。"""
        if not has_target or target_name is None:
            return self._empty_risk_corr_reference_table(target_name)

        benchmark_prepared = normalize_binary_target_column(
            benchmark_df,
            target_name,
        )
        benchmark_binned = binner.transform(benchmark_prepared, return_type="index")
        benchmark_binned = benchmark_binned.with_columns(pl.lit("Benchmark").alias(self.MARS_GROUP_COL))
        benchmark_stats = aggregate_basic_stats(
            benchmark_binned,
            group_col=self.MARS_GROUP_COL,
            features=features,
            target_col=target_name,
            weights_col=weights_col,
        )
        reference_df = (
            benchmark_stats
            .group_by(["feature", "bin_index"])
            .agg(
                [
                    pl.col("count").sum().alias("count"),
                    pl.col("observed_count").sum().alias("observed_count"),
                    pl.col("bad").sum().alias("bad"),
                ]
            )
            .with_columns(bad_rate_expr(output_col="base_br"))
        )
        return self._attach_risk_corr_reference_context(
            reference_df,
            target_name=target_name,
            source="benchmark_df",
        )

    def _build_risk_corr_reference_table(
        self,
        *,
        target_name: str | None,
        metrics_groups: pl.DataFrame,
        metrics_total: pl.DataFrame,
        group_col: str,
        risk_corr_baseline: RiskCorrBaseline,
        benchmark_df: pl.DataFrame | None,
        benchmark_features: List[str],
        benchmark_weights_col: str | None,
        feature_start_reference: dict[str, Any] | None,
        binner: MarsBinnerBase,
        has_target: bool,
    ) -> tuple[pl.DataFrame, str]:
        """按统一语义选择 RC 参考表。"""
        if not has_target:
            return self._empty_risk_corr_reference_table(target_name), "total"

        if risk_corr_baseline == "total":
            reference_df = metrics_total.select(
                [
                    "feature",
                    "bin_index",
                    pl.col("bad_rate").alias("base_br"),
                ]
            )
            return (
                self._attach_risk_corr_reference_context(
                    reference_df,
                    target_name=target_name,
                    source="total",
                ),
                "total",
            )

        if risk_corr_baseline == "first_group":
            first_group = metrics_groups.select(pl.col(group_col).min()).item()
            reference_df = (
                metrics_groups
                .filter(pl.col(group_col) == first_group)
                .select(
                    [
                        "feature",
                        "bin_index",
                        pl.col("bad_rate").alias("base_br"),
                    ]
                )
            )
            return (
                self._attach_risk_corr_reference_context(
                    reference_df,
                    target_name=target_name,
                    source="first_group",
                ),
                "first_group",
            )

        if benchmark_df is not None:
            reference_df = self._build_benchmark_risk_corr_reference(
                benchmark_df,
                binner=binner,
                has_target=has_target,
                features=benchmark_features,
                weights_col=benchmark_weights_col,
                target_name=target_name,
            )
            return reference_df, "benchmark_df"

        if feature_start_reference is not None:
            baseline_df = feature_start_reference.get("baseline_bad_rate")
            if baseline_df is not None and not baseline_df.is_empty():
                return (
                    self._attach_risk_corr_reference_context(
                        baseline_df,
                        target_name=target_name,
                        source="feature_start_reference",
                    ),
                    "feature_start_reference",
                )

        raise ValueError(
            "`risk_corr_baseline='benchmark'` requires `benchmark_df` or "
            "`feature_start_aware_reference=True` with a valid feature-start reference.",
        )

    def _build_risk_corr_long(
        self,
        metrics_df: pl.DataFrame,
        baseline_df: pl.DataFrame,
        *,
        group_col: str,
    ) -> pl.DataFrame:
        """基于参考坏率表计算分组级 RC 长表。"""
        return (
            metrics_df
            .filter(pl.col("bin_index") >= 0)
            .join(baseline_df, on=["feature", "bin_index"], how="left")
            .group_by(["feature", group_col])
            .agg(risk_corr_expr())
        )

    @staticmethod
    def _detect_feature_start_index(
        inactive_flags: List[bool],
        *,
        leading_inactive_ratio: float = 0.90,
        sustain_window: int = 3,
        sustain_active_ratio: float = 2.0 / 3.0,
    ) -> int | None:
        """识别特征从长期未覆盖状态切换到持续有效覆盖的首个位置。"""
        if not inactive_flags:
            return None

        inactive_prefix = [0]
        for flag in inactive_flags:
            inactive_prefix.append(inactive_prefix[-1] + int(flag))

        n_days = len(inactive_flags)
        for idx, is_inactive in enumerate(inactive_flags):
            if is_inactive:
                continue

            prefix_days = idx
            prefix_ratio = 1.0 if prefix_days == 0 else inactive_prefix[idx] / prefix_days
            if prefix_ratio < leading_inactive_ratio:
                continue

            window_end = min(n_days, idx + sustain_window)
            window_flags = inactive_flags[idx:window_end]
            active_days = sum(not flag for flag in window_flags)
            required_active_days = max(1, int(np.ceil(len(window_flags) * sustain_active_ratio)))
            if active_days >= required_active_days:
                return idx

        return None

    @staticmethod
    def _merge_feature_expected_dist(
        default_expected_dist: pl.DataFrame,
        feature_expected_dist: pl.DataFrame,
    ) -> pl.DataFrame:
        """用特征级基准分布覆盖全局默认基准分布。"""
        if feature_expected_dist.is_empty():
            return default_expected_dist

        override_features = feature_expected_dist.get_column("feature").unique().to_list()
        retained_default = default_expected_dist.filter(
            ~pl.col("feature").is_in(pl.Series(override_features).implode())
        )
        return pl.concat([retained_default, feature_expected_dist], how="vertical_relaxed")

    @staticmethod
    def _merge_feature_frame(
        default_df: pl.DataFrame,
        override_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """按 feature 维度用覆盖表替换默认表中的同名特征记录。"""
        if override_df is None or override_df.is_empty():
            return default_df

        override_features = override_df.get_column("feature").unique().to_list()
        retained_default = default_df.filter(
            ~pl.col("feature").is_in(pl.Series(override_features).implode())
        )
        return pl.concat([retained_default, override_df], how="vertical_relaxed")

    def _build_feature_start_baseline_reference(
        self,
        *,
        df_binned: pl.DataFrame,
        missing_by_day_table: Union[pl.DataFrame, pd.DataFrame] | None,
        features: List[str],
        dt_col: str,
        profile_by: str | None,
        group_col: str,
        weights_col: str | None,
        target: str,
        has_target: bool,
    ) -> Dict[str, Any] | None:
        """
        基于特征上线起始日推导 PSI 基准分布覆盖表。

        当存在时间列且可识别特征从长期缺失转为持续活跃时，该方法会为
        相关特征构造独立的 expected distribution 与 bad rate 基准。
        若时间列不可解析或没有可用起始点，则返回 ``None``。
        """
        _ = missing_by_day_table

        if dt_col not in df_binned.columns:
            return None

        dt_alias = "__mars_feature_start_dt"
        working_df = df_binned.with_columns(MarsDate.smart_parse_expr(dt_col).alias(dt_alias))
        if working_df.select(pl.col(dt_alias).is_not_null().any()).item() is not True:
            return None

        missing_idx = -1
        try:
            missing_idx = int(MarsBinnerBase.IDX_MISSING)
        except Exception:
            missing_idx = -1

        expected_frames: List[pl.DataFrame] = []
        baseline_bad_rate_frames: List[pl.DataFrame] = []
        valid_group_frames: List[pl.DataFrame] = []
        monitor_group_stats_frames: List[pl.DataFrame] = []
        feature_start_dates: Dict[str, str] = {}
        for feature in features:
            bin_col = f"{feature}_bin"
            if bin_col not in working_df.columns:
                continue

            select_cols = [dt_alias, group_col, bin_col]
            if weights_col and weights_col in working_df.columns:
                select_cols.append(weights_col)
            if has_target and target in working_df.columns:
                select_cols.append(target)

            feature_df = (
                working_df
                .select(select_cols)
                .rename({bin_col: "bin_index"})
                .filter(pl.col(dt_alias).is_not_null())
            )
            if feature_df.is_empty():
                continue

            daily_missing = (
                feature_df
                .group_by(dt_alias)
                .agg([
                    pl.len().alias("_day_count"),
                    pl.when(pl.col("bin_index") == missing_idx).then(1).otherwise(0).sum().alias("_missing_count"),
                ])
                .sort(dt_alias)
                .with_columns(
                    pl.when(pl.col("_day_count") > 0)
                    .then(pl.col("_missing_count") / pl.col("_day_count"))
                    .otherwise(1.0)
                    .alias("_missing_rate")
                )
            )
            if daily_missing.is_empty():
                continue

            inactive_flags = [
                bool(rate >= 0.99)
                for rate in daily_missing.get_column("_missing_rate").to_list()
            ]
            start_idx = self._detect_feature_start_index(inactive_flags)
            if start_idx is None:
                continue

            start_dt = daily_missing.get_column(dt_alias).to_list()[start_idx]
            post_start_df = feature_df.filter(pl.col(dt_alias) >= pl.lit(start_dt))
            if post_start_df.is_empty():
                continue

            baseline_group = (
                post_start_df
                .sort(dt_alias)
                .select(pl.col(group_col).first())
                .item()
            )
            if baseline_group is None:
                continue

            baseline_rows = post_start_df.filter(pl.col(group_col) == pl.lit(str(baseline_group)))

            if baseline_rows.is_empty():
                continue

            if weights_col and weights_col in post_start_df.columns:
                monitor_count_expr = pl.col(weights_col).cast(pl.Float64).sum().alias("count")
            else:
                monitor_count_expr = pl.len().cast(pl.Float64).alias("count")

            if has_target and target in post_start_df.columns:
                if weights_col and weights_col in post_start_df.columns:
                    monitor_observed_expr = (
                        pl.when(pl.col(target).is_not_null())
                        .then(pl.col(weights_col).cast(pl.Float64))
                        .otherwise(0)
                        .sum()
                        .alias("observed_count")
                    )
                    monitor_bad_expr = (
                        pl.col(target).fill_null(0).cast(pl.Float64)
                        * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("bad")
                else:
                    monitor_observed_expr = pl.col(target).is_not_null().sum().alias("observed_count")
                    monitor_bad_expr = pl.col(target).fill_null(0).cast(pl.Float64).sum().alias("bad")
            else:
                monitor_observed_expr = monitor_count_expr.alias("observed_count")
                monitor_bad_expr = pl.lit(0.0).alias("bad")

            monitor_group_stats_df = (
                post_start_df
                .group_by([group_col, "bin_index"])
                .agg([monitor_count_expr, monitor_observed_expr, monitor_bad_expr])
                .select([
                    pl.col(group_col).cast(pl.String).alias(group_col),
                    pl.lit(feature).alias("feature"),
                    pl.col("bin_index"),
                    pl.col("count").cast(pl.Float64),
                    pl.col("observed_count").cast(pl.Float64),
                    pl.col("bad").cast(pl.Float64),
                ])
            )
            if not monitor_group_stats_df.is_empty():
                monitor_group_stats_frames.append(monitor_group_stats_df)

            if weights_col and weights_col in baseline_rows.columns:
                expected_count_expr = pl.col(weights_col).cast(pl.Float64).sum().alias("expected_count")
            else:
                expected_count_expr = pl.len().cast(pl.Float64).alias("expected_count")

            expected_dist_df = (
                baseline_rows
                .group_by("bin_index")
                .agg(expected_count_expr)
                .with_columns(global_distribution_expr(count_col="expected_count", output_col="expected_dist"))
                .select([
                    pl.lit(feature).alias("feature"),
                    pl.col("bin_index"),
                    pl.col("expected_dist"),
                ])
            )
            if expected_dist_df.is_empty():
                continue
            expected_frames.append(expected_dist_df)
            valid_groups_df = (
                post_start_df
                .select([
                    pl.lit(feature).alias("feature"),
                    pl.col(group_col).cast(pl.String).alias(group_col),
                ])
                .unique()
            )
            if not valid_groups_df.is_empty():
                valid_group_frames.append(valid_groups_df)
            feature_start_dates[feature] = str(start_dt)

            if has_target and target in baseline_rows.columns:
                if weights_col and weights_col in baseline_rows.columns:
                    bad_expr = (
                        pl.col(target).fill_null(0).cast(pl.Float64)
                        * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("base_bad")
                    total_expr = (
                        pl.when(pl.col(target).is_not_null())
                        .then(pl.col(weights_col).cast(pl.Float64))
                        .otherwise(0)
                        .sum()
                        .alias("base_total")
                    )
                else:
                    bad_expr = pl.col(target).fill_null(0).cast(pl.Float64).sum().alias("base_bad")
                    total_expr = pl.col(target).is_not_null().sum().cast(pl.Float64).alias("base_total")

                baseline_bad_rate_df = (
                    baseline_rows
                    .filter(pl.col("bin_index") >= 0)
                    .group_by("bin_index")
                    .agg([bad_expr, total_expr])
                    .with_columns(
                        pl.when(pl.col("base_total") > 0)
                        .then(pl.col("base_bad") / pl.col("base_total"))
                        .otherwise(None)
                        .alias("base_br")
                    )
                    .select([
                        pl.lit(feature).alias("feature"),
                        pl.col("bin_index"),
                        pl.col("base_br"),
                    ])
                )
                if not baseline_bad_rate_df.is_empty():
                    baseline_bad_rate_frames.append(baseline_bad_rate_df)

        if not expected_frames:
            return None

        expected_dist = pl.concat(expected_frames, how="vertical_relaxed")
        if baseline_bad_rate_frames:
            baseline_bad_rate = pl.concat(baseline_bad_rate_frames, how="vertical_relaxed")
        else:
            baseline_bad_rate = pl.DataFrame(
                schema={"feature": pl.String, "bin_index": pl.Int16, "base_br": pl.Float64}
            )
        if valid_group_frames:
            valid_groups = pl.concat(valid_group_frames, how="vertical_relaxed")
        else:
            valid_groups = pl.DataFrame(
                schema={"feature": pl.String, group_col: pl.String}
            )
        if monitor_group_stats_frames:
            monitor_group_stats_raw = pl.concat(monitor_group_stats_frames, how="vertical_relaxed")
        else:
            monitor_group_stats_raw = pl.DataFrame(
                schema={
                    group_col: pl.String,
                    "feature": pl.String,
                    "bin_index": pl.Int16,
                    "count": pl.Float64,
                    "observed_count": pl.Float64,
                    "bad": pl.Float64,
                }
            )

        return {
            "expected_dist": expected_dist,
            "baseline_bad_rate": baseline_bad_rate,
            "valid_groups": valid_groups,
            "monitor_group_stats_raw": monitor_group_stats_raw,
            "feature_start_dates": feature_start_dates,
        }

    def _build_bin_label_map(
        self,
        stats_long: pl.DataFrame,
        *,
        binner: MarsBinnerBase,
    ) -> pl.DataFrame:
        """构建明细报告使用的特征和分箱索引到标签的映射。"""
        map_rows: list[dict[str, Any]] = []
        features = set(stats_long["feature"].unique().to_list())

        for feature, mapping in binner.bin_mappings_.items():
            if feature not in features:
                continue

            for bin_index, label in mapping.items():
                try:
                    map_rows.append(
                        {
                            "feature": feature,
                            "bin_index": int(bin_index),
                            "bin_label": str(label),
                        }
                    )
                except (ValueError, TypeError):
                    continue

        map_schema = {
            "feature": pl.String,
            "bin_index": pl.Int16,
            "bin_label": pl.String,
        }
        if not map_rows:
            return pl.DataFrame([], schema=map_schema)
        return pl.DataFrame(map_rows, schema=map_schema)

    def _format_report(
        self,
        stats_long: pl.DataFrame,
        metrics_groups: pl.DataFrame,
        metrics_total: pl.DataFrame,
        group_col: str,
        monotonicity_df: pl.DataFrame,
        *,
        binner: MarsBinnerBase,
        target_name: str,
        feature_source_map: Dict[str, str] | None = None,
        display_group_col: str | None = None,
        dt_col: str | None = None,
        missing_by_day_table: Union[pl.DataFrame, pd.DataFrame] | None = None,
        risk_corr_reference_table: pl.DataFrame | None = None,
        feature_valid_groups_df: pl.DataFrame | None = None,
        monitor_metrics_groups: pl.DataFrame | None = None,
        monitor_metrics_total: pl.DataFrame | None = None,
    ) -> "MarsBinningReport":
        """
        构建评估报告的明细、汇总与趋势数据。

        该方法负责将向量化计算的中间结果重塑为具备业务深度的三层报表体系：
        明细层 (Detail)、审计层 (Summary) 和趋势层 (Trend)。

        产出的 Summary 报表坚持极简与客观原则，仅保留最核心的预测力与稳定性评估指标。

        Parameters
        ----------
        stats_long : pl.DataFrame
            全量分箱统计长表。包含每个特征、每个分组、每个分箱的原始统计量及分箱级指标。
        metrics_groups : pl.DataFrame
            仅包含分组数据（如 Monthly）的长表。用于计算跨期稳定性。
        metrics_total : pl.DataFrame
            仅包含全量（total）统计的数据。用于获取特征全局区分度。
        group_col : str
            分组维度列名（如 'month'）。
        monotonicity_df : pl.DataFrame
            单调性检查结果。包含特征在全量分组下的 Spearman 相关系数 (mono)。
        feature_source_map : Dict[str, str] | None
            特征到来源分组的映射，用于在汇总报告中保留来源字段。
        display_group_col : str | None
            报告层对外展示的分组语义列名。
        dt_col : str | None
            原始日期列名，用于生成按日缺失率附表。
        missing_by_day_table : Union[pl.DataFrame, pd.DataFrame] | None
            已计算好的按日缺失率附表。
        risk_corr_reference_table : pl.DataFrame | None
            RiskCorr 基准分布表。
        feature_valid_groups_df : pl.DataFrame | None
            特征有效分组数量表。
        monitor_metrics_groups : pl.DataFrame | None
            分组粒度监控指标表。
        monitor_metrics_total : pl.DataFrame | None
            全量粒度监控指标表。

        Returns
        -------
        MarsBinningReport
            报告容器实例。包含 Summary, Trend, Detail 三张重塑后的报表。
        """
        map_df = self._build_bin_label_map(stats_long, binner=binner)

        # 此时 stats_long 和 map_df 都是 Int16，Join 安全，不会发生类型提升
        detail_base = (
            stats_long
            .join(map_df, on=["feature", "bin_index"], how="left")
            .with_columns(pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)))
        )
        amount_detail_cols = (
            ["tot_amt", "good_amt", "bad_amt", "avg_amt", "amt_bad_rate", "lift_amt"]
            if {"tot_amt", "good_amt", "bad_amt"}.issubset(detail_base.columns)
            else []
        )

        # 提取 WOE 序列
        trend_source = (
            metrics_total  # 使用 total 数据
            .lazy()
            .filter(pl.col("bin_index") >= 0)
            .sort(["feature", "bin_index"])
            .select(["feature", "woe"])
        )

        # 调用 Binner 中的静态方法进行判断
        from mars.feature.binning.base import MarsBinnerBase

        trend_shape_df = MarsBinnerBase._build_trend_shape_frame(
            trend_source.group_by("feature").agg(pl.col("woe")).collect(),
            trend_col_name="trend",
        )
        detail_base = detail_base.join(trend_shape_df, on="feature", how="left")

        #  构建自定义排序键 (Sort Key)
        # 显式 cast(pl.Int32) 解决 SchemaError
        detail_table = (
            detail_base
            .with_columns([
                # 构建排序辅助列 (0:Normal, 1:Special/Missing, 2:Total)
                pl.when(pl.col("bin_index") >= 0).then(0).otherwise(1).cast(pl.Int32).alias("_sort_group"),

                # 针对非正常箱的内部排序：
                # -1（Missing）映射到 10000。
                # -2（Other）映射到 10001。
                # 小于 -2（Special）映射到 20000 + abs。
                pl.when(pl.col("bin_index") >= 0).then(pl.col("bin_index").cast(pl.Int32)) # 显式转 Int32
                  .when(pl.col("bin_index") == -1).then(10000)
                  .when(pl.col("bin_index") == -2).then(10001)
                  .otherwise(20000 + pl.col("bin_index").abs().cast(pl.Int32))
                  .alias("_sort_idx")
            ])
            .sort(["feature", group_col, "_sort_group", "_sort_idx"]) # 执行物理排序
        )

        #  计算累积指标
        detail_table = detail_table.with_columns([
            *ordered_count_metric_exprs(["feature", group_col]),
            (pl.col("observed_count") - pl.col("bad")).cum_sum().over(["feature", group_col]).alias("cum_good"),
            distribution_rate_expr(
                numerator_col="count",
                denominator_col="total_count",
                output_col="pct",
            ),
            pl.col("bin_index").max().over(["feature", group_col]).alias("bin_index_max"),
        ]).with_columns([
            pl.when(
                (pl.col("bin_index") == pl.col("bin_index_max")) | (pl.col("bin_index") == 0)
            )
            .then(pl.lit("首尾组"))
            .when(
                pl.col("bin_index") == -1
            )
            .then(pl.lit("空值组"))
            .when(
                pl.col("bin_index") == -2
            )
            .then(pl.lit("其他组"))
            .when(
                pl.col("bin_index") <= -3
            )
            .then(pl.lit("特殊组"))
            .otherwise(pl.lit("正常组"))
            .alias("bin_type")

        ])

        # 构造 total 汇总行 )
        # 对每个 (feature, group) 生成一行汇总数据，包含计算后的综合指标
        total_rows = (
            stats_long
            .group_by(["feature", group_col])
            .agg([
                # 基础统计量汇总
                pl.col("count").sum().alias("count"),
                pl.col("observed_count").sum().alias("observed_count"),
                pl.col("bad").sum().alias("bad"),
                pl.col("iv_bin").sum().alias("iv_bin"),
                pl.col("psi_bin").sum().alias("psi_bin"),
                pl.col("auc_bin").sum().alias("auc_bin"),
                pl.col("ks_bin").max().alias("ks_bin"),
                pl.col("lift").max().alias("lift"),
                pl.col("count").sum().alias("total_count")
            ])
            .with_columns([
                # 衍生列
                (pl.col("observed_count") - pl.col("bad")).alias("good"),
                bad_rate_expr(),

                # total 行代表全量样本，占比固定为 1.0。
                pl.lit(1.0).alias("pct"),

                # 累积列 (对于 total 行，累积值等于自身)
                pl.col("count").alias("cum_count"),
                pl.col("observed_count").alias("cum_observed_count"),
                pl.col("bad").alias("cum_bad"),
                bad_rate_expr(output_col="cum_bad_rate"),

                # AUC 方向修正
                normalized_auc_expr(auc_col="auc_bin", output_col="auc_bin"),

                # 标识列与排序键 (确保排在最后)
                pl.lit(9999).cast(pl.Int16).alias("bin_index"),
                pl.lit("Total").alias("bin_label"),

                pl.lit("汇总组").alias("bin_type"),

                pl.lit(2).cast(pl.Int32).alias("_sort_group"), # 明确为 Int32，避免排序键类型漂移。
                pl.lit(0).cast(pl.Int32).alias("_sort_idx")    # 明确为 Int32，避免排序键类型漂移。
            ])
        )

        # total 行也要 join trend 列
        if amount_detail_cols:
            amount_totals = (
                stats_long
                .group_by(["feature", group_col])
                .agg([
                    pl.col("count").sum().alias("count"),
                    pl.col("tot_amt").sum().alias("tot_amt"),
                    pl.col("good_amt").sum().alias("good_amt"),
                    pl.col("bad_amt").sum().alias("bad_amt"),
                ])
                .with_columns(amount_distribution_exprs(["feature", group_col]))
                .with_columns(amount_metric_exprs())
                .select(["feature", group_col] + amount_detail_cols)
            )
            total_rows = total_rows.join(
                amount_totals,
                on=["feature", group_col],
                how="left",
            )
        total_rows = total_rows.join(trend_shape_df, on="feature", how="left")

        targets = [
            "feature", group_col, "bin_index", "bin_label", "_sort_group", "_sort_idx",
            "count", "observed_count", "pct", "bad", "good", "bad_rate", "lift", "trend",
            "cum_count", "cum_observed_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ]
        if amount_detail_cols:
            targets.extend(amount_detail_cols)

        detail_table = (
            pl.concat([
                detail_table.select(targets),
                total_rows.select(targets)
            ])
            .sort(["feature", group_col, "_sort_group", "_sort_idx"])
        )

        detail_table = detail_table.select([
            pl.lit(target_name).alias("y"),
            "feature", "trend", group_col, "bin_index", "bin_label",
            "count", "observed_count", "bad", "good", "pct", "bad_rate", "lift",
            "cum_count", "cum_observed_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ] + amount_detail_cols)

        if feature_source_map:
            source_df = pl.DataFrame({
                "feature": list(feature_source_map.keys()),
                "data_source": [feature_source_map[feature] for feature in feature_source_map],
            })
            detail_table = detail_table.join(source_df, on="feature", how="left").with_columns(
                pl.col("data_source").fill_null("UNMAPPED")
            )

        # 中间指标计算
        # RiskCorr (RC) 跨期稳定性逻辑
        # 确定基准序列 (选取时间最早的一组)
        baseline_df = (
            risk_corr_reference_table
            if risk_corr_reference_table is not None
            else pl.DataFrame(
                schema={
                    "y": pl.String,
                    "feature": pl.String,
                    "bin_index": pl.Int16,
                    "base_br": pl.Float64,
                    "source": pl.String,
                }
            )
        )
        baseline_df = baseline_df.select(["feature", "bin_index", "base_br"])

        _ = feature_valid_groups_df
        monitoring_groups = self._merge_feature_frame(metrics_groups, monitor_metrics_groups)
        monitoring_total = self._merge_feature_frame(metrics_total, monitor_metrics_total)

        valid_group_lookup = None
        valid_group_feature_flags = None
        if feature_valid_groups_df is not None and not feature_valid_groups_df.is_empty():
            valid_group_lookup = (
                feature_valid_groups_df
                .select(["feature", group_col])
                .unique()
                .with_columns(pl.lit(True).alias("_mars_valid_group"))
            )
            valid_group_feature_flags = (
                feature_valid_groups_df
                .select(["feature"])
                .unique()
                .with_columns(pl.lit(True).alias("_mars_feature_start_override"))
            )

        def _null_metric_for_invalid_groups(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
            """对特征启用前的无效分组置空指定趋势指标。"""
            if (
                df.is_empty()
                or metric_col not in df.columns
                or valid_group_lookup is None
                or valid_group_feature_flags is None
            ):
                return df
            return (
                df
                .join(valid_group_feature_flags, on="feature", how="left")
                .join(valid_group_lookup, on=["feature", group_col], how="left")
                .with_columns(
                    pl.when(
                        pl.col("_mars_feature_start_override").fill_null(False)
                        & pl.col("_mars_valid_group").is_null()
                    )
                    .then(pl.lit(None).cast(pl.Float64))
                    .otherwise(pl.col(metric_col))
                    .alias(metric_col)
                )
                .drop(["_mars_feature_start_override", "_mars_valid_group"])
            )

        # 构造用于计算相关性的全量数据流
        all_metrics_for_corr = pl.concat([
            monitoring_groups.select(["feature", group_col, "bin_index", "bad_rate", "observed_count"]),
            monitoring_total.select(["feature", group_col, "bin_index", "bad_rate", "observed_count"])
        ])

        risk_corr_long = build_risk_corr_long(
            all_metrics_for_corr,
            baseline_df,
            group_col=group_col,
        )
        _ = _null_metric_for_invalid_groups

        # 分组指标聚合
        # 将分箱粒度聚合为分组粒度 (如: Month Level)
        group_level_metrics = (
            metrics_groups
            .group_by(["feature", group_col])
            .agg([
                observed_iv_agg_expr(),
                observed_auc_agg_expr(),
                bin_missing_rate_expr(missing_bin_index=MarsBinnerBase.IDX_MISSING),
                observed_lift_max_agg_expr(),
            ])
            # 确保 AUC 方向正确 (>= 0.5)
            .with_columns(normalized_auc_expr())
        )
        monitor_group_level_metrics = (
            monitoring_groups
            .group_by(["feature", group_col])
            .agg(pl.col("psi_bin").sum().alias("psi"))
        )
        group_level_metrics = (
            group_level_metrics
            .join(monitor_group_level_metrics, on=["feature", group_col], how="left")
            .join(risk_corr_long, on=["feature", group_col], how="left")
        )

        total_missing_metrics = (
            stats_long
            .group_by("feature")
            .agg(bin_missing_rate_expr(missing_bin_index=MarsBinnerBase.IDX_MISSING))
        )

        total_real_bin_lift_metrics = (
            metrics_total
            .filter(pl.col("bin_index") >= 0)
            .group_by("feature")
            .agg([
                observed_lift_min_agg_expr(output_col="lift_min"),
                observed_lift_max_agg_expr(output_col="lift_max"),
            ])
        )

        # 第三阶段：汇总表
        total_metrics_agg = (
            metrics_total.group_by("feature")
            .agg([
                observed_iv_agg_expr(),
                observed_ks_agg_expr(),
                observed_auc_agg_expr(),
            ])
            .with_columns(
                normalized_auc_expr()
            )
        )

        if not group_level_metrics.is_empty():
            summary_audit = (
                group_level_metrics
                .group_by("feature")
                .agg([
                    pl.col("psi").max().fill_null(0.0).alias("psi_max"),
                    pl.col("risk_corr").min().fill_null(1.0).alias("rc_min"),
                    pl.col("missing").min().alias("missing_min"),
                    pl.col("missing").max().alias("missing_max"),
                ])
            )
        else:
            # 单点评估模式兜底
            summary_audit = pl.DataFrame({
                "feature": total_metrics_agg["feature"],
                "psi_max": [0.0] * len(total_metrics_agg),
                "rc_min": [1.0] * len(total_metrics_agg),
                "missing_min": [0.0] * len(total_metrics_agg),
                "missing_max": [0.0] * len(total_metrics_agg),
            })

        summary_df = (
            total_metrics_agg
            .join(summary_audit, on="feature", how="left")
            .join(total_missing_metrics, on="feature", how="left")
            .join(total_real_bin_lift_metrics, on="feature", how="left")
            .join(monotonicity_df, on="feature", how="left")
            .with_columns([
                # 空值兜底，防备部分单点评估或无数据的极端情况
                pl.col("psi_max").fill_null(0.0),
                pl.col("rc_min").fill_null(1.0),
                pl.col("missing").fill_null(0.0),
                pl.col("missing_min").fill_null(0.0),
                pl.col("missing_max").fill_null(0.0),
                pl.col("mono").fill_null(1.0)
            ])
            .sort(["iv", "rc_min"], descending=[True, True]) # 按预测力和稳定性双重降序
            .select([
                "feature", "iv", "ks", "auc",
                "psi_max", "rc_min",
                "lift_min", "lift_max",
                "missing", "missing_min", "missing_max",
                "mono"
            ])
        )

        if feature_source_map:
            source_df = pl.DataFrame({
                "feature": list(feature_source_map.keys()),
                "data_source": [feature_source_map[feature] for feature in feature_source_map],
            })
            summary_df = summary_df.join(source_df, on="feature", how="left").with_columns(
                pl.col("data_source").fill_null("UNMAPPED")
            )
            summary_df = summary_df.select(
                ["feature", "data_source"] + [col for col in summary_df.columns if col not in {"feature", "data_source"}]
            )

        trend_tables = {
            metric: self._format_output(table)
            for metric, table in build_binning_trend_tables(
                stats_long=stats_long,
                risk_corr_long=risk_corr_long,
                monitor_group_level_metrics=monitor_group_level_metrics,
                monitoring_total=monitoring_total,
                group_col=group_col,
                missing_bin_index=MarsBinnerBase.IDX_MISSING,
            ).items()
        }

        return MarsBinningReport(
            summary_table=self._format_output(summary_df),
            trend_tables=trend_tables,
            detail_table=self._format_output(detail_table),
            group_col=display_group_col,
            detail_group_col=group_col,
            feature_data_source=feature_source_map or {},
            dt_col=dt_col,
            missing_by_day_table=missing_by_day_table,
            risk_corr_reference_table=self._format_output(
                risk_corr_reference_table
                if risk_corr_reference_table is not None
                else empty_risk_corr_reference_table(target_name),
            ),
        )

from mars.analysis._risk_profile import profile_risk as profile_risk  # noqa: E402
