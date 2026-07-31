"""MARS 特征分箱评估模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd
import polars as pl

from mars._compat import polars_is_in, remove_suffix
from mars.analysis._evaluation.aggregation import (
    aggregate_basic_stats,
    build_missing_by_day_table,
    get_benchmark_dist,
    rollup_total_stats,
)
from mars.analysis._evaluation.context import (
    binning_requires_target,
    build_binner,
    count_observed_target_classes,
    normalize_binary_target_column,
    normalize_feature_data_source,
    prepare_benchmark_frame,
    prepare_group_context,
    resolve_date_bounds,
    resolve_profile_by,
)
from mars.analysis._evaluation.metrics import (
    calculate_metrics_from_stats,
    ensure_woe_info,
)
from mars.analysis._evaluation.references import (
    build_risk_corr_reference_table,
    empty_risk_corr_reference_table,
)
from mars.analysis._evaluation.report_parts import build_binning_report_parts
from mars.compute import (
    OrderedMetricSortBy,
    RiskCorrBaseline,
    global_distribution_expr,
    normalize_ordered_metric_sort_by,
    normalize_risk_corr_baseline,
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
        ordered_metric_sort_by: OrderedMetricSortBy = "woe",
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
        ordered_metric_sort_by : OrderedMetricSortBy
            KS/AUC 的默认排序口径。`"woe"` 适合普通特征预测力评估；
            `"bin_index"` 适合概率、分数和强有序变量。

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
        self.ordered_metric_sort_by = normalize_ordered_metric_sort_by(ordered_metric_sort_by)

    @time_it
    def evaluate(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        target: str | None = None,
        features: list[str] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        feature_data_source: dict[str, list[str]] | None = None,
        weights_col: str | None = None,
        amount_col: str | None = None,
        binner: MarsBinnerBase | None = None,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        feature_start_aware_reference: bool | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        ordered_metric_sort_by: OrderedMetricSortBy | None = None,
        batch_size: int = 100,
    ) -> MarsRiskProfile:
        """
        对一次数据上下文执行分箱评估。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待评估样本表。
        target : str | None
            二分类目标列名。为 `None` 或列不存在时会进入无标签模式，只计算分布类指标
            和 PSI，不计算 IV、KS、AUC 等依赖标签的指标。
        features : list[str] | None
            本次评估的特征列；不传时自动排除目标列和分组列后选择候选特征。
        group_col : str | None
            已存在的分组列名，例如月份、客群或样本切片。
        time_col : str | None
            原始日期列名；与 `time_grain` 配合时会生成临时时间分组列。
        time_grain : str | None
            时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
            仅在传入 `time_col` 时生效，默认按 `"month"` 聚合。
        feature_data_source : dict[str, list[str]] | None
            特征来源映射，只对本次 active features 生效，用于报告中保留来源分层。
        weights_col : str | None
            样本权重列名。
        amount_col : str | None
            金额列名；传入后会在 `detail_table` 中额外产出金额视角指标。
        binner : MarsBinnerBase | None
            显式复用的分箱器；传入后不会再根据 `binning_type` 和 `binner_params`
            构造新分箱器。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            基准期样本。未传入 ``binner`` 时用其拟合分箱规则，并始终作为 PSI
            expected distribution；已传入 ``binner`` 时只按该固定规则构造基准分布。
            active features 与 ``weights_col`` 必须存在。监督分箱或
            ``risk_corr_baseline="benchmark"`` 时还必须包含至少两个有效类别的 target。
        psi_include_missing : bool
            计算 PSI 时是否单独保留缺失值分布。
        psi_include_special : bool
            计算 PSI 时是否单独保留特殊值分布。
        feature_start_aware_reference : bool | None
            是否按特征首次出现的分组选择 PSI 基准。传入 `None` 时沿用实例初始化时保存的默认值。
        risk_corr_baseline : RiskCorrBaseline | None
            本次评估使用的 RC 基准；传入 `None` 时沿用实例初始化时保存的默认值。
        ordered_metric_sort_by : OrderedMetricSortBy | None
            本次评估使用的 KS/AUC 排序口径；传入 `None` 时沿用实例初始化时保存的默认值。
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
        # 先把输入统一成内部 Polars 表，并解析本次画像的分组口径。
        working_df = self._ensure_polars_dataframe(df)
        benchmark_pl = (
            self._ensure_polars_dataframe(benchmark_df)
            if benchmark_df is not None
            else None
        )
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
        effective_ordered_metric_sort_by = normalize_ordered_metric_sort_by(
            ordered_metric_sort_by or self.ordered_metric_sort_by,
        )

        # 评估标签与拟合标签独立：表现期未成熟时仍可复用 benchmark 的监督分箱规则。
        has_target = target is not None and target in working_df.columns
        if has_target:
            working_df = normalize_binary_target_column(working_df, effective_target)
            observed_classes = count_observed_target_classes(working_df, effective_target)
            if observed_classes == 0:
                logger.warning(
                    "Target column '%s' has no observed values. Falling back to label-free mode.",
                    target,
                )
                has_target = False
            elif observed_classes < 2:
                raise ValueError(
                    f"Target column '{effective_target}' must have at least 2 observed classes "
                    "after excluding null / NaN values."
                )
        elif target is not None:
            logger.warning(
                "Target column '%s' was not found. Falling back to label-free mode.",
                target,
            )

        if not has_target:
            # 注入常量标签可以复用同一套聚合链路，后续再擦除标签依赖指标。
            working_df = working_df.with_columns(pl.lit(0).cast(pl.Int32).alias(effective_target))

        working_df, group_col = prepare_group_context(
            working_df,
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
            mars_group_col=self.MARS_GROUP_COL,
        )
        if amount_col is not None and amount_col not in working_df.columns:
            raise ValueError(f"Amount column '{amount_col}' was not found in dataframe.")

        # 未显式传 features 时，从分析上下文列之外自动推断候选特征。
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

        needs_supervised_fit = binning_requires_target(
            binning_type=self.binning_type,
            binner_params=self.binner_params,
        )
        needs_benchmark_risk_corr = (
            has_target and effective_risk_corr_baseline == "benchmark"
        )
        benchmark_prepared = None
        if benchmark_pl is not None:
            benchmark_prepared = prepare_benchmark_frame(
                benchmark_pl,
                features=target_features,
                weights_col=weights_col,
                target=original_target,
                require_binary_target=(
                    (binner is None and needs_supervised_fit)
                    or needs_benchmark_risk_corr
                ),
            )

        active_binner = binner
        binning_fit_source = "explicit_binner" if active_binner is not None else "df"
        if active_binner is None:
            fit_df = benchmark_prepared if benchmark_prepared is not None else working_df
            fit_has_target = (
                benchmark_prepared is not None and needs_supervised_fit
            ) or (benchmark_prepared is None and has_target)
            if benchmark_prepared is not None:
                binning_fit_source = "benchmark_df"
            active_binner = build_binner(
                binning_type=self.binning_type,
                binner_params=dict(self.binner_params),
                fit_has_target=fit_has_target,
                fit_df=fit_df,
                target=effective_target,
                features=target_features,
            )

        # 后续评估只消费分箱索引列，原始特征取值不再参与指标计算。
        df_binned = active_binner.transform(working_df, return_type="index")
        benchmark_binned = (
            active_binner.transform(benchmark_prepared, return_type="index")
            if benchmark_prepared is not None
            else None
        )
        if benchmark_binned is not None:
            expected_bin_cols = {f"{feature}_bin" for feature in target_features}
            missing_bin_cols = sorted(expected_bin_cols - set(benchmark_binned.columns))
            if missing_bin_cols:
                failed_features = [remove_suffix(col, "_bin") for col in missing_bin_cols]
                fit_failures = getattr(active_binner, "fit_failures_", {})
                raise ValueError(
                    "`benchmark_df` could not produce bins for active features "
                    f"{failed_features}. Fit failures: {fit_failures}."
                )
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

        # 先聚合到 feature/group/bin 粒度，所有后续指标都从这张长表派生。
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

        # PSI expected distribution 独立于 RC 基准：无 benchmark 时沿用最早分组。
        expected_dist = get_benchmark_dist(
            group_stats_raw=group_stats_raw,
            benchmark_binned=benchmark_binned,
            group_col=group_col,
            features=target_features,
            weights_col=weights_col,
        )
        feature_start_reference = None
        if effective_feature_start_reference:
            if benchmark_prepared is not None:
                logger.warning(
                    "`feature_start_aware_reference=True` was ignored because `benchmark_df` was provided."
                )
            elif not dt_col or dt_col not in working_df.columns:
                logger.warning(
                    "`feature_start_aware_reference=True` requires a valid `dt_col`; "
                    "falling back to the default reference logic."
                )
            else:
                # 对接入较晚的特征，按其上线后的首个稳定分组重锚 PSI 与 benchmark RC 参考。
                feature_start_reference = self._build_feature_start_reference(
                    df_binned=df_binned,
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

        # Total 统计用于全局预测力、默认 RC 基准和报告最右侧 Total 面板。
        total_stats_raw = rollup_total_stats(
            group_stats_raw,
            group_col=group_col,
        )
        # 分组指标保留跨期趋势，用于 trend_tables、RC 和图表面板。
        metrics_groups = (
            calculate_metrics_from_stats(
                binner=active_binner,
                stats_df=group_stats_raw,
                expected_dist=expected_dist,
                group_col=group_col,
                include_missing=psi_include_missing,
                include_special=psi_include_special,
                ordered_metric_sort_by=effective_ordered_metric_sort_by,
            )
            .with_columns(pl.col(group_col).cast(pl.String))
        )

        # Total 指标与分组指标保持同列结构，便于后续拼接和统一报表装配。
        metrics_total = calculate_metrics_from_stats(
            binner=active_binner,
            stats_df=total_stats_raw,
            expected_dist=expected_dist,
            group_col=group_col,
            include_missing=psi_include_missing,
            include_special=psi_include_special,
            ordered_metric_sort_by=effective_ordered_metric_sort_by,
        )

        if feature_start_reference is not None:
            monitor_group_stats_raw = feature_start_reference.get("monitor_group_stats_raw")
            if monitor_group_stats_raw is not None and not monitor_group_stats_raw.is_empty():
                # feature-start 特征只用上线后的有效窗口参与趋势监控，避免上线前全缺失拖低稳定性。
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
                        ordered_metric_sort_by=effective_ordered_metric_sort_by,
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
                    ordered_metric_sort_by=effective_ordered_metric_sort_by,
                ).select(monitor_metrics_groups.columns)

        metrics_total = metrics_total.select(metrics_groups.columns)
        # RC reference 在这里一次性确定，报告和绘图后续都复用同一份参考表。
        risk_corr_reference_table, risk_corr_reference_source = (
            build_risk_corr_reference_table(
                target_name=effective_target,
                metrics_groups=metrics_groups,
                metrics_total=metrics_total,
                group_col=group_col,
                risk_corr_baseline=effective_risk_corr_baseline,
                benchmark_binned=benchmark_binned,
                benchmark_features=target_features,
                benchmark_weights_col=weights_col,
                feature_start_reference=feature_start_reference,
                has_target=has_target,
                mars_group_col=self.MARS_GROUP_COL,
            )
        )

        # 单点评估只有 Total 一个分组，避免重复拼接语义相同的全量行。
        is_single_snapshot = (
            metrics_groups.select(pl.col(group_col).n_unique()).item() == 1 and
            metrics_groups.select(pl.col(group_col).first()).item() == "Total"
        )

        if is_single_snapshot:
            stats_long = metrics_groups
        else:
            stats_long = pl.concat([metrics_total, metrics_groups])

        # 单调性只看 Total 正常箱，避免跨期波动、Missing 和 Special 影响趋势方向判断。
        if has_target:
            monotonicity_df = (
                stats_long
                .filter((pl.col("bin_index") >= 0) & (pl.col(group_col) == "Total"))
                .group_by("feature")
                .agg(pl.corr("bin_index", "bad_rate", method="spearman").fill_nan(1.0).alias("mono"))
            )
        else:
            # 无标签模式没有坏率序列，单调性只保留中性占位值。
            monotonicity_df = pl.DataFrame({"feature": target_features, "mono": [1.0] * len(target_features)})

        # 报告对象只做结构化承载；各类导出和绘图入口在 reporting 层继续消费这些表。
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
            monitor_metrics_groups=monitor_metrics_groups,
            monitor_metrics_total=monitor_metrics_total,
        )
        profile_label = profile_by
        if dt_col and not profile_by:
            profile_label = "month (auto)"
        report._report_meta = {
            "row_count": int(working_df.height),
            "benchmark_row_count": (
                int(benchmark_prepared.height) if benchmark_prepared is not None else None
            ),
            "binning_fit_source": binning_fit_source,
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
        start_dt, end_dt = resolve_date_bounds(working_df, dt_col)
        report._report_meta["start_dt"] = start_dt
        report._report_meta["end_dt"] = end_dt
        if has_target and original_target and original_target in working_df.columns:
            try:
                event_rate = float(working_df.select(pl.col(original_target).cast(pl.Float64).mean()).item())
            except Exception:
                event_rate = None
            report._report_meta["event_rate_by_target"] = {str(original_target): event_rate}
        targets = [str(original_target)] if has_target and original_target else []

        # 无标签模式下擦除所有标签依赖指标，防止 dummy_target 产生误导性坏率。
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

            # detail 保留样本分布和 PSI，坏率、Lift、KS、AUC、IV 等统一置空。
            dt_cols = [c for c in null_cols if c in report._detail.columns]
            if isinstance(report._detail, pd.DataFrame):
                for c in dt_cols:
                    report._detail[c] = np.nan
            else:
                report._detail = report._detail.with_columns([
                    pl.lit(None).cast(pl.Float64).alias(c) for c in dt_cols
                ])

            # summary 只保留可解释的分布稳定性字段。
            sum_cols = ["iv", "ks", "auc", "rc_min", "lift_min", "lift_max", "mono"]
            sum_cols = [c for c in sum_cols if c in report._summary.columns]
            if isinstance(report._summary, pd.DataFrame):
                for c in sum_cols:
                    report._summary[c] = np.nan
            else:
                report._summary = report._summary.with_columns([
                    pl.lit(None).cast(pl.Float64).alias(c) for c in sum_cols
                ])

            # trend_tables 中也只保留不依赖标签的 PSI 趋势。
            if "psi" in report._trend_dict:
                report._trend_dict = {"psi": report._trend_dict["psi"]}
            else:
                report._trend_dict = {}

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
            ~polars_is_in(pl.col("feature"), pl.Series(override_features))
        )
        return pl.concat([retained_default, feature_expected_dist], how="vertical_relaxed")

    def _build_feature_start_baseline_reference(
        self,
        *,
        df_binned: pl.DataFrame,
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

        # 每个特征的上线时间不同，参考分布和坏率基准必须逐特征构造。
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

            # 先按自然日识别长期全缺失到持续活跃的切换点。
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

            # 起点之后才视为该特征的有效监控窗口，上线前全缺失不参与稳定性口径。
            start_dt = daily_missing.get_column(dt_alias).to_list()[start_idx]
            post_start_df = feature_df.filter(pl.col(dt_alias) >= pl.lit(start_dt))
            if post_start_df.is_empty():
                continue

            # feature-start 的 PSI 基准取上线后的首个分组，而不是全局最早分组。
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

            # 监控覆盖表保留上线后的真实分组序列，用来覆盖默认全量趋势表。
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

            # PSI expected distribution 使用上线首期分箱分布，避免上线前缺失期污染基准。
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

            # benchmark RC 模式可复用同一份 feature-start 首期坏率参考。
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
        monitor_metrics_groups: pl.DataFrame | None = None,
        monitor_metrics_total: pl.DataFrame | None = None,
    ) -> MarsBinningReport:
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
        monitor_metrics_groups : pl.DataFrame | None
            分组粒度监控指标表。
        monitor_metrics_total : pl.DataFrame | None
            全量粒度监控指标表。

        Returns
        -------
        MarsBinningReport
            报告容器实例。包含 Summary, Trend, Detail 三张重塑后的报表。
        """
        # 表构造全部下沉到 _evaluation.report_parts，evaluator 只负责输出格式和 report 对象装配。
        reference_table = (
            risk_corr_reference_table
            if risk_corr_reference_table is not None
            else empty_risk_corr_reference_table(target_name)
        )
        report_parts = build_binning_report_parts(
            stats_long=stats_long,
            metrics_groups=metrics_groups,
            metrics_total=metrics_total,
            group_col=group_col,
            monotonicity_df=monotonicity_df,
            binner=binner,
            target_name=target_name,
            feature_source_map=feature_source_map,
            risk_corr_reference_table=reference_table,
            monitor_metrics_groups=monitor_metrics_groups,
            monitor_metrics_total=monitor_metrics_total,
        )
        trend_tables = {
            metric: self._format_output(table)
            for metric, table in report_parts.trend_tables.items()
        }

        return MarsBinningReport(
            summary_table=self._format_output(report_parts.summary_table),
            trend_tables=trend_tables,
            detail_table=self._format_output(report_parts.detail_table),
            group_col=display_group_col,
            detail_group_col=group_col,
            feature_data_source=feature_source_map or {},
            dt_col=dt_col,
            missing_by_day_table=missing_by_day_table,
            risk_corr_reference_table=self._format_output(report_parts.risk_corr_reference_table),
        )
