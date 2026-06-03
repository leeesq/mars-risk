"""MARS 特征分箱评估模块。"""

import inspect
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.analysis.report import MarsEvaluationReport
from mars.core.base import MarsBaseEstimator
from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.utils.date import MarsDate
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsBinEvaluator(MarsBaseEstimator):
    """
    特征分箱评估器。

    该组件负责对特征分箱规则进行多维度的质量检验。系统通过无缝集成底层离散化引擎
    （原生分箱器或数学规划最优分箱器），在极速的单次扫描（One-Pass Scan）架构下，
    完成信息值 (IV)、群体稳定性 (PSI)、ROC曲线下面积 (AUC) 及 KS 统计量等关键风控
    度量指标的聚合计算。

    支持单点截面评估，以及基于时间窗口的跨期分布漂移与逻辑单调性追踪分析。

    Parameters
    ----------
    target : str, default "target"
        目标变量列名。在纯分布监测场景（如缺乏标签的 OOT 监控集）下，若该列不存在，
        引擎将自动降级为无标签模式（Label-Free Mode），仅执行 PSI 等分布指标的推演计算。

    binner : MarsBinnerBase, optional
        预实例化的底层分箱器对象（如 `MarsNativeBinner` 或 `MarsOptimalBinner`）。
        若提供此参数，评估器将直接复用该分箱器内部的区间边界与映射规则（`bin_cuts_` 等）；
        若为 None，评估器将在执行 `evaluate` 时，依据后续配置自动实例化并拟合全新的分箱器。

    binning_type : {"native", "opt"}, optional
        底层离散化引擎的动态路由选择。仅在 `binner` 为 None 时生效。
        - 'native': 调用基于 Scikit-Learn 与 Polars 的原生分箱器 (`MarsNativeBinner`)。
        - 'opt': 调用基于数学规划的最优分箱器 (`MarsOptimalBinner`)。
    bining_type : {"native", "opt"}, optional
        ``binning_type`` 的历史兼容别名，已弃用。
    feature_data_source : dict of str to list of str, optional
        特征到数据源标签的映射，可用于多源特征的分析与展示。
    **binner_kwargs
        任意透传至底层分箱器初始化方法的超参数字典（例如 `n_bins`, `method`, `min_bin_size` 等）。
        仅在 `binner` 为 None 且需要引擎内部动态构建分箱器时生效。系统会自动过滤目标
        分箱器不支持的冗余参数。

    Attributes
    ----------
    target : str
        实例绑定的目标变量列名。
    binner : MarsBinnerBase
        评估器当前持有的底层分箱器物理实体引用。
    binning_type : str
        初始化声明的底层离散化引擎类型。
    binner_kwargs : dict
        捕获的透传超参数映射字典。
    has_target_ : bool
        内部状态标识。指示在最近一次调用 `evaluate` 时，目标变量列是否实际存在并参与了
        有监督指标（IV/AUC等）的计算。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import MarsBinEvaluator
    >>> df = pl.read_parquet("credit_risk_data.parquet")
    >>>
    >>> # 初始化评估器并透传原生分箱器参数
    >>> evaluator = MarsBinEvaluator(
    ...     target="is_default",
    ...     binning_type="native",
    ...     method="quantile",
    ...     n_bins=10
    ... )
    >>>
    >>> # 执行评估流水线并生成多维分析报告
    >>> report = evaluator.evaluate(df, profile_by="month")
    >>>
    >>> # 导出评估报告至电子表格
    >>> report.write_excel("risk_evaluation_report.xlsx")
    """

    MARS_GROUP_COL = "mars_group"

    def __init__(
        self,
        target: str = "target",
        *,
        binner: MarsBinnerBase | None = None,
        binning_type: Literal["native", "opt"] | None = None,
        bining_type: Literal["native", "opt"] | None = None,
        feature_data_source: Dict[str, List[str]] | None = None,
        **binner_kwargs: Any,
    ) -> None:
        """
        初始化特征分箱评估器。

        Parameters
        ----------
        target : str, default "target"
            目标变量列名。
        binner : MarsBinnerBase, optional
            已拟合或待复用的分箱器实例。提供后将优先复用其分箱规则。
        binning_type : {"native", "opt"}, optional
            在未提供 ``binner`` 时，内部动态构建分箱器所采用的类型。
        bining_type : {"native", "opt"}, optional
            ``binning_type`` 的历史兼容别名，已弃用。
        feature_data_source : dict of str to list of str, optional
            特征到数据源标签的映射，可用于多源特征的分析与展示。
        **binner_kwargs
            透传给底层分箱器构造函数的额外参数。

        Raises
        ------
        ValueError
            当同时传入 ``binning_type`` 和 ``bining_type`` 且取值冲突时抛出。
        """
        super().__init__()
        self.target = target
        self.binner = binner
        self.feature_data_source = feature_data_source or {}
        self.binner_kwargs = binner_kwargs
        if bining_type is not None:
            warnings.warn(
                "`bining_type` is deprecated and will be removed in a future version. "
                "Use `binning_type` instead.",
                FutureWarning,
                stacklevel=2,
            )
            if binning_type is not None and binning_type != bining_type:
                raise ValueError("Received conflicting values for 'binning_type' and deprecated 'bining_type'.")

        resolved_binning_type = binning_type if binning_type is not None else bining_type
        if resolved_binning_type is None:
            resolved_binning_type = "native"

        self.binning_type = resolved_binning_type
        self.bining_type = resolved_binning_type

    @time_it
    def evaluate(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        features: List[str] | None = None,
        *,
        profile_by: str | None = None,
        dt_col: str | None = None,
        feature_start_aware_baseline: bool = False,
        feature_data_source: Dict[str, List[str]] | None = None,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        weights_col: str | None = None,
        batch_size: int = 100
    ) -> "MarsEvaluationReport":
        """
        执行特征分箱评估并生成报告。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            待评估的数据集，支持 Polars 与 Pandas DataFrame。
        features : Optional[List[str]]
            指定需要评估的特征列表。若为 ``None``，将自动扫描除目标列、
            分组列和权重列外的全部候选特征。
        profile_by : Optional[str]
            趋势分析的分组维度，可以是数据中已有列名，也可以是
            ``"day"``、``"week"``、``"month"`` 或 ``"7d"`` 这类时间粒度指令。
        dt_col : Optional[str]
            用于辅助 ``profile_by`` 生成时间切片的日期列名。
        feature_start_aware_baseline : bool, default False
            是否按特征首次出现的分组切片作为稳定性计算基准。
        feature_data_source : dict of str to list of str, optional
            本次评估中使用的特征数据源标签映射。若未提供，则回退到评估器实例级配置。
        psi_include_missing : bool, default=False
            计算 PSI 时是否包含缺失值箱。
        psi_include_special : bool, default=False
            计算 PSI 时是否包含特殊值箱。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame, None]
            PSI 计算使用的基准数据集。未提供时默认使用 ``df`` 中最早的分组切片。
        weights_col : Optional[str]
            样本权重列名。提供后，IV、AUC、KS、PSI、Lift 等指标
            将基于加权频数计算。
        batch_size : int, default 100
            特征切片的批处理大小。

        Returns
        -------
        MarsEvaluationReport
            特征评估报告对象，包含汇总表、趋势表和分箱明细表。

        Raises
        ------
        ValueError
            当真实目标列存在但取值少于两个类别时抛出。
        """

        # 上下文准备
        working_df = self._ensure_polars_dataframe(df)
        if benchmark_df is not None:
            benchmark_df = self._ensure_polars_dataframe(benchmark_df)
        original_target = self.target
        effective_target = self.target if self.target else "dummy_target"

        # 允许 target 为空，或数据集中本就不存在目标列。
        self.has_target_ = self.target is not None and self.target in working_df.columns

        if not self.has_target_:
            logger.info(
                f"Label-free mode enabled: target '{self.target}' was not found. "
                "A dummy target will be injected and only distribution metrics plus PSI will be evaluated."
            )
            # 临时注入常量标签，保持下游统计链路可复用。
            working_df = working_df.with_columns(pl.lit(0).cast(pl.Int32).alias(effective_target))
        self.target = effective_target

        # 检查 Target 有效性 (仅在有真实标签时检查)
        if self.has_target_:
            n_unique = working_df.select(pl.col(self.target).n_unique()).item()
            if n_unique < 2:
                raise ValueError(f"Target column '{self.target}' must have at least 2 unique values for evaluation.")

        working_df, group_col = self._prepare_context(working_df, profile_by, dt_col)

        # 自动识别特征列
        # 排除 target, weights, 和刚刚生成的统一 mars_group 列
        exclude_cols = {self.target, group_col}
        if weights_col:
            exclude_cols.add(weights_col)

        target_features = features if features else [
            c for c in working_df.columns if c not in exclude_cols
        ]

        effective_feature_data_source = feature_data_source if feature_data_source is not None else self.feature_data_source
        feature_source_map = self._normalize_feature_data_source(effective_feature_data_source, target_features)

        if self.binner is None:
            fit_kwargs = self.binner_kwargs if self.binner_kwargs is not None else {}

            binner_factory = {
                "native": MarsNativeBinner,
                "opt": MarsOptimalBinner
            }

            # 确定分箱器类型
            binner_cls = binner_factory.get(self.binning_type)
            if binner_cls is None:
                logger.warning(f"Unknown binning_type '{self.binning_type}'. Falling back to 'native'.")
                binner_cls = MarsNativeBinner

            # 获取目标类的构造函数签名
            # inspect.signature 会分析 __init__(self, n_bins, min_bin_size, ...) 到底有哪些参数
            sig = inspect.signature(binner_cls.__init__)
            valid_keys = set(sig.parameters.keys())

            # 过滤参数：只保留目标类支持的参数
            # 排除 'self' 和 'features' (因为 features 我们是显式传递的)
            valid_keys.discard("self")
            valid_keys.discard("features")

            clean_kwargs = {k: v for k, v in fit_kwargs.items() if k in valid_keys}

            # 记录被丢弃的参数，方便调试
            ignored_keys = set(fit_kwargs.keys()) - set(clean_kwargs.keys())
            if ignored_keys:
                logger.debug(f"Auto-cleaned kwargs for {binner_cls.__name__}. Ignored: {ignored_keys}")

            logger.info(f"Auto-fitting {binner_cls.__name__} internally with params: {clean_kwargs}.")

            # 实例化并拟合分箱器
            self.binner = binner_cls(features=target_features, **clean_kwargs)
            y_series = working_df.get_column(self.target)
            self.binner.fit(working_df, y_series)

        # 将原始特征映射为分箱索引列 `{feature}_bin`。
        logger.debug("Transforming features to bin indices.")
        df_binned = self.binner.transform(working_df, return_type="index")
        missing_by_day_table = self._build_missing_by_day_table(
            df=working_df,
            features=target_features,
            dt_col=dt_col,
            output_kind="pandas" if isinstance(df, pd.DataFrame) else "polars",
        )

        # 流式扫描并聚合到最细粒度统计表。
        logger.debug("Step 1: scanning grouped bin statistics.")
        group_stats_raw = self._agg_basic_stats(
            df_binned, group_col, target_features, self.target, weights_col,
            batch_size=batch_size
        )

        # [Reduce Phase A] 补全 WOE 信息
        # 计算 KS/AUC 依赖 WOE 排序。若分箱器无 WOE，利用 group_stats_raw 内存计算，无需扫原表。
        self._ensure_woe_info(group_stats_raw)

        # 获取 PSI 基准分布。若无外部基准，默认取最早一组。
        expected_dist = self._get_benchmark_dist(
            group_stats_raw, benchmark_df, group_col, target_features, weights_col
        )
        feature_start_reference = None
        if feature_start_aware_baseline:
            if benchmark_df is not None:
                logger.warning(
                    "`feature_start_aware_baseline=True` was ignored because `benchmark_df` was provided."
                )
            elif not dt_col or dt_col not in working_df.columns:
                logger.warning(
                    "`feature_start_aware_baseline=True` requires a valid `dt_col`; falling back to the default baseline logic."
                )
            else:
                feature_start_reference = self._build_feature_start_baseline_reference(
                    df_binned=df_binned,
                    missing_by_day_table=missing_by_day_table,
                    features=target_features,
                    dt_col=dt_col,
                    profile_by=profile_by,
                    group_col=group_col,
                    weights_col=weights_col,
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
        total_stats_raw = (
            group_stats_raw
            .group_by(["feature", "bin_index"])
            .agg([
                pl.col("count").sum(),
                pl.col("bad").sum()
            ])
            .with_columns(pl.lit("Total").alias(group_col)) # 显式标记为全量
        )

        # 指标计算
        logger.debug("Step 3: calculating metrics.")

        # 计算 Trend 数据
        metrics_groups = (
            self._calc_metrics_from_stats(
                group_stats_raw, expected_dist, group_col,
                # 传参
                include_missing=psi_include_missing,
                include_special=psi_include_special
            )
            .with_columns(pl.col(group_col).cast(pl.String))
        )

        # 计算 total 数据
        metrics_total = self._calc_metrics_from_stats(
            total_stats_raw, expected_dist, group_col,
            # 传参
            include_missing=psi_include_missing,
            include_special=psi_include_special
        )

        # 合并分组与总体结果
        if feature_start_reference is not None:
            monitor_group_stats_raw = feature_start_reference.get("monitor_group_stats_raw")
            if monitor_group_stats_raw is not None and not monitor_group_stats_raw.is_empty():
                monitor_total_stats_raw = (
                    monitor_group_stats_raw
                    .group_by(["feature", "bin_index"])
                    .agg([
                        pl.col("count").sum().alias("count"),
                        pl.col("bad").sum().alias("bad"),
                    ])
                    .with_columns(pl.lit("Total").alias(group_col))
                )
                monitor_metrics_groups = (
                    self._calc_metrics_from_stats(
                        monitor_group_stats_raw,
                        expected_dist,
                        group_col,
                        include_missing=psi_include_missing,
                        include_special=psi_include_special,
                    )
                    .with_columns(pl.col(group_col).cast(pl.String))
                )
                monitor_metrics_total = self._calc_metrics_from_stats(
                    monitor_total_stats_raw,
                    expected_dist,
                    group_col,
                    include_missing=psi_include_missing,
                    include_special=psi_include_special,
                ).select(monitor_metrics_groups.columns)

        metrics_total = metrics_total.select(metrics_groups.columns)

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
        if self.has_target_:
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
            feature_source_map=feature_source_map,
            dt_col=dt_col,
            missing_by_day_table=missing_by_day_table,
            risk_corr_baseline_df=feature_start_reference["baseline_bad_rate"] if feature_start_reference else None,
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
            "targets": [original_target] if self.has_target_ and original_target else [],
            "event_rate_by_target": {},
            "feature_start_aware_baseline": bool(feature_start_reference),
            "feature_start_baseline_features": sorted((feature_start_reference or {}).get("feature_start_dates", {}).keys()),
            "feature_start_baseline_dates": dict((feature_start_reference or {}).get("feature_start_dates", {})),
        }
        if dt_col and dt_col in working_df.columns:
            try:
                report._report_meta["start_dt"] = str(working_df.select(pl.col(dt_col).min()).item())
                report._report_meta["end_dt"] = str(working_df.select(pl.col(dt_col).max()).item())
            except Exception:
                report._report_meta["start_dt"] = None
                report._report_meta["end_dt"] = None
        if self.has_target_ and original_target and original_target in working_df.columns:
            try:
                event_rate = float(working_df.select(pl.col(original_target).cast(pl.Float64).mean()).item())
            except Exception:
                event_rate = None
            report._report_meta["event_rate_by_target"] = {str(original_target): event_rate}
        self.target = original_target

        # 无标签模式下擦除依赖真实坏样本标签的指标，保留分布类结果。
        if not self.has_target_:
            null_cols = ["bad", "bad_rate", "lift", "trend", "cum_bad", "cum_bad_rate", "ks_bin", "auc_bin", "iv_bin", "mono"]

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
        return report

    def _agg_basic_stats(
        self,
        df_binned: pl.DataFrame,
        group_col: str,
        features: List[str],
        y_col: str,
        weights_col: str | None,
        batch_size: int = 500
    ) -> pl.DataFrame:
        """
        [Map Phase] 全量数据扫描，计算最重要的统计量：样本数和坏样本数。

        Parameters
        ----------
        df_binned : pl.DataFrame
            已经过分箱索引转换的数据集。
        group_col : str
            分组列。
        features : List[str]
            特征名列表。
        y_col : str
            目标变量列。
        weights_col : Optional[str]
            权重列。
        batch_size : int
            每次聚合处理的特征数量。

        Returns
        -------
        pl.DataFrame
            长表格式的统计汇总表，包含 [group_col, feature, bin_index, count, bad]。
        """
        # 构造 bin 列名
        theoretical_bin_cols = [f"{f}_bin" for f in features]

        # 获取实际存在的列名
        # 使用 set 运算过滤，防止传入了未被分箱的特征导致报错
        existing_cols = set(df_binned.columns)
        actual_bin_cols = [c for c in theoretical_bin_cols if c in existing_cols]

        # 记录丢失的列
        missing_cols = set(theoretical_bin_cols) - set(actual_bin_cols)
        if missing_cols:
            logger.warning(
                f"{len(missing_cols)} features were not binned and will be skipped in evaluation. "
                f"All missing: {list(missing_cols)}"
            )

        if not actual_bin_cols:
            raise ValueError("No valid binned columns found in dataframe. Check your binner fit results.")

        # 使用实际存在的列进行后续操作
        bin_cols = actual_bin_cols

        # 确定必须要保留的索引列 (Group, Target, Weight)
        index_cols = [group_col, y_col] # 注意这里 y_col 被放到 index 是为了 unpivot 后不丢失信息
        if weights_col:
            index_cols.append(weights_col)

        # 预定义聚合表达式 (Lazy Expr)，避免在循环中重复构建
        # 统计样本数 (Count)
        expr_count = pl.col(weights_col).sum() if weights_col else pl.len()
        # 统计坏样本数 (Bad)
        expr_bad = (pl.col(y_col) * pl.col(weights_col)).sum() if weights_col else pl.col(y_col).sum()

        agg_exprs = [
            expr_count.alias("count"),
            expr_bad.alias("bad")
        ]

        result_frames: List[pl.DataFrame] = []

        # 分批处理特征
        for i in range(0, len(bin_cols), batch_size):
            # 切片：获取当前批次的特征列
            batch_bins = bin_cols[i : i + batch_size]

            # 构造查询计划 (Lazy Plan)
            # 这里的 .lazy() 它允许 Polars 优化器仅针对当前切片进行内存规划
            batch_res = (
                df_binned.lazy()
                .select([pl.col(c).cast(pl.Int16) for c in batch_bins] + [pl.col(c) for c in index_cols])
                .unpivot(
                    index=index_cols,
                    on=batch_bins,
                    variable_name="feature_bin",
                    value_name="bin_index"
                )
                # 还原原始特征名 (去除 _bin 后缀)
                .with_columns(
                    pl.col("feature_bin").str.replace("_bin", "").alias("feature")
                )
                # 聚合至最小粒度：(Group x Feature x Bin)
                .group_by([group_col, "feature", "bin_index"])
                .agg(agg_exprs)
                # 执行并物化 (Streaming 模式防止大聚合 OOM)
                .collect(engine="streaming")
            )

            result_frames.append(batch_res)

        if not result_frames:
            return pl.DataFrame()

        # 合并结果：将所有批次的小表 (Reduced Tables) 纵向合并
        return pl.concat(result_frames)

    def _ensure_woe_info(self, group_stats_raw: pl.DataFrame) -> None:
        """
        内存内 WOE 反向补全.

        当评估器检测到 `binner` 实例中缺失某些特征的 WOE 映射表时（例如：仅做了 transform 但未在
        当前 Label 上 fit，或者直接加载了无 WOE 的分箱规则），该方法利用 **已聚合的统计长表** 反向计算 WOE。

        Parameters
        ----------
        group_stats_raw : pl.DataFrame
            Map 阶段产出的统计长表。必须包含以下列：`['feature', 'bin_index', 'count', 'bad']`。

        Returns
        -------
        None
            该方法为 **In-place** 操作，计算结果将直接更新至 `self.binner.bin_woes_` 字典中。
        """
        features = group_stats_raw["feature"].unique().to_list()
        missing_woe_feats = [
            f for f in features
            if f not in self.binner.bin_woes_ or not self.binner.bin_woes_[f]
        ]

        if not missing_woe_feats:
            return

        logger.debug(f"Calculating missing WOEs for {len(missing_woe_feats)} features.")

        # 过滤出需要计算的特征
        target_stats = group_stats_raw.filter(pl.col("feature").is_in(missing_woe_feats))

        epsilon = 1e-6 # 平滑因子，防止除零或对0取对数

        # 计算 WOE
        woe_df = (
            target_stats
            .group_by(["feature", "bin_index"])
            .agg([
                pl.col("bad").sum().alias("bin_bad"),
                pl.col("count").sum().alias("bin_total")
            ])
            .with_columns([
                (pl.col("bin_total") - pl.col("bin_bad")).alias("bin_good")
            ])
            .with_columns([
                pl.col("bin_bad").sum().over("feature").alias("feature_total_bad"),
                pl.col("bin_good").sum().over("feature").alias("feature_total_good")
            ])
            .with_columns([
                (
                    ((pl.col("bin_bad") + epsilon) / (pl.col("feature_total_bad") + epsilon)) /
                    ((pl.col("bin_good") + epsilon) / (pl.col("feature_total_good") + epsilon))
                ).log().cast(pl.Float32).alias("woe")
            ])
        )
        # 提取数据并更新到分箱器
        # 使用 to_dict(as_series=False) 避免 Python 对象开销
        woe_data = woe_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)

        # 使用 defaultdict 简化映射构建，免去初始化判断
        temp_woe_map: Dict[str, Dict[int, float]] = defaultdict(dict)

        for feature, bin_index, woe in zip(
            woe_data["feature"],
            woe_data["bin_index"],
            woe_data["woe"],
            strict=False,
        ):
            # 过滤掉非法的 bin_index (如 Null 或 NaN)
            if bin_index is not None and not (isinstance(bin_index, float) and np.isnan(bin_index)):
                temp_woe_map[feature][int(bin_index)] = woe

        self.binner.bin_woes_.update(temp_woe_map)

    def _get_benchmark_dist(
        self,
        group_stats_raw: pl.DataFrame,
        bench_df: pl.DataFrame | None,
        group_col: str,
        features: List[str],
        w_col: str
    ) -> pl.DataFrame:
        """
        获取用于 PSI 计算的基准分布 (Expected Distribution).

        该方法负责计算 PSI 公式 $\\sum (A - E) \\times \\ln(A/E)$ 中的 $E$ (Expected Distribution)。
        支持两种基准策略，自动根据 `bench_df` 是否传入进行切换。

        Parameters
        ----------
        group_stats_raw : pl.DataFrame
            当前数据集的统计长表 (Actual Data Stats)。
            仅在 `bench_df` 为 None 时使用，用于提取时间最早的分组作为基准。
        bench_df : Optional[pl.DataFrame]
            外部基准数据集 (OOT/Training Set)。
            若提供，将对其执行 `transform` -> `unpivot` -> `agg` 流程以获取基准分布。
        group_col : str
            分组列名 (如 'month')。用于在内部基准模式下定位 "Earliest Group"。
        features : List[str]
            需要计算的特征列表。
        w_col : str
            权重列名。若存在，基准分布将基于权重求和计算；否则基于样本计数。

        Returns
        -------
        pl.DataFrame
            包含基准分布占比的长表。
            - Schema: `['feature', 'bin_index', 'expected_dist']`
            - `expected_dist`: 该分箱在基准集中的占比 (Float32, Sum=1.0)。

        Notes
        -----
        **策略详情 (Strategy Details):**
        1. **External Mode (外部基准)**:
           当传入 `bench_df` 时，系统会复用当前的 Binner 对其进行分箱转换，
           并执行与 `_agg_basic_stats` 类似的 Unpivot-Agg 操作。
           适用于 "Train vs OOT" 或 "Train vs Test" 的 PSI 计算。

        2. **Internal Mode (内部基准)**:
           当 `bench_df` 为 None 时，系统默认假设 `group_stats_raw` 是含有时间维度的。
           它会自动筛选 `group_col` 最小的组 (e.g., '2023-01') 作为基准。
           适用于 "Month vs Baseline Month" 的跨期稳定性监控。
        """
        if bench_df is not None:
            # Case A: 处理外部基准集
            bench_binned = self.binner.transform(bench_df, return_type="index")
            theoretical_bin_cols = [f"{f}_bin" for f in features]
            existing_cols = set(bench_binned.columns)
            bin_cols = [c for c in theoretical_bin_cols if c in existing_cols]
            missing_cols = set(theoretical_bin_cols) - set(bin_cols)
            if missing_cols:
                logger.warning(
                    f"{len(missing_cols)} benchmark features were not binned and will be skipped in PSI baseline. "
                    f"All missing: {list(missing_cols)}"
                )
            if not bin_cols:
                raise ValueError(
                    "No valid benchmark bin columns found in dataframe. "
                    "Check that benchmark_df includes the fitted feature set."
                )
            agg_expr = pl.col(w_col).sum().alias("expected_count") if w_col else pl.len().alias("expected_count")
            idx_cols = [w_col] if w_col else []

            return (
                bench_binned.select(bin_cols + idx_cols)
                .unpivot(index=idx_cols, on=bin_cols, variable_name="feat_bin", value_name="bin_index")
                .with_columns(pl.col("feat_bin").str.replace("_bin", "").alias("feature"))
                .group_by(["feature", "bin_index"])
                .agg(agg_expr)
                .with_columns((pl.col("expected_count") / pl.col("expected_count").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )
        else:
            # Case B: 内部基准
            min_group = group_stats_raw.select(pl.col(group_col).min()).item()
            logger.debug(f"[BASELINE] Using earliest group '{min_group}' as baseline (from stats cache).")

            return (
                group_stats_raw
                .filter(pl.col(group_col) == min_group)
                .group_by(["feature", "bin_index"])
                .agg(pl.col("count").sum().alias("expected_count"))
                .with_columns((pl.col("expected_count") / pl.col("expected_count").sum().over("feature")).alias("expected_dist"))
                .select(["feature", "bin_index", "expected_dist"])
            )

    def _calc_metrics_from_stats(
        self,
        stats_df: pl.DataFrame,
        expected_dist: pl.DataFrame,
        group_col: str,
        include_missing: bool = True,
        include_special: bool = True
    ) -> pl.DataFrame:
        """
        [Math Core] 基于聚合结果计算统计指标

        Parameters
        ----------
        stats_df : pl.DataFrame
            基础统计长表。
            必须包含列：`[group_col, 'feature', 'bin_index', 'count', 'bad']`。
        expected_dist : pl.DataFrame
            PSI 基准分布表。
            必须包含列：`['feature', 'bin_index', 'expected_dist']`。
        group_col : str
            分组维度列名 (如 'month')。
            计算累积指标 (KS/AUC) 时，会以此列和 'feature' 作为窗口分区 (Partition)。

        Returns
        -------
        pl.DataFrame
            包含分箱级指标详情的 DataFrame。
        """
        # 构建 WOE 映射表
        woe_data = [
            {"feature": f, "bin_index": i, "woe": w}
            for f, m in self.binner.bin_woes_.items() for i, w in m.items()
        ]
        schema = {"feature": pl.String, "bin_index": pl.Int16, "woe": pl.Float64}
        woe_df = pl.DataFrame(woe_data, schema=schema) if woe_data else pl.DataFrame([], schema=schema)

        # 合并统计量、基准分布与 WOE
        base_df = (
            stats_df
            .join(expected_dist, on=["feature", "bin_index"], how="left")
            .join(woe_df, on=["feature", "bin_index"], how="left")
            .with_columns([
                (pl.col("count") - pl.col("bad")).alias("good"),
                pl.col("expected_dist").fill_null(1e-9),
                pl.col("woe").fill_null(0)
            ])
        )

        epsilon = 1e-6

        # 构建 PSI 专用计算域
        # 定义哪些箱子参与 PSI 计算
        # 约定: Missing=-1, Special <= -3, Normal >= 0, Other=-2
        psi_valid_cond = pl.lit(True)

        if not include_missing:
            psi_valid_cond &= (pl.col("bin_index") != -1)

        if not include_special:
            psi_valid_cond &= (pl.col("bin_index") > -3)

        # 计算双套分布
        # 全量分布 (用于 IV, BadRate, Lift)
        base_df = base_df.with_columns([
            pl.col("count").sum().over([group_col, "feature"]).alias("total_count"),
            pl.col("bad").sum().over([group_col, "feature"]).alias("total_bad"),
            pl.col("good").sum().over([group_col, "feature"]).alias("total_good"),
        ])

        # PSI 专用分布
        base_df = base_df.with_columns([
            # 动态计算 Actual 的有效总数
            pl.col("count")
              .filter(psi_valid_cond)
              .sum()
              .over([group_col, "feature"])
              .alias("total_count_psi"),

            # 动态计算 Expected 的有效总占比 (因为 expected_dist 是比例，sum 可能等于 0.8)
            pl.col("expected_dist")
              .filter(psi_valid_cond)
              .sum()
              .over([group_col, "feature"])
              .alias("total_expected_dist_psi")
        ])

        # 指标计算
        base_df = base_df.with_columns([
            ((pl.col("count") + epsilon) / (pl.col("total_count") + epsilon)).alias("actual_dist"),
            (pl.col("bad") / (pl.col("total_bad") + epsilon)).alias("bad_dist"),
            (pl.col("good") / (pl.col("total_good") + epsilon)).alias("good_dist"),
            (pl.col("bad") / (pl.col("count") + epsilon)).alias("bad_rate"),

            # PSI
            # 计算归一化后的 Actual% (只针对有效箱)
            (pl.col("count") / (pl.col("total_count_psi") + epsilon)).alias("act_prob_clean"),
            # 计算归一化后的 Expected% (只针对有效箱)
            #    例如：如果剔除缺失值后，剩余 expected_dist 之和为 0.8，则每一项除以 0.8 放大
            (pl.col("expected_dist") / (pl.col("total_expected_dist_psi") + epsilon)).alias("exp_prob_clean")
        ])

        # PSI bin contribution
        base_df = base_df.with_columns([
            # 仅在有效箱上计算 PSI，无效箱置为 None
            pl.when(psi_valid_cond)
            .then(
                (pl.col("act_prob_clean") - pl.col("exp_prob_clean"))
                *
                (pl.col("act_prob_clean") / (pl.col("exp_prob_clean") + epsilon)).log()
            )
            .otherwise(None)
            .alias("psi_bin"),

            # Lift
            (
                pl.col("bad_rate")
                /
                ((pl.col("total_bad") + epsilon) / (pl.col("total_count") + epsilon))
            ).alias("lift"),

            # IV
            (
                (pl.col("bad_dist") - pl.col("good_dist"))
                *
                ((pl.col("bad_dist") + epsilon) / (pl.col("good_dist") + epsilon)).log()
            ).cast(pl.Float32).alias("iv_bin")
        ])

        # 计算有序指标 (AUC, KS, IV)：必须按 WOE 风险程度排序
        sorted_df = base_df.sort([group_col, "feature", "woe"])

        # 累积分布用于计算 KS 和 AUC
        sorted_df = sorted_df.with_columns([
            pl.col("bad_dist").cum_sum().over([group_col, "feature"]).alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over([group_col, "feature"]).alias("cum_good_dist"),
        ])

        sorted_df = sorted_df.with_columns([

            ((pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs() * 100).alias("ks_bin"),

            # AUC 梯形法则计算面积
            (
                (pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1, fill_value=0).over([group_col, "feature"]))
                *
                (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1, fill_value=0).over([group_col, "feature"]))
                / 2
            ).alias("auc_bin")
        ])

        sorted_df = sorted_df.with_columns([
            pl.when(pl.col("psi_bin").abs() < 1e-12)
              .then(0.0)
              .otherwise(pl.col("psi_bin"))
              .alias("psi_bin")
        ])

        return sorted_df

    def _prepare_context(self,
                         df: pl.DataFrame,
                         profile_by: str | None,
                         dt_col: str | None
                         ) -> Tuple[pl.DataFrame, str]:
        """
        标准化评估所需的分组上下文。

        该方法负责解析用户传入的 `profile_by` 和 `dt_col` 参数，确定最终用于趋势分析的分组列。
        如果需要基于时间切片（如按月、按周），它会自动调用 `MarsDate` 生成派生列。

        Parameters
        ----------
        df : pl.DataFrame
            输入的 Polars DataFrame。
        profile_by : Optional[str]
            分组指令。可以是具体的列名（如 'channel'），也可以是时间粒度指令（'day', 'week', 'month'）。
        dt_col : Optional[str]
            日期列名。仅当 `profile_by` 为时间粒度指令时必须提供。

        Returns
        -------
        Tuple[pl.DataFrame, str]
            - **pl.DataFrame**: 处理后的 DataFrame。如果涉及时间截断或兜底逻辑，会新增一列。
            - **str**: 最终确定的分组列名（可能是原始列，也可能是新增的临时列）。

        Notes
        -----
        **策略优先级 (Strategy Priority):**

        1. **智能默认 (Smart Default)**:
           若仅提供 `dt_col` 但未指定 `profile_by`，系统默认按 **'month'** (月度) 进行聚合。

        2. **自动时间聚合 (Auto Date Truncation)**:
           若 `profile_by` 为 `['day', 'week', 'month']` 且提供了 `dt_col`，
           系统会自动生成一个新的时间切片列 (e.g., `_mars_auto_month`) 并以此分组。

        3. **常规分组 (Explicit Grouping)**:
           若 `profile_by` 指定了具体的现有列 (e.g., 'city', 'vintage')，则直接使用该列。

        4. **全局兜底 (Global Fallback)**:
           若未提供任何分组信息，系统生成一个包含常量值 "total" 的列 `_mars_auto_total`，
           将所有样本视为同一个分组（适用于单点评估）。
        """

        # 有时间没分组 -> 默认按月
        if dt_col and not profile_by:
            logger.info("`dt_col` was provided without `profile_by`; defaulting trend grouping to 'month'.")
            profile_by = "month"

        # 新逻辑：支持正则匹配 'Nd' (如 '3d', '14d')
        is_date_granularity = profile_by in ["day", "week", "month"] or (
            isinstance(profile_by, str) and re.match(r"^\d+d$", profile_by.lower())
        )

        # 处理时间切片
        if dt_col and is_date_granularity:
            if profile_by == "month":
                date_expr = MarsDate.dt2month(dt_col).alias(self.MARS_GROUP_COL)
            elif profile_by == "week":
                date_expr = MarsDate.dt2week(dt_col).alias(self.MARS_GROUP_COL)
            else:
                # 把未命中的 day / 3d / 14d 直接丢给 dt2day 处理
                date_expr = MarsDate.dt2day(dt_col, interval=profile_by).alias(self.MARS_GROUP_COL)
            return df.with_columns(date_expr), self.MARS_GROUP_COL

        # 常规分组：按现有列
        if profile_by:
            if profile_by in df.columns:
                return df.with_columns(pl.col(profile_by).cast(pl.String).alias(self.MARS_GROUP_COL)), self.MARS_GROUP_COL
            else:
                logger.warning(f"Column '{profile_by}' was not found. Falling back to snapshot mode.")

        # 兜底逻辑：单点评估
        return df.with_columns(pl.lit("Total").alias(self.MARS_GROUP_COL)), self.MARS_GROUP_COL

    @staticmethod
    def _normalize_feature_data_source(
        feature_data_source: Dict[str, List[str]] | None,
        features: List[str],
    ) -> Dict[str, str]:
        feature_set = set(features)
        if not feature_data_source:
            return {feature: "UNMAPPED" for feature in features}

        normalized: Dict[str, str] = {}
        mapped_features = set()
        for data_source, source_features in feature_data_source.items():
            for feature in source_features or []:
                if feature not in feature_set:
                    raise ValueError(
                        "feature_data_source contains features outside the active evaluation feature set: "
                        f"{feature}"
                    )
                normalized[feature] = str(data_source)
                mapped_features.add(feature)

        for feature in feature_set - mapped_features:
            normalized[feature] = "UNMAPPED"

        return normalized

    def _build_missing_by_day_table(
        self,
        *,
        df: pl.DataFrame,
        features: List[str],
        dt_col: str | None,
        output_kind: str,
    ) -> Union[pl.DataFrame, pd.DataFrame] | None:
        if not dt_col or dt_col not in df.columns:
            return None

        try:
            from mars.analysis.profiler import profile_stats

            missing_values = getattr(self.binner, "missing_values", None)
            if missing_values is None:
                missing_values = self.binner_kwargs.get("missing_values")
            special_values = getattr(self.binner, "special_values", None)
            if special_values is None:
                special_values = self.binner_kwargs.get("special_values")

            report = profile_stats(
                df,
                metrics=["missing"],
                features=features,
                profile_by="day",
                dt_col=dt_col,
                missing_values=missing_values,
                special_values=special_values,
                enable_sparkline=False,
            )
            missing_table = report.dq_tables.get("missing")
            if missing_table is None:
                return None
            if output_kind == "pandas" and isinstance(missing_table, pl.DataFrame):
                return missing_table.to_pandas()
            if output_kind == "polars" and isinstance(missing_table, pd.DataFrame):
                return pl.from_pandas(missing_table)
            return missing_table
        except Exception as exc:
            logger.warning("Missing-by-day trend generation skipped due to error: %s", exc)
            return None

    @staticmethod
    def _is_time_granularity(profile_by: str | None) -> bool:
        if profile_by in {"day", "week", "month"}:
            return True
        return isinstance(profile_by, str) and re.match(r"^\d+d$", profile_by.lower()) is not None

    @staticmethod
    def _detect_feature_start_index(
        inactive_flags: List[bool],
        *,
        leading_inactive_ratio: float = 0.90,
        sustain_window: int = 3,
        sustain_active_ratio: float = 2.0 / 3.0,
    ) -> int | None:
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
        if feature_expected_dist.is_empty():
            return default_expected_dist

        override_features = feature_expected_dist.get_column("feature").unique().to_list()
        retained_default = default_expected_dist.filter(~pl.col("feature").is_in(override_features))
        return pl.concat([retained_default, feature_expected_dist], how="vertical_relaxed")

    @staticmethod
    def _merge_feature_frame(
        default_df: pl.DataFrame,
        override_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        if override_df is None or override_df.is_empty():
            return default_df

        override_features = override_df.get_column("feature").unique().to_list()
        retained_default = default_df.filter(~pl.col("feature").is_in(override_features))
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
    ) -> Dict[str, Any] | None:
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
            if self.has_target_ and self.target in working_df.columns:
                select_cols.append(self.target)

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

            if self.has_target_ and self.target in post_start_df.columns:
                if weights_col and weights_col in post_start_df.columns:
                    monitor_bad_expr = (
                        pl.col(self.target).cast(pl.Float64) * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("bad")
                else:
                    monitor_bad_expr = pl.col(self.target).cast(pl.Float64).sum().alias("bad")
            else:
                monitor_bad_expr = pl.lit(0.0).alias("bad")

            monitor_group_stats_df = (
                post_start_df
                .group_by([group_col, "bin_index"])
                .agg([monitor_count_expr, monitor_bad_expr])
                .select([
                    pl.col(group_col).cast(pl.String).alias(group_col),
                    pl.lit(feature).alias("feature"),
                    pl.col("bin_index"),
                    pl.col("count").cast(pl.Float64),
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
                .with_columns(
                    (pl.col("expected_count") / (pl.col("expected_count").sum() + 1e-9)).alias("expected_dist")
                )
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

            if self.has_target_ and self.target in baseline_rows.columns:
                if weights_col and weights_col in baseline_rows.columns:
                    bad_expr = (
                        pl.col(self.target).cast(pl.Float64) * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("base_bad")
                    total_expr = pl.col(weights_col).cast(pl.Float64).sum().alias("base_total")
                else:
                    bad_expr = pl.col(self.target).cast(pl.Float64).sum().alias("base_bad")
                    total_expr = pl.len().cast(pl.Float64).alias("base_total")

                baseline_bad_rate_df = (
                    baseline_rows
                    .filter(pl.col("bin_index") >= 0)
                    .group_by("bin_index")
                    .agg([bad_expr, total_expr])
                    .with_columns(
                        (pl.col("base_bad") / (pl.col("base_total") + 1e-9)).alias("base_br")
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

    def _build_bin_label_map(self, stats_long: pl.DataFrame) -> pl.DataFrame:
        """Build the feature/bin-index label lookup used by detail reports."""
        map_rows: list[dict[str, Any]] = []
        features = set(stats_long["feature"].unique().to_list())

        for feature, mapping in self.binner.bin_mappings_.items():
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
        feature_source_map: Dict[str, str] | None = None,
        dt_col: str | None = None,
        missing_by_day_table: Union[pl.DataFrame, pd.DataFrame] | None = None,
        risk_corr_baseline_df: pl.DataFrame | None = None,
        feature_valid_groups_df: pl.DataFrame | None = None,
        monitor_metrics_groups: pl.DataFrame | None = None,
        monitor_metrics_total: pl.DataFrame | None = None,
    ) -> "MarsEvaluationReport":
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

        Returns
        -------
        MarsEvaluationReport
            报告容器实例。包含 Summary, Trend, Detail 三张重塑后的报表。
        """
        map_df = self._build_bin_label_map(stats_long)

        # 此时 stats_long 和 map_df 都是 Int16，Join 安全，不会发生类型提升
        detail_base = (
            stats_long
            .join(map_df, on=["feature", "bin_index"], how="left")
            .with_columns(pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)))
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
        from mars.feature.binner import MarsBinnerBase

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

                # 针对非 Normal 箱的内部排序:
                # -1 (Missing) -> 10000
                # -2 (Other) -> 10001
                # < -2 (Special) -> 20000 + abs
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
            # 累积样本数
            pl.col("count").cum_sum().over(["feature", group_col]).alias("cum_count"),
            # 累积坏样本数
            pl.col("bad").cum_sum().over(["feature", group_col]).alias("cum_bad"),
            # 累积好样本数
            (pl.col("count") - pl.col("bad")).cum_sum().over(["feature", group_col]).alias("cum_good")
        ]).with_columns([
            # 累积坏账率 = 累积坏 / 累积总
            (pl.col("cum_bad") / (pl.col("cum_count") + 1e-9)).alias("cum_bad_rate"),

            # 计算箱占比 pct = count / total_count
            # total_count 已经在 _calc_metrics_from_stats 中计算并包含在 stats_long 中
            (pl.col("count") / (pl.col("total_count") + 1e-9)).alias("pct"),

            pl.col("bin_index").max().over(["feature", group_col]).alias("bin_index_max")
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
                (pl.col("count") - pl.col("bad")).alias("good"),
                (pl.col("bad") / (pl.col("count") + 1e-9)).alias("bad_rate"),

                # total 行代表全量样本，占比固定为 1.0。
                pl.lit(1.0).alias("pct"),

                # 累积列 (对于 total 行，累积值等于自身)
                pl.col("count").alias("cum_count"),
                pl.col("bad").alias("cum_bad"),
                (pl.col("bad") / (pl.col("count") + 1e-9)).alias("cum_bad_rate"),

                # AUC 方向修正
                pl.when(pl.col("auc_bin") < 0.5)
                  .then(pl.lit(1) - pl.col("auc_bin"))
                  .otherwise(pl.col("auc_bin"))
                  .alias("auc_bin"),

                # 标识列与排序键 (确保排在最后)
                pl.lit(9999).cast(pl.Int16).alias("bin_index"),
                pl.lit("Total").alias("bin_label"),

                pl.lit("汇总组").alias("bin_type"),

                pl.lit(2).cast(pl.Int32).alias("_sort_group"), # [Fix] Int32
                pl.lit(0).cast(pl.Int32).alias("_sort_idx")    # [Fix] Int32
            ])
        )

        # total 行也要 join trend 列
        total_rows = total_rows.join(trend_shape_df, on="feature", how="left")

        targets = [
            "feature", group_col, "bin_index", "bin_label", "_sort_group", "_sort_idx",
            "count", "pct", "bad", "good", "bad_rate", "lift", "trend",
            "cum_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ]

        detail_table = (
            pl.concat([
                detail_table.select(targets),
                total_rows.select(targets)
            ])
            .sort(["feature", group_col, "_sort_group", "_sort_idx"])
        )

        detail_table = detail_table.select([
            pl.lit(self.target).alias("y"),
            "feature", "trend", group_col, "bin_index", "bin_label",
            "count", "bad", "good", "pct", "bad_rate", "lift",
            "cum_count", "cum_bad", "cum_bad_rate",
            "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count",
            "bin_type"
        ])

        if feature_source_map:
            source_df = pl.DataFrame({
                "feature": list(feature_source_map.keys()),
                "data_source": [feature_source_map[feature] for feature in feature_source_map],
            })
            detail_table = detail_table.join(source_df, on="feature", how="left").with_columns(
                pl.col("data_source").fill_null("UNMAPPED")
            )

        #  Intermediate Calculations
        # RiskCorr (RC) 跨期稳定性逻辑
        # 确定基准序列 (选取时间最早的一组)
        first_group = metrics_groups.select(pl.col(group_col).min()).item()
        default_baseline_df = (
            metrics_groups
            .filter((pl.col(group_col) == first_group) & (pl.col("bin_index") >= 0))
            .select(["feature", "bin_index", "bad_rate"])
            .rename({"bad_rate": "base_br"})
        )
        if risk_corr_baseline_df is not None and not risk_corr_baseline_df.is_empty():
            override_features = risk_corr_baseline_df.get_column("feature").unique().to_list()
            baseline_df = pl.concat(
                [
                    default_baseline_df.filter(~pl.col("feature").is_in(override_features)),
                    risk_corr_baseline_df.select(["feature", "bin_index", "base_br"]),
                ],
                how="vertical_relaxed",
            )
        else:
            baseline_df = default_baseline_df

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
            monitoring_groups.select(["feature", group_col, "bin_index", "bad_rate"]),
            monitoring_total.select(["feature", group_col, "bin_index", "bad_rate"])
        ])

        # 计算 RiskCorr 长表: [feature, group_col, risk_corr]
        risk_corr_long = (
            all_metrics_for_corr
            .filter(pl.col("bin_index") >= 0)
            .join(baseline_df, on=["feature", "bin_index"], how="left")
            .group_by(["feature", group_col])
            .agg(
                # 1. 只有当正常箱数 > 1 时，才去计算皮尔逊相关系数
                # 2. 如果正常箱 <= 1, 直接赋予 1.0 放行
                # 3. .fill_nan(1.0) 用于兜底多箱但坏率完全一致导致方差为 0 报错的情况
                pl.when(pl.len() > 1)
                  .then(pl.corr("bad_rate", "base_br", method="pearson"))
                  .otherwise(pl.lit(1.0))
                  .fill_nan(1.0)
                  .fill_null(1.0)
                  .alias("risk_corr")
            )
        )
        _ = _null_metric_for_invalid_groups

        # 分组指标聚合
        # 将分箱粒度聚合为分组粒度 (如: Month Level)
        group_level_metrics = (
            metrics_groups
            .group_by(["feature", group_col])
            .agg([
                pl.col("iv_bin").sum().alias("iv"),
                pl.col("auc_bin").sum().alias("auc"),
                (
                    pl.when(pl.col("bin_index") == MarsBinnerBase.IDX_MISSING)
                    .then(pl.col("count"))
                    .otherwise(0)
                    .sum()
                    /
                    (pl.col("count").sum() + 1e-9)
                ).alias("missing"),
                pl.col("lift").max().alias("lift"),
            ])
            # 确保 AUC 方向正确 (>= 0.5)
            .with_columns(
                pl.when(pl.col("auc") < 0.5).then(pl.lit(1) - pl.col("auc")).otherwise(pl.col("auc")).alias("auc")
            )
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
            .agg(
                (
                    pl.when(pl.col("bin_index") == MarsBinnerBase.IDX_MISSING)
                    .then(pl.col("count"))
                    .otherwise(0)
                    .sum()
                    /
                    (pl.col("count").sum() + 1e-9)
                ).alias("missing")
            )
        )

        total_real_bin_lift_metrics = (
            metrics_total
            .filter(pl.col("bin_index") >= 0)
            .group_by("feature")
            .agg([
                pl.col("lift").min().alias("lift_min"),
                pl.col("lift").max().alias("lift_max"),
            ])
        )

        # Part 3: Summary Table
        total_metrics_agg = (
            metrics_total.group_by("feature")
            .agg([
                pl.col("iv_bin").sum().alias("iv"),
                pl.col("ks_bin").max().alias("ks"),
                pl.col("auc_bin").sum().alias("auc")
            ])
            .with_columns(
                pl.when(pl.col("auc") < 0.5)
                .then(pl.lit(1.0) - pl.col("auc"))
                .otherwise(pl.col("auc"))
                .alias("auc")
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

        # Trend Tables
        trend_tables = {}
        target_metrics = ["psi", "auc", "ks", "iv", "missing", "lift", "bad_rate", "risk_corr"]

        for metric in target_metrics:
            if metric == "risk_corr":
                pivot_src = risk_corr_long
            elif metric == "psi":
                psi_total_src = (
                    monitoring_total
                    .group_by(["feature", group_col])
                    .agg(pl.col("psi_bin").sum().alias("psi"))
                )
                pivot_src = pl.concat(
                    [
                        monitor_group_level_metrics.select(["feature", group_col, "psi"]),
                        psi_total_src.select(["feature", group_col, "psi"]),
                    ],
                    how="vertical_relaxed",
                )
            else:
                if metric == "bad_rate":
                    agg_func = (pl.col("bad").sum() / (pl.col("count").sum() + 1e-9))
                elif metric == "missing":
                    agg_func = (
                        pl.when(pl.col("bin_index") == MarsBinnerBase.IDX_MISSING)
                        .then(pl.col("count"))
                        .otherwise(0)
                        .sum()
                        /
                        (pl.col("count").sum() + 1e-9)
                    )
                elif metric == "lift":
                    agg_func = pl.col("lift").max()
                elif metric == "ks":
                    agg_func = pl.col(f"{metric}_bin").max()
                else:
                    agg_func = pl.col(f"{metric}_bin").sum()

                pivot_src = stats_long.group_by([group_col, "feature"]).agg(agg_func.alias(metric))

                # Pivot 前的方向校正
                if metric == "auc":
                    pivot_src = pivot_src.with_columns(
                        pl.when(pl.col(metric) < 0.5).then(pl.lit(1) - pl.col(metric)).otherwise(pl.col(metric)).alias(metric)
                    )

            # 执行 Pivot
            pivot_df = pivot_src.pivot(
                index="feature", on=group_col, values=metric
            ).sort("feature").with_columns(pl.lit("Float64").alias("dtype"))

            # 排序列顺序，确保 total 在最右侧
            cols = [c for c in pivot_df.columns if c not in ["feature", "dtype"]]
            sorted_cols = sorted([c for c in cols if c != "Total"]) + (["Total"] if "Total" in cols else [])

            trend_tables[metric] = self._format_output(pivot_df.select(["feature", "dtype"] + sorted_cols))

        return MarsEvaluationReport(
            summary_table=self._format_output(summary_df),
            trend_tables=trend_tables,
            detail_table=self._format_output(detail_table),
            group_col=group_col,
            feature_data_source=feature_source_map or {},
            dt_col=dt_col,
            missing_by_day_table=missing_by_day_table,
        )

    def plot_feature_binning_risk_trends(
        self,
        *,
        report: Optional["MarsEvaluationReport"] = None,
        df_detail: Union[pl.DataFrame, pd.DataFrame, None] = None,
        features: Union[str, List[str], None] = None,
        group_col: str | None = None,
        target_name: str | None = None,
        sort_by: str = "iv",
        ascending: bool = False,
        dpi: int = 150,
    ) -> None:
        """
        批量绘制特征分箱风险趋势图。

        该方法用于展示特征在不同分组切片下的样本分布与坏率走势，
        便于快速识别风险倒挂、稳定性漂移和分箱效果异常。

        Parameters
        ----------
        report : MarsEvaluationReport, optional
            由 ``evaluate`` 生成的评估报告对象。
        df_detail : Union[pl.DataFrame, pd.DataFrame], optional
            直接传入分箱明细表；当未提供 ``report`` 时使用。
        features : str or List[str], optional
            需要绘图的特征名称。若为 ``None``，绘制明细表中的全部特征。
        group_col : str, optional
            分组列名。当直接传入 ``df_detail`` 且无法自动推断时可显式指定。
        target_name : str, optional
            图表标题中展示的目标名称。未提供时默认使用 ``self.target``。
        sort_by : str, default "iv"
            绘图特征的排序依据。
        ascending : bool, default False
            是否按 ``sort_by`` 升序绘制。
        dpi : int, default 150
            图像分辨率。

        Raises
        ------
        ValueError
            当 ``report`` 和 ``df_detail`` 同时缺失时抛出。
        """
        target_df = None
        target_group_col = None

        # 尝试从 Report 提取绘图所需数据
        if report is not None:
            dt_table = report.detail_table
            # 兼容 Pandas DataFrame
            target_df = self._ensure_polars_dataframe(dt_table).filter(pl.col("bin_index") != 9999)
            target_group_col = report.group_col

        # 尝试从 df_detail 提取数据
        elif df_detail is not None:
            target_df = self._ensure_polars_dataframe(df_detail).filter(pl.col("bin_index") != 9999)

            if group_col:
                target_group_col = group_col
            else:
                # 自动推断分组列
                # 排除已知列，剩下的通常就是分组列
                known = {"feature", "bin_index", "bin_label", "count", "bad", "bad_rate", "lift", "psi_bin", "ks_bin", "auc_bin", "iv_bin", "total_count", "trend", "y"}
                candidates = [c for c in target_df.columns if c not in known]
                target_group_col = candidates[0] if candidates else "month"
                logger.debug(f"Auto-inferred group_col: '{target_group_col}'")
        else:
            raise ValueError("Must provide either 'report' or 'df_detail' to plot.")

        if features is None:
            features = target_df["feature"].unique().to_list()
        elif isinstance(features, str):
            features = [features]

        # 确定最终显示的 Target Name
        # 优先使用传入的 target_name (多目标循环时传入)，否则使用实例绑定的 target
        final_target_name = target_name if target_name else self.target

        from mars.utils.plotter import MarsPlotter

        # 调用 MarsPlotter 绘图组件进行渲染
        MarsPlotter.plot_feature_binning_risk_trend_batch(
            df_detail=target_df,
            features=features,
            group_col=target_group_col,
            target_name=final_target_name, # 透传参数
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi
        )

def profile_risk(
    df: Union[pl.DataFrame, pd.DataFrame],
    *,
    target: Union[str, List[str]] | None = None, # 放宽约束
    features: List[str] | None = None,
    feature_data_source: Dict[str, List[str]] | None = None,
    profile_by: str | None = None,
    dt_col: str | None = None,
    feature_start_aware_baseline: bool = False,

    binning_type: Literal["native", "opt"] = "native",
    n_bins: int = 10,
    min_bin_size: float = 0.02,
    monotonic_trend: str = "auto_asc_desc",
    special_values: List[Any] | None = None,
    missing_values: List[Any] | None = None,
    binner_kwargs: Dict[str, Any] | None = None,

    benchmark_df: Union[pl.DataFrame, pd.DataFrame] | None = None,
    weights_col: str | None = None,

    plot: bool = True,
    plot_target: Union[str, List[str], None] = None,
    max_plots: int = 10,
    sort_by: str = "iv",
    ascending: bool = False,
    dpi: int = 300,

    n_jobs: int = -1,
    batch_size: int = 100
) -> Tuple[MarsEvaluationReport, MarsBinEvaluator]:
    """
    自动化特征分箱与效能评估管线。

    该函数封装了底层特征分箱、指标计算与趋势可视化的全流程逻辑。支持在给定的时间
    或群体切片下，快速评估特征的风险区分度、逻辑单调性以及跨期分布稳定性。

    系统原生支持多目标评估模式：在传入多个目标变量时，底层引擎将严格基于主目标
    （Primary Target）训练分箱边界规则，并将该规则直接映射至全部副目标
    （Secondary Targets）执行同步统计与报表汇总，以保障不同目标定义下的基准一致性。

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        待评估的数据集。
    target : str or list of str, optional
        目标变量列名。支持单目标和多目标模式；在多目标模式下，首个目标会被用作
        主目标来训练分箱边界。若为 ``None``，将启用无标签模式，仅输出分布类指标。
    features : list of str, optional
        需要评估的特征候选集合。默认为自动推断的全部候选特征。
    feature_data_source : dict of str to list of str, optional
        特征到数据源标签的映射，用于报告展示或分源分析。
    profile_by : str, optional
        稳定性分析的分组维度，可为业务分群列或时间粒度指令。
    dt_col : str, optional
        配合 ``profile_by`` 进行时间聚合的日期列名。
    feature_start_aware_baseline : bool, default False
        是否按特征首次出现的分组切片作为基准，计算稳定性指标。
    binning_type : {'native', 'opt'}, default 'native'
        底层分箱器类型。
    n_bins : int, default 10
        分箱的目标最大区间数。
    min_bin_size : float, default 0.02
        单一分箱的最小样本占比约束。
    monotonic_trend : str, default 'auto_asc_desc'
        最优分箱时使用的单调性约束方向。
    special_values : list of Any, optional
        需要单独隔离的特殊值集合。
    missing_values : list of Any, optional
        需要额外识别为缺失的值集合。
    binner_kwargs : dict, optional
        透传到底层分箱器的额外构造参数。
    benchmark_df : pl.DataFrame or pd.DataFrame, optional
        群体稳定性计算的基准数据集。
    weights_col : str, optional
        样本权重列名。
    plot : bool, default True
        是否在评估完成后自动绘制风险趋势图。
    plot_target : str or list of str, optional
        多目标模式下需要绘图的目标名称集合。默认为所有目标。
    max_plots : int, default 10
        自动绘图时展示的特征数量上限。
    sort_by : str, default 'iv'
        自动绘图时的特征排序依据。
    ascending : bool, default False
        自动绘图时是否按 ``sort_by`` 升序排列。
    dpi : int, default 300
        绘图输出分辨率。
    n_jobs : int, default -1
        并行计算使用的核心数限制。
    batch_size : int, default 100
        底层评估引擎处理特征切片时的批处理大小。

    Returns
    -------
    tuple of (MarsEvaluationReport, MarsBinEvaluator)
        评估报告对象与对应的评估器实例。

    Notes
    -----
    当系统检测到 `target` 为空时，将自动关闭包含决策树与数学规划在内的有监督底层
    算法，强制回退至基于等频策略的无监督原生分箱模式，以保障数据集探查的连续性。
    在多目标评估场景中，鉴于主副目标的评价基准差异，报告对象内的趋势宽表（Trend Tables）
    将默认仅保留对主目标维度的时序分析结果。

    Raises
    ------
    ValueError
        当底层评估流程校验失败时，由 ``MarsBinEvaluator`` 继续向上抛出。
    """

    input_is_pandas = isinstance(df, pd.DataFrame)

    # 兼容 target 为空的无标签模式。
    if target is None or target == []:
        target_list = ["dummy_target"]
        primary_target = "dummy_target"
        is_multi_target = False

        # 无标签模式下无法运行 CART/OptBinning，统一降级为无监督等频分箱。
        if binning_type == "opt" or (binner_kwargs and binner_kwargs.get("method") == "cart"):
            logger.warning("No target provided. Forcing `binning_type='native'` and `method='quantile'`.")
            binning_type = "native"
            if binner_kwargs is None:
                binner_kwargs = {}
            binner_kwargs["method"] = "quantile"
    else:
        target_list = [target] if isinstance(target, str) else target
        primary_target = target_list[0]
        is_multi_target = len(target_list) > 1

    primary_target = target_list[0]
    is_multi_target = len(target_list) > 1

    fit_params = {
        "n_bins": n_bins,
        "min_bin_size": min_bin_size,
        "monotonic_trend": monotonic_trend,
        "special_values": special_values,
        "missing_values": missing_values,
        "n_jobs": n_jobs
    }
    if binner_kwargs:
        fit_params.update(binner_kwargs)

    # 拟合 main binner
    primary_evaluator = MarsBinEvaluator(
        target=primary_target,
        binning_type=binning_type,
        feature_data_source=feature_data_source,
        **fit_params
    )

    primary_report = primary_evaluator.evaluate(
        df=df,
        features=features,
        profile_by=profile_by,
        dt_col=dt_col,
        feature_start_aware_baseline=feature_start_aware_baseline,
        feature_data_source=feature_data_source,
        benchmark_df=benchmark_df,
        weights_col=weights_col,
        batch_size=batch_size
    )

    # 如果只有单目标，无需合并，直接准备绘图
    if not is_multi_target:
        # 统一使用最后的绘图逻辑
        final_report = primary_report
        # 这里的 target_list 就是 [primary_target]

    else:

        # 获取已经训练好的分箱器 (Trained Binner)
        trained_binner = primary_evaluator.binner

        def to_pl(d: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
            """将中间结果统一转换为 Polars DataFrame。"""
            return pl.from_pandas(d) if isinstance(d, pd.DataFrame) else d

        # 处理主目标的表：添加 target 列以区分来源
        p_summary = to_pl(primary_report.summary_table).with_columns(pl.lit(primary_target).alias("target"))
        p_detail = to_pl(primary_report.detail_table)

        all_details: List[pl.DataFrame] = [p_detail]
        all_summaries: List[pl.DataFrame] = [p_summary]

        # 循环评估其余 Target
        for sec_target in target_list[1:]:

            # 实例化一个新的 Evaluator，但传入**已训练好的 Binner**, 确保分箱规则复用
            sec_evaluator = MarsBinEvaluator(
                target=sec_target,
                feature_data_source=feature_data_source,
                binner=trained_binner # 复用分箱规则
            )

            sec_report = sec_evaluator.evaluate(
                df=df,
                features=features,
                profile_by=profile_by,
                dt_col=dt_col,
                feature_start_aware_baseline=feature_start_aware_baseline,
                feature_data_source=feature_data_source,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                batch_size=batch_size
            )

            # 同样转换并标记副目标的表
            s_detail = to_pl(sec_report.detail_table)
            s_summary = to_pl(sec_report.summary_table).with_columns(pl.lit(sec_target).alias("target"))

            all_details.append(s_detail)
            all_summaries.append(s_summary)

        # 纵向合并所有 Detail 和 Summary 表
        final_detail = pl.concat(all_details, how="vertical_relaxed")
        final_summary = pl.concat(all_summaries, how="vertical_relaxed")

        # 如果原始输入是 Pandas，则保持最终报告的 Pandas 返回风格。
        if input_is_pandas:
            final_detail = final_detail.to_pandas()
            final_summary = final_summary.to_pandas()

        logger.info("`trend_tables` in the merged report contains primary-target data only.")
        merged_meta = dict(primary_report.report_meta or {})
        meta_df = primary_evaluator._ensure_polars_dataframe(df)
        merged_meta["targets"] = [str(t) for t in target_list]
        event_rate_map = {}
        for target_name in target_list:
            if target_name in meta_df.columns:
                try:
                    event_rate_map[str(target_name)] = float(meta_df.select(pl.col(target_name).cast(pl.Float64).mean()).item())
                except Exception:
                    event_rate_map[str(target_name)] = None
        merged_meta["event_rate_by_target"] = event_rate_map

        final_report = MarsEvaluationReport(
            summary_table=final_summary,
            trend_tables=primary_report.trend_tables,
            detail_table=final_detail,
            group_col=primary_report.group_col,
            feature_data_source=primary_report.feature_data_source,
            dt_col=primary_report.dt_col,
            missing_by_day_table=primary_report.missing_by_day_table,
            report_meta=merged_meta,
        )

    if plot:
        # 解析需要绘图的 Target 列表
        targets_to_plot = []
        if plot_target is None or plot_target == "all":
            targets_to_plot = target_list
        elif plot_target == "primary":
            targets_to_plot = [primary_target]
        elif isinstance(plot_target, str):
            targets_to_plot = [plot_target]
        elif isinstance(plot_target, list):
            targets_to_plot = plot_target

        # 过滤无效 Target
        targets_to_plot = [t for t in targets_to_plot if t in target_list]

        if not targets_to_plot:
            logger.warning("No valid targets specified for plotting.")
        else:
            _plot_report_helper(
                evaluator=primary_evaluator, # 复用这个实例的 plot 方法即可
                report=final_report,
                target_list=targets_to_plot,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi
            )

    return final_report, primary_evaluator

def _plot_report_helper(
    evaluator: MarsBinEvaluator,
    report: MarsEvaluationReport,
    target_list: List[str],
    sort_by: str,
    ascending: bool,
    max_plots: int,
    dpi: int
) -> None:
    """
    辅助绘图函数，处理多 Target 循环与 Top-N 筛选逻辑。

    参数
    ----
    evaluator : MarsBinEvaluator
        用于调用底层 plot_feature_binning_risk_trends 方法的实例。
    report : MarsEvaluationReport
        包含汇总数据的报告对象。
    target_list : List[str]
        需要绘制的目标变量列表。
    """
    # 准备 Summary 表用于排序
    summary_all = evaluator._ensure_polars_dataframe(report.summary_table)

    # 准备 Detail 表用于绘图数据源
    detail_all = evaluator._ensure_polars_dataframe(report.detail_table)

    # 映射排序简码到真实列名
    sort_map = {
        "iv": "iv", "psi": "psi_max", "ks": "ks",
        "auc": "auc", "rc": "rc_min", "mono": "mono"
    }
    sort_key = sort_map.get(sort_by.lower(), "iv")

    for current_target in target_list:
        logger.info(f"Plotting target '{current_target}'.")

        # 筛选当前 Target 的 Summary 数据（用于排序）
        # 注意：不同 Target 下特征的 IV/AUC 是不一样的，所以 Top 特征可能不同
        # 单目标模式下 summary 可能没有 target 列，需要兼容
        if "target" in summary_all.columns:
            curr_summary = summary_all.filter(pl.col("target") == current_target)
        else:
            curr_summary = summary_all

        plot_features = None
        if sort_key in curr_summary.columns:
            sorted_feats = curr_summary.sort(sort_key, descending=not ascending)["feature"].to_list()

            if len(sorted_feats) > max_plots:
                logger.info(f"Selecting top {max_plots} features by '{sort_key}' for plotting.")
                plot_features = sorted_feats[:max_plots]
            else:
                plot_features = sorted_feats

        # 筛选当前 Target 的 Detail 数据（用于绘图）
        # 利用 Evaluator 的 plot 方法支持传入 df_detail 的特性
        # 注意这里的 filter("y") ，在 Evaluator._format_report 里加了这一列
        curr_detail = detail_all.filter(pl.col("y") == current_target)

        if curr_detail.is_empty():
            logger.warning(f"No detail data found for target '{current_target}'. Skipping plotting.")
            continue

        # 调用底层绘图
        # 注意：这里我们手动传入 df_detail，从而绕过 report.detail_table，那是全量的
        evaluator.plot_feature_binning_risk_trends(
            report=None, # 不传 report，使用 df_detail
            df_detail=curr_detail, # 传筛选后的 detail
            features=plot_features,
            group_col=report.group_col,
            target_name=current_target, # 标题显示当前 Target 名
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi
        )

        # 分隔线，方便区分不同 Target 的图
        if len(target_list) > 1:
            logger.info(f"{'-'*40}")
