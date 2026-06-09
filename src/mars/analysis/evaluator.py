"""MARS 特征分箱评估模块。"""

import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.analysis.report import MarsEvaluationReport
from mars.core.base import MarsBaseEstimator
from mars.feature.base import MarsBinnerBase
from mars.feature.lite_opt_binner import MarsLiteOptBinner
from mars.feature.native_binner import MarsNativeBinner
from mars.feature.optimal_binner import MarsOptimalBinner
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
    report : MarsEvaluationReport
        风险评估报告，包含汇总表、明细表、趋势表和导出方法。
    binner : MarsBinnerBase
        本次评估拟合或显式传入的分箱器。
    targets : list of str
        本次画像覆盖的目标列列表；无标签画像时为空列表。
    metadata : dict
        本次运行的列名、特征范围、分箱配置和其他上下文信息。
    """

    report: MarsEvaluationReport
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
    ) -> None:
        """
        初始化分箱评估器。

        Parameters
        ----------
        binning_type : Literal['native', 'optimal', 'lite_opt']
            未显式传入分箱器时使用的默认分箱策略。
        binner_params : Dict[str, Any] | None
            构造默认分箱器时使用的参数。

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
        feature_start_aware_baseline: bool = False,
        psi_include_missing: bool = False,
        psi_include_special: bool = False,
        benchmark_df: Union[pl.DataFrame, pd.DataFrame, None] = None,
        weights_col: str | None = None,
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
        feature_start_aware_baseline : bool
            是否按特征首次出现的分组选择 PSI 基准，适合特征上线时间不一致的场景。
        psi_include_missing : bool
            计算 PSI 时是否单独保留缺失值分布。
        psi_include_special : bool
            计算 PSI 时是否单独保留特殊值分布。
        benchmark_df : Union[pl.DataFrame, pd.DataFrame, None]
            外部 benchmark 样本；传入后分布稳定性可与该样本进行对比。
        weights_col : str | None
            样本权重列名。
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
        prev_target = self.target
        prev_binner = self.binner
        prev_has_target = self.has_target_
        original_target = target
        effective_target = target if target else "dummy_target"
        profile_by = self._resolve_profile_by(
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
        )
        dt_col = time_col
        self.target = target

        # 允许 target 为空，或数据集中本就不存在目标列。
        self.has_target_ = self.target is not None and self.target in working_df.columns

        if not self.has_target_:
            logger.info(
                f"Label-free mode enabled: target '{target}' was not found. "
                "A dummy target will be injected and only distribution metrics plus PSI will be evaluated."
            )
            # 临时注入常量标签，保持下游统计链路可复用。
            working_df = working_df.with_columns(pl.lit(0).cast(pl.Int32).alias(effective_target))
        self.target = effective_target

        # 检查 Target 有效性，并把未到表现期的空值保留为 null。
        if self.has_target_:
            working_df = self._normalize_binary_target_column(working_df, self.target)
            n_unique = (
                working_df
                .filter(pl.col(self.target).is_not_null())
                .select(pl.col(self.target).n_unique())
                .item()
            )
            if n_unique < 2:
                raise ValueError(
                    f"Target column '{self.target}' must have at least 2 observed classes "
                    "after excluding null / NaN values."
                )

        working_df, group_col = self._prepare_context(working_df, profile_by, dt_col)

        # 自动识别特征列
        # 排除 target, weights, 和刚刚生成的统一 mars_group 列
        exclude_cols = {self.target, group_col}
        if weights_col:
            exclude_cols.add(weights_col)

        target_features = features if features else [
            c for c in working_df.columns if c not in exclude_cols
        ]

        effective_feature_data_source = feature_data_source if feature_data_source is not None else {}
        feature_source_map = self._normalize_feature_data_source(effective_feature_data_source, target_features)

        if binner is not None and self.binner_params:
            raise ValueError("`binner` and evaluator-level `binner_params` cannot be provided together.")

        active_binner = binner
        if active_binner is None:
            fit_kwargs = dict(self.binner_params)

            binner_factory = {
                "native": MarsNativeBinner,
                "optimal": MarsOptimalBinner,
                "lite_opt": MarsLiteOptBinner,
            }

            # 确定分箱器类型
            binner_cls = binner_factory[self.binning_type]

            # 获取目标类的构造函数签名
            # inspect.signature 会分析 __init__(self, n_bins, min_bin_size, ...) 到底有哪些参数
            sig = inspect.signature(binner_cls.__init__)
            valid_keys = set(sig.parameters.keys())

            # 过滤参数：只保留目标类支持的参数
            # 排除 'self' 和 'features' (因为 features 我们是显式传递的)
            valid_keys.discard("self")
            valid_keys.discard("features")
            valid_keys.discard("cat_features")

            clean_kwargs = {k: v for k, v in fit_kwargs.items() if k in valid_keys}

            # 记录被丢弃的参数，方便调试
            ignored_keys = set(fit_kwargs.keys()) - set(clean_kwargs.keys())
            if ignored_keys:
                logger.debug(f"Auto-cleaned kwargs for {binner_cls.__name__}. Ignored: {ignored_keys}")

            logger.info(f"Auto-fitting {binner_cls.__name__} internally with params: {clean_kwargs}.")

            # 实例化并拟合分箱器
            if not self.has_target_:
                if binner_cls in {MarsOptimalBinner, MarsLiteOptBinner}:
                    logger.warning("No target provided. Falling back to native quantile binning.")
                    binner_cls = MarsNativeBinner
                    clean_kwargs["method"] = "quantile"
                elif clean_kwargs.get("method") == "cart":
                    logger.warning("No target provided. Forcing native method='quantile'.")
                    clean_kwargs["method"] = "quantile"

            active_binner = binner_cls(**clean_kwargs)
            fit_df = working_df
            y_series = None

            # 有监督分箱只能使用已表现样本；无监督分箱保留全量样本拟合切点，
            # 后续 WOE 和风险指标会基于 observed_count 重新计算。
            if self.has_target_:
                is_supervised_binner = (
                    binner_cls is MarsOptimalBinner
                    or binner_cls is MarsLiteOptBinner
                    or clean_kwargs.get("method") == "cart"
                )
                if is_supervised_binner:
                    fit_df = working_df.filter(pl.col(self.target).is_not_null())
                    y_series = fit_df.get_column(self.target)
            active_binner.fit(fit_df, y_series, features=target_features)

        self.binner = active_binner

        # 将原始特征映射为分箱索引列 `{feature}_bin`。
        logger.debug("Transforming features to bin indices.")
        df_binned = active_binner.transform(working_df, return_type="index")
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
                pl.col("observed_count").sum(),
                pl.col("bad").sum(),
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
                        pl.col("observed_count").sum().alias("observed_count"),
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
        targets = [str(original_target)] if self.has_target_ and original_target else []

        # 无标签模式下擦除依赖真实坏样本标签的指标，保留分布类结果。
        if not self.has_target_:
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
        self.target = prev_target
        self.binner = prev_binner
        self.has_target_ = prev_has_target
        return run

    @staticmethod
    def _normalize_binary_target_column(df: pl.DataFrame, target: str) -> pl.DataFrame:
        """
        校验并归一化二分类 target 列。

        风控样本经常存在尚未到表现期的最新数据，这类样本应以 ``null`` 或 ``NaN``
        表达。该方法只接受 ``0``、``1``、``True``、``False`` 和空值；其他非空值
        一律视为上游清洗问题。

        Parameters
        ----------
        df : pl.DataFrame
            待校验的数据集。
        target : str
            目标变量列名。

        Returns
        -------
        pl.DataFrame
            target 已归一为 ``Int8`` 且保留空值的数据集。

        Raises
        ------
        ValueError
            当 target 包含非 ``0/1/True/False/null`` 的非空值时抛出。
        """
        dtype = df.schema[target]

        if dtype in {pl.Float32, pl.Float64}:
            df = df.with_columns(pl.col(target).fill_nan(None).alias(target))
            dtype = df.schema[target]

        if dtype == pl.Boolean:
            return df.with_columns(pl.col(target).cast(pl.Int8).alias(target))

        if dtype.is_numeric():
            invalid_values = (
                df
                .filter(pl.col(target).is_not_null() & ~pl.col(target).is_in([0, 1]))
                .select(pl.col(target).unique().head(5))
                .to_series()
                .to_list()
            )
            if invalid_values:
                raise ValueError(
                    f"Target column '{target}' contains invalid values {invalid_values}. "
                    "Please clean it to 0/1/True/False/null before evaluation."
                )
            return df.with_columns(pl.col(target).cast(pl.Int8).alias(target))

        invalid_values = (
            df
            .filter(pl.col(target).is_not_null())
            .select(pl.col(target).unique().head(5))
            .to_series()
            .to_list()
        )
        if invalid_values:
            raise ValueError(
                f"Target column '{target}' contains invalid values {invalid_values}. "
                "Please clean it to 0/1/True/False/null before evaluation."
            )
        return df.with_columns(pl.lit(None).cast(pl.Int8).alias(target))

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
        weights_col : str | None
            权重列。
        batch_size : int
            每次聚合处理的特征数量。

        Returns
        -------
        pl.DataFrame
            长表格式的统计汇总表，包含 [group_col, feature, bin_index, count, observed_count, bad]。
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

        # 预定义聚合表达式，count 保留全量分布，observed_count 只记录已表现样本。
        expr_count = pl.col(weights_col).sum() if weights_col else pl.len()
        if weights_col:
            expr_observed_count = (
                pl.when(pl.col(y_col).is_not_null())
                .then(pl.col(weights_col))
                .otherwise(0)
                .sum()
            )
            expr_bad = (
                (pl.col(y_col).fill_null(0).cast(pl.Float64) * pl.col(weights_col).cast(pl.Float64))
                .sum()
            )
        else:
            expr_observed_count = pl.col(y_col).is_not_null().sum()
            expr_bad = pl.col(y_col).fill_null(0).cast(pl.Float64).sum()

        agg_exprs = [
            expr_count.alias("count"),
            expr_observed_count.alias("observed_count"),
            expr_bad.alias("bad"),
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
            Map 阶段产出的统计长表。必须包含以下列：
            `['feature', 'bin_index', 'count', 'observed_count', 'bad']`。

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
                    pl.col("observed_count").sum().alias("bin_observed"),
                ])
                .with_columns([
                    (pl.col("bin_observed") - pl.col("bin_bad")).alias("bin_good")
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

        该方法负责计算 PSI 公式中的期望分布项 E，并支持两种基准策略，
        自动根据 `bench_df` 是否传入进行切换。

        Parameters
        ----------
        group_stats_raw : pl.DataFrame
            当前数据集的统计长表 (Actual Data Stats)。
            仅在 `bench_df` 为 None 时使用，用于提取时间最早的分组作为基准。
        bench_df : pl.DataFrame | None
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
            必须包含列：`[group_col, 'feature', 'bin_index', 'count', 'observed_count', 'bad']`。
        expected_dist : pl.DataFrame
            PSI 基准分布表。
            必须包含列：`['feature', 'bin_index', 'expected_dist']`。
        group_col : str
            分组维度列名 (如 'month')。
            计算累积指标 (KS/AUC) 时，会以此列和 'feature' 作为窗口分区 (Partition)。
        include_missing : bool
            计算 PSI 时是否保留缺失值箱。
        include_special : bool
            计算 PSI 时是否保留特殊值箱。

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

        if "observed_count" not in stats_df.columns:
            stats_df = stats_df.with_columns(pl.col("count").alias("observed_count"))

        # 合并统计量、基准分布与 WOE
        base_df = (
            stats_df
            .join(expected_dist, on=["feature", "bin_index"], how="left")
            .join(woe_df, on=["feature", "bin_index"], how="left")
            .with_columns([
                (pl.col("observed_count") - pl.col("bad")).alias("good"),
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

        # 计算双套分布：count 负责全量分布，observed_count 负责监督指标。
        base_df = base_df.with_columns([
            pl.col("count").sum().over([group_col, "feature"]).alias("total_count"),
            pl.col("observed_count").sum().over([group_col, "feature"]).alias("total_observed"),
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
            pl.when(pl.col("observed_count") > 0)
            .then(pl.col("bad") / pl.col("observed_count"))
            .otherwise(None)
            .alias("bad_rate"),

            # PSI 概率基准
            # 计算归一化后的 Actual% (只针对有效箱)
            (pl.col("count") / (pl.col("total_count_psi") + epsilon)).alias("act_prob_clean"),
            # 计算归一化后的 Expected% (只针对有效箱)
            #    例如：如果剔除缺失值后，剩余 expected_dist 之和为 0.8，则每一项除以 0.8 放大
            (pl.col("expected_dist") / (pl.col("total_expected_dist_psi") + epsilon)).alias("exp_prob_clean")
        ])

        # 计算 PSI 分箱贡献
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

            # 计算 Lift
            pl.when(pl.col("total_observed") > 0)
            .then(
                pl.col("bad_rate")
                /
                ((pl.col("total_bad") + epsilon) / (pl.col("total_observed") + epsilon))
            )
            .otherwise(None)
            .alias("lift"),

            # IV
            pl.when(pl.col("total_observed") > 0)
            .then(
                (
                    (pl.col("bad_dist") - pl.col("good_dist"))
                    *
                    ((pl.col("bad_dist") + epsilon) / (pl.col("good_dist") + epsilon)).log()
                ).cast(pl.Float32)
            )
            .otherwise(None)
            .alias("iv_bin")
        ])

        # 计算有序指标 (AUC, KS, IV)：必须按 WOE 风险程度排序
        sorted_df = base_df.sort([group_col, "feature", "woe"])

        # 累积分布用于计算 KS 和 AUC
        sorted_df = sorted_df.with_columns([
            pl.col("bad_dist").cum_sum().over([group_col, "feature"]).alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over([group_col, "feature"]).alias("cum_good_dist"),
        ])

        sorted_df = sorted_df.with_columns([

            pl.when(pl.col("total_observed") > 0)
            .then((pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs() * 100)
            .otherwise(None)
            .alias("ks_bin"),

            # AUC 梯形法则计算面积
            pl.when(pl.col("total_observed") > 0)
            .then(
                (pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1, fill_value=0).over([group_col, "feature"]))
                *
                (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1, fill_value=0).over([group_col, "feature"]))
                / 2
            )
            .otherwise(None)
            .alias("auc_bin")
        ])

        sorted_df = sorted_df.with_columns([
            pl.when(pl.col("psi_bin").abs() < 1e-12)
              .then(0.0)
              .otherwise(pl.col("psi_bin"))
              .alias("psi_bin")
        ])

        return sorted_df

    @staticmethod
    def _resolve_profile_by(
        *,
        group_col: str | None,
        time_col: str | None,
        time_grain: str | None,
    ) -> str | None:
        """把新的分组/时间参数映射到内部趋势维度。"""
        if group_col:
            return group_col
        if time_col:
            return time_grain or "month"
        return None

    def _prepare_context(self,
                         df: pl.DataFrame,
                         profile_by: str | None,
                         dt_col: str | None
                         ) -> Tuple[pl.DataFrame, str]:
        """
        构造内部趋势分组上下文。

        Parameters
        ----------
        df : pl.DataFrame
            输入样本表。
        profile_by : str | None
            由公开参数 ``group_col`` 或 ``time_grain`` 解析出的内部分组指令。
        dt_col : str | None
            由公开参数 ``time_col`` 传入的原始日期列名。

        Returns
        -------
        tuple of (polars.DataFrame, str)
            包含派生分组列的样本表，以及最终使用的内部分组列名。
        """
        # 有时间没分组 -> 默认按月
        if dt_col and not profile_by:
            logger.info("`dt_col` was provided without `profile_by`; defaulting trend grouping to 'month'.")
            profile_by = "month"

        # 统一识别日、周、月粒度，避免 1m 被误当成普通分组列。
        is_date_granularity = MarsDate.is_time_grain(profile_by)

        # 处理时间切片
        if dt_col and is_date_granularity:
            date_expr = MarsDate.from_grain(dt_col, profile_by).alias(self.MARS_GROUP_COL)
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
        """
        将外部数据源到特征列表的映射标准化为特征到数据源的字典。

        未显式映射的特征会标记为 ``"UNMAPPED"``；若外部映射包含当前评估
        特征集合之外的字段，则立即抛出异常，避免报告中出现不可追溯来源。
        """
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
        """构建按日缺失率趋势表，失败时降级为不输出该附表。"""
        if not dt_col or dt_col not in df.columns:
            return None

        try:
            from mars.analysis.profiler import profile_stats

            missing_values = getattr(self.binner, "missing_values", None)
            if missing_values is None:
                missing_values = self.binner_params.get("missing_values")
            special_values = getattr(self.binner, "special_values", None)
            if special_values is None:
                special_values = self.binner_params.get("special_values")

            report = profile_stats(
                df,
                metrics=["missing"],
                features=features,
                time_col=dt_col,
                time_grain="day",
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
        """判断分组配置是否表示内置时间粒度。"""
        return MarsDate.is_time_grain(profile_by)

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
        retained_default = default_expected_dist.filter(~pl.col("feature").is_in(override_features))
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
                    monitor_observed_expr = (
                        pl.when(pl.col(self.target).is_not_null())
                        .then(pl.col(weights_col).cast(pl.Float64))
                        .otherwise(0)
                        .sum()
                        .alias("observed_count")
                    )
                    monitor_bad_expr = (
                        pl.col(self.target).fill_null(0).cast(pl.Float64)
                        * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("bad")
                else:
                    monitor_observed_expr = pl.col(self.target).is_not_null().sum().alias("observed_count")
                    monitor_bad_expr = pl.col(self.target).fill_null(0).cast(pl.Float64).sum().alias("bad")
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
                        pl.col(self.target).fill_null(0).cast(pl.Float64)
                        * pl.col(weights_col).cast(pl.Float64)
                    ).sum().alias("base_bad")
                    total_expr = (
                        pl.when(pl.col(self.target).is_not_null())
                        .then(pl.col(weights_col).cast(pl.Float64))
                        .otherwise(0)
                        .sum()
                        .alias("base_total")
                    )
                else:
                    bad_expr = pl.col(self.target).fill_null(0).cast(pl.Float64).sum().alias("base_bad")
                    total_expr = pl.col(self.target).is_not_null().sum().cast(pl.Float64).alias("base_total")

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

    def _build_bin_label_map(self, stats_long: pl.DataFrame) -> pl.DataFrame:
        """构建明细报告使用的特征和分箱索引到标签的映射。"""
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
        feature_source_map : Dict[str, str] | None
            特征到来源分组的映射，用于在汇总报告中保留来源字段。
        dt_col : str | None
            原始日期列名，用于生成按日缺失率附表。
        missing_by_day_table : Union[pl.DataFrame, pd.DataFrame] | None
            已计算好的按日缺失率附表。
        risk_corr_baseline_df : pl.DataFrame | None
            RiskCorr 基准分布表。
        feature_valid_groups_df : pl.DataFrame | None
            特征有效分组数量表。
        monitor_metrics_groups : pl.DataFrame | None
            分组粒度监控指标表。
        monitor_metrics_total : pl.DataFrame | None
            全量粒度监控指标表。

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
        from mars.feature.base import MarsBinnerBase

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
            # 累积样本数
            pl.col("count").cum_sum().over(["feature", group_col]).alias("cum_count"),
            # 累积已表现样本数
            pl.col("observed_count").cum_sum().over(["feature", group_col]).alias("cum_observed_count"),
            # 累积坏样本数
            pl.col("bad").cum_sum().over(["feature", group_col]).alias("cum_bad"),
            # 累积好样本数
            (pl.col("observed_count") - pl.col("bad")).cum_sum().over(["feature", group_col]).alias("cum_good")
        ]).with_columns([
            # 累积坏账率 = 累积坏 / 累积已表现样本
            pl.when(pl.col("cum_observed_count") > 0)
            .then(pl.col("cum_bad") / pl.col("cum_observed_count"))
            .otherwise(None)
            .alias("cum_bad_rate"),

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
                pl.when(pl.col("observed_count") > 0)
                .then(pl.col("bad") / pl.col("observed_count"))
                .otherwise(None)
                .alias("bad_rate"),

                # total 行代表全量样本，占比固定为 1.0。
                pl.lit(1.0).alias("pct"),

                # 累积列 (对于 total 行，累积值等于自身)
                pl.col("count").alias("cum_count"),
                pl.col("observed_count").alias("cum_observed_count"),
                pl.col("bad").alias("cum_bad"),
                pl.when(pl.col("observed_count") > 0)
                .then(pl.col("bad") / pl.col("observed_count"))
                .otherwise(None)
                .alias("cum_bad_rate"),

                # AUC 方向修正
                pl.when(pl.col("auc_bin") < 0.5)
                  .then(pl.lit(1) - pl.col("auc_bin"))
                  .otherwise(pl.col("auc_bin"))
                  .alias("auc_bin"),

                # 标识列与排序键 (确保排在最后)
                pl.lit(9999).cast(pl.Int16).alias("bin_index"),
                pl.lit("Total").alias("bin_label"),

                pl.lit("汇总组").alias("bin_type"),

                pl.lit(2).cast(pl.Int32).alias("_sort_group"), # 明确为 Int32，避免排序键类型漂移。
                pl.lit(0).cast(pl.Int32).alias("_sort_idx")    # 明确为 Int32，避免排序键类型漂移。
            ])
        )

        # total 行也要 join trend 列
        total_rows = total_rows.join(trend_shape_df, on="feature", how="left")

        targets = [
            "feature", group_col, "bin_index", "bin_label", "_sort_group", "_sort_idx",
            "count", "observed_count", "pct", "bad", "good", "bad_rate", "lift", "trend",
            "cum_count", "cum_observed_count", "cum_bad", "cum_bad_rate",
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
            "count", "observed_count", "bad", "good", "pct", "bad_rate", "lift",
            "cum_count", "cum_observed_count", "cum_bad", "cum_bad_rate",
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

        # 中间指标计算
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
                pl.when(pl.col("observed_count").sum() <= 0)
                  .then(pl.lit(None).cast(pl.Float64))
                  .when(pl.len() > 1)
                  .then(pl.corr("bad_rate", "base_br", method="pearson"))
                  .otherwise(pl.lit(1.0))
                  .fill_nan(1.0)
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
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("iv_bin").sum())
                .otherwise(None)
                .alias("iv"),
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("auc_bin").sum())
                .otherwise(None)
                .alias("auc"),
                (
                    pl.when(pl.col("bin_index") == MarsBinnerBase.IDX_MISSING)
                    .then(pl.col("count"))
                    .otherwise(0)
                    .sum()
                    /
                    (pl.col("count").sum() + 1e-9)
                ).alias("missing"),
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("lift").max())
                .otherwise(None)
                .alias("lift"),
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
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("lift").min())
                .otherwise(None)
                .alias("lift_min"),
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("lift").max())
                .otherwise(None)
                .alias("lift_max"),
            ])
        )

        # 第三阶段：汇总表
        total_metrics_agg = (
            metrics_total.group_by("feature")
            .agg([
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("iv_bin").sum())
                .otherwise(None)
                .alias("iv"),
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("ks_bin").max())
                .otherwise(None)
                .alias("ks"),
                pl.when(pl.col("observed_count").sum() > 0)
                .then(pl.col("auc_bin").sum())
                .otherwise(None)
                .alias("auc"),
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

        # 构建趋势表
        trend_tables = {}
        target_metrics = ["psi", "auc", "ks", "iv", "missing", "lift", "bad_rate", "risk_corr"]

        for metric in target_metrics:
            if metric == "risk_corr":
                pivot_src = risk_corr_long
            elif metric == "psi":
                # 单快照模式下分组侧已经可能是 Total，拼接前先剔除，避免 pivot 出现重复键。
                psi_group_src = monitor_group_level_metrics.filter(pl.col(group_col) != "Total")
                psi_total_src = (
                    monitoring_total
                    .group_by(["feature", group_col])
                    .agg(pl.col("psi_bin").sum().alias("psi"))
                )
                pivot_src = pl.concat(
                    [
                        psi_group_src.select(["feature", group_col, "psi"]),
                        psi_total_src.select(["feature", group_col, "psi"]),
                    ],
                    how="vertical_relaxed",
                )
            else:
                if metric == "bad_rate":
                    agg_func = (
                        pl.when(pl.col("observed_count").sum() > 0)
                        .then(pl.col("bad").sum() / pl.col("observed_count").sum())
                        .otherwise(None)
                    )
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
                    agg_func = (
                        pl.when(pl.col("observed_count").sum() > 0)
                        .then(pl.col("lift").max())
                        .otherwise(None)
                    )
                elif metric == "ks":
                    agg_func = (
                        pl.when(pl.col("observed_count").sum() > 0)
                        .then(pl.col(f"{metric}_bin").max())
                        .otherwise(None)
                    )
                else:
                    agg_func = (
                        pl.when(pl.col("observed_count").sum() > 0)
                        .then(pl.col(f"{metric}_bin").sum())
                        .otherwise(None)
                    )

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
        report : Optional['MarsEvaluationReport']
            由 ``evaluate`` 生成的评估报告对象。
        df_detail : Union[pl.DataFrame, pd.DataFrame, None]
            直接传入分箱明细表；当未提供 ``report`` 时使用。
        features : Union[str, List[str], None]
            需要绘图的特征名称。若为 ``None``，绘制明细表中的全部特征。
        group_col : str | None
            分组列名。当直接传入 ``df_detail`` 且无法自动推断时可显式指定。
        target_name : str | None
            图表标题中展示的目标名称。未提供时默认使用 ``self.target``。
        sort_by : str
            绘图特征的排序依据。
        ascending : bool
            是否按 ``sort_by`` 升序绘制。
        dpi : int
            图像分辨率。

        Raises
        ------
        ValueError
            当 ``report`` 和 ``df_detail`` 同时缺失时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> detail = pl.DataFrame(
        ...     {
        ...         "feature": ["age"],
        ...         "bin_index": [0],
        ...         "bin_label": ["[20, 40)"],
        ...         "month": ["2026-01"],
        ...         "count": [100],
        ...         "bad": [12],
        ...         "bad_rate": [0.12],
        ...         "lift": [1.0],
        ...         "iv_bin": [0.02],
        ...     }
        ... )
        >>> evaluator = MarsBinEvaluator()
        >>> evaluator.plot_feature_binning_risk_trends(
        ...     df_detail=detail,
        ...     features="age",
        ...     group_col="month",
        ... ) is None
        True
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
                known = {
                    "feature",
                    "bin_index",
                    "bin_label",
                    "count",
                    "observed_count",
                    "bad",
                    "bad_rate",
                    "lift",
                    "psi_bin",
                    "ks_bin",
                    "auc_bin",
                    "iv_bin",
                    "total_count",
                    "trend",
                    "y",
                }
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
    target: Union[str, List[str]] | None = None,  # 放宽约束，支持无标签画像和多目标画像。
    features: List[str] | None = None,
    feature_data_source: Dict[str, List[str]] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    feature_start_aware_baseline: bool = False,

    binning_type: Literal["native", "optimal", "lite_opt"] = "native",
    binner: MarsBinnerBase | None = None,
    binner_params: Dict[str, Any] | None = None,

    benchmark_df: Union[pl.DataFrame, pd.DataFrame] | None = None,
    weights_col: str | None = None,

    plot: bool = True,
    plot_target: Union[str, List[str], None] = None,
    max_plots: int = 10,
    sort_by: str = "iv",
    ascending: bool = False,
    dpi: int = 300,

    batch_size: int = 100
) -> MarsRiskProfile:
    """
    运行高层风险画像工作流。

    `profile_risk` 是面向风控分析场景的轻量入口：调用方传入 `df`、`target`、
    特征范围和分组上下文，函数内部负责构造评估器、拟合或复用分箱器，并返回
    `MarsRiskProfile`。底层分箱器仍然遵循 `X, y` 风格，高层入口只暴露
    `df, target`，避免同一个 public method 同时出现两套目标变量语义。

    多目标画像时，主目标先拟合分箱器，其他目标显式复用同一个分箱器进行评估，
    最终在返回结果的 `targets` 中记录完整目标列表。

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        待画像样本表。
    target : Union[str, List[str]] | None
        二分类目标列名或目标列列表；`None` 表示无标签画像。
    features : List[str] | None
        本次参与画像的特征列。
    feature_data_source : Dict[str, List[str]] | None
        特征来源映射，只保留方法级入口，因为它依赖本次 active features。
    group_col : str | None
        已存在的分组列名。
    time_col : str | None
        原始日期列名。
    time_grain : str | None
        时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
    feature_start_aware_baseline : bool
        是否按特征首次出现的分组选择 PSI 基准。
    binning_type : Literal['native', 'optimal', 'lite_opt']
        未显式传入 `binner` 时使用的分箱器类型。
    binner : MarsBinnerBase | None
        显式复用的分箱器；传入后不允许再传 `binner_params`。
    binner_params : Dict[str, Any] | None
        构造默认分箱器时使用的参数。
    benchmark_df : Union[pl.DataFrame, pd.DataFrame] | None
        外部 benchmark 样本。
    weights_col : str | None
        样本权重列名。
    plot : bool
        是否生成图表明细。
    plot_target : Union[str, List[str], None]
        指定需要绘图的目标列。
    max_plots : int
        最多绘制的特征数量。
    sort_by : str
        绘图特征排序指标。
    ascending : bool
        是否按 `sort_by` 升序排序。
    dpi : int
        图表分辨率。
    batch_size : int
        批量评估时的特征批大小。

    Returns
    -------
    MarsRiskProfile
        单次风险画像结果，包含报告、分箱器、目标列表和运行元数据。

    Raises
    ------
    ValueError
        当 `binner` 与 `binner_params` 同时传入或输入列配置不合法时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis.evaluator import profile_risk
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> profile = profile_risk(df, target="y", features=["age"], plot=False)
    >>> profile.report.summary_table is not None and profile.targets == ["y"]
    True
    """
    input_is_pandas = isinstance(df, pd.DataFrame)
    if binner is not None and binner_params:
        raise ValueError("`binner` and `binner_params` cannot be provided together.")

    effective_binner_params = dict(binner_params or {})
    if target is None or target == []:
        target_list: list[str] = []
        primary_target: str | None = None
        is_multi_target = False
        if binning_type in {"optimal", "lite_opt"} or effective_binner_params.get("method") == "cart":
            logger.warning("No target provided. Forcing `binning_type='native'` and `method='quantile'`.")
            binning_type = "native"
            effective_binner_params["method"] = "quantile"
    else:
        target_list = [target] if isinstance(target, str) else list(target)
        primary_target = target_list[0]
        is_multi_target = len(target_list) > 1

    primary_evaluator = MarsBinEvaluator(
        binning_type=binning_type,
        binner_params=effective_binner_params,
    )
    primary_run = primary_evaluator.evaluate(
        df=df,
        target=primary_target,
        features=features,
        binner=binner,
        feature_data_source=feature_data_source,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        feature_start_aware_baseline=feature_start_aware_baseline,
        benchmark_df=benchmark_df,
        weights_col=weights_col,
        batch_size=batch_size,
    )
    primary_report = primary_run.report
    trained_binner = primary_run.binner

    if not is_multi_target:
        final_report = primary_report
        final_targets = primary_run.targets
    else:

        def to_pl(d: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
            """将报告中间表统一转换为 Polars DataFrame。"""
            return pl.from_pandas(d) if isinstance(d, pd.DataFrame) else d

        p_summary = to_pl(primary_report.summary_table).with_columns(pl.lit(primary_target).alias("target"))
        p_detail = to_pl(primary_report.detail_table)
        all_details: List[pl.DataFrame] = [p_detail]
        all_summaries: List[pl.DataFrame] = [p_summary]

        for sec_target in target_list[1:]:
            sec_run = MarsBinEvaluator(binning_type=binning_type).evaluate(
                df=df,
                target=sec_target,
                features=features,
                binner=trained_binner,
                feature_data_source=feature_data_source,
                group_col=group_col,
                time_col=time_col,
                time_grain=time_grain,
                feature_start_aware_baseline=feature_start_aware_baseline,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                batch_size=batch_size,
            )
            all_details.append(to_pl(sec_run.report.detail_table))
            all_summaries.append(
                to_pl(sec_run.report.summary_table).with_columns(pl.lit(sec_target).alias("target"))
            )

        final_detail = pl.concat(all_details, how="vertical_relaxed")
        final_summary = pl.concat(all_summaries, how="vertical_relaxed")
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
                    event_rate_map[str(target_name)] = float(
                        meta_df.select(pl.col(target_name).cast(pl.Float64).mean()).item()
                    )
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
        final_targets = [str(t) for t in target_list]

    if plot and final_targets:
        targets_to_plot = []
        if plot_target is None or plot_target == "all":
            targets_to_plot = final_targets
        elif plot_target == "primary" and final_targets:
            targets_to_plot = [final_targets[0]]
        elif isinstance(plot_target, str):
            targets_to_plot = [plot_target]
        elif isinstance(plot_target, list):
            targets_to_plot = plot_target

        targets_to_plot = [t for t in targets_to_plot if t in final_targets]
        if not targets_to_plot:
            logger.warning("No valid targets specified for plotting.")
        else:
            _plot_report_helper(
                evaluator=primary_evaluator,
                report=final_report,
                target_list=targets_to_plot,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi,
            )

    return MarsRiskProfile(
        report=final_report,
        binner=trained_binner,
        targets=final_targets,
        metadata=dict(final_report.report_meta or {}),
    )

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
