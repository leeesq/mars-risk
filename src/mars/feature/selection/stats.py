"""MARS 特征筛选器实现模块。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Union

import pandas as pd
import polars as pl

from mars.compute import RiskCorrBaseline, normalize_risk_corr_baseline
from mars.feature.binning.base import MarsBinnerBase
from mars.feature.binning.native import MarsNativeBinner
from mars.feature.selection.base import MarsBaseSelector
from mars.reporting import MarsBinningReport
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsStatsSelector(MarsBaseSelector):
    """
    基于风控统计指标的特征筛选器。

    该筛选器将数据质量、IV/Lift、PSI、相关性和白黑名单规则串成一个漏斗式筛选流程。
    构造函数只保存阈值、分箱策略、缺失/特殊值配置以及运行资源参数；样本数据、
    目标列、特征范围、分组列和时间列都由 `fit` 传入。

    典型用法是先用粗分箱低成本压缩特征空间，再用精细分箱和稳定性规则做最终筛选。
    `white_list` 中的特征会尽量绕过自动剔除规则，`black_list` 中的特征会被强制排除。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> selector = MarsStatsSelector(skip_fine_scan=True, psi_thr=None, rc_thr=None)
    >>> selector.fit(df, target="y", features=["age"]).selected_features_
    ['age']
    """

    def __init__(
        self,
        *,
        missing_thr: float = 0.90,
        zeros_thr: float = 0.90,
        mode_thr: float = 0.90,

        iv_thr: float = 0.01,
        lift_thr: float | None = 1.2,
        min_sample_rate: float = 0.05,

        psi_thr: float | None = 0.25,
        rc_thr: float | None = 0.5,
        corr_thr: float | None = 0.95,
        feature_start_aware_reference: bool = False,
        risk_corr_baseline: RiskCorrBaseline = "total",
        psi_include_missing: bool = False,
        psi_include_special: bool = False,

        skip_rough_scan: bool = False,
        skip_fine_scan: bool = False,
        rough_iv_thr: float = 0.01,
        rough_lift_thr: float = 1.2,
        rough_min_sample_rate: float = 0.02,

        missing_values: List[Any] | None = None,
        special_values: List[Any] | None = None,

        binning_params: Dict[str, Any] | None = None,
        rough_binning_params: Dict[str, Any] | None = None,

        batch_size: int | None = 100,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化统计筛选策略。

        Parameters
        ----------
        missing_thr : float
            缺失率剔除阈值。
        zeros_thr : float
            零值率剔除阈值。
        mode_thr : float
            单一众数占比剔除阈值。
        iv_thr : float
            精筛阶段保留特征所需的最低 IV。
        lift_thr : float | None
            精筛阶段保留特征所需的最低 Lift。
        min_sample_rate : float
            计算 Lift 时单个分箱所需的最低样本占比。
        psi_thr : float | None
            稳定性筛选的 PSI 上限。
        rc_thr : float | None
            排名变化率筛选阈值。
        corr_thr : float | None
            WOE 相关性筛选阈值。
        feature_start_aware_reference : bool
            是否默认启用 feature-start aware reference，用于 PSI 基准重锚。
        risk_corr_baseline : RiskCorrBaseline
            RC 的默认基准选择方式。
        psi_include_missing : bool
            稳定性筛选和评估报告计算 PSI 时是否纳入缺失值箱。
        psi_include_special : bool
            稳定性筛选和评估报告计算 PSI 时是否纳入特殊值箱。
        skip_rough_scan : bool
            是否跳过粗筛分箱阶段。
        skip_fine_scan : bool
            是否跳过精筛分箱阶段。
        rough_iv_thr : float
            粗筛阶段保留特征所需的最低 IV。
        rough_lift_thr : float
            粗筛阶段保留特征所需的最低 Lift。
        rough_min_sample_rate : float
            粗筛阶段计算 Lift 时单个分箱所需的最低样本占比。
        missing_values : List[Any] | None
            需要视为缺失的取值列表。
        special_values : List[Any] | None
            需要单独处理的特殊值列表。
        binning_params : Dict[str, Any] | None
            精筛阶段分箱器参数。支持增量更新，传入的字典将与默认配置合并，未指定的参数将保留默认值。
            默认配置为：`{"prebinning_method": "cart", "n_bins": 10, "min_bin_size": 0.05}`。
        rough_binning_params : Dict[str, Any] | None
            粗筛阶段分箱器参数。支持增量更新，传入的字典将与默认配置合并，未指定的参数将保留默认值。
            默认配置为：`{"method": "quantile", "n_bins": 20, "min_bin_size": 0.02, "merge_small_bins": True}`。
        batch_size : int | None
            批量评估时的特征批大小。
        n_jobs : int
            并行任务数，含义遵循 joblib 约定。
        """
        super().__init__()

        self.features: list[str] | None = None
        self.feature_data_source: dict[str, list[str]] = {}
        self.group_col: str | None = None
        self.time_col: str | None = None
        self.time_grain: str | None = None
        self.profile_by: str | None = None
        self.feature_start_aware_reference = bool(feature_start_aware_reference)
        self.risk_corr_baseline = normalize_risk_corr_baseline(risk_corr_baseline)
        self.white_list: list[str] = []
        self.black_list: list[str] = []

        self.missing_values = missing_values if missing_values else []
        self.special_values = special_values if special_values else []

        self.missing_thr = missing_thr
        self.mode_thr = mode_thr
        self.zeros_thr = zeros_thr

        self.skip_rough_scan = skip_rough_scan
        default_rough_params = {
            "method": "quantile",
            "n_bins": 20,
            "min_bin_size": 0.02,
            "merge_small_bins": True
        }
        # 如果用户传了参数，就更新对应的值，否则保持默认
        if rough_binning_params:
            default_rough_params.update(rough_binning_params)
        self.rough_binning_params = default_rough_params
        self.rough_iv_thr = rough_iv_thr
        self.rough_lift_thr = rough_lift_thr
        self.rough_min_sample_rate = rough_min_sample_rate

        self.skip_fine_scan = skip_fine_scan
        default_fine_params = {
            "prebinning_method": "cart",
            "n_bins": 10,
            "min_bin_size": 0.05,
        }
        if binning_params:
            default_fine_params.update(binning_params)
        self.binning_params = default_fine_params
        self.iv_thr = iv_thr
        self.lift_thr = lift_thr
        self.min_sample_rate = min_sample_rate

        self.psi_thr = psi_thr
        self.rc_thr = rc_thr
        self.corr_thr = corr_thr
        self.psi_include_missing = psi_include_missing
        self.psi_include_special = psi_include_special

        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self._rough_binner: MarsNativeBinner | None = None
        self._stage3_binner: MarsBinnerBase | None = None
        self._feature_iv_dict: Dict[str, float] = {}
        self._feature_source_map: Dict[str, str] = {}
        self._stability_report: MarsBinningReport | None = None
        self._fit_used_benchmark = False
        self._benchmark_row_count: int | None = None
        self._binning_fit_source = "df"

        self._funnel_stats = []

    @time_it
    def fit(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        target: str,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
        features: list[str] | None = None,
        feature_data_source: dict[str, list[str]] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        white_list: list[str] | None = None,
        black_list: list[str] | None = None,
        feature_start_aware_reference: bool | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
    ) -> MarsStatsSelector:
        """
        在一次样本上下文中执行统计特征筛选。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待筛选样本表。
        target : str
            二分类目标列名。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            基准样本表。传入后用于拟合粗筛和精筛分箱规则，并提供 PSI expected
            distribution；质量、IV、Lift、RC 和相关性仍在 ``df`` 上计算。
        features : list[str] | None
            候选特征列；不传时会从样本表中自动推断。
        feature_data_source : dict[str, list[str]] | None
            特征来源映射，用于报告中追踪特征所属来源。
        group_col : str | None
            已存在的分组列名，用于趋势和稳定性筛选。
        time_col : str | None
            原始日期列名；与 `time_grain` 配合时生成时间分组。
        time_grain : str | None
            时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
        white_list : list[str] | None
            白名单特征，尽量绕过自动剔除规则。
        black_list : list[str] | None
            黑名单特征，会被强制剔除。
        feature_start_aware_reference : bool | None
            是否按特征首次出现分组选择 PSI 基准。传入 `None` 时沿用实例初始化时保存的默认值。
        risk_corr_baseline : RiskCorrBaseline | None
            本次筛选使用的 RC 基准；传入 `None` 时沿用实例初始化时保存的默认值。

        Returns
        -------
        MarsStatsSelector
            拟合后的筛选器，`selected_features_` 中保存最终特征列表。

        Raises
        ------
        ValueError
            当标签、稳定性分组、benchmark schema 或分箱结果不满足要求时抛出。

        Notes
        -----
        ``psi_thr`` 或 ``rc_thr`` 任一非 ``None`` 时，必须提供 ``group_col`` 或
        ``time_col``。传入 benchmark 后不会把其行合并进 ``df`` 的趋势分组或 Total。

        Examples
        --------
        >>> benchmark = pl.DataFrame(
        ...     {"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]}
        ... )
        >>> df = benchmark.with_columns(pl.lit("current").alias("period"))
        >>> selector = MarsStatsSelector(
        ...     skip_fine_scan=True,
        ...     rough_iv_thr=-1.0,
        ...     psi_thr=100.0,
        ...     rc_thr=None,
        ...     corr_thr=None,
        ... )
        >>> selector.fit(
        ...     df,
        ...     target="y",
        ...     benchmark_df=benchmark,
        ...     features=["age"],
        ...     group_col="period",
        ... ).selected_features_
        ['age']
        """
        # 拦截互斥的配置项
        if self.skip_rough_scan and self.skip_fine_scan:
            raise ValueError("Cannot skip both rough scan and fine scan. At least one binning stage is required.")

        self.target = target
        self.features = features
        self.feature_data_source = feature_data_source or {}
        self.group_col = group_col
        self.time_col = time_col
        self.time_grain = (time_grain or "month") if time_col and not group_col else time_grain
        self.profile_by = group_col or ((time_grain or "month") if time_col else None)
        self.white_list = white_list if white_list else []
        self.black_list = black_list if black_list else []
        self.feature_start_aware_reference = (
            self.feature_start_aware_reference
            if feature_start_aware_reference is None
            else bool(feature_start_aware_reference)
        )
        self.risk_corr_baseline = normalize_risk_corr_baseline(
            risk_corr_baseline or self.risk_corr_baseline,
        )

        X = self._ensure_polars_dataframe(df)
        X = self._prepare_evaluation_frame(X)
        self._validate_group_context(X)
        self._funnel_stats = []
        self._feature_iv_dict = {}
        self._stability_report = None
        self.report_records_ = []
        self._initial_feature_count = 0

        exclude_cols = {self.target}
        if self.time_col:
            exclude_cols.add(self.time_col)
        if self.group_col:
            exclude_cols.add(self.group_col)

        source_features = self.features if self.features else X.columns
        candidate_features = [
            col for col in source_features if col in X.columns and col not in exclude_cols
        ]
        self._feature_source_map = self._normalize_feature_data_source(candidate_features)
        valid_white_list = [
            feature for feature in self.white_list if feature in candidate_features
        ]
        current_features = [
            feature for feature in candidate_features if feature not in self.black_list
        ]
        benchmark_pl = self._prepare_selector_benchmark(
            benchmark_df,
            features=current_features,
        )
        self._fit_used_benchmark = benchmark_pl is not None
        self._benchmark_row_count = benchmark_pl.height if benchmark_pl is not None else None
        self._binning_fit_source = "benchmark_df" if benchmark_pl is not None else "df"
        if benchmark_pl is not None and self.feature_start_aware_reference:
            logger.warning(
                "`feature_start_aware_reference=True` was ignored because "
                "`benchmark_df` was provided."
            )
        self._record_funnel(
            "Init",
            "Blacklist & Exclusions",
            {"black_list_len": len(self.black_list)},
            len(candidate_features),
            len(current_features),
        )

        # 执行数据质量探查
        if current_features:
            prev_count = len(current_features)
            current_features = self._filter_quality(X, current_features)

            thr_msg = (
                f"miss < {self.missing_thr} & "
                f"zero < {self.zeros_thr} & "
                f"mode < {self.mode_thr}"
            )
            self._record_funnel(
                stage="Stage 1",
                description="Data Quality Check",
                thresholds=thr_msg,
                count_before=prev_count,
                count_after=len(current_features)
            )

        # 执行轻量级分布区间探查
        if not self.skip_rough_scan and current_features:
            prev_count = len(current_features)

            if not self.skip_fine_scan:
                # 动态路由分配：白名单特征旁路以缩减运算开销
                scan_features = [f for f in current_features if not self._should_bypass_filter(f)]
                white_features = [f for f in current_features if self._should_bypass_filter(f)]

                if scan_features:
                    scan_features = self._filter_rough(
                        X,
                        scan_features,
                        benchmark_df=benchmark_pl,
                    )

                current_features = scan_features + white_features
            else:
                # 探查流水线终点：白名单特征强制并入获取分箱实体
                current_features = self._filter_rough(
                    X,
                    current_features,
                    benchmark_df=benchmark_pl,
                )

            thr_msg = f"iv >= {self.rough_iv_thr} | (lift >= {self.rough_lift_thr} & sample >= {self.rough_min_sample_rate})"
            self._record_funnel("Stage 2", "Rough Scan (Native)", thr_msg, prev_count, len(current_features))

        # 执行精确分布区间与区分度评估
        if not self.skip_fine_scan and current_features:
            prev_count = len(current_features)
            current_features = self._filter_fine(
                X,
                current_features,
                benchmark_df=benchmark_pl,
            )
            thr_msg = f"iv >= {self.iv_thr} | (lift >= {self.lift_thr} & sample >= {self.min_sample_rate})"
            self._record_funnel("Stage 3", "Fine Scan (Optimal)", thr_msg, prev_count, len(current_features))

        # 跨阶段分箱器实例继承机制
        elif self.skip_fine_scan:
            self._stage3_binner = self._rough_binner

        stability_report: MarsBinningReport | None = None
        if current_features and (self.psi_thr is not None or self.rc_thr is not None):
            stability_report = self._get_stability_report(
                X,
                current_features,
                benchmark_df=benchmark_pl,
            )

        # 验证截面分布漂移指标
        if current_features and self.psi_thr is not None and stability_report is not None:
            prev_count = len(current_features)
            current_features = self._filter_psi(stability_report, current_features)
            self._record_funnel(
                stage="Stage 4",
                description="Stability Check (PSI)",
                thresholds={"psi": self.psi_thr},
                count_before=prev_count,
                count_after=len(current_features)
            )

        # 验证截面逻辑相关性指标
        if current_features and self.rc_thr is not None and stability_report is not None:
            prev_count = len(current_features)
            current_features = self._filter_rc(stability_report, current_features)
            self._record_funnel(
                stage="Stage 5",
                description="Risk Consistency (RiskCorr)",
                thresholds={"rc": self.rc_thr},
                count_before=prev_count,
                count_after=len(current_features)
            )

        # 验证交叉项共线性指标
        if current_features and self.corr_thr is not None:
            prev_count = len(current_features)
            current_features = self._filter_corr(X, current_features)
            self._record_funnel("Stage 6", "Correlation Filter",
                                {"corr": self.corr_thr},
                                prev_count, len(current_features))

        # 执行特征集终态覆盖映射
        selected_features = list(current_features)
        selected_set = set(selected_features)
        for feature in valid_white_list:
            if feature not in selected_set:
                selected_features.append(feature)
                selected_set.add(feature)
        self.selected_features_ = selected_features
        self._record_funnel("Final", "White List Forcing",
                            {"white_list_len": len(valid_white_list)},
                            len(current_features), len(self.selected_features_))

        # 触发底层引擎缓存销毁与空间压缩
        if self._stage3_binner is not None:
            self._stage3_binner.prune(self.selected_features_)
            self.clear_cache()

        self._is_fitted = True
        self.show_summary()
        return self

    def _prepare_evaluation_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """校验并归一化当前筛选样本的二分类标签。"""
        from mars.analysis._evaluation.context import (
            count_observed_target_classes,
            normalize_binary_target_column,
        )

        if self.target is None or self.target not in df.columns:
            raise ValueError(
                "`df` must contain the requested target column for IV/Lift feature selection."
            )

        normalized_df = normalize_binary_target_column(df, self.target)
        observed_classes = count_observed_target_classes(normalized_df, self.target)
        if observed_classes < 2:
            raise ValueError(
                f"Target column {self.target!r} in `df` must have at least 2 observed "
                "classes after excluding null / NaN values."
            )
        return normalized_df

    def _validate_group_context(self, df: pl.DataFrame) -> None:
        """校验稳定性筛选所需的分组参数及声明列。"""
        if self.time_grain is not None and self.time_col is None:
            raise ValueError("`time_grain` requires `time_col`.")
        if self.group_col is not None and self.group_col not in df.columns:
            raise ValueError(f"Group column {self.group_col!r} was not found in `df`.")
        if self.time_col is not None and self.time_col not in df.columns:
            raise ValueError(f"Time column {self.time_col!r} was not found in `df`.")

        stability_enabled = self.psi_thr is not None or self.rc_thr is not None
        if stability_enabled and self.group_col is None and self.time_col is None:
            raise ValueError(
                "`group_col` or `time_col` is required when `psi_thr` or `rc_thr` is enabled. "
                "Set both thresholds to `None` for snapshot-only feature selection."
            )

    def _prepare_selector_benchmark(
        self,
        benchmark_df: pl.DataFrame | pd.DataFrame | None,
        *,
        features: list[str],
    ) -> pl.DataFrame | None:
        """校验 selector benchmark，并按监督分箱需求归一化标签。"""
        if benchmark_df is None:
            return None

        from mars.analysis._evaluation.context import prepare_benchmark_frame

        benchmark_pl = self._ensure_polars_dataframe(benchmark_df)
        rough_requires_target = (
            not self.skip_rough_scan
            and self.rough_binning_params.get("method") == "cart"
        )
        require_binary_target = (
            not self.skip_fine_scan
            or rough_requires_target
            or self.risk_corr_baseline == "benchmark"
        )
        return prepare_benchmark_frame(
            benchmark_pl,
            features=features,
            weights_col=None,
            target=self.target,
            require_binary_target=require_binary_target,
        )

    def _effective_feature_start_reference(
        self,
        benchmark_df: pl.DataFrame | None,
    ) -> bool:
        """有显式 benchmark 时关闭 feature-start 基准，避免重复提示。"""
        return self.feature_start_aware_reference if benchmark_df is None else False

    @staticmethod
    def _validate_benchmark_bins(
        binner: MarsBinnerBase,
        benchmark_df: pl.DataFrame,
        features: list[str],
    ) -> None:
        """确认 benchmark 中每个活跃特征都能按拟合规则产出分箱列。"""
        transformed = binner.transform(benchmark_df, return_type="index")
        benchmark_binned = transformed.collect() if isinstance(transformed, pl.LazyFrame) else transformed
        expected_cols = {f"{feature}_bin" for feature in features}
        missing_cols = sorted(expected_cols - set(benchmark_binned.columns))
        if not missing_cols:
            return

        failed_features = [column.removesuffix("_bin") for column in missing_cols]
        raise ValueError(
            "`benchmark_df` could not produce bins for active features "
            f"{failed_features}. Fit failures: {getattr(binner, 'fit_failures_', {})}."
        )

    def _annotate_selector_report(
        self,
        report: MarsBinningReport,
        *,
        benchmark_row_count: int | None,
    ) -> MarsBinningReport:
        """补充 selector 规则来源，避免复用 binner 后元数据退化为显式分箱器。"""
        report.report_meta["binning_fit_source"] = self._binning_fit_source
        report.report_meta["benchmark_row_count"] = benchmark_row_count
        report.report_meta["selection_metric_source"] = "df"
        return report

    def _normalize_feature_data_source(self, features: List[str]) -> Dict[str, str]:
        """
        将选择器的数据源配置转换为特征到数据源的稳定映射。

        未配置来源的特征统一标记为 ``"UNMAPPED"``；配置中出现当前候选特征
        之外的字段时立即失败，避免导出的选择器报告来源错位。
        """
        if not self.feature_data_source:
            return {feature: "UNMAPPED" for feature in features}

        feature_set = set(features)
        mapped_features = set()
        normalized: Dict[str, str] = {}

        for data_source, source_features in self.feature_data_source.items():
            for feature in source_features or []:
                if feature not in feature_set:
                    raise ValueError(
                        "feature_data_source contains features outside the active selector feature set: "
                        f"{feature}"
                    )
                normalized[feature] = str(data_source)
                mapped_features.add(feature)

        for feature in feature_set - mapped_features:
            normalized[feature] = "UNMAPPED"

        return normalized

    def _feature_source_for(self, feature: str) -> str:
        """返回特征所属数据源，未映射时使用统一兜底标签。"""
        return self._feature_source_map.get(feature, "UNMAPPED")

    def _feature_data_source_for(self, features: List[str]) -> Dict[str, List[str]]:
        """按当前活跃特征裁剪数据源配置，避免评估器接收到已过滤字段。"""
        if not self.feature_data_source:
            return {}

        active_features = set(features)
        filtered_source: Dict[str, List[str]] = {}
        for data_source, source_features in self.feature_data_source.items():
            matched_features = [
                feature
                for feature in source_features or []
                if feature in active_features
            ]
            if matched_features:
                filtered_source[str(data_source)] = matched_features

        return filtered_source

    def _register_feature_decision(
        self,
        feature: str,
        status: str,
        stage: str,
        reason: str = "",
        value: float = -1.0,
        desc: str = "",
    ) -> None:
        """记录特征筛选决策并补充数据源标签。"""
        self._register_decision(
            feature,
            status,
            stage,
            reason,
            value,
            desc,
            data_source=self._feature_source_for(feature),
        )

    def transform(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        *,
        keep_target: bool = True,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """根据筛选结果裁剪数据，可选择保留目标列。"""
        result = super().transform(df)
        if not keep_target or self.target is None:
            return result

        df_pl = self._ensure_polars_dataframe(df)
        out_pl = self._ensure_polars_dataframe(result)
        if self.target in df_pl.columns and self.target not in out_pl.columns:
            out_pl = out_pl.with_columns(df_pl.get_column(self.target))
        return self._format_output(out_pl)

    def _record_funnel(
        self,
        stage: str,
        description: str,
        thresholds: dict[str, Any] | str,
        count_before: int,
        count_after: int,
    ) -> None:
        """内部方法：序列化记录筛选节点的快照度量。"""
        if not hasattr(self, "_initial_feature_count") or self._initial_feature_count == 0:
            self._initial_feature_count = count_before

        if isinstance(thresholds, dict):
            thr_str = ", ".join([f"{k}={v}" for k, v in thresholds.items()])
        else:
            thr_str = thresholds

        self._funnel_stats.append({
            "Stage": stage,
            "Description": description,
            "Thresholds": thr_str,
            "Input": count_before,
            "Dropped": count_before - count_after,
            "Remaining": count_after,
            "Retention %": f"{(count_after / count_before * 100):.1f}%" if count_before > 0 else "0%",
            "Cumulative %": f"{(count_after / self._initial_feature_count * 100):.1f}%" if self._initial_feature_count > 0 else "0%"
        })

    def show_summary(self) -> None:
        """
        展示特征筛选漏斗摘要。

        Returns
        -------
        None
            函数仅展示或记录漏斗摘要，不返回表格对象。

        Notes
        -----
        在 Notebook 环境中优先返回富样式表格；若环境不支持，则退化为日志打印。

        Examples
        --------
        >>> selector = MarsStatsSelector()
        >>> selector._record_funnel("Init", "Demo", {"iv": 0.02}, 2, 1)
        >>> selector.show_summary() is None
        True
        """
        if not self._funnel_stats:
            logger.warning("No funnel stats available. Run .fit() first.")
            return

        df_summary = pd.DataFrame(self._funnel_stats)

        def _color_retention(val: str) -> str:
            """为留存率文本生成条件样式。"""
            try:
                p = float(val.rstrip('%'))
                if p < 30:
                    return 'color: #d32f2f; font-weight: bold;'
                if p < 70:
                    return 'color: #ed6c02; font-weight: bold;'
                return 'color: #2e7d32; font-weight: bold;'
            except ValueError:
                return 'color: #757575;'

        def _style_logic_text(v: Any) -> str:
            """为阈值逻辑文本生成高亮样式。"""
            if not isinstance(v, str):
                return ''
            logic_ops = ['&', '|', '<', '>', '=']
            if any(op in v for op in logic_ops):
                return 'color: #7b1fa2; font-weight: bold; font-family: "Courier New", Courier, monospace;'
            return ''

        dropped_style = 'color: #d32f2f; font-weight: bold;'
        muted_style = 'color: #999;'
        remaining_style = 'color: #1565c0; font-weight: bold;'
        cumulative_style = 'font-weight: bold; color: #1b5e20; background-color: #f1f8e9;'
        table_styles: list[dict[str, Any]] = [
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#f5f5f5'),
                    ('color', '#333'),
                    ('border-bottom', '2px solid #ddd'),
                    ('padding', '12px'),
                    ('text-align', 'center'),
                ],
            },
            {
                'selector': 'th.col0, th.col1, th.col2',
                'props': [('text-align', 'left')],
            },
            {
                'selector': 'caption',
                'props': [
                    ('font-size', '18px'),
                    ('font-weight', 'bold'),
                    ('padding', '12px'),
                    ('color', '#1a237e'),
                    ('text-align', 'left'),
                ],
            },
        ]
        styler = (
            df_summary.style
            .set_caption("Mars Stats Selector: Feature Selection Funnel")
            .bar(subset=['Dropped'], color='#ffdbdb', vmin=0)
            .bar(subset=['Remaining'], color='#e1f5fe', vmin=0)
            .set_properties(**{'text-align': 'left'}, subset=['Stage', 'Description', 'Thresholds'])
            .set_properties(
                **{'text-align': 'center'},
                subset=['Input', 'Dropped', 'Remaining', 'Retention %', 'Cumulative %'],
            )
            .map(lambda v: dropped_style if v > 0 else muted_style, subset=['Dropped'])
            .map(lambda v: remaining_style, subset=['Remaining'])
            .map(_color_retention, subset=['Retention %'])
            .map(lambda v: cumulative_style, subset=['Cumulative %'])
            .map(_style_logic_text, subset=['Thresholds'])
            .set_table_styles(table_styles)
        )

        try:
            from IPython.display import display
            display(styler)
        except ImportError:
            return None

    def _should_bypass_filter(self, feat: str) -> bool:
        """内部方法：解析特征实体是否命中免检逻辑池。"""
        return feat in self.white_list

    def clear_cache(self) -> None:
        """
        释放缓存的分箱器上下文。

        Returns
        -------
        None
            函数仅释放缓存资源，并记录调试日志。

        Notes
        -----
        该方法会清理最终分箱器中缓存的数据引用，并主动触发一次垃圾回收。

        Examples
        --------
        >>> selector = MarsStatsSelector()
        >>> selector.clear_cache() is None
        True
        """
        if self._stage3_binner is not None:
            self._stage3_binner.clear_cache()

        import gc
        gc.collect()
        logger.debug("Selector cache cleared.")

    def _filter_quality(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：计算并核验数据质量约束向量。"""
        from mars.analysis.profiler import MarsDataProfiler
        profiler = MarsDataProfiler(
            missing_values=self.missing_values,
            special_values=self.special_values
        )
        report = profiler.generate_profile(
            df,
            metrics=["missing", "zeros", "mode"],
            features=features,
            enable_sparkline=False,
        )

        stats_records = report.overview_table.select([
            "feature", "missing_rate", "mode_rate", "zeros_rate"
        ]).to_dicts()
        kept_features = []

        for row in stats_records:
            feat = row["feature"]
            missing = row["missing_rate"]
            mode_rate = row["mode_rate"]
            zeros_rate = row["zeros_rate"]

            # 实施特定属性旁路绕过数据分布的边界校验
            if self._should_bypass_filter(feat):
                self._register_feature_decision(feat, "Selected", "Quality", "White List", missing)
                kept_features.append(feat)
                continue

            if missing > self.missing_thr:
                self._register_feature_decision(feat, "Dropped", "Quality", "High Missing", missing)
            elif zeros_rate > self.zeros_thr:
                self._register_feature_decision(feat, "Dropped", "Quality", "High Zero Rate", zeros_rate)
            elif mode_rate > self.mode_thr:
                self._register_feature_decision(feat, "Dropped", "Quality", "Single Value (Mode)", mode_rate)
            else:
                self._register_feature_decision(feat, "Selected", "Quality", "Pass", missing)
                kept_features.append(feat)

        return kept_features

    def _filter_rough(
        self,
        df: pl.DataFrame,
        features: list[str],
        *,
        benchmark_df: pl.DataFrame | None,
    ) -> list[str]:
        """在 benchmark 或当前样本上拟合粗分箱，并始终用当前样本计算 IV/Lift。"""
        if not features:
            return []

        fit_df = benchmark_df if benchmark_df is not None else df
        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if fit_df.schema[c] in cat_types]

        binner = MarsNativeBinner(
            missing_values=self.missing_values,
            special_values=self.special_values,
            n_jobs=self.n_jobs,
            **self.rough_binning_params,
        )
        fit_target = (
            fit_df.get_column(self.target)
            if self.rough_binning_params.get("method") == "cart" and self.target is not None
            else None
        )
        binner.fit(
            fit_df,
            fit_target,
            features=features,
            cat_features=cat_features,
        )
        if benchmark_df is not None:
            self._validate_benchmark_bins(binner, benchmark_df, features)

        self._rough_binner = binner

        if self.target is None:
            raise ValueError("Selector target is unavailable after input validation.")
        target = df.get_column(self.target)
        # 分箱边界来自 fit_df，但 WOE 和筛选指标必须基于当前 df 物化。
        stats_df = binner.profile_bin_performance(df, target, update_woe=True)

        lift_recall_cond = (pl.col("Lift") > self.rough_lift_thr) & (pl.col("count_dist") > self.rough_min_sample_rate)
        feat_stats = (
            stats_df.group_by("feature")
            .agg([
                pl.col("IV").max().alias("IV_total"),
                lift_recall_cond.any().alias("has_high_lift"),
                pl.col("Lift").max().alias("max_lift")
            ])
        ).to_dicts()

        kept_features = []

        for row in feat_stats:
            feat, iv, has_high_lift, max_lift = row["feature"], row["IV_total"], row["has_high_lift"], row["max_lift"]

            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                self._feature_iv_dict[feat] = iv
                continue

            if iv > self.rough_iv_thr or has_high_lift:
                reason = f"Pass (IV={iv:.3f})" if iv > self.rough_iv_thr else f"Pass (Lift={max_lift:.2f})"
                self._register_feature_decision(feat, "Selected", "Rough_Scan", reason, iv)
                kept_features.append(feat)
                self._feature_iv_dict[feat] = iv
            else:
                self._register_feature_decision(feat, "Dropped", "Rough_Scan", "Low IV & Low Lift", iv)

        # 构建分箱失败时的容错矩阵，确保声明属性不发生物理丢失
        kept_set = set(kept_features)
        for f in features:
            if self._should_bypass_filter(f) and f not in kept_set:
                kept_features.append(f)
                self._feature_iv_dict[f] = 0.0
                self._register_feature_decision(f, "Selected", "Rough_Scan", "White List (Binner Fallback)", 0.0)

        return kept_features

    def _filter_fine(
        self,
        df: pl.DataFrame,
        features: list[str],
        *,
        benchmark_df: pl.DataFrame | None,
    ) -> list[str]:
        """在 benchmark 上拟合最优分箱，并基于当前样本指标执行精筛。"""
        from mars.analysis.evaluator import MarsBinEvaluator

        binner_params = {
            **self.binning_params,
            "missing_values": self.missing_values,
            "special_values": self.special_values,
        }
        evaluator = MarsBinEvaluator(
            binning_type="optimal",
            binner_params=binner_params,
            feature_start_aware_reference=self.feature_start_aware_reference,
            risk_corr_baseline=self.risk_corr_baseline,
        )

        run = evaluator.evaluate(
            df=df,
            target=self.target,
            features=features,
            group_col=self.group_col,
            time_col=self.time_col,
            time_grain=self.time_grain,
            feature_data_source=self._feature_data_source_for(features),
            benchmark_df=benchmark_df,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
            feature_start_aware_reference=self._effective_feature_start_reference(
                benchmark_df,
            ),
            risk_corr_baseline=self.risk_corr_baseline,
            batch_size=self.batch_size,
        )
        report = self._annotate_selector_report(
            run.report,
            benchmark_row_count=(benchmark_df.height if benchmark_df is not None else None),
        )

        self._stage3_binner = run.binner
        self._stability_report = report

        lift_recall_set = set()
        if self.lift_thr is not None:
            group_col = report.detail_group_col
            if group_col is None:
                raise ValueError("Binning report detail group column is missing.")
            lift_cond = (
                (pl.col("lift") > self.lift_thr)
                & (pl.col("pct") > self.min_sample_rate)
            )

            lift_passed = report.detail_table.filter(
                (pl.col(group_col) == "Total") & lift_cond
            )
            lift_recall_set = set(lift_passed["feature"].unique().to_list())

        kept_features = []
        for row in report.summary_table.to_dicts():
            feat = row["feature"]
            iv_total = row.get("iv", 0.0)

            if self._should_bypass_filter(feat):
                self._feature_iv_dict[feat] = iv_total
                kept_features.append(feat)
                continue

            is_iv_ok = iv_total >= self.iv_thr
            is_lift_recall = feat in lift_recall_set

            if is_iv_ok or is_lift_recall:
                decision_reason = "Pass (IV)" if is_iv_ok else "Pass (Lift Recall)"
                self._register_feature_decision(feat, "Selected", "Fine_Scan", decision_reason, iv_total)

                self._feature_iv_dict[feat] = iv_total
                kept_features.append(feat)
            else:
                self._register_feature_decision(feat, "Dropped", "Fine_Scan", "Low IV & No Lift Recall", iv_total)

        # 执行针对数学期望异常结果的系统级强制路由分配
        kept_set = set(kept_features)
        for f in features:
            if self._should_bypass_filter(f) and f not in kept_set:
                kept_features.append(f)
                self._feature_iv_dict[f] = 0.0
                self._register_feature_decision(f, "Selected", "Fine_Scan", "White List (Binner Fallback)", 0.0)

        return kept_features

    def _get_stability_report(
        self,
        df: pl.DataFrame,
        features: list[str],
        *,
        benchmark_df: pl.DataFrame | None,
    ) -> MarsBinningReport:
        """复用精筛报告，或按最终粗分箱器一次性计算 PSI 与 RC。"""
        if self._stability_report is not None:
            return self._stability_report
        if self._stage3_binner is None:
            raise ValueError("No fitted binner is available for stability evaluation.")

        from mars.analysis.evaluator import MarsBinEvaluator

        run = MarsBinEvaluator(
            feature_start_aware_reference=self.feature_start_aware_reference,
            risk_corr_baseline=self.risk_corr_baseline,
        ).evaluate(
            df=df,
            target=self.target,
            features=features,
            group_col=self.group_col,
            time_col=self.time_col,
            time_grain=self.time_grain,
            feature_data_source=self._feature_data_source_for(features),
            binner=self._stage3_binner,
            benchmark_df=benchmark_df,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
            feature_start_aware_reference=self._effective_feature_start_reference(
                benchmark_df,
            ),
            risk_corr_baseline=self.risk_corr_baseline,
            batch_size=self.batch_size,
        )
        self._stability_report = self._annotate_selector_report(
            run.report,
            benchmark_row_count=(benchmark_df.height if benchmark_df is not None else None),
        )
        return self._stability_report

    def _filter_psi(
        self,
        report: MarsBinningReport,
        features: list[str],
    ) -> list[str]:
        """基于共享稳定性报告执行 PSI 上限筛选。"""
        psi_map = {
            row["feature"]: row["psi_max"]
            for row in report.summary_table.select(["feature", "psi_max"]).to_dicts()
        }
        threshold = self.psi_thr
        if threshold is None:
            return features

        kept_features = []
        for feat in features:
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue

            psi_val = psi_map.get(feat, 0.0)
            if psi_val < threshold:
                self._register_feature_decision(feat, "Selected", "Stability", "Stable PSI", psi_val)
                kept_features.append(feat)
            else:
                self._register_feature_decision(feat, "Dropped", "Stability", f"High PSI ({psi_val:.2f})", psi_val)

        return kept_features

    def _filter_rc(
        self,
        report: MarsBinningReport,
        features: list[str],
    ) -> list[str]:
        """基于共享稳定性报告执行 RC 下限筛选。"""
        if "rc_min" in report.summary_table.columns:
            rc_map = {
                row["feature"]: row["rc_min"]
                for row in report.summary_table.select(["feature", "rc_min"]).to_dicts()
            }
        else:
            rc_map = {}
            logger.warning("rc_min metric not found in report. Skipping RC check.")
        threshold = self.rc_thr
        if threshold is None:
            return features

        kept_features = []
        for feat in features:
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue

            rc_val = rc_map.get(feat, 1.0)
            if rc_val is None or rc_val >= threshold:
                self._register_feature_decision(feat, "Selected", "RiskCorr", "Stable Logic", rc_val if rc_val is not None else 1.0)
                kept_features.append(feat)
            else:
                self._register_feature_decision(feat, "Dropped", "RiskCorr", f"Logic Broken (RC={rc_val:.2f})", rc_val)

        return kept_features

    def _filter_corr(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：执行目标感知导向的共线性惩罚计算。"""
        if len(features) < 2:
            return features

        woe_cols = [f"{c}_woe" for c in features]
        df_woe = self._stage3_binner.transform(df.select(features), return_type="woe")
        corr_matrix_df = df_woe.select(woe_cols).fill_null(0.0).corr()

        corr_matrix_with_names = corr_matrix_df.with_columns(
            feature_name=pl.Series(woe_cols)
        )

        sorted_feats = sorted(
            features,
            key=lambda f: (-self._feature_iv_dict.get(f, 0.0), f)
        )

        kept_features_set = set()
        dropped_features = set()

        for feat in sorted_feats:
            if self._should_bypass_filter(feat):
                kept_features_set.add(feat)
                continue

            if feat in dropped_features:
                continue

            kept_features_set.add(feat)
            self._register_feature_decision(feat, "Selected", "Corr_Filter", "Independent", self._feature_iv_dict.get(feat, 0))

            target_woe_name = f"{feat}_woe"
            high_corr_row = corr_matrix_with_names.filter(pl.col("feature_name") == target_woe_name)

            for other_feat_woe in woe_cols:
                if other_feat_woe == target_woe_name:
                    continue

                corr_val = abs(high_corr_row.get_column(other_feat_woe)[0])
                if corr_val > self.corr_thr:
                    orig_f = other_feat_woe[:-4]
                    if orig_f not in dropped_features and not self._should_bypass_filter(orig_f):
                        dropped_features.add(orig_f)
                        self._register_feature_decision(orig_f, "Dropped", "Corr_Filter", f"Correlated with '{feat}'", corr_val)

        return [f for f in features if f in kept_features_set]

    def get_binning_report(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    ) -> MarsBinningReport:
        """
        为已选中特征生成最终风险评估报告。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            用于重新评估已选中特征的样本表。
        benchmark_df : pl.DataFrame | pd.DataFrame | None
            PSI expected distribution 和可选 benchmark RC 的基准样本。若 ``fit``
            使用过 benchmark，本方法必须重新传入；固定分箱器不会重新拟合。

        Returns
        -------
        MarsBinningReport
            基于 `selected_features_` 生成的风险评估报告。

        Raises
        ------
        ValueError
            当筛选器尚未拟合、没有选中特征、缺少所需 benchmark 或输入无效时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> selector = MarsStatsSelector(skip_fine_scan=True, psi_thr=None, rc_thr=None)
        >>> selector.fit(df, target="y", features=["age"])
        >>> report = selector.get_binning_report(df)
        >>> isinstance(report, MarsBinningReport)
        True

        Notes
        -----
        selector 不保存原始 benchmark。重新生成报告时显式重传可避免 PSI 基准静默
        回退到 ``df`` 的最早分组。
        """
        self._check_is_fitted()

        if not self.selected_features_:
            raise ValueError("No selected features found. Cannot generate report.")
        if self._fit_used_benchmark and benchmark_df is None:
            raise ValueError(
                "`benchmark_df` must be provided to `get_binning_report` because this selector "
                "was fitted with a benchmark. Re-pass the intended baseline to preserve PSI."
            )

        X_pl = self._prepare_evaluation_frame(self._ensure_polars_dataframe(df))
        self._validate_group_context(X_pl)

        benchmark_pl: pl.DataFrame | None = None
        if benchmark_df is not None:
            from mars.analysis._evaluation.context import prepare_benchmark_frame

            benchmark_pl = prepare_benchmark_frame(
                self._ensure_polars_dataframe(benchmark_df),
                features=self.selected_features_,
                weights_col=None,
                target=self.target,
                require_binary_target=self.risk_corr_baseline == "benchmark",
            )

        from mars.analysis.evaluator import MarsBinEvaluator

        if self._stage3_binner is None:
            raise ValueError("No fitted binner is available for report generation.")

        evaluator = MarsBinEvaluator(
            feature_start_aware_reference=self.feature_start_aware_reference,
            risk_corr_baseline=self.risk_corr_baseline,
        )

        if self._return_pandas:
            evaluator.set_output("pandas")

        run = evaluator.evaluate(
            df=X_pl,
            target=self.target,
            features=self.selected_features_,
            group_col=self.group_col,
            time_col=self.time_col,
            time_grain=self.time_grain,
            feature_data_source=self._feature_data_source_for(self.selected_features_),
            binner=self._stage3_binner,
            benchmark_df=benchmark_pl,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
            feature_start_aware_reference=self._effective_feature_start_reference(
                benchmark_pl,
            ),
            risk_corr_baseline=self.risk_corr_baseline,
        )
        report = self._annotate_selector_report(
            run.report,
            benchmark_row_count=(benchmark_pl.height if benchmark_pl is not None else None),
        )

        return report

    def export_selector_report(self, path: str = "mars_selector_report.xlsx") -> None:
        """
        导出选择器决策报告。

        Parameters
        ----------
        path : str
            持久化导出路径。引擎根据扩展名执行 `.csv` 或复合样式 `.xlsx` 的落盘处理。

        Returns
        -------
        None
            报告直接写入 ``path``，函数不返回文件句柄或表格对象。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> selector = MarsStatsSelector()
        >>> selector._is_fitted = True
        >>> selector._register_feature_decision("age", "Selected", "demo", "demo")
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "selector.csv"
        ...     selector.export_selector_report(str(path))
        ...     path.exists()
        True
        """
        report_df = self.get_report()
        if isinstance(report_df, pd.DataFrame):
            if report_df.empty:
                logger.warning("No report to export.")
                return
            pd_df = report_df
        else:
            if report_df.height == 0:
                logger.warning("No report to export.")
                return
            pd_df = report_df.to_pandas()

        if path.endswith(".csv"):
            pd_df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            try:
                styler = pd_df.style.map(
                    lambda v: 'color: green; font-weight: bold' if v == 'Selected' else 'color: red',
                    subset=['status']
                )
                styler.to_excel(path, index=False, engine="openpyxl")
            except Exception as e:
                logger.warning(f"Failed to export styled excel, falling back to basic export. Error: {e}")
                pd_df.to_excel(path, index=False)

    def save_selector_lists(
        self,
        path: str = "mars_lists.json",
        blacklist_stages: List[str] | None = None
    ) -> None:
        """
        保存当前筛选结果中的白名单与黑名单。

        Parameters
        ----------
        path : str
            JSON 结构存储路径。
        blacklist_stages : List[str] | None
            界定需写入惩罚名单的阶段。支持字符串模糊匹配（例如 'quality' 匹配质量校验环节）。

        Returns
        -------
        None
            白名单与黑名单直接写入 JSON 文件。

        Notes
        -----
        导出的 ``white_list`` 为当前最终入选特征；``black_list`` 为被剔除特征与
        用户预设黑名单的并集。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> selector = MarsStatsSelector()
        >>> selector._is_fitted = True
        >>> selector.selected_features_ = ["age"]
        >>> selector.report_records_ = [{"feature": "income", "status": "Dropped", "stage": "Quality"}]
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "lists.json"
        ...     selector.save_selector_lists(str(path))
        ...     MarsStatsSelector.load_lists_from_json(str(path))["white_list"]
        ['age']
        """
        self._check_is_fitted()

        if blacklist_stages:
            patterns = [p.lower() for p in blacklist_stages]
        else:
            patterns = []

        def is_stage_matched(actual_stage: str) -> bool:
            """判断当前阶段名是否命中黑名单过滤规则。"""
            if not patterns:
                return True
            actual_stage_lower = actual_stage.lower()
            return any(p in actual_stage_lower for p in patterns)

        dropped_records = [
            r["feature"] for r in self.report_records_
            if r["status"] == "Dropped" and is_stage_matched(r["stage"])
        ]

        data = {
            "white_list": self.selected_features_,
            "black_list": list(set(dropped_records + self.black_list))
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        match_msg = f"matching {blacklist_stages}" if blacklist_stages else "from all stages"
        logger.info(f"Black/White lists saved to {path}. (Blacklisted features {match_msg})")

    @classmethod
    def load_lists_from_json(
        cls: type[MarsStatsSelector],
        path: str,
    ) -> Dict[str, List[str]]:
        """
        从 JSON 文件加载白名单与黑名单。

        Parameters
        ----------
        path : str
            JSON 结构存储路径。

        Returns
        -------
        dict
            包含 ``white_list`` 与 ``black_list`` 的字典。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "lists.json"
        ...     _ = path.write_text(
        ...         json.dumps({"white_list": ["age"], "black_list": []}),
        ...         encoding="utf-8",
        ...     )
        ...     MarsStatsSelector.load_lists_from_json(str(path))["white_list"]
        ['age']
        """
        if not os.path.exists(path):
            logger.warning(f"File {path} not found. Returning empty lists.")
            return {"white_list": [], "black_list": []}

        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def print_stats(self, iv_thresholds: List[float] | None = None) -> None:
        """
        打印最终入选特征的统计摘要。

        Parameters
        ----------
        iv_thresholds : List[float] | None
            自定义统计边界截断数组。默认渲染 [0.02, 0.05, 0.10] 区间梯度。

        Returns
        -------
        None
            函数仅通过日志输出统计摘要。

        Examples
        --------
        >>> selector = MarsStatsSelector()
        >>> selector._is_fitted = True
        >>> selector.selected_features_ = ["age"]
        >>> selector._feature_iv_dict = {"age": 0.12}
        >>> selector.print_stats([0.05]) is None
        True
        """
        self._check_is_fitted()

        if not self.selected_features_:
            logger.warning("No features survived the selection funnel.")
            return

        if iv_thresholds is None:
            iv_thresholds = [0.02, 0.05, 0.10]
        else:
            iv_thresholds = sorted(iv_thresholds)

        final_ivs = [self._feature_iv_dict.get(f, 0.0) for f in self.selected_features_]
        total = len(self.selected_features_)

        max_iv = max(final_ivs) if final_ivs else 0.0
        mean_iv = sum(final_ivs) / total if total > 0 else 0.0

        stats_msg = [
            f"\n{'='*50}",
            "Mars Feature Selection Final Stats",
            f"{'-'*50}",
            f"Survived Features : {total}",
            f"Maximum IV        : {max_iv:.4f}",
            f"Average IV        : {mean_iv:.4f}"
        ]

        for thr in iv_thresholds:
            count = sum(1 for iv in final_ivs if iv >= thr)
            stats_msg.append(f"IV >= {thr:<13} : {count} ({count/total:.1%})")

        stats_msg.append(f"{'='*50}")
