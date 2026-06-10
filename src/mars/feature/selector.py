"""MARS 特征筛选器实现模块。"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.analysis.report import MarsEvaluationReport
from mars.core.base import MarsBaseSelector
from mars.feature.base import MarsBinnerBase
from mars.feature.native_binner import MarsNativeBinner
from mars.modeling.utils import require_optional_module
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsStatsSelector(MarsBaseSelector):
    """
    基于风控统计指标的特征筛选器。

    该筛选器将数据质量、IV/Lift、PSI、相关性和白黑名单规则串成一个漏斗式筛选流程。
    构造函数只保存阈值、分箱策略、缺失/特殊值配置以及运行资源参数；样本数据、
    目标列、特征范围、分组列、时间列和最大抽样量都由 `fit` 传入。

    典型用法是先用粗分箱低成本压缩特征空间，再用精细分箱和稳定性规则做最终筛选。
    `white_list` 中的特征会尽量绕过自动剔除规则，`black_list` 中的特征会被强制排除。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> selector = MarsStatsSelector(skip_fine_scan=True)
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
            精筛阶段分箱器参数。
        rough_binning_params : Dict[str, Any] | None
            粗筛阶段分箱器参数。
        batch_size : int | None
            批量评估时的特征批大小。
        n_jobs : int
            并行任务数，含义遵循 joblib 约定。
        """
        super().__init__()

        self.features: list[str] | None = None
        self.feature_data_source: dict[str, list[str]] = {}
        self.time_col: str | None = None
        self.profile_by: str | None = None
        self.feature_start_aware_baseline = False
        self.white_list: list[str] = []
        self.black_list: list[str] = []

        self.missing_values = missing_values if missing_values else []
        self.special_values = special_values if special_values else []

        self.missing_thr = missing_thr
        self.mode_thr = mode_thr
        self.zeros_thr = zeros_thr

        self.skip_rough_scan = skip_rough_scan
        self.rough_binning_params = rough_binning_params or {
            "method": "quantile",
            "n_bins": 20,
            "min_bin_size": 0.01,
            "merge_small_bins": True
        }
        self.rough_iv_thr = rough_iv_thr
        self.rough_lift_thr = rough_lift_thr
        self.rough_min_sample_rate = rough_min_sample_rate

        self.skip_fine_scan = skip_fine_scan
        self.binning_params = binning_params or {
            "prebinning_method": "cart",
            "n_bins": 10,
            "min_bin_size": 0.05,
        }
        self.iv_thr = iv_thr
        self.lift_thr = lift_thr
        self.min_sample_rate = min_sample_rate

        self.psi_thr = psi_thr
        self.rc_thr = rc_thr
        self.corr_thr = corr_thr
        self.psi_include_missing = psi_include_missing
        self.psi_include_special = psi_include_special

        self.max_samples: int | None = None
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self._rough_binner: MarsNativeBinner | None = None
        self._stage3_binner: MarsBinnerBase | None = None
        self._feature_iv_dict: Dict[str, float] = {}
        self._feature_source_map: Dict[str, str] = {}

        self._funnel_stats = []

    @time_it
    def fit(
        self,
        df: pl.DataFrame | pd.DataFrame,
        *,
        target: str,
        features: List[str] | None = None,
        feature_data_source: Dict[str, List[str]] | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        white_list: List[str] | None = None,
        black_list: List[str] | None = None,
        max_samples: int | None = None,
        feature_start_aware_baseline: bool = False,
    ) -> MarsStatsSelector:
        """
        在一次样本上下文中执行统计特征筛选。

        Parameters
        ----------
        df : pl.DataFrame | pd.DataFrame
            待筛选样本表。
        target : str
            二分类目标列名。
        features : List[str] | None
            候选特征列；不传时会从样本表中自动推断。
        feature_data_source : Dict[str, List[str]] | None
            特征来源映射，用于报告中追踪特征所属来源。
        group_col : str | None
            已存在的分组列名，用于趋势和稳定性筛选。
        time_col : str | None
            原始日期列名；与 `time_grain` 配合时生成时间分组。
        time_grain : str | None
            时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
        white_list : List[str] | None
            白名单特征，尽量绕过自动剔除规则。
        black_list : List[str] | None
            黑名单特征，会被强制剔除。
        max_samples : int | None
            本次筛选允许使用的最大样本量。
        feature_start_aware_baseline : bool
            是否按特征首次出现分组选择 PSI 基准。

        Returns
        -------
        MarsStatsSelector
            拟合后的筛选器，`selected_features_` 中保存最终特征列表。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        # 拦截互斥的配置项
        if self.skip_rough_scan and self.skip_fine_scan:
            raise ValueError("Cannot skip both rough scan and fine scan. At least one binning stage is required.")

        self.target = target
        self.features = features
        self.feature_data_source = feature_data_source or {}
        self.time_col = time_col
        self.profile_by = (time_grain or "month") if time_col else group_col
        self.white_list = white_list if white_list else []
        self.black_list = black_list if black_list else []
        self.max_samples = max_samples
        self.feature_start_aware_baseline = feature_start_aware_baseline

        X = self._ensure_polars_dataframe(df)
        self._funnel_stats = []
        self._feature_iv_dict = {}

        # 初始化候选特征空间，剥离非特征维度
        exclude_cols = {self.target}
        if self.time_col:
            exclude_cols.add(self.time_col)
        if self.profile_by:
            exclude_cols.add(self.profile_by)

        candidate_features = [c for c in (self.features if self.features else X.columns)
                              if c in X.columns and c not in exclude_cols]
        self._feature_source_map = self._normalize_feature_data_source(candidate_features)

        # 校准白名单，剔除数据集中不存在的幽灵声明
        valid_white_list = [f for f in self.white_list if f in candidate_features]

        # 应用静态黑名单约束
        current_features = [c for c in candidate_features if c not in self.black_list]
        self._record_funnel("Init", "Blacklist & Exclusions",
                            {"black_list_len": len(self.black_list)},
                            len(candidate_features), len(current_features))

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
                    scan_features = self._filter_rough(X, scan_features)

                current_features = scan_features + white_features
            else:
                # 探查流水线终点：白名单特征强制并入获取分箱实体
                current_features = self._filter_rough(X, current_features)

            thr_msg = f"iv >= {self.rough_iv_thr} | (lift >= {self.rough_lift_thr} & sample >= {self.rough_min_sample_rate})"
            self._record_funnel("Stage 2", "Rough Scan (Native)", thr_msg, prev_count, len(current_features))

        # 执行精确分布区间与区分度评估
        if not self.skip_fine_scan and current_features:
            prev_count = len(current_features)
            current_features = self._filter_fine(X, current_features)
            thr_msg = f"iv >= {self.iv_thr} | (lift >= {self.lift_thr} & sample >= {self.min_sample_rate})"
            self._record_funnel("Stage 3", "Fine Scan (Optimal)", thr_msg, prev_count, len(current_features))

        # 跨阶段分箱器实例继承机制
        elif self.skip_fine_scan:
            logger.info("Fine Scan skipped. Promoting Rough Scan Binner to main pipeline.")
            self._stage3_binner = self._rough_binner

        # 验证截面分布漂移指标
        if current_features and (self.time_col or self.profile_by) and self.psi_thr is not None:
            prev_count = len(current_features)
            current_features = self._filter_psi(X, current_features)
            self._record_funnel(
                stage="Stage 4",
                description="Stability Check (PSI)",
                thresholds={"psi": self.psi_thr},
                count_before=prev_count,
                count_after=len(current_features)
            )

        # 验证截面逻辑相关性指标
        if current_features and (self.time_col or self.profile_by) and self.rc_thr is not None:
            prev_count = len(current_features)
            current_features = self._filter_rc(X, current_features)
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
            logger.info(f"Pruning internal binner state to {len(self.selected_features_)} selected features...")
            self._stage3_binner.prune(self.selected_features_)
            self.clear_cache()

        self._is_fitted = True
        self.show_summary()
        return self

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
            logger.info("Selector summary:\n%s", df_summary.to_string(index=False))

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
            features=features,
            config_overrides={
                "dq_metrics": ["missing", "zeros", "top1"],
                "stat_metrics": [],
                "enable_sparkline": False
            }
        )

        stats_records = report.overview_table.select([
            "feature", "missing_rate", "top1_ratio", "zeros_rate"
        ]).to_dicts()
        kept_features = []

        for row in stats_records:
            feat = row["feature"]
            missing = row["missing_rate"]
            mode_rate = row["top1_ratio"]
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

    def _filter_rough(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：执行基于高并发原生分箱器的粗略信息增益下限校验。"""
        if not features:
            return []

        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if df.schema[c] in cat_types]

        if cat_features:
            logger.info(f"Native Binner will also evaluate {len(cat_features)} categorical features.")

        binner = MarsNativeBinner(
            missing_values=self.missing_values,
            special_values=self.special_values,
            n_jobs=self.n_jobs,
            **self.rough_binning_params
        )
        target = df.get_column(self.target)
        binner.fit(df, target, features=features, cat_features=cat_features)

        self._rough_binner = binner

        stats_df = binner.profile_bin_performance(df, target, update_woe=False)

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

    def _filter_fine(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：执行具备全量约束的最优分箱核查及区间效能召回。"""
        from mars.analysis.evaluator import MarsBinEvaluator

        binner_params = {
            **self.binning_params,
            "missing_values": self.missing_values,
            "special_values": self.special_values,
        }
        evaluator = MarsBinEvaluator(
            binning_type="optimal",
            binner_params=binner_params,
        )

        run = evaluator.evaluate(
            df=df,
            target=self.target,
            features=features,
            feature_data_source=self._feature_data_source_for(features),
            group_col=None if self.time_col else self.profile_by,
            time_col=self.time_col,
            time_grain=self.profile_by if self.time_col else None,
            batch_size=self.batch_size,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
        )
        report = run.report

        self._stage3_binner = run.binner

        lift_recall_set = set()
        if self.lift_thr is not None:
            group_col = report.group_col
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

    def _filter_psi(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：跨维度投射特征区间计算群体偏移极值。"""
        from mars.analysis.evaluator import MarsBinEvaluator
        run = MarsBinEvaluator().evaluate(
            df=df,
            target=self.target,
            features=features,
            binner=self._stage3_binner,
            feature_data_source=self._feature_data_source_for(features),
            group_col=None if self.time_col else self.profile_by,
            time_col=self.time_col,
            time_grain=self.profile_by if self.time_col else None,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
        )
        report = run.report
        psi_map = {r["feature"]: r["psi_max"] for r in report.summary_table.select(["feature", "psi_max"]).to_dicts()}

        kept_features = []
        for feat in features:
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue

            psi_val = psi_map.get(feat, 0.0)
            if psi_val < self.psi_thr:
                self._register_feature_decision(feat, "Selected", "Stability", "Stable PSI", psi_val)
                kept_features.append(feat)
            else:
                self._register_feature_decision(feat, "Dropped", "Stability", f"High PSI ({psi_val:.2f})", psi_val)

        return kept_features

    def _filter_rc(self, df: pl.DataFrame, features: List[str]) -> List[str]:
        """内部方法：追踪特征序列区间逻辑相关性的变异下限。"""
        from mars.analysis.evaluator import MarsBinEvaluator
        run = MarsBinEvaluator().evaluate(
            df=df,
            target=self.target,
            features=features,
            binner=self._stage3_binner,
            feature_data_source=self._feature_data_source_for(features),
            group_col=None if self.time_col else self.profile_by,
            time_col=self.time_col,
            time_grain=self.profile_by if self.time_col else None,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
        )
        report = run.report

        if "rc_min" in report.summary_table.columns:
            rc_map = {r["feature"]: r["rc_min"] for r in report.summary_table.select(["feature", "rc_min"]).to_dicts()}
        else:
            rc_map = {}
            logger.warning("rc_min metric not found in report. Skipping RC check.")

        kept_features = []
        for feat in features:
            if self._should_bypass_filter(feat):
                kept_features.append(feat)
                continue

            rc_val = rc_map.get(feat, 1.0)
            if rc_val is None or rc_val >= self.rc_thr:
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

    def get_eval_report(self, df: Union[pl.DataFrame, pd.DataFrame]) -> MarsEvaluationReport:
        """
        为已选中特征生成最终风险评估报告。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            用于重新评估已选中特征的样本表。

        Returns
        -------
        MarsEvaluationReport
            基于 `selected_features_` 生成的风险评估报告。

        Raises
        ------
        ValueError
            当筛选器尚未拟合或没有选中特征时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> selector = MarsStatsSelector(skip_fine_scan=True)
        >>> selector.fit(df, target="y", features=["age"])
        >>> report = selector.get_eval_report(df)
        >>> isinstance(report, MarsEvaluationReport)
        True
        """
        self._check_is_fitted()

        if not self.selected_features_:
            raise ValueError("No selected features found. Cannot generate report.")

        X_pl = self._ensure_polars_dataframe(df)

        from mars.analysis.evaluator import MarsBinEvaluator

        if self._stage3_binner is not None:
            evaluator = MarsBinEvaluator()
            eval_binner = self._stage3_binner
        else:
            logger.warning("Cached binner not found. Re-fitting Binner for the selected features...")
            binning_type = "native" if self.skip_fine_scan else "optimal"
            binning_params = self.rough_binning_params if self.skip_fine_scan else self.binning_params
            evaluator = MarsBinEvaluator(
                binning_type=binning_type,
                binner_params={
                    **binning_params,
                    "missing_values": self.missing_values,
                    "special_values": self.special_values,
                },
            )
            eval_binner = None

        if self._return_pandas:
            evaluator.set_output("pandas")

        logger.info(f"Generating final evaluation report for {len(self.selected_features_)} selected features...")

        run = evaluator.evaluate(
            df=X_pl,
            target=self.target,
            features=self.selected_features_,
            binner=eval_binner,
            group_col=None if self.time_col else self.profile_by,
            time_col=self.time_col,
            time_grain=self.profile_by if self.time_col else None,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
            feature_data_source=self._feature_data_source_for(self.selected_features_),
            psi_include_missing=self.psi_include_missing,
            psi_include_special=self.psi_include_special,
        )
        report: MarsEvaluationReport = run.report

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

        logger.info(f"Exporting Selection Report to {path}...")

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

        logger.info("Export Complete.")

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

        logger.info("\n%s", "\n".join(stats_msg))


class MarsLinearSelector(MarsBaseSelector):
    """
    面向传统 LR 建模的线性特征筛选器。

    该选择器按相关性过滤、VIF 过滤和逐步回归三个阶段收敛候选特征。
    输入可以是 Polars 或 Pandas；统计建模边界会转换为 Pandas/NumPy，
    以复用 statsmodels 的 Logit、AIC/BIC 和 VIF 实现。

    Attributes
    ----------
    selected_features_ : list of str
        最终入选特征。
    coef_table_ : pandas.DataFrame
        最终 Logit 模型的系数、标准误和 p-value。
    vif_table_ : pandas.DataFrame
        VIF 阶段的最终候选特征 VIF 表。
    stepwise_history_ : pandas.DataFrame
        逐步回归每一步的 add/drop 决策记录。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> selector = MarsLinearSelector(corr_thr=0.95)
    >>> selector.fit(df[["age"]], df["y"], features=["age"]).selected_features_
    ['age']
    """

    def __init__(
        self,
        enable_corr_filter: bool = True,
        corr_thr: float = 0.8,
        corr_method: str = "spearman",
        enable_vif_filter: bool = False,
        vif_threshold: float = 5.0,
        enable_stepwise: bool = False,
        stepwise_direction: str = "forward",
        stepwise_criterion: str = "aic",
        max_features: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化线性筛选器配置。

        Parameters
        ----------
        enable_corr_filter : bool
            是否启用相关性去重阶段。
        corr_thr : float
            相关性去重阈值。
        corr_method : str
            相关性计算方法。
        enable_vif_filter : bool
            是否启用 VIF 筛查阶段。
        vif_threshold : float
            VIF 阈值。
        enable_stepwise : bool
            是否启用逐步回归阶段。
        stepwise_direction : str
            逐步回归方向。
        stepwise_criterion : str
            逐步回归优化准则。
        max_features : int | None
            最终保留特征数上限。
        n_jobs : int
            并行任务数量。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        super().__init__()
        self.enable_corr_filter = bool(enable_corr_filter)
        self.corr_thr = float(corr_thr)
        self.corr_method = str(corr_method).lower()
        self.enable_vif_filter = bool(enable_vif_filter)
        self.vif_threshold = float(vif_threshold)
        self.enable_stepwise = bool(enable_stepwise)
        self.stepwise_direction = str(stepwise_direction).lower()
        self.stepwise_criterion = str(stepwise_criterion).lower()
        self.max_features = max_features
        self.n_jobs = int(n_jobs)

        if self.stepwise_direction not in {"forward", "backward", "both"}:
            raise ValueError("stepwise_direction must be one of {'forward', 'backward', 'both'}.")
        if self.stepwise_criterion not in {"aic", "bic"}:
            raise ValueError("stepwise_criterion must be one of {'aic', 'bic'}.")

        self.coef_table_: pd.DataFrame = pd.DataFrame()
        self.vif_table_: pd.DataFrame = pd.DataFrame()
        self.stepwise_history_: pd.DataFrame = pd.DataFrame()

    def _prepare_xy(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any,
        features: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """将输入表转换为干净的数值建模矩阵。"""
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

        if y is None:
            raise ValueError("MarsLinearSelector.fit requires `y`.")

        target_col = "__mars_target__"
        df[target_col] = np.asarray(y)
        candidate_features = list(features) if features is not None else [
            feature for feature in df.columns if feature != target_col
        ]
        numeric_data: dict[str, pd.Series] = {}
        for feature in candidate_features:
            series = pd.to_numeric(df[feature], errors="coerce")
            if series.notna().sum() == 0:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="precheck",
                    reason="non_numeric",
                    desc="Feature cannot be converted to numeric values.",
                )
                continue
            numeric_data[feature] = series

        target_series = pd.to_numeric(df[target_col], errors="coerce")
        clean = pd.DataFrame(numeric_data)
        clean[target_col] = target_series
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if clean.empty:
            raise ValueError("No complete numeric rows are available for MarsLinearSelector.")
        if clean[target_col].nunique() < 2:
            raise ValueError("MarsLinearSelector requires a binary target with both classes present.")

        features = [feature for feature in candidate_features if feature in clean.columns]
        return clean.loc[:, features], clean[target_col].astype(int), features

    @staticmethod
    def _target_strength(X: pd.DataFrame, y: pd.Series, features: Sequence[str]) -> dict[str, float]:
        """按特征与目标的一元绝对关联强度生成排序分值。"""
        strengths: dict[str, float] = {}
        for feature in features:
            corr = pd.Series(X[feature]).corr(y, method="spearman")
            strengths[feature] = 0.0 if pd.isna(corr) else float(abs(corr))
        return strengths

    def _apply_corr_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """在高度相关的特征对中剔除一侧特征。"""
        if not self.enable_corr_filter or len(features) <= 1:
            return list(features)

        corr = X.loc[:, features].corr(method=self.corr_method).abs()
        strengths = self._target_strength(X, y, features)
        dropped: set[str] = set()
        for left_idx, left_feature in enumerate(features):
            if left_feature in dropped:
                continue
            for right_feature in features[left_idx + 1 :]:
                if right_feature in dropped:
                    continue
                value = float(corr.loc[left_feature, right_feature])
                if pd.isna(value) or value < self.corr_thr:
                    continue
                drop_feature = (
                    right_feature
                    if strengths[left_feature] >= strengths[right_feature]
                    else left_feature
                )
                dropped.add(drop_feature)
                self._register_decision(
                    drop_feature,
                    status="Dropped",
                    stage="corr",
                    reason=f"corr_with_{left_feature if drop_feature == right_feature else right_feature}",
                    value=value,
                    desc=f"Absolute {self.corr_method} correlation exceeded {self.corr_thr:.4f}.",
                )
                if drop_feature == left_feature:
                    break
        return [feature for feature in features if feature not in dropped]

    @staticmethod
    def _compute_vif_table(X: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """计算当前候选特征集合的 VIF 表。"""
        if not features:
            return pd.DataFrame(columns=["feature", "vif"])
        if len(features) == 1:
            return pd.DataFrame([{"feature": str(features[0]), "vif": 1.0}])

        vif_module = require_optional_module("statsmodels.stats.outliers_influence")
        variance_inflation_factor = vif_module.variance_inflation_factor
        values = X.loc[:, list(features)].astype(float).to_numpy()
        rows = []
        for idx, feature in enumerate(features):
            try:
                vif_value = float(variance_inflation_factor(values, idx))
            except Exception:
                vif_value = float("inf")
            rows.append({"feature": str(feature), "vif": vif_value})
        return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)

    def _apply_vif_filter(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        """迭代剔除 VIF 最高且超过阈值的特征。"""
        if not self.enable_vif_filter or len(features) <= 1:
            self.vif_table_ = self._compute_vif_table(X, features)
            return list(features)

        remaining = list(features)
        while len(remaining) > 1:
            vif_table = self._compute_vif_table(X, remaining)
            max_row = vif_table.iloc[0]
            max_vif = float(max_row["vif"])
            if max_vif <= self.vif_threshold:
                self.vif_table_ = vif_table
                return remaining
            drop_feature = str(max_row["feature"])
            remaining.remove(drop_feature)
            self._register_decision(
                drop_feature,
                status="Dropped",
                stage="vif",
                reason="high_vif",
                value=max_vif,
                desc=f"VIF exceeded {self.vif_threshold:.4f}.",
            )
        self.vif_table_ = self._compute_vif_table(X, remaining)
        return remaining

    def _fit_logit_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: Sequence[str],
    ) -> tuple[float, Any | None]:
        """拟合 statsmodels Logit 并返回当前配置的信息准则值。"""
        sm = require_optional_module("statsmodels.api")
        design = X.loc[:, list(features)] if features else pd.DataFrame(index=X.index)
        design = sm.add_constant(design, has_constant="add")
        try:
            result = sm.Logit(y, design).fit(disp=False, maxiter=200)
        except Exception:
            return float("inf"), None
        return float(getattr(result, self.stepwise_criterion)), result

    def _apply_stepwise(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """执行 forward、backward 或双向 AIC/BIC 逐步筛选。"""
        if not self.enable_stepwise or not features:
            return list(features)

        history: list[dict[str, Any]] = []

        def record(action: str, feature: str | None, score: float, selected: Sequence[str]) -> None:
            """记录逐步回归每一步的特征集合与信息准则分数。"""
            history.append(
                {
                    "action": action,
                    "feature": feature,
                    "criterion": self.stepwise_criterion,
                    "score": score,
                    "n_features": len(selected),
                    "selected_features": json.dumps(list(selected), ensure_ascii=False),
                }
            )

        if self.stepwise_direction == "backward":
            selected = list(features)
        else:
            selected = []

        current_score, current_result = self._fit_logit_score(X, y, selected)
        record("start", None, current_score, selected)

        def try_add() -> bool:
            """尝试加入一个能继续降低信息准则的候选特征。"""
            nonlocal current_score, current_result, selected
            remaining = [feature for feature in features if feature not in selected]
            if self.max_features is not None and len(selected) >= int(self.max_features):
                return False
            candidates = []
            for feature in remaining:
                score, result = self._fit_logit_score(X, y, [*selected, feature])
                candidates.append((score, feature, result))
            if not candidates:
                return False
            best_score, best_feature, best_result = min(candidates, key=lambda item: item[0])
            if best_score + 1e-9 >= current_score:
                return False
            selected.append(best_feature)
            current_score = best_score
            current_result = best_result
            record("add", best_feature, current_score, selected)
            return True

        def try_drop() -> bool:
            """尝试移除一个能继续降低信息准则的已选特征。"""
            nonlocal current_score, current_result, selected
            if len(selected) <= 1:
                return False
            candidates = []
            for feature in selected:
                trial_features = [item for item in selected if item != feature]
                score, result = self._fit_logit_score(X, y, trial_features)
                candidates.append((score, feature, result, trial_features))
            best_score, best_feature, best_result, best_features = min(
                candidates,
                key=lambda item: item[0],
            )
            if best_score + 1e-9 >= current_score:
                return False
            selected = list(best_features)
            current_score = best_score
            current_result = best_result
            record("drop", best_feature, current_score, selected)
            return True

        if self.stepwise_direction == "forward":
            while try_add():
                pass
        elif self.stepwise_direction == "backward":
            while try_drop():
                pass
        else:
            changed = True
            while changed:
                changed = try_add()
                while try_drop():
                    changed = True

        self.stepwise_history_ = pd.DataFrame(history)
        selected_set = set(selected)
        for feature in features:
            if feature not in selected_set:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="stepwise",
                    reason=f"not_selected_by_{self.stepwise_criterion}",
                    desc=f"Excluded by {self.stepwise_direction} stepwise regression.",
                )
        if current_result is not None and selected:
            params = current_result.params.reindex(["const", *selected])
            pvalues = current_result.pvalues.reindex(["const", *selected])
            stderr = current_result.bse.reindex(["const", *selected])
            self.coef_table_ = pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "coefficient": float(params.get(feature, np.nan)),
                        "abs_coefficient": abs(float(params.get(feature, np.nan))),
                        "p_value": float(pvalues.get(feature, np.nan)),
                        "std_err": float(stderr.get(feature, np.nan)),
                    }
                    for feature in selected
                ]
            )
        else:
            self.coef_table_ = pd.DataFrame(
                columns=["feature", "coefficient", "abs_coefficient", "p_value", "std_err"]
            )
        return selected

    def _apply_max_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> list[str]:
        """在未启用 stepwise 或保留过多特征时应用最终 Top-N 限制。"""
        if self.max_features is None or len(features) <= int(self.max_features):
            return list(features)
        strengths = self._target_strength(X, y, features)
        ranked = sorted(features, key=lambda feature: strengths.get(feature, 0.0), reverse=True)
        selected = ranked[: int(self.max_features)]
        selected_set = set(selected)
        for feature in features:
            if feature not in selected_set:
                self._register_decision(
                    feature,
                    status="Dropped",
                    stage="max_features",
                    reason="rank_cap",
                    value=float(strengths.get(feature, 0.0)),
                    desc=f"Feature rank exceeded max_features={self.max_features}.",
                )
        return [feature for feature in features if feature in selected_set]

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any,
        *,
        features: Sequence[str] | None = None,
    ) -> MarsLinearSelector:
        """
        执行相关性、VIF 与可选 stepwise 线性特征筛选。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征表。
        y : Any
            二分类目标数组。
        features : Sequence[str] | None
            本次参与筛选的特征列；不传时使用输入表中的全部候选列。

        Returns
        -------
        MarsLinearSelector
            已拟合的线性筛选器实例。

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> selector = MarsLinearSelector().fit(X, y)
        >>> selector.selected_features_
        ['age']
        """
        self.report_records_ = []
        X_numeric, target_series, features = self._prepare_xy(X, y, features)
        self.n_features_in_ = len(features)

        selected = self._apply_corr_filter(X_numeric, target_series, features)
        if self.enable_corr_filter:
            corr_frame = X_numeric.loc[:, features].corr(method=self.corr_method).abs()
            for feature in selected:
                other_features = [item for item in features if item != feature]
                max_corr = (
                    float(corr_frame.loc[feature, other_features].max())
                    if other_features
                    else 0.0
                )
                self._register_decision(
                    feature,
                    status="Checked",
                    stage="corr",
                    reason="within_threshold",
                    value=max_corr,
                    desc=f"Maximum absolute {self.corr_method} correlation stayed below threshold.",
                )

        selected = self._apply_vif_filter(X_numeric, selected)
        if self.enable_vif_filter and not self.vif_table_.empty:
            selected_set = set(selected)
            for row in self.vif_table_.to_dict("records"):
                feature = str(row["feature"])
                if feature not in selected_set:
                    continue
                self._register_decision(
                    feature,
                    status="Checked",
                    stage="vif",
                    reason="within_threshold",
                    value=float(row["vif"]),
                    desc=f"VIF stayed below {self.vif_threshold:.4f}.",
                )

        selected = self._apply_stepwise(X_numeric, target_series, selected)
        if self.enable_stepwise:
            for feature in selected:
                self._register_decision(
                    feature,
                    status="Selected",
                    stage="stepwise",
                    reason=f"selected_by_{self.stepwise_criterion}",
                    desc=f"Retained by {self.stepwise_direction} stepwise regression.",
                )

        selected = self._apply_max_features(X_numeric, target_series, selected)

        self.selected_features_ = [feature for feature in features if feature in set(selected)]
        for feature in self.selected_features_:
            self._register_decision(
                feature,
                status="Selected",
                stage="final",
                reason="kept",
                desc="Feature survived linear selector filters.",
            )

        if self.coef_table_.empty and self.selected_features_:
            _, result = self._fit_logit_score(X_numeric, target_series, self.selected_features_)
            if result is not None:
                params = result.params.reindex(["const", *self.selected_features_])
                self.coef_table_ = pd.DataFrame(
                    [
                        {
                            "feature": feature,
                            "coefficient": float(params.get(feature, np.nan)),
                            "abs_coefficient": abs(float(params.get(feature, np.nan))),
                            "p_value": float(result.pvalues.get(feature, np.nan)),
                            "std_err": float(result.bse.get(feature, np.nan)),
                        }
                        for feature in self.selected_features_
                    ]
                )

        self._is_fitted = True
        return self


class MarsImportanceSelector(MarsBaseSelector):
    """
    基于模型重要性或 SHAP 的特征筛选器。

    该选择器支持直接消费已有 importance table，也可以训练 sklearn/树模型
    读取 ``feature_importances_`` 或 ``coef_``。当 ``method="shap"`` 时，
    选择器计算 mean absolute SHAP value 并统一输出 MARS importance table。

    Attributes
    ----------
    selected_features_ : list of str
        最终入选特征。
    importance_table_ : pandas.DataFrame
        标准化后的重要性表。
    estimator_ : Any or None
        由选择器训练得到的 estimator；使用外部 importance table 时为 ``None``。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> importance = pd.DataFrame({"feature": ["age"], "importance": [1.0]})
    >>> selector = MarsImportanceSelector()
    >>> selector.fit(df[["age"]], df["y"], features=["age"], importance_table=importance).selected_features_
    ['age']
    """

    def __init__(
        self,
        estimator: Union[str, Any] = "lgbm",
        estimator_params: dict | None = None,
        method: Literal["importance", "shap", "rfe", "sfm"] = "importance",
        selection_mode: Literal["top_k", "threshold", "percentile"] = "top_k",
        selection_threshold: Union[int, float, str] = 50,
        cv: int = 3,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        """
        初始化重要性筛选器配置。

        Parameters
        ----------
        estimator : Union[str, Any]
            底层模型类型或实例。
        estimator_params : dict | None
            底层模型初始化参数。
        method : Literal['importance', 'shap', 'rfe', 'sfm']
            重要性筛选策略。
        selection_mode : Literal['top_k', 'threshold', 'percentile']
            特征保留模式。
        selection_threshold : Union[int, float, str]
            对应筛选模式下的阈值。
        cv : int
            交叉验证折数。
        n_jobs : int
            并行任务数量。
        random_state : int
            随机种子。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        super().__init__()
        self.estimator = estimator
        self.estimator_params = dict(estimator_params or {})
        self.method = str(method).lower()
        self.selection_mode = str(selection_mode).lower()
        self.selection_threshold = selection_threshold
        self.cv = int(cv)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

        if self.method not in {"importance", "shap", "rfe", "sfm"}:
            raise ValueError("method must be one of {'importance', 'shap', 'rfe', 'sfm'}.")
        if self.selection_mode not in {"top_k", "threshold", "percentile"}:
            raise ValueError("selection_mode must be one of {'top_k', 'threshold', 'percentile'}.")

        self.importance_table_: pd.DataFrame = pd.DataFrame()
        self.estimator_: Any | None = None

    def _prepare_xy(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any | None,
        features: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
        """将输入数据转为 Pandas，并解析目标列与候选特征。"""
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(X)!r}.")

        raw_features = list(features) if features is not None else list(df.columns)
        if y is None:
            return df.loc[:, raw_features], None, raw_features

        target_series = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce")
        valid_mask = target_series.notna().to_numpy()
        if int(valid_mask.sum()) == 0:
            raise ValueError("Target contains no valid numeric labels.")
        return df.loc[valid_mask, raw_features], target_series.loc[valid_mask].astype(int), raw_features

    @staticmethod
    def _encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """编码混合类型特征，并保留编码列到原始特征的映射。"""
        encoded_parts: list[pd.DataFrame] = []
        mapping: dict[str, str] = {}
        for feature in X.columns:
            series = X[feature]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                encoded_col = pd.to_numeric(series, errors="coerce")
                fill_value = encoded_col.median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                encoded_parts.append(pd.DataFrame({feature: encoded_col.fillna(fill_value)}))
                mapping[feature] = feature
                continue

            dummies = pd.get_dummies(
                series.astype("string").fillna("__MISSING__"),
                prefix=feature,
                prefix_sep="__",
                dtype=float,
            )
            encoded_parts.append(dummies)
            for encoded_feature in dummies.columns:
                mapping[str(encoded_feature)] = feature

        if not encoded_parts:
            raise ValueError("At least one feature is required for MarsImportanceSelector.")
        encoded = pd.concat(encoded_parts, axis=1)
        return encoded.astype(float), mapping

    def _build_estimator(self) -> Any:
        """实例化受支持的 estimator，或复制用户传入的 estimator 对象。"""
        if not isinstance(self.estimator, str):
            return copy.deepcopy(self.estimator)

        estimator_name = self.estimator.lower()
        params = dict(self.estimator_params)
        if estimator_name in {"rf", "random_forest", "randomforest"}:
            ensemble = require_optional_module("sklearn.ensemble")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            return ensemble.RandomForestClassifier(**params)
        if estimator_name in {"extra_trees", "extratrees", "et"}:
            ensemble = require_optional_module("sklearn.ensemble")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            return ensemble.ExtraTreesClassifier(**params)
        if estimator_name in {"lr", "logit", "logistic", "logistic_regression"}:
            linear = require_optional_module("sklearn.linear_model")
            params.setdefault("solver", "liblinear")
            params.setdefault("random_state", self.random_state)
            return linear.LogisticRegression(**params)
        if estimator_name in {"lgb", "lgbm", "lightgbm"}:
            lgb = require_optional_module("lightgbm")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            params.setdefault("verbosity", -1)
            return lgb.LGBMClassifier(**params)
        if estimator_name in {"xgb", "xgboost"}:
            xgb = require_optional_module("xgboost")
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            params.setdefault("eval_metric", "logloss")
            return xgb.XGBClassifier(**params)
        if estimator_name in {"cat", "catboost", "cbt"}:
            catboost = require_optional_module("catboost")
            params.setdefault("iterations", 100)
            params.setdefault("random_seed", self.random_state)
            params.setdefault("verbose", False)
            return catboost.CatBoostClassifier(**params)
        raise ValueError(
            "Unsupported estimator. Expected one of "
            "{'rf', 'extra_trees', 'lr', 'lgbm', 'xgb', 'cat'} or an estimator object."
        )

    @staticmethod
    def _aggregate_importance(
        encoded_features: Sequence[str],
        values: Sequence[float],
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> dict[str, float]:
        """将编码列级别的重要性聚合回原始特征名。"""
        importance_map = {feature: 0.0 for feature in raw_features}
        for encoded_feature, value in zip(encoded_features, values, strict=False):
            raw_feature = mapping.get(str(encoded_feature), str(encoded_feature))
            if raw_feature in importance_map:
                importance_map[raw_feature] += float(value)
        return importance_map

    def _build_importance_table(
        self,
        importance_map: Mapping[str, float],
        importance_type: str,
    ) -> pd.DataFrame:
        """将重要性映射标准化为 MARS importance table 结构。"""
        rows = [
            {
                "feature": feature,
                "importance": float(importance),
                "importance_type": importance_type,
                "model_type": str(
                    self.estimator
                    if isinstance(self.estimator, str)
                    else type(self.estimator).__name__
                ),
            }
            for feature, importance in importance_map.items()
        ]
        table = pd.DataFrame(rows)
        table = table.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1, dtype=int)
        return table[["feature", "importance", "importance_type", "model_type", "rank"]]

    def _importance_from_estimator(
        self,
        estimator: Any,
        X_encoded: pd.DataFrame,
        y: pd.Series,
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """拟合 estimator 并提取内置特征重要性或系数。"""
        estimator.fit(X_encoded, y)
        self.estimator_ = estimator
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
            importance_type = "feature_importance"
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.ravel(estimator.coef_)).astype(float)
            importance_type = "abs_coef"
        else:
            raise ValueError(
                "Estimator must expose feature_importances_ or coef_ for method='importance'."
            )
        importance_map = self._aggregate_importance(
            list(X_encoded.columns),
            values,
            mapping,
            raw_features,
        )
        return self._build_importance_table(importance_map, importance_type)

    def _importance_from_shap(
        self,
        estimator: Any,
        X_encoded: pd.DataFrame,
        y: pd.Series,
        mapping: Mapping[str, str],
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """拟合 estimator 并计算 mean absolute SHAP value。"""
        shap = require_optional_module("shap")
        estimator.fit(X_encoded, y)
        self.estimator_ = estimator
        sample = X_encoded.head(min(len(X_encoded), 300))

        try:
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(sample)
        except Exception:
            explainer = shap.Explainer(estimator.predict_proba, sample)
            explanation = explainer(sample)
            values = getattr(explanation, "values", explanation)

        if isinstance(values, list):
            values_arr = np.asarray(values[-1])
        else:
            values_arr = np.asarray(values)
        if values_arr.ndim == 3:
            values_arr = values_arr[:, :, -1]
        mean_abs = np.abs(values_arr).mean(axis=0)
        importance_map = self._aggregate_importance(
            list(X_encoded.columns),
            mean_abs,
            mapping,
            raw_features,
        )
        return self._build_importance_table(importance_map, "mean_abs_shap")

    @staticmethod
    def _normalize_importance_table(
        table: pd.DataFrame | pl.DataFrame,
        raw_features: Sequence[str],
    ) -> pd.DataFrame:
        """校验并标准化用户传入的重要性表。"""
        table_pd = table.to_pandas() if isinstance(table, pl.DataFrame) else table.copy()
        if "feature" not in table_pd.columns or "importance" not in table_pd.columns:
            raise ValueError("importance_table must contain 'feature' and 'importance' columns.")
        table_pd["feature"] = table_pd["feature"].astype(str)
        table_pd["importance"] = pd.to_numeric(table_pd["importance"], errors="coerce").fillna(0.0)
        table_pd = table_pd[table_pd["feature"].isin(set(raw_features))].copy()
        if "importance_type" not in table_pd.columns:
            table_pd["importance_type"] = "provided"
        if "model_type" not in table_pd.columns:
            table_pd["model_type"] = "provided"
        table_pd = table_pd.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
        table_pd["rank"] = np.arange(1, len(table_pd) + 1, dtype=int)
        return table_pd[["feature", "importance", "importance_type", "model_type", "rank"]]

    def _select_features(self, table: pd.DataFrame) -> list[str]:
        """按 top-k、绝对阈值或百分位阈值选择特征。"""
        if table.empty:
            return []
        if self.selection_mode == "top_k":
            k = max(int(float(self.selection_threshold)), 0)
            return table.head(k)["feature"].astype(str).tolist()
        if self.selection_mode == "threshold":
            threshold = float(self.selection_threshold)
            return table.loc[table["importance"] >= threshold, "feature"].astype(str).tolist()

        raw_threshold = self.selection_threshold
        if isinstance(raw_threshold, str) and raw_threshold.endswith("%"):
            percentile = float(raw_threshold.rstrip("%")) / 100.0
        else:
            threshold_value = float(raw_threshold)
            percentile = threshold_value / 100.0 if threshold_value > 1 else threshold_value
        percentile = min(max(percentile, 0.0), 1.0)
        keep_count = int(np.ceil(len(table) * percentile))
        return table.head(keep_count)["feature"].astype(str).tolist()

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any | None = None,
        *,
        features: Sequence[str] | None = None,
        importance_table: pd.DataFrame | pl.DataFrame | None = None,
    ) -> MarsImportanceSelector:
        """
        执行基于模型重要性或 SHAP 的特征筛选。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征表。若未显式传入 ``y``，则必须包含目标列。
        y : Any | None
            二分类目标数组。
        features : Sequence[str] | None
            本次参与筛选的特征列；不传时使用输入表中的全部候选列。
        importance_table : pd.DataFrame | pl.DataFrame | None
            预先计算好的重要性表，需包含 ``feature`` 与 ``importance`` 列。

        Returns
        -------
        MarsImportanceSelector
            已拟合的重要性筛选器实例。

        Raises
        ------
        NotImplementedError
            当当前选项尚未实现时抛出。
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
        >>> importance = pd.DataFrame({"feature": ["age"], "importance": [1.0]})
        >>> selector = MarsImportanceSelector().fit(X, importance_table=importance)
        >>> selector.selected_features_
        ['age']
        """
        if self.method in {"rfe", "sfm"}:
            raise NotImplementedError(
                f"MarsImportanceSelector method={self.method!r} is not implemented in v1."
            )

        self.report_records_ = []
        X_pd, y_series, raw_features = self._prepare_xy(X, y, features)
        self.n_features_in_ = len(raw_features)

        provided_table = importance_table
        if provided_table is not None:
            table = self._normalize_importance_table(provided_table, raw_features)
        else:
            if y_series is None:
                raise ValueError("MarsImportanceSelector.fit requires y when `importance_table` is not provided.")
            X_encoded, mapping = self._encode_features(X_pd)
            estimator = self._build_estimator()
            if self.method == "importance":
                table = self._importance_from_estimator(
                    estimator,
                    X_encoded,
                    y_series,
                    mapping,
                    raw_features,
                )
            else:
                table = self._importance_from_shap(
                    estimator,
                    X_encoded,
                    y_series,
                    mapping,
                    raw_features,
                )

        selected = self._select_features(table)
        selected_set = set(selected)
        self.importance_table_ = table.copy()
        self.selected_features_ = [feature for feature in raw_features if feature in selected_set]

        importance_lookup = dict(zip(table["feature"], table["importance"], strict=False))
        for feature in raw_features:
            status = "Selected" if feature in selected_set else "Dropped"
            reason = self.selection_mode if feature in selected_set else f"below_{self.selection_mode}"
            self._register_decision(
                feature,
                status=status,
                stage=self.method,
                reason=reason,
                value=float(importance_lookup.get(feature, 0.0)),
                desc="Feature selection based on normalized importance table.",
            )

        self._is_fitted = True
        return self
