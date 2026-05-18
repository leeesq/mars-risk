from __future__ import annotations
import os
import json
from typing import List, Optional, Union, Any, Literal, Dict, Tuple, TYPE_CHECKING
import polars as pl
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from mars.core.base import MarsBaseSelector
from mars.feature.binner import MarsBinnerBase, MarsNativeBinner, MarsOptimalBinner
from mars.analysis.report import MarsEvaluationReport
from mars.utils.logger import logger
from mars.utils.decorators import time_it

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsBinEvaluator


class MarsStatsSelector(MarsBaseSelector):
    """
    全流程自动化特征筛选评估器。

    该组件采用多阶段过滤架构，通过连续的条件测试对特征空间进行降维，
    提取具备高区分度、高稳定性且符合业务逻辑单调性的特征子集。

    核心过滤维度包括数据质量校验、轻量级分布探查、精确区间区分度计算、
    群体分布偏移评估、风险逻辑一致性检验以及共线性降维。

    Parameters
    ----------
    target : str
        目标变量列名。
    features : list of str, optional
        特征候选池。若为 None，将自动扫描输入数据集中剔除目标列与时间（或分组）列后的全量特征。
    feature_data_source : dict of str to list of str, optional
        特征到数据源标签的映射。
    time_col : str, optional
        时间切片列名。用于界定特征表现的时间序列，支撑跨期稳定性与风险逻辑一致性的测算。
    profile_by : str, optional
        稳定性探查的分组聚合维度。配置此参数将激活截面分布稳定性 (PSI) 及风险逻辑一致性 (RiskCorr) 的校验。
        该参数支持两类映射形态的入参：
        - 数据集内已存在的物理列名（如 'customer_segment', 'vintage'），用于执行基于特定客群或静态批次的截面分析；
        - 时间粒度聚合指令（如 'month', 'week', 'day' 及自定义滚动周期的 '7d', '14d' 等），用于执行时间序列分析。
        - 当使用时间粒度指令时，必须协同配置 `time_col` 参数提供原始时间基准，底层引擎将基于该时间戳列自动完成时间窗口的动态截断与衍生列生成。
        - 若仅配置了 `time_col` 参数而未显式声明本参数，系统将默认采用 'month' 级别的时间跨期聚合策略。
    feature_start_aware_baseline : bool, default False
        是否按特征首次出现时点感知稳定性基准。
    missing_thr : float, default 0.90
        缺失率控制阈值。超过该阈值的特征将在数据质量校验阶段被剔除。
    zeros_thr : float, default 0.90
        零值率控制阈值，仅对数值型特征生效。
    mode_thr : float, default 0.90
        众数占比控制阈值，用于过滤低方差的常量特征。
    iv_thr : float, default 0.01
        基于最优分箱器评估的信息值 (IV) 准入下限。
    lift_thr : float, optional, default 1.2
        基于最优分箱器评估的单分箱提升度 (Lift) 召回下限阈值。
    min_sample_rate : float, default 0.05
        触发最优分箱评估提升度召回机制的最小物理箱占比前提。
    psi_thr : float, optional, default 0.25
        群体稳定性指标 (PSI) 容忍上限。超过该值的特征将被判定为发生分布漂移 (Data Drift)。
    rc_thr : float, optional, default 0.5
        风险逻辑一致性相关系数 (RiskCorr) 下限，用于防范特征分箱的违约率排序逻辑随时间发生跨期翻转。
    corr_thr : float, optional, default 0.95
        皮尔逊相关系数上限。用于执行基于信息值的贪心共线性去重。
    skip_rough_scan : bool, default False
        控制是否跳过基于原生分箱器 (`MarsNativeBinner`) 的特征筛选阶段。
    skip_fine_scan : bool, default False
        控制是否跳过基于最优分箱器 (`MarsOptimalBinner`) 的特征筛选阶段。
    rough_iv_thr : float, default 0.01
        基于原生分箱器评估的信息值 (IV) 准入下限。
    rough_lift_thr : float, default 1.2
        基于原生分箱器评估的单分箱提升度 (Lift) 召回下限阈值。
    rough_min_sample_rate : float, default 0.02
        触发原生分箱评估提升度召回机制的最小物理箱占比前提。
    white_list : list of str, optional
        白名单特征列表。列表内的特征将无视各项统计与稳定性约束，强制保留至结果集。
    black_list : list of str, optional
        黑名单特征列表。列表内的特征将在评估管道启动前被物理剥离。
    missing_values : list, optional
        领域自定义缺失值定义（如 [-999, 'unknown']）。将合并计入数据质量模块的缺失率统计，并分配至独立的缺失值箱。
    special_values : list, optional
        领域自定义特殊值定义。底层分箱引擎将对此类数值执行强制物理隔离，不参与正常区间切分。
    binning_params : dict, optional
        透传至最优分箱器 (`MarsOptimalBinner`) 的初始化超参数字典。
    rough_binning_params : dict, optional
        透传至原生分箱器 (`MarsNativeBinner`) 的初始化超参数字典。
    max_samples : int, optional
        全局随机采样上限，用于控制超大规模数据集下的运算时间边界。
    batch_size : int, optional, default 100
        向量化计算执行时的特征列并发分块大小，防范底层查询优化器 (Query Planner) 解析耗时爆炸。
    n_jobs : int, default -1
        并行计算的分配核心数限制。
        
    Attributes
    ----------
    selected_features_ : list of str
        最终通过所有过滤阶段保留的入模特征列表（包含匹配成功的白名单特征）。
    report_records_ : list of dict
        特征级评估决策明细记录。包含每个特征在评估管线中的最终状态（保留或截断）、触发该决策的逻辑节点、决策原因及决定性度量指标。
    _funnel_stats : list of dict
        阶段级评估漏斗统计快照。记录评估管线中各节点的输入数量、截断数量、留存绝对值及累积留存率。
    _stage3_binner : MarsBinnerBase
        承载最终入选特征分箱映射规则的基础分箱器实例。可调用该实例对未见数据执行区间离散化转换，或导出模型部署脚本。

    Notes
    -----
    评估器内部持有了最优分箱规则状态。在调用 `fit` 完成特征筛选后，系统会自动触发 
    `prune` 方法裁剪冗余的非入模特征状态，以收敛序列化后的模型体积。
    """
    def __init__(
        self,
        *,
        target: str,
        features: Optional[List[str]] = None,        
        feature_data_source: Optional[Dict[str, List[str]]] = None,
        time_col: Optional[str] = None,
        profile_by: Optional[str] = None, 
        feature_start_aware_baseline: bool = False,
        
        missing_thr: float = 0.90,
        zeros_thr: float = 0.90,  
        mode_thr: float = 0.90,
        
        iv_thr: float = 0.01, 
        lift_thr: Optional[float] = 1.2,      
        min_sample_rate: float = 0.05,               
        
        psi_thr: Optional[float] = 0.25,      
        rc_thr: Optional[float] = 0.5,           
        corr_thr: Optional[float] = 0.95,    

        skip_rough_scan: bool = False,
        skip_fine_scan: bool = False, 
        rough_iv_thr: float = 0.01,
        rough_lift_thr: float = 1.2,
        rough_min_sample_rate: float = 0.02,  
        
        white_list: Optional[List[str]] = None,
        black_list: Optional[List[str]] = None,

        missing_values: Optional[List[Any]] = None,   
        special_values: Optional[List[Any]] = None,  
        
        binning_params: Optional[Dict[str, Any]] = None,    
        rough_binning_params: Optional[Dict[str, Any]] = None,  

        max_samples: Optional[int] = None,
        batch_size: Optional[int] = 100,    
        n_jobs: int = -1
    ):
        """
        初始化统计筛选器配置。

        Parameters
        ----------
        target : str
            目标变量列名。
        features : list of str, optional
            候选特征列表。
        feature_data_source : dict of str to list of str, optional
            特征到数据源标签的映射。
        time_col : str, optional
            时间列名。
        profile_by : str, optional
            分组或时间聚合维度。
        feature_start_aware_baseline : bool, default False
            是否按特征首次出现时点感知稳定性基准。
        missing_thr : float, default 0.90
            缺失率阈值。
        zeros_thr : float, default 0.90
            零值率阈值。
        mode_thr : float, default 0.90
            众数占比阈值。
        iv_thr : float, default 0.01
            精筛阶段 IV 阈值。
        lift_thr : float, optional, default 1.2
            精筛阶段 Lift 阈值。
        min_sample_rate : float, default 0.05
            精筛阶段最小样本占比。
        psi_thr : float, optional, default 0.25
            PSI 阈值。
        rc_thr : float, optional, default 0.5
            风险相关性阈值。
        corr_thr : float, optional, default 0.95
            相关性去重阈值。
        skip_rough_scan : bool, default False
            是否跳过粗筛。
        skip_fine_scan : bool, default False
            是否跳过精筛。
        rough_iv_thr : float, default 0.01
            粗筛阶段 IV 阈值。
        rough_lift_thr : float, default 1.2
            粗筛阶段 Lift 阈值。
        rough_min_sample_rate : float, default 0.02
            粗筛阶段最小样本占比。
        white_list : list of str, optional
            白名单特征列表。
        black_list : list of str, optional
            黑名单特征列表。
        missing_values : list, optional
            自定义缺失值集合。
        special_values : list, optional
            自定义特殊值集合。
        binning_params : dict, optional
            精筛分箱参数。
        rough_binning_params : dict, optional
            粗筛分箱参数。
        max_samples : int, optional
            抽样样本上限。
        batch_size : int, optional, default 100
            批处理大小。
        n_jobs : int, default -1
            并行任务数量。
        """
        super().__init__(target=target)
        
        self.features = features
        self.feature_data_source = feature_data_source or {}
        self.time_col = time_col
        self.profile_by = profile_by
        self.feature_start_aware_baseline = feature_start_aware_baseline
        self.white_list = white_list if white_list else []
        self.black_list = black_list if black_list else []
        
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
        
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self._rough_binner: Optional[MarsNativeBinner] = None 
        self._stage3_binner: Optional[MarsBinnerBase] = None
        self._feature_iv_dict: Dict[str, float] = {}  
        self._feature_source_map: Dict[str, str] = {}
        
        self._funnel_stats = []

    @time_it
    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "MarsStatsSelector":
        """
        触发自动化特征筛选流程。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据集上下文。
        y : array-like, optional
            目标变量数组，默认通过内部解析目标列获取。

        Returns
        -------
        MarsStatsSelector
            完成拟合与裁剪的自身实例。

        Raises
        ------
        ValueError
            当粗筛和精筛同时被禁用时抛出。
        """
        # 拦截互斥的配置项
        if self.skip_rough_scan and self.skip_fine_scan:
            raise ValueError("Cannot skip both rough scan and fine scan. At least one binning stage is required.")

        X = self._ensure_polars_dataframe(X)
        self._funnel_stats = [] 
        self._feature_iv_dict = {} 
        
        # 初始化候选特征空间，剥离非特征维度
        exclude_cols = {self.target}
        if self.time_col: exclude_cols.add(self.time_col)
        if self.profile_by: exclude_cols.add(self.profile_by)
        
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
        self.selected_features_ = list(set(current_features + valid_white_list))
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
        return self._feature_source_map.get(feature, "UNMAPPED")

    def _register_feature_decision(
        self,
        feature: str,
        status: str,
        stage: str,
        reason: str = "",
        value: float = -1.0,
        desc: str = "",
    ) -> None:
        self._register_decision(
            feature,
            status,
            stage,
            reason,
            value,
            desc,
            data_source=self._feature_source_for(feature),
        )
    
    def _record_funnel(self, stage: str, description: str, thresholds: Union[Dict, str], count_before: int, count_after: int):
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
    
    def show_summary(self):
        """
        展示特征筛选漏斗摘要。

        Notes
        -----
        在 Notebook 环境中优先返回富样式表格；若环境不支持，则退化为日志打印。
        """
        if not self._funnel_stats:
            logger.warning("No funnel stats available. Run .fit() first.")
            return

        df_summary = pd.DataFrame(self._funnel_stats)
        
        def _color_retention(val):
            """为留存率文本生成条件样式。"""
            try:
                p = float(val.rstrip('%'))
                if p < 30: return 'color: #d32f2f; font-weight: bold;'
                if p < 70: return 'color: #ed6c02; font-weight: bold;'
                return 'color: #2e7d32; font-weight: bold;'
            except: return 'color: #757575;'

        def _style_logic_text(v):
            """为阈值逻辑文本生成高亮样式。"""
            if not isinstance(v, str): return ''
            logic_ops = ['&', '|', '<', '>', '=']
            if any(op in v for op in logic_ops):
                return 'color: #7b1fa2; font-weight: bold; font-family: "Courier New", Courier, monospace;'
            return ''

        styler = (
            df_summary.style
            .set_caption("Mars Stats Selector: Feature Selection Funnel")
            .bar(subset=['Dropped'], color='#ffdbdb', vmin=0)
            .bar(subset=['Remaining'], color='#e1f5fe', vmin=0)
            .set_properties(**{'text-align': 'left'}, subset=['Stage', 'Description', 'Thresholds'])
            .set_properties(**{'text-align': 'center'}, 
                            subset=['Input', 'Dropped', 'Remaining', 'Retention %', 'Cumulative %'])
            .map(lambda v: 'color: #d32f2f; font-weight: bold;' if v > 0 else 'color: #999;', subset=['Dropped'])
            .map(lambda v: 'color: #1565c0; font-weight: bold;', subset=['Remaining'])
            .map(_color_retention, subset=['Retention %'])
            .map(lambda v: 'font-weight: bold; color: #1b5e20; background-color: #f1f8e9;', subset=['Cumulative %'])
            .map(_style_logic_text, subset=['Thresholds'])
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#f5f5f5'), ('color', '#333'), ('border-bottom', '2px solid #ddd'),
                    ('padding', '12px'), ('text-align', 'center')
                ]},
                {'selector': 'th.col0, th.col1, th.col2', 'props': [('text-align', 'left')]},
                {'selector': 'caption', 'props': [
                    ('font-size', '18px'), ('font-weight', 'bold'), ('padding', '12px'), ('color', '#1a237e'), ('text-align', 'left')
                ]}
            ])
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

        Notes
        -----
        该方法会清理最终分箱器中缓存的数据引用，并主动触发一次垃圾回收。
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
            df, 
            features=features,
            missing_values=self.missing_values,
            special_values=self.special_values
        )
        report = profiler.generate_profile(
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
            features=features, 
            cat_features=cat_features,
            missing_values=self.missing_values,
            special_values=self.special_values,
            n_jobs=self.n_jobs,
            **self.rough_binning_params
        )
        target = df.get_column(self.target)
        binner.fit(df, target)
        
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

        cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        cat_features = [c for c in features if df.schema[c] in cat_types]

        evaluator = MarsBinEvaluator(
            target=self.target, 
            bining_type="opt", 
            feature_data_source=self.feature_data_source,
            cat_features=cat_features,
            missing_values=self.missing_values, 
            special_values=self.special_values,
            **self.binning_params
        )
        
        report = evaluator.evaluate(
            df, 
            features=features, 
            dt_col=self.time_col, 
            profile_by=self.profile_by,
            batch_size=self.batch_size,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
        )
        
        self._stage3_binner = evaluator.binner

        lift_recall_set = set()
        if self.lift_thr is not None:
            group_col = report.group_col
            lift_cond = (pl.col("lift") > self.lift_thr) & \
                        (pl.col("pct") > self.min_sample_rate)
            
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
        evaluator = MarsBinEvaluator(
            target=self.target,
            feature_data_source=self.feature_data_source,
            binner=self._stage3_binner 
        )
        
        report = evaluator.evaluate(
            df,
            features=features,
            dt_col=self.time_col,
            profile_by=self.profile_by,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
        )
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
        evaluator = MarsBinEvaluator(
            target=self.target,
            feature_data_source=self.feature_data_source,
            binner=self._stage3_binner 
        )
        
        report = evaluator.evaluate(
            df,
            features=features,
            dt_col=self.time_col,
            profile_by=self.profile_by,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
        )
        
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

    def get_eval_report(self, df: Union[pl.DataFrame, pd.DataFrame]) -> Tuple["MarsEvaluationReport", "MarsBinEvaluator"]:
        """
        基于当前筛选结果生成最终评估报告。

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            评估数据上下文环境。

        Returns
        -------
        tuple of (MarsEvaluationReport, MarsBinEvaluator)
            包含汇总统计报表与趋势透视表的报告容器，以及执行规则推演的评估器实例。

        Raises
        ------
        ValueError
            当当前选择器尚未拟合，或没有任何入选特征时抛出。
        """
        self._check_is_fitted()
        
        if not self.selected_features_:
            raise ValueError("No selected features found. Cannot generate report.")
            
        X_pl = self._ensure_polars_dataframe(df)
        
        from mars.analysis.evaluator import MarsBinEvaluator
        
        if self._stage3_binner is not None:
            evaluator = MarsBinEvaluator(
                target=self.target,
                binner=self._stage3_binner,
                feature_data_source=self.feature_data_source,
            )
        else:
            logger.warning("Cached binner not found. Re-fitting Binner for the selected features...")
            cat_types = [pl.Utf8, pl.Categorical, pl.Boolean]
            cat_features = [c for c in self.selected_features_ if X_pl.schema[c] in cat_types]
            
            bining_type = "native" if self.skip_fine_scan else "opt"
            binning_params = self.rough_binning_params if self.skip_fine_scan else self.binning_params
            
            evaluator = MarsBinEvaluator(
                target=self.target,
                bining_type=bining_type,
                feature_data_source=self.feature_data_source,
                cat_features=cat_features,
                missing_values=self.missing_values,
                special_values=self.special_values,
                **binning_params
            )
            
        if self._return_pandas:
            evaluator.set_output("pandas")

        logger.info(f"Generating final evaluation report for {len(self.selected_features_)} selected features...")
        
        report: MarsEvaluationReport = evaluator.evaluate(
            X_pl, 
            features=self.selected_features_, 
            dt_col=self.time_col, 
            profile_by=self.profile_by,
            feature_start_aware_baseline=self.feature_start_aware_baseline,
            feature_data_source=self.feature_data_source,
        )
        
        return report, evaluator

    def export_selector_report(self, path: str = "mars_selector_report.xlsx") -> None:
        """
        导出选择器决策报告。

        Parameters
        ----------
        path : str, default "mars_selector_report.xlsx"
            持久化导出路径。引擎根据扩展名执行 `.csv` 或复合样式 `.xlsx` 的落盘处理。
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
        blacklist_stages: Optional[List[str]] = None
    ):
        """
        保存当前筛选结果中的白名单与黑名单。

        Parameters
        ----------
        path : str, default "mars_lists.json"
            JSON 结构存储路径。
        blacklist_stages : list of str, optional
            界定需写入惩罚名单的阶段。支持字符串模糊匹配（例如 'quality' 匹配质量校验环节）。

        Notes
        -----
        导出的 ``white_list`` 为当前最终入选特征；``black_list`` 为被剔除特征与
        用户预设黑名单的并集。
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
    def load_lists_from_json(cls, path: str) -> Dict[str, List[str]]:
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
        """
        if not os.path.exists(path):
            logger.warning(f"File {path} not found. Returning empty lists.")
            return {"white_list": [], "black_list": []}
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def print_stats(self, iv_thresholds: Optional[List[float]] = None) -> None:
        """
        打印最终入选特征的统计摘要。

        Parameters
        ----------
        iv_thresholds : list of float, optional
            自定义统计边界截断数组。默认渲染 [0.02, 0.05, 0.10] 区间梯度。
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
            f"Mars Feature Selection Final Stats",
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
    线性模型场景下的特征筛选器占位实现。

    Notes
    -----
    当前类主要用于固定接口与参数语义，具体筛选流程仍待实现。
    """

    def __init__(
        self,
        target: str,

        # --- Stage 1: 相关性去重 (Correlation Filter) ---
        # 作用: 快速去除高度相关的特征 (如 app_count_7d vs app_count_30d)
        enable_corr_filter: bool = True,
        corr_thr: float = 0.8,         # 相关性 > 0.8 视为冗余
        corr_method: str = "spearman",       # 推荐 spearman (分箱后是非线性的)
        
        # --- Stage 2: 多重共线性筛查 (VIF Filter) ---
        # 作用: 解决多重线性依赖 (A = B + C)
        # 注意: 计算矩阵逆非常慢，建议只对 < 1000 个特征开启
        enable_vif_filter: bool = False,
        vif_threshold: float = 5.0,          # 业界通常取 5.0 或 10.0
        
        # --- Stage 3: 逐步回归 (Stepwise Selection) ---
        # 作用: 基于 AIC/BIC 的包裹式筛选
        enable_stepwise: bool = False,
        stepwise_direction: str = "forward", # forward (快) / backward (慢但准)
        stepwise_criterion: str = "aic",     # 优化目标: aic / bic
        max_features: Optional[int] = None,  # 限制最终入模特征数
        
        n_jobs: int = -1
    ):
        """
        初始化线性筛选器配置。

        Parameters
        ----------
        target : str
            目标变量列名。
        enable_corr_filter : bool, default True
            是否启用相关性去重阶段。
        corr_thr : float, default 0.8
            相关性去重阈值。
        corr_method : str, default "spearman"
            相关性计算方法。
        enable_vif_filter : bool, default False
            是否启用 VIF 筛查阶段。
        vif_threshold : float, default 5.0
            VIF 阈值。
        enable_stepwise : bool, default False
            是否启用逐步回归阶段。
        stepwise_direction : str, default "forward"
            逐步回归方向。
        stepwise_criterion : str, default "aic"
            逐步回归优化准则。
        max_features : int, optional
            最终保留特征数上限。
        n_jobs : int, default -1
            并行任务数量。
        """
        super().__init__(target=target)
        # ...
        

class MarsImportanceSelector(MarsBaseSelector):
    """
    基于模型重要性的特征筛选器占位实现。

    Notes
    -----
    当前类用于固定接口与参数语义，具体训练与筛选流程仍待实现。
    """

    def __init__(
        self,
        target: str,
        
        # --- 模型配置 (Estimator) ---
        # 支持字符串简写，也支持传入实例化好的 sklearn/lgbm 对象
        estimator: Union[str, Any] = "lgbm", # lgbm, xgb, cat, rf, extra_trees
        estimator_params: Optional[dict] = None, # 比如 {'learning_rate': 0.05, 'n_estimators': 100}
        
        # --- 筛选策略 (Method) ---
        # 1. importance: 使用模型自带的 feature_importances_ (gain/split)
        # 2. shap: 训练后计算 SHAP mean values (更准但更慢)
        # 3. rfe: 递归特征消除 (反复训练剔除尾部，最慢但效果最好)
        # 4. sfm: SelectFromModel (基于阈值的一次性筛选)
        method: Literal["importance", "shap", "rfe", "sfm"] = "importance",
        
        # --- 阈值控制 (Criteria) ---
        # 决定保留多少特征
        selection_mode: Literal["top_k", "threshold", "percentile"] = "top_k",
        selection_threshold: Union[int, float, str] = 50, 
        # 解释:
        # - top_k: 保留前 50 个 (最常用)
        # - threshold: 重要性 > 0.01
        # - percentile: 保留前 20%
        
        # --- 交叉验证 (CV) ---
        # 是否使用 CV 后的平均重要性来减少随机性 (建议开启)
        cv: int = 3, 
        
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        初始化重要性筛选器配置。

        Parameters
        ----------
        target : str
            目标变量列名。
        estimator : str or Any, default "lgbm"
            底层模型类型或实例。
        estimator_params : dict, optional
            底层模型初始化参数。
        method : {"importance", "shap", "rfe", "sfm"}, default "importance"
            重要性筛选策略。
        selection_mode : {"top_k", "threshold", "percentile"}, default "top_k"
            特征保留模式。
        selection_threshold : int or float or str, default 50
            对应筛选模式下的阈值。
        cv : int, default 3
            交叉验证折数。
        n_jobs : int, default -1
            并行任务数量。
        random_state : int, default 42
            随机种子。
        """
        super().__init__(target=target)
        # ...
