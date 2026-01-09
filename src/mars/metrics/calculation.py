import numpy as np
import polars as pl
from typing import Union, Dict, Tuple
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr

class MarsMetrics:
    """
    [数学计算层] MarsMetrics
    
    集成风控建模核心指标计算。
    - 连续指标 (AUC, Continuous KS): 基于 NumPy/Sklearn (精确, 适合评估模型能力)
    - 离散指标 (WOE, Lift, Binned KS): 基于 Polars (高速, 适合评估分箱效果)
    """

    # ==========================================
    # 1. 连续值指标 (基于 y_true, y_prob)
    # ==========================================

    @staticmethod
    def calculate_ks_continuous(
        y_true: Union[pl.Series, np.ndarray], 
        y_prob: Union[pl.Series, np.ndarray]
    ) -> float:
        """
        [连续 KS] 基于 ROC 曲线计算 KS (Kolmogorov-Smirnov)。
        
        KS = max(TPR - FPR)
        这是评估模型排序能力最准确的方法，不依赖分箱方式。
        """
        # 数据转换与对齐
        y_true, y_prob = MarsMetrics._to_numpy(y_true, y_prob)
        
        # 使用 sklearn 的 roc_curve 获取所有阈值下的 fpr, tpr
        # 相比手动排序，sklearn 底层经过高度优化
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # 计算 KS
        ks_value = np.max(np.abs(tpr - fpr))
        return float(ks_value)

    @staticmethod
    def calculate_auc(
        y_true: Union[pl.Series, np.ndarray], 
        y_prob: Union[pl.Series, np.ndarray]
    ) -> float:
        """计算 AUC (Area Under Curve)。"""
        y_true, y_prob = MarsMetrics._to_numpy(y_true, y_prob)
        try:
            return float(roc_auc_score(y_true, y_prob))
        except ValueError:
            return 0.5

    # ==========================================
    # 2. 分箱/离散指标 (基于 DataFrame GroupBy)
    # ==========================================

    @staticmethod
    def calculate_bin_stats(
        df: pl.DataFrame, 
        bin_col: str, 
        target_col: str,
        sort_ascending: bool = True
    ) -> pl.DataFrame:
        """
        [核心聚合] 一次性计算分箱后的所有核心指标：
        - Count, Good, Bad
        - Bad Rate
        - WOE (Weight of Evidence)
        - IV (Information Value)
        - Lift
        - Binned KS (分箱 KS)
        
        Parameters
        ----------
        df : pl.DataFrame
            包含分箱列和目标列的数据。
        bin_col : str
            分箱列名。
        target_col : str
            目标列名 (0/1)。
        sort_ascending : bool
            是否按 bin_col 排序 (计算累积 KS 时顺序很重要)。

        Returns
        -------
        pl.DataFrame
            包含所有统计指标的汇总表。
        """
        # 1. 基础聚合 (利用 Polars 并行能力)
        stats = df.group_by(bin_col).agg([
            pl.len().alias("count"),
            (pl.col(target_col) == 1).sum().alias("bad"),
            (pl.col(target_col) == 0).sum().alias("good"),
            pl.col(target_col).mean().alias("bad_rate")
        ]).sort(bin_col, descending=not sort_ascending)

        # 2. 全局统计量 (用于计算占比)
        total_bad = stats["bad"].sum()
        total_good = stats["good"].sum()
        total_count = stats["count"].sum()
        overall_bad_rate = total_bad / total_count

        # 防止 log(0) 或 div/0 的平滑项
        EPS = 1e-10

        # 3. 向量化计算复杂指标
        return stats.with_columns([
            # 占比计算
            (pl.col("bad") / total_bad).alias("pct_bad"),
            (pl.col("good") / total_good).alias("pct_good"),
            (pl.col("count") / total_count).alias("pct_total")
        ]).with_columns([
            # WOE = ln(%Good / %Bad)
            # 注意: 风控中常用 bad 为分子还是分母有不同习惯，这里采用标准 ln(Good/Bad) * -1 或者 ln(Bad/Good)
            # 通常：WOE = ln(Distr Good / Distr Bad)
            (pl.col("pct_good") / (pl.col("pct_bad") + EPS)).log().alias("woe"),
            
            # Lift = Bin_BadRate / Total_BadRate
            (pl.col("bad_rate") / (overall_bad_rate + EPS)).alias("lift")
        ]).with_columns([
            # IV Component = (Distr Good - Distr Bad) * WOE
            ((pl.col("pct_good") - pl.col("pct_bad")) * pl.col("woe")).alias("iv_contribution")
        ]).with_columns([
            # 累积指标 (用于计算 Binned KS)
            pl.col("pct_bad").cum_sum().alias("cum_pct_bad"),
            pl.col("pct_good").cum_sum().alias("cum_pct_good")
        ]).with_columns([
            # Binned KS = max |Cum%Bad - Cum%Good|
            (pl.col("cum_pct_bad") - pl.col("cum_pct_good")).abs().alias("ks_spread")
        ])

    @staticmethod
    def calculate_ks_binned(
        df: pl.DataFrame, 
        bin_col: str, 
        target_col: str
    ) -> float:
        """
        [分箱 KS] 仅返回分箱 KS 的最大值。
        如果需要画图或查看每箱详情，请使用 calculate_bin_stats。
        """
        stats = MarsMetrics.calculate_bin_stats(df, bin_col, target_col)
        return stats["ks_spread"].max()

    # ==========================================
    # 3. 辅助工具
    # ==========================================

    @staticmethod
    def _to_numpy(
        y_true: Union[pl.Series, np.ndarray], 
        y_prob: Union[pl.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[Internal] 统一转换为 Numpy 且移除 NaN"""
        if isinstance(y_true, pl.Series): y_true = y_true.to_numpy()
        if isinstance(y_prob, pl.Series): y_prob = y_prob.to_numpy()
        
        # 移除 NaN (Sklearn 不接受 NaN)
        mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        return y_true[mask], y_prob[mask]