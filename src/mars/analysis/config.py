from dataclasses import dataclass, field
from typing import List

@dataclass
class MarsProfileConfig:
    """
    [配置类] Mars analysis profiler 的全局配置对象。
    
    Attributes
    ----------
    stat_metrics : List[str]
        需要计算的数值统计指标列表。
    dq_metrics : List[str]
        需要计算的数据质量(DQ)指标列表。
    enable_sparkline : bool
        是否生成迷你分布图。默认 True。
    sparkline_bins : int
        分布图的分箱数量 (字符画的精度)。默认 8 (对应 _▂▃▄▅▆▇█)。
    """
    
    # --- 计算范围 ---
    # "psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"
    stat_metrics: List[str] = field(default_factory=lambda: ["psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"])
    dq_metrics: List[str] = field(default_factory=lambda: ["missing", "zeros", "unique", "top1"])

    # --- 可视化 ---
    enable_sparkline: bool = True # 是否启用迷你分布图
    sparkline_bins: int = 8  # 分布图分箱数
    sparkline_sample_size: int = 200_000 # 采样上限