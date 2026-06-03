from dataclasses import dataclass, field
from typing import List


@dataclass
class MarsProfileConfig:
    """
    `MarsDataProfiler` 的全局配置对象。

    该配置用于统一控制画像流程中的统计指标范围、数据质量指标范围、
    Sparkline 渲染行为，以及 PSI 计算时是否纳入缺失值箱与特殊值箱。

    Attributes
    ----------
    stat_metrics : List[str]
        需要计算的统计指标列表，例如 ``psi``、``mean``、``median``。
    dq_metrics : List[str]
        需要计算的数据质量指标列表，例如 ``missing``、``zeros``、``top1``。
    enable_sparkline : bool
        是否在概览表中生成字符画形式的分布图。
    sparkline_bins : int
        Sparkline 的分箱数量，控制字符画分布图的精度。
    sparkline_sample_size : int
        生成 Sparkline 时允许使用的最大采样行数。
    psi_include_missing : bool
        计算 PSI 时是否包含缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否包含特殊值箱。
    """

    # "psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"
    stat_metrics: List[str] = field(default_factory=lambda: ["psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"])
    dq_metrics: List[str] = field(default_factory=lambda: ["missing", "zeros", "unique", "top1"])

    enable_sparkline: bool = True # 是否启用迷你分布图
    sparkline_bins: int = 8  # 分布图分箱数
    sparkline_sample_size: int = 200_000 # 采样上限

    psi_include_missing: bool = False  # 计算 PSI 时是否包含缺失值箱
    psi_include_special: bool = False  # 计算 PSI 时是否包含特殊值箱
