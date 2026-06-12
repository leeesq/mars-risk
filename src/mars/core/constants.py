"""MARS 内部数值稳定性常量。"""

from __future__ import annotations

# 风险指标平滑项：用于 WOE、IV、PSI、Lift 等概率/占比分布指标，避免对 0 取对数。
METRIC_EPSILON: float = 1e-6

# 分母保护项：用于占比、均值、相对变化等普通除法，不改变指标平滑口径。
DIVISION_EPSILON: float = 1e-9

# 浮点容差：用于近零判断、单调性比较和展示比例防抖。
FLOAT_TOLERANCE: float = 1e-12

# 概率裁剪项：用于 LogLoss 等对概率边界敏感的指标。
PROBABILITY_EPSILON: float = 1e-15

# 最小方差阈值：用于常量列和低方差特征的稳定性判断。
MIN_VARIANCE: float = 1e-6
