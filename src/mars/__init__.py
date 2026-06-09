"""MARS 风险建模工具包的公开导出入口。"""

from .analysis import (
    MarsBinEvaluator,
    MarsDataProfiler,
    MarsEvaluationReport,
    MarsProfileConfig,
    MarsProfileReport,
    MarsRiskProfile,
    profile_risk,
    profile_stats,
)
from .feature import MarsLiteOptBinner, MarsNativeBinner, MarsOptimalBinner, MarsStatsSelector
from .modeling import MarsModelingSession
from .monitoring import (
    MarsMonitor,
    MarsMonitoringAlertConfig,
    MarsMonitoringAlerter,
    MarsMonitoringData,
    MarsMonitoringReport,
    generate_monitoring_alert,
)
from .scoring import MarsScorecard, build_scorecard
from .utils import logger, set_log_level

__version__ = "0.0.16"

_BANNER = rf"""
    __________________________________________________________________________
       __  ___ ___    ____  _____
      /  |/  //   |  / __ \/ ___/
     / /|_/ // /| | / /_/ /\__ \
    / /  / // ___ |/ _, _/___/ /
   /_/  /_//_/  |_/_/ |_|/____/

    MODELING ANALYSIS RISK SCORE
    __________________________________________________________________________
    Version: {__version__} | Copyright (c) 2026 Christian Li
    High-performance Risk Modeling Toolkit powered by Polars
    __________________________________________________________________________
"""


def __repr__() -> str:
    """返回包级欢迎横幅字符串。"""
    return _BANNER


def __str__() -> str:
    """返回包级欢迎横幅字符串。"""
    return _BANNER


__all__ = [
    "MarsDataProfiler",
    "MarsProfileConfig",
    "MarsProfileReport",
    "MarsRiskProfile",
    "MarsLiteOptBinner",
    "MarsNativeBinner",
    "MarsOptimalBinner",
    "MarsBinEvaluator",
    "MarsEvaluationReport",
    "profile_stats",
    "profile_risk",
    "MarsStatsSelector",
    "MarsModelingSession",
    "MarsMonitor",
    "MarsMonitoringAlertConfig",
    "MarsMonitoringAlerter",
    "MarsMonitoringData",
    "MarsMonitoringReport",
    "generate_monitoring_alert",
    "MarsScorecard",
    "build_scorecard",
    "logger",
    "set_log_level",
]
