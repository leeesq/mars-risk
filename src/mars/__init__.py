"""
MARS: MODELING ANALYSIS RISK SCORE

High-performance Risk Modeling Toolkit powered by Polars

Copyright (c) 2026 Christian and Tina
"""

from .analysis import (
    MarsBinEvaluator,
    MarsDataProfiler,
    MarsRiskProfile,
    profile_risk,
    profile_stats,
)
from .feature import (
    MarsLiteOptBinner,
    MarsNativeBinner,
    MarsOptimalBinner,
    MarsStatsSelector,
)
from .modeling import MarsModelingSession
from .monitoring import (
    MarsMonitor,
    MarsMonitoringAlertConfig,
    MarsMonitoringAlerter,
    MarsMonitoringData,
    MarsMonitoringReport,
    generate_monitoring_alert,
)
from .pipeline import (
    MarsModelingPipeline,
    MarsModelingStep,
    MarsPipelineResult,
    MarsPipelineStep,
    MarsSelectionStep,
    MarsStepResult,
    MarsWOEBinningStep,
)
from .reporting import MarsBinningReport, MarsProfileReport
from .scoring import MarsScorecard, build_scorecard
from .utils import logger, set_log_level

__version__ = "0.0.19"

_BANNER = rf"""
    __________________________________________________________________________
       __  ___ ___    ____  _____
      /  |/  //   |  / __ \/ ___/
     / /|_/ // /| | / /_/ /\__ \
    / /  / // ___ |/ _, _/___/ /
   /_/  /_//_/  |_/_/ |_|/____/

    MODELING ANALYSIS RISK SCORE
    __________________________________________________________________________
    Version: {__version__} | Copyright (c) 2026 Christian and Tina
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
    "MarsProfileReport",
    "MarsRiskProfile",
    "MarsLiteOptBinner",
    "MarsNativeBinner",
    "MarsOptimalBinner",
    "MarsBinEvaluator",
    "MarsBinningReport",
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
    "MarsModelingPipeline",
    "MarsModelingStep",
    "MarsPipelineStep",
    "MarsPipelineResult",
    "MarsSelectionStep",
    "MarsStepResult",
    "MarsWOEBinningStep",
    "MarsScorecard",
    "build_scorecard",
    "logger",
    "set_log_level",
]
