"""Public package exports for MARS."""

from .analysis import (
    MarsDataProfiler,
    MarsProfileConfig,
    MarsProfileReport,
    MarsBinEvaluator,
    MarsEvaluationReport,
    profile_stats,
    profile_risk,
)
from .feature import MarsNativeBinner, MarsOptimalBinner, MarsStatsSelector
from .modeling import MarsModelingSession
from .scoring import MarsScorecard, build_scorecard
from .utils import logger, set_log_level

__version__ = "0.0.15"

_BANNER = r"""
    __________________________________________________________________________
       __  ___ ___    ____  _____
      /  |/  //   |  / __ \/ ___/
     / /|_/ // /| | / /_/ /\__ \ 
    / /  / // ___ |/ _, _/___/ / 
   /_/  /_//_/  |_/_/ |_|/____/  
                                 
    MODELING ANALYSIS RISK SCORE 
    __________________________________________________________________________
    Version: {ver} | Copyright (c) 2026 Christian Li
    High-performance Risk Modeling Toolkit powered by Polars
    __________________________________________________________________________
""".format(ver=__version__)


def __repr__() -> str:
    """Return the package banner string."""
    return _BANNER


def __str__() -> str:
    """Return the package banner string."""
    return _BANNER


__all__ = [
    "MarsDataProfiler",
    "MarsProfileConfig",
    "MarsProfileReport",
    "MarsNativeBinner",
    "MarsOptimalBinner",
    "MarsBinEvaluator",
    "MarsEvaluationReport",
    "profile_stats",
    "profile_risk",
    "MarsStatsSelector",
    "MarsModelingSession",
    "MarsScorecard",
    "build_scorecard",
    "logger",
    "set_log_level",
]
