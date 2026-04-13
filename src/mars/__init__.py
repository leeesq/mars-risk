import sys
from .analysis import (
    MarsDataProfiler, MarsProfileConfig, MarsProfileReport,
    MarsBinEvaluator, MarsEvaluationReport,  profile_risk
)
from .feature import MarsNativeBinner, MarsOptimalBinner, MarsStatsSelector
from .utils import logger, set_log_level

__version__ = "0.0.14" 

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

# if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
#     print(_BANNER)

def __repr__():
    return _BANNER

def __str__():
    return _BANNER

__all__ = [
    "MarsDataProfiler",
    "MarsProfileConfig",
    "MarsProfileReport",
    
    "MarsNativeBinner",
    "MarsOptimalBinner",
    "MarsBinEvaluator",
    "MarsEvaluationReport",
    "profile_risk",
    
    "MarsStatsSelector",
    
    "logger",
    "set_log_level",
]