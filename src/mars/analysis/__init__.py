"""MARS Stable 数据画像与分箱评估公开入口。"""

from ._risk_profile import profile_risk
from .evaluator import MarsBinEvaluator, MarsRiskProfile
from .profiler import MarsDataProfiler, profile_stats
from .score_cross import (
    ScoreCrossReport,
    cross_scores,
    evaluate_score_policy,
    get_score_bin_definitions,
)
from .score_cross_view import get_score_cell, show_score_matrix, write_score_cross_html

__all__ = [
    "MarsDataProfiler",
    "MarsBinEvaluator",
    "MarsRiskProfile",
    "profile_stats",
    "profile_risk",
    "ScoreCrossReport",
    "cross_scores",
    "evaluate_score_policy",
    "get_score_bin_definitions",
    "get_score_cell",
    "show_score_matrix",
    "write_score_cross_html",
    "get_score_cell",
    "show_score_matrix",
    "write_score_cross_html",
]
