"""MARS 核心基类与异常定义。"""

from .base import MarsBaseEstimator, MarsTransformer
from .constants import (
    DIVISION_EPSILON,
    FLOAT_TOLERANCE,
    METRIC_EPSILON,
    MIN_VARIANCE,
    PROBABILITY_EPSILON,
)
from .exceptions import DataTypeError, MarsError, NotFittedError

__all__ = [
    "MarsBaseEstimator",
    "MarsTransformer",
    "METRIC_EPSILON",
    "DIVISION_EPSILON",
    "FLOAT_TOLERANCE",
    "PROBABILITY_EPSILON",
    "MIN_VARIANCE",
    "MarsError",
    "NotFittedError",
    "DataTypeError"
]
