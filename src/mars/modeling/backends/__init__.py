"""内置建模后端、注册表工具与预测适配器导出入口。"""

from mars.modeling.backends.adapters import get_prediction_adapter
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.catboost import MarsCatBoostStrategy
from mars.modeling.backends.lightgbm import MarsLGBStrategy
from mars.modeling.backends.logistic import MarsLogisticRegressionStrategy
from mars.modeling.backends.registry import (
    BACKEND_REGISTRY,
    backend_map,
    get_backend_spec,
    get_backend_strategy,
    has_backend,
    register_backend,
    registered_backend_names,
    resolve_backend_name,
)
from mars.modeling.backends.xgboost import MarsXGBStrategy

__all__ = [
    "BACKEND_REGISTRY",
    "MarsBaseModelStrategy",
    "MarsCatBoostStrategy",
    "MarsLGBStrategy",
    "MarsLogisticRegressionStrategy",
    "MarsXGBStrategy",
    "backend_map",
    "get_backend_spec",
    "get_backend_strategy",
    "get_prediction_adapter",
    "has_backend",
    "register_backend",
    "registered_backend_names",
    "resolve_backend_name",
]
