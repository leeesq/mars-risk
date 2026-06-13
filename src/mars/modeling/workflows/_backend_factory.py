"""建模后端实例化的内部工具。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from mars.compute import FrameLike
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.backends.registry import get_backend_strategy, resolve_backend_name
from mars.modeling.contracts.specs import ModelingSpec
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection


def build_backend_from_spec(
    spec: ModelingSpec,
    df: FrameLike,
    *,
    param_space: Mapping[str, Any] | None = None,
    max_diff: float = 3.0,
    use_oot_penalty: bool = False,
    optimize_metric: str | None = None,
    seed: int | None = None,
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
    metric_directions: Mapping[str, MetricDirection] | None = None,
    training_metric: str | None = None,
    backend_metric: Any | None = None,
    keep_top_n_models: int = 0,
) -> MarsBaseModelStrategy:
    """
    根据建模配置创建具体后端策略实例。

    Parameters
    ----------
    spec : ModelingSpec
        已标准化的建模规格对象。
    df : FrameLike
        建模数据表。
    param_space : Mapping[str, Any] | None
        Optuna 搜索空间。
    max_diff : float
        train/oot 最大允许差异阈值。
    use_oot_penalty : bool
        是否在优化指标中加入 OOT 惩罚。
    optimize_metric : str | None
        本次运行覆盖的优化指标。
    seed : int | None
        本次运行覆盖的随机种子。
    metric_params : Mapping[str, Any] | None
        指标附加参数。
    custom_metrics : Mapping[str, MetricCallable] | None
        自定义评估指标。
    metric_directions : Mapping[str, MetricDirection] | None
        各指标的排序方向。
    training_metric : str | None
        训练过程的后端原生指标名。
    backend_metric : Any | None
        后端特定指标对象。
    keep_top_n_models : int
        需要动态保留的候选模型数量。

    Returns
    -------
    MarsBaseModelStrategy
        已实例化的后端策略对象。
    """
    backend_cls = get_backend_strategy(spec.model_type)
    backend_kwargs: Dict[str, Any] = {
        "df": df,
        "features": spec.features,
        "target": spec.target,
        "optimize_metric": (optimize_metric or spec.optimize_metric).lower(),
        "param_space": param_space,
        "max_diff": max_diff,
        "seed": spec.seed if seed is None else int(seed),
        "use_oot_penalty": use_oot_penalty,
        "dataset_flag_col": spec.dataset_flag_col,
        "categorical_features": spec.categorical_features,
        "metric_params": metric_params,
        "custom_metrics": custom_metrics,
        "metric_directions": metric_directions,
        "training_metric": training_metric,
        "backend_metric": backend_metric,
        "keep_top_n_models": keep_top_n_models,
    }
    if resolve_backend_name(spec.model_type) == "lr":
        backend_kwargs.update(
            {
                "lr_feature_mode": spec.lr_feature_mode,
                "lr_binning_type": spec.lr_binning_type,
                "lr_binner_kwargs": spec.lr_binner_kwargs,
                "lr_binner": spec.lr_binner,
            }
        )
    return backend_cls(**backend_kwargs)
