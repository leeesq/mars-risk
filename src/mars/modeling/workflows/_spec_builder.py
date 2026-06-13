"""建模规格构造的内部工具。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mars.modeling.backends.registry import has_backend, registered_backend_names
from mars.modeling.contracts.specs import ModelingSpec


def build_modeling_spec(
    *,
    model_type: str,
    features: Sequence[str],
    target: str,
    dataset_flag_col: str = "dataset_flag",
    categorical_features: Sequence[str] | None = None,
    optimize_metric: str = "ks",
    seed: int = 1206,
    lr_feature_mode: str = "numeric",
    lr_binning_type: str = "native",
    lr_binner_kwargs: Mapping[str, Any] | None = None,
    lr_binner: Any | None = None,
) -> ModelingSpec:
    """
    校验建模配置并构造共享规格对象。

    Parameters
    ----------
    model_type : str
        模型后端类型。
    features : Sequence[str]
        特征列名。
    target : str
        目标列名。
    dataset_flag_col : str
        建模样本切片标记列名。
    categorical_features : Sequence[str] | None
        类别特征列名。
    optimize_metric : str
        调参优化指标，可使用内置指标或后续传入的自定义指标名。
    seed : int
        随机种子。
    lr_feature_mode : str
        Logistic Regression 特征模式，支持 ``"numeric"`` 和 ``"woe"``。
    lr_binning_type : str
        LR WOE 模式使用的分箱器类型，支持 ``native``、``optimal`` 和 ``lite_opt``。
    lr_binner_kwargs : Mapping[str, Any] | None
        构造 LR 分箱器时使用的参数。
    lr_binner : Any | None
        已拟合或待复用的 LR 分箱器实例。

    Returns
    -------
    ModelingSpec
        标准化后的建模配置。

    Raises
    ------
    ValueError
        当后端类型、`lr_feature_mode` 或 `lr_binning_type` 不在支持范围内时抛出。
    """
    spec = ModelingSpec(
        model_type=model_type.lower(),
        features=list(features),
        target=target,
        dataset_flag_col=dataset_flag_col,
        categorical_features=list(categorical_features or []),
        optimize_metric=optimize_metric.lower(),
        seed=int(seed),
        lr_feature_mode=str(lr_feature_mode).lower(),
        lr_binning_type=str(lr_binning_type).lower(),
        lr_binner_kwargs=dict(lr_binner_kwargs or {}),
        lr_binner=lr_binner,
    )
    if not has_backend(spec.model_type):
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. Expected one of {registered_backend_names()}."
        )
    if spec.lr_feature_mode not in {"numeric", "woe"}:
        raise ValueError("lr_feature_mode must be one of {'numeric', 'woe'}.")
    if spec.lr_binning_type not in {"native", "optimal", "lite_opt"}:
        raise ValueError(
            "lr_binning_type must be one of {'native', 'optimal', 'lite_opt'}."
        )
    return spec
