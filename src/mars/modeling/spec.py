"""建模工作流的轻量配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(slots=True)
class ModelingSpec:
    """
    建模 session 级上下文配置。

    Attributes
    ----------
    model_type : str
        模型后端类型。
    features : list of str
        特征列名。
    target : str
        目标变量列名。
    dataset_flag_col : str
        数据集切片标识列名。
    categorical_features : list of str
        类别特征列名。
    optimize_metric : str
        优化指标。
    seed : int
        随机种子。
    lr_feature_mode : str
        逻辑回归后端使用的特征模式。
    lr_binning_type : str
        逻辑回归 WOE 模式使用的分箱器类型。

    Examples
    --------
    >>> spec = ModelingSpec(model_type="xgb", features=["age"], target="y")
    >>> spec.dataset_flag_col
    'dataset_flag'
    """

    model_type: str
    features: List[str]
    target: str
    dataset_flag_col: str = "dataset_flag"
    categorical_features: List[str] = field(default_factory=list)
    optimize_metric: str = "ks"
    seed: int = 1206
    lr_feature_mode: str = "numeric"
    lr_binning_type: str = "native"
    lr_binner_kwargs: dict[str, Any] = field(default_factory=dict)
    lr_binner: Any | None = None


@dataclass(slots=True)
class SplitSpec:
    """
    数据切分配置。

    Attributes
    ----------
    time_col : str
        时间列名。
    label_col : str
        标签列名。
    mode : str
        切分模式。
    train_key : str
        训练集标识。
    val_key : str
        验证集标识。
    random_seed : int
        hybrid 模式下建模窗口随机切分种子。

    Examples
    --------
    >>> spec = SplitSpec(time_col="apply_dt", label_col="y", mode="hybrid")
    >>> spec.train_key
    'train'
    """

    time_col: str
    label_col: str
    mode: str = "strict"
    train_key: str = "train"
    val_key: str = "val"
    random_seed: int = 42


@dataclass(slots=True)
class ReplaySpec:
    """
    Top-K replay 配置。

    Attributes
    ----------
    top_k : int
        需要回放的 trial 数量。
    sort_metric : str
        排序指标。
    include_val : bool
        排序均值是否包含验证集。
    num_boost_round : int
        最大训练轮数。
    early_stopping_rounds : int
        早停轮数。
    optimize_metric : str
        replay 阶段优化指标。

    Examples
    --------
    >>> spec = ReplaySpec(top_k=3)
    >>> spec.top_k
    3
    """

    top_k: int = 5
    sort_metric: str = "ks"
    include_val: bool = True
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    optimize_metric: str = "auc"
