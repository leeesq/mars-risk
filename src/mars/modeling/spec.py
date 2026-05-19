"""建模工作流的轻量配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class ModelingSpec:
    """
    建模 session 级上下文配置。

    Parameters
    ----------
    model_type : str
        模型后端类型。
    features : list of str
        特征列名。
    target : str
        目标变量列名。
    dataset_flag_col : str, default "dataset_flag"
        数据集切片标识列名。
    categorical_features : list of str, optional
        类别特征列名。
    optimize_metric : str, default "ks"
        优化指标。
    seed : int, default 1206
        随机种子。
    benchmark_col : str, optional
        基准模型分数列。
    time_col : str, optional
        时间列名。
    """

    model_type: str
    features: List[str]
    target: str
    dataset_flag_col: str = "dataset_flag"
    categorical_features: List[str] = field(default_factory=list)
    optimize_metric: str = "ks"
    seed: int = 1206
    benchmark_col: Optional[str] = None
    time_col: Optional[str] = None


@dataclass(slots=True)
class SplitSpec:
    """
    数据切分配置。

    Parameters
    ----------
    time_col : str
        时间列名。
    label_col : str
        标签列名。
    mode : {"strict", "hybrid"}, default "strict"
        切分模式。
    train_key : str, default "train"
        训练集标识。
    val_key : str, default "val"
        验证集标识。
    random_seed : int, default 42
        hybrid 模式下建模窗口随机切分种子。
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

    Parameters
    ----------
    top_k : int, default 5
        需要回放的 trial 数量。
    sort_metric : str, default "ks"
        排序指标。
    include_val : bool, default True
        排序均值是否包含验证集。
    num_boost_round : int, default 500
        最大训练轮数。
    early_stopping_rounds : int, default 50
        早停轮数。
    optimize_metric : str, default "auc"
        replay 阶段优化指标。
    """

    top_k: int = 5
    sort_metric: str = "ks"
    include_val: bool = True
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    optimize_metric: str = "auc"
