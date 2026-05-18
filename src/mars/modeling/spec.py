"""MARS 建模任务工作流的轻量规格对象。"""

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
        底层模型类型。
    features : list of str
        特征列名。
    target : str
        目标变量列名。
    dataset_flag_col : str
        数据集切分标识列名。
    categorical_features : list of str
        类别特征列名。
    optimize_metric : str
        默认优化指标。
    seed : int
        默认随机种子。
    benchmark_col : str, optional
        默认基线分数列名。
    time_col : str, optional
        默认时间列名。
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
    数据切分规格。

    Parameters
    ----------
    time_col : str
        时间列名。
    label_col : str
        标签列名。
    mode : {"strict", "hybrid"}
        切分模式。
    train_key : str
        训练集标识名。
    val_key : str
        验证集标识名。
    random_seed : int
        hybrid 模式下建模区随机切分的随机种子。
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
    Top-K 回放重训规格。

    Parameters
    ----------
    top_k : int
        需要回放的 Trial 数量。
    sort_metric : str
        排序指标。
    include_val : bool
        计算排序均值时是否包含验证集。
    num_boost_round : int
        最大训练轮数。
    early_stopping_rounds : int
        早停轮数。
    optimize_metric : str
        回放阶段的最终优化指标。
    """

    top_k: int = 5
    sort_metric: str = "ks"
    include_val: bool = True
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    optimize_metric: str = "auc"
