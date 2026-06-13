"""MARS Pipeline 数据结构与基础抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

import polars as pl

from mars.modeling.contracts.tuning_result import MarsModelTuningResult


@dataclass(slots=True)
class MarsStepResult:
    """
    单个 Pipeline step 的结构化执行结果。

    Attributes
    ----------
    name : str
        step 名称。
    step_type : str
        step 类型，当前包括 ``selection``、``woe_binning`` 和 ``modeling``。
    input_features : list of str
        step 执行前的 active features。
    output_features : list of str
        step 执行后的 active features。
    dropped_features : list of str
        本 step 从 active features 中剔除的特征。
    report : Any
        step 产生的报告对象或明细表；无报告时为 ``None``。
    metadata : dict of str to Any
        step 级元数据。
    """

    name: str
    step_type: str
    input_features: list[str]
    output_features: list[str]
    dropped_features: list[str] = field(default_factory=list)
    report: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarsPipelineResult:
    """
    MarsModelingPipeline 的结构化运行结果。

    Attributes
    ----------
    active_features : list of str
        Pipeline 最终输出、并进入建模阶段的特征列。
    selected_features : list of str
        最后一层筛选或 WOE 转换后的特征列。
    feature_map : dict of str to str
        原始特征到派生特征的映射，例如 ``age -> age_woe``。
    step_results : list of MarsStepResult
        每个 step 的输入、输出、剔除特征、报告和元数据。
    modeling_result : MarsModelTuningResult | None
        建模 step 的调参结果；Pipeline 不包含建模 step 时为 ``None``。
    metadata : dict of str to Any
        Pipeline 级元数据。
    """

    active_features: list[str]
    selected_features: list[str]
    feature_map: dict[str, str]
    step_results: list[MarsStepResult]
    modeling_result: MarsModelTuningResult | None
    metadata: dict[str, Any] = field(default_factory=dict)


class MarsPipelineStep(ABC):
    """
    Pipeline step 抽象基类。

    子类负责管理自身的 ``fit_transform`` 和 ``transform`` 生命周期；Pipeline 主类只负责
    调度、拓扑校验和最终结果组装。
    """

    def __init__(self, name: str) -> None:
        """
        初始化 step 名称。

        Parameters
        ----------
        name : str
            step 唯一名称。
        """
        self.name = name

    @abstractmethod
    def fit_transform(
        self,
        df: pl.DataFrame,
        *,
        target: str,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str], MarsStepResult]:
        """
        拟合当前 step，并返回转换后的数据、active features 和 step 结果。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        target : str
            建模主目标列。
        active_features : Sequence[str]
            当前 step 可消费的特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级可变状态，例如特征映射和是否已执行 WOE step。

        Returns
        -------
        tuple of polars.DataFrame, list of str, MarsStepResult
            转换后的工作表、更新后的 active features 和 step 结果。
        """

    @abstractmethod
    def transform(
        self,
        df: pl.DataFrame,
        *,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        使用已拟合状态转换新样本。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        active_features : Sequence[str]
            当前 step 可消费的特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级可变状态。

        Returns
        -------
        tuple of polars.DataFrame and list of str
            转换后的工作表和更新后的 active features。
        """


def pipeline_state_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """复制 Pipeline 状态中的可序列化摘要，供结果对象写入 metadata。"""
    return {
        "has_woe_step": bool(state.get("has_woe_step", False)),
        "feature_map": dict(state.get("feature_map", {})),
    }
