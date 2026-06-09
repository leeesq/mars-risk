"""MARS 建模全链路 Pipeline 主类。"""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import pandas as pd
import polars as pl

from mars.modeling.prediction import ModelPredictor
from mars.modeling.utils import FrameLike, is_polars_dataframe, restore_frame_type
from mars.pipeline.base import MarsPipelineResult, MarsPipelineStep, MarsStepResult
from mars.pipeline.steps import MarsModelingStep, MarsSelectionStep, MarsWOEBinningStep
from mars.utils.logger import logger


class MarsModelingPipeline:
    """
    串联特征筛选、可选 WOE 分箱和 Modeling 建模的高层 Pipeline 编排器。

    Pipeline 面向风控建模宽表场景，默认保持树模型链路简单：先筛选特征，再进入建模。
    LR 或评分卡链路可以显式加入 ``MarsWOEBinningStep``，把原始特征转换为 ``*_woe`` 后再筛选
    或建模。该类不是 sklearn ``Pipeline`` 的严格子类，但保留熟悉的 ``fit``、``transform`` 和
    ``predict`` 调用方式。

    Examples
    --------
    >>> from mars.feature import MarsStatsSelector
    >>> pipeline = MarsModelingPipeline(
    ...     target="target",
    ...     features=["age"],
    ...     steps=[MarsSelectionStep(name="stats", selector=MarsStatsSelector(skip_fine_scan=True))],
    ... )
    >>> pipeline.features
    ['age']
    """

    def __init__(
        self,
        *,
        target: str,
        features: Sequence[str],
        steps: Sequence[MarsPipelineStep],
    ) -> None:
        """
        初始化建模编排器并校验 step 拓扑。

        Parameters
        ----------
        target : str
            建模主目标列。
        features : Sequence[str]
            初始候选特征列。
        steps : Sequence[MarsPipelineStep]
            按顺序执行的 step 列表。``MarsSelectionStep`` 可出现多次；
            ``MarsModelingStep`` 最多出现一次且必须放在最后。
        """
        self.target = target
        self.features = list(features)
        self.steps = list(steps)
        self._validate_steps()

        self.result_: MarsPipelineResult | None = None
        self.fitted_steps_: list[MarsPipelineStep] = []
        self.named_steps_: dict[str, MarsPipelineStep] = {}
        self._prefer_polars: bool = True

    def fit(self, df: FrameLike) -> MarsPipelineResult:
        """
        按 step 顺序拟合 Pipeline 并返回结构化结果。

        Parameters
        ----------
        df : FrameLike
            包含特征、目标列以及可选切片列的建模样本。

        Returns
        -------
        MarsPipelineResult
            Pipeline 执行结果，包含最终特征、每步报告和建模调参结果。
        """
        self._prefer_polars = is_polars_dataframe(df)
        working_df = self._to_polars(df)
        self._validate_input_columns(working_df, self.features + [self.target])

        logger.info(
            "Starting MarsModelingPipeline fit on %s rows and %s initial features.",
            working_df.height,
            len(self.features),
        )

        active_features = list(self.features)
        pipeline_state: MutableMapping[str, Any] = {
            "feature_map": {feature: feature for feature in active_features},
            "has_woe_step": False,
        }
        step_results: list[MarsStepResult] = []
        modeling_result = None

        self.fitted_steps_ = []
        self.named_steps_ = {}

        for step in self.steps:
            working_df, active_features, step_result = step.fit_transform(
                working_df,
                target=self.target,
                active_features=active_features,
                pipeline_state=pipeline_state,
            )
            step_results.append(step_result)
            self.fitted_steps_.append(step)
            self.named_steps_[step.name] = step
            if isinstance(step, MarsModelingStep):
                modeling_result = step_result.report

        result = MarsPipelineResult(
            active_features=list(active_features),
            selected_features=list(active_features),
            feature_map=dict(pipeline_state["feature_map"]),
            step_results=step_results,
            modeling_result=modeling_result,
            metadata={
                "target": self.target,
                "features": list(self.features),
                "step_names": [step.name for step in self.steps],
                "has_woe_step": bool(pipeline_state.get("has_woe_step", False)),
                "has_modeling_step": modeling_result is not None,
            },
        )
        self.result_ = result
        logger.info("MarsModelingPipeline fit completed with %s active features.", len(active_features))
        return result

    def transform(self, df: FrameLike) -> FrameLike:
        """
        使用已拟合的分箱和筛选步骤转换样本。

        Parameters
        ----------
        df : FrameLike
            待转换样本。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            保留原始上下文列，并追加已拟合 WOE 特征后的样本。返回类型尽量与输入一致。
        """
        self._check_is_fitted()
        working_df = self._to_polars(df)
        active_features = list(self.features)
        self._validate_input_columns(working_df, active_features)
        pipeline_state: MutableMapping[str, Any] = {
            "feature_map": dict(self.result_.feature_map if self.result_ is not None else {}),
            "has_woe_step": False,
        }

        for step in self.fitted_steps_:
            if isinstance(step, MarsModelingStep):
                break
            working_df, active_features = step.transform(
                working_df,
                active_features=active_features,
                pipeline_state=pipeline_state,
            )
        return restore_frame_type(working_df, is_polars_dataframe(df))

    def predict(
        self,
        df: FrameLike,
        *,
        pred_col: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """
        对新样本执行 Pipeline 转换并追加模型预测分。

        Parameters
        ----------
        df : FrameLike
            待打分样本。
        pred_col : str
            预测分列名。
        inplace : bool
            当输入为 Pandas DataFrame 时，是否允许在输入对象上原地写入预测列。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            追加预测分后的样本。

        Raises
        ------
        RuntimeError
            Pipeline 尚未拟合或没有 modeling step 时抛出。
        """
        self._check_is_fitted()
        if self.result_ is None or self.result_.modeling_result is None:
            raise RuntimeError("MarsModelingPipeline.predict requires a fitted MarsModelingStep.")

        transformed = self.transform(df)
        transformed_input = transformed
        if inplace and isinstance(df, pd.DataFrame):
            transformed_input = transformed

        run = self.result_.modeling_result
        category_levels: dict[str, Sequence[Any]] = {
            feature: levels for feature, levels in run.category_levels.items()
        }
        predictor = ModelPredictor(
            model=run.best_model,
            feature_list=run.features,
            categorical_features=run.categorical_features,
            category_levels=category_levels,
        )
        return predictor.predict(transformed_input, pred_col=pred_col, inplace=inplace)

    def _validate_steps(self) -> None:
        """校验 step 名称唯一、建模 step 数量和位置。"""
        names = [step.name for step in self.steps]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"Pipeline step names must be unique. Duplicates: {duplicate_names}.")

        modeling_positions = [
            index
            for index, step in enumerate(self.steps)
            if isinstance(step, MarsModelingStep)
        ]
        if len(modeling_positions) > 1:
            raise ValueError("MarsModelingPipeline accepts at most one MarsModelingStep.")
        if modeling_positions and modeling_positions[0] != len(self.steps) - 1:
            raise ValueError("MarsModelingStep must be the last step in MarsModelingPipeline.")

    def _validate_input_columns(self, df: pl.DataFrame, columns: Sequence[str]) -> None:
        """校验输入数据包含指定列。"""
        missing = sorted(set(columns).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}.")

    def _check_is_fitted(self) -> None:
        """校验 Pipeline 已经完成拟合。"""
        if self.result_ is None:
            raise RuntimeError("MarsModelingPipeline is not fitted. Call fit(df) first.")

    def _to_polars(self, df: FrameLike) -> pl.DataFrame:
        """将输入表转换为 Polars 副本。"""
        if isinstance(df, pl.DataFrame):
            return df.clone()
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


__all__ = [
    "MarsModelingPipeline",
    "MarsModelingStep",
    "MarsSelectionStep",
    "MarsWOEBinningStep",
]
