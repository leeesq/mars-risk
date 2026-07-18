---
description: Experimental Modeling/Pipeline API：切分、调参、replay、评估、预测与编排。
---

# Modeling / Pipeline

!!! warning "Experimental"

    参数、结果对象和 artifact 结构仍可能变化。使用前阅读
    [Modeling / Pipeline 指南](../user-guide/modeling-pipeline.md)和
    [稳定性与兼容性](../project/stability.md)。

## Pipeline

::: mars.pipeline.MarsModelingPipeline

::: mars.pipeline.MarsPipelineStep

::: mars.pipeline.MarsSelectionStep

::: mars.pipeline.MarsWOEBinningStep

::: mars.pipeline.MarsModelingStep

::: mars.pipeline.MarsPipelineResult

::: mars.pipeline.MarsStepResult

## Session 与工作流

::: mars.modeling.MarsModelingSession

::: mars.modeling.MarsModelDataSplitter

::: mars.modeling.MarsModelTuner

::: mars.modeling.MarsModelReplayRunner

::: mars.modeling.MarsFeatureIncrementalTuner

## Evaluation 与 Prediction

::: mars.modeling.MarsModelEvaluator

::: mars.modeling.ModelPredictor

## Result Objects

::: mars.modeling.MarsModelTuningResult

::: mars.modeling.MarsModelReplayResult

::: mars.modeling.MarsFeatureGrowthResult

::: mars.modeling.MarsModelingReport
