# Modeling Pipeline

!!! warning "快速迭代模块"
    Modeling Pipeline 仍在快速迭代中，接口约定、结果对象和调参参数后续可能发生较大变化。生产流程建议固定版本，并在升级前检查返回对象和字段名称。

## 能力索引

| 能力 | 主要入口 | 说明 |
| --- | --- | --- |
| 样本切分 | `MarsModelingSession.slice` / `MarsModelDataSplitter` | 按时间严格切分 train/val/oot，或为长短 y 生成独立切片列 |
| 模型调参 | `MarsModelingSession.tune` / `MarsModelTuner.tune` | 支持 XGBoost、LightGBM、CatBoost、Logistic Regression |
| 指标体系 | `optimize_metric` / `custom_metrics` | 内置 `auc`、`ks`、`f1`，支持自定义 metric 和 maximize/minimize |
| 调参产物 | `artifact_dir` | 每次调参生成独立目录；`artifact_dir=None` 表示完全不落盘 |
| 模型保留 | `keep_top_n_models` | 调参过程中动态保留当前最优 N 个有效 trial 模型 |
| Replay | `MarsModelReplayRunner.run` | 支持 Top-K replay，也支持 `trial_nums=[...]` 指定编号 replay |
| 特征重要性 | `importance_methods` | 默认 native importance，显式请求可计算 SHAP importance |
| 多口径评估 | `benchmark_cols` / `aux_targets` / `target_group_cols` | 多 benchmark、多辅助 target 和长短 y 独立评估 |

## Pipeline 编排

::: mars.pipeline.MarsModelingPipeline

::: mars.pipeline.MarsPipelineStep

::: mars.pipeline.MarsSelectionStep

::: mars.pipeline.MarsWOEBinningStep

::: mars.pipeline.MarsModelingStep

::: mars.pipeline.MarsPipelineResult

::: mars.pipeline.MarsStepResult

## Session

::: mars.modeling.MarsModelingSession

## Data Splitter

::: mars.modeling.slicing.MarsModelDataSplitter

## Tuning

::: mars.modeling.tuning.MarsModelTuner

::: mars.modeling.tuning.MarsModelReplayRunner

## Evaluation 与 Prediction

::: mars.modeling.evaluation.MarsModelEvaluator

::: mars.modeling.prediction.ModelPredictor

## Result Objects

::: mars.modeling.results.MarsModelTuningResult

::: mars.modeling.results.MarsModelReplayResult

::: mars.modeling.feature_growth.MarsFeatureGrowthResult

::: mars.modeling.report.MarsModelingReport
