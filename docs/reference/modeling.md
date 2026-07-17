# Modeling / Pipeline

!!! warning "快速迭代模块"
    Modeling 建模和 Pipeline 编排的接口、结果对象和调参参数仍可能变化。生产流程请固定依赖版本；
    升级前在测试环境核对所依赖的返回字段和产物路径。

## 能力索引

| 能力 | 主要入口 | 说明 |
| --- | --- | --- |
| Pipeline 编排 | `MarsModelingPipeline` / `MarsSelectionStep` / `MarsWOEBinningStep` / `MarsModelingStep` | 串联多层筛选、可选 WOE 分箱和最终建模 |
| 样本切分 | `MarsModelingSession.slice` / `MarsModelDataSplitter` | 按时间严格切分 train/val/oot，或为长短 y 生成独立切片列 |
| 模型调参 | `MarsModelingSession.tune` / `MarsModelTuner.tune` | 支持 XGBoost、LightGBM、CatBoost、Logistic Regression |
| 指标体系 | `optimize_metric` / `custom_metrics` | 内置 `auc`、`ks`、`f1`，支持自定义 metric 和 maximize/minimize |
| 调参产物 | `artifact_dir` | 每次调参生成独立目录；`artifact_dir=None` 表示完全不落盘 |
| 模型保留 | `keep_top_n_models` | 调参过程中动态保留当前最优 N 个有效 trial 模型 |
| Replay | `MarsModelReplayRunner.replay` | 支持 Top-K replay，也支持 `trial_nums=[...]` 指定编号 replay |
| 特征重要性 | `importance_methods` | 默认 native importance，显式请求可计算 SHAP importance |
| 多口径评估 | `benchmark_cols` / `aux_targets` / `target_group_cols` | 多 benchmark、多辅助 target 和长短 y 独立评估 |
| PSI 口径 | `psi_include_missing` | 建模评估复用分箱评估器计算 Score/Feature PSI，只控制缺失箱是否纳入 |

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

::: mars.modeling.MarsModelDataSplitter

## Tuning

::: mars.modeling.MarsModelTuner

::: mars.modeling.MarsModelReplayRunner

## Evaluation 与 Prediction

::: mars.modeling.MarsModelEvaluator

::: mars.modeling.ModelPredictor

## Result Objects

::: mars.modeling.MarsModelTuningResult

::: mars.modeling.MarsModelReplayResult

::: mars.modeling.MarsFeatureGrowthResult

::: mars.modeling.MarsModelingReport
