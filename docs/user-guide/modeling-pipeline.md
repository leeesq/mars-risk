---
description: 使用 Experimental Modeling/Pipeline 完成样本切分、调参、replay 和预测。
---

# Modeling / Pipeline

!!! warning "Experimental"

    Modeling 和 Pipeline 的参数、结果对象与 artifact 结构仍可能在 `0.0.x` 版本间调整。生产流程应
    固定精确版本，并为依赖字段增加契约测试。

## 适用场景

`mars.modeling` 负责样本切分、模型调参、trial replay、评估和特征重要性；`mars.pipeline` 将筛选、
可选 WOE 分箱和建模串成一个可复用流程。

支持 LightGBM、XGBoost、CatBoost 和 Logistic Regression。使用前安装：

```bash
pip install "mars-risk[ml,tuning]==0.0.26"
```

## Pipeline 完整调用

```python
--8<-- "docs/snippets/modeling_pipeline.py"
```

示例使用时间列严格切分 train/val/oot，只运行一次轻量 LightGBM trial，并设置
`artifact_dir=None` 避免写文件。

## Session 工作流

不需要 Pipeline 时，可以直接使用 `MarsModelingSession`：

| 阶段 | 方法 | 输出 |
| --- | --- | --- |
| 时间切分 | `slice()` | 带 `dataset_flag` 的样本 |
| 调参 | `tune()` | `MarsModelTuningResult` |
| 指定 trial 复盘 | `replay()` / `MarsModelReplayRunner` | `MarsModelReplayResult` |
| 建模评估 | `evaluate()` | `MarsModelingReport` |
| 特征增长 | `tune_incrementally()` | `MarsFeatureGrowthResult` |

构造 Session 时固定模型类型、特征、target、优化指标和随机种子；数据、时间列、切分比例和输出目录
属于单次方法调用。

## Pipeline 约束

- `MarsSelectionStep` 可以出现多次，每步只消费上一阶段 active features。
- `MarsWOEBinningStep` 主要服务 LR/评分卡；树模型通常直接使用筛选后的原始特征。
- `MarsModelingStep` 最多出现一次且必须位于最后。
- 任一筛选步骤筛空特征时立即抛出 `ValueError`。

## Artifact 与 Replay

`artifact_dir=None` 表示完全不落盘。指定目录时，每次调参生成独立运行目录，保存 history、配置、
元数据、重要性和保留模型。Replay 可以按 Top-K 或 `trial_nums` 复现候选模型；未保留模型需要设置
`retrain=True`。

## 常见失败

- 缺少 `ml,tuning` extra：按本页安装命令补齐模型和 Optuna 依赖。
- 时间切分没有足够的 train/val/oot 样本：扩大确定性示例或调整切分比例。
- Pipeline 提供 `split_ratios` 却没有 `time_col`：两者必须同时配置。
- 升级后读取旧 artifact 失败：按[稳定性与兼容性](../project/stability.md)固定版本并检查 Release Notes。

## 下一步

- 查看完整端到端流程：[LightGBM 建模与监控示例](../demos/lgb-modeling-monitoring.ipynb)。
- 对模型分和入模特征做周期监控：[特征与模型监控](monitoring.md)。
- 查询全部结果对象：[Modeling / Pipeline API](../reference/modeling.md)。
