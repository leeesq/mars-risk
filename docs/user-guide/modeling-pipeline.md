---
description: 使用 Experimental Modeling/Pipeline 完成样本切分、调参、replay 和预测。
---

# Modeling / Pipeline

建模（包含 Modeling 和 Pipeline）暂时停止功能迭代，现有入口与使用文档保留。只处理必要的正确性、运行修复和上游适配。
Experimental 说明接口成熟度，与暂停独立。可借助编程型 AI、MARS 分析与报告能力及自己的模型工具构建定制流程。
本模块直接适配上游，规则见[稳定性](../project/stability.md#暂停模块的下游适配)。

!!! warning "Experimental"

    Modeling 和 Pipeline 的参数、结果对象与 artifact 结构仍可能在 `0.0.x` 版本间调整。生产流程应
    固定精确版本，并为依赖字段增加契约测试。

## 适用场景

`mars.modeling` 负责样本切分、模型调参、trial replay、评估和特征重要性；`mars.pipeline` 将筛选、
可选 WOE 分箱和建模串成一个可复用流程。

支持 LightGBM、XGBoost、CatBoost 和 Logistic Regression。使用前安装：

```bash
pip install "mars-risk[ml,tuning] @ git+https://github.com/leeesq/mars-risk.git"
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

## 候选排序与外部选择政策

`tune()` 的优化目标主要来自 val；默认 replay 并非“先按 TEST 取 10 个，再综合 OOT 选最终模型”。
它先保留 history 中 `trial_state=COMPLETE` 且 `is_valid` 的行，再寻找列名含 `oot`
且以 `_<sort_metric>` 结尾的列；`include_val=True` 时另加精确的 `val_<sort_metric>` 列，
按 Pandas 逐行均值（跳过缺失值）计算 `custom_mean_score`。本轮未改变切片和公式。
没有识别到排序列时仍报错，即使显式传了 trial_nums 也须有这些列。

| 参数 | 实际作用 |
| --- | --- |
| `top_k` | 未提供 trial_nums 时取均值排序前 K 行 |
| `sort_metric` | 选择排序列的指标后缀，名称小写规范化 |
| `include_val` | 是否加入识别到的 val 列，不会自动加入 TEST |
| `metric_directions` | 本次逐项覆盖 > 历史配置 > 项目默认；排序和 backend 共用解析结果 |
| `trial_nums` | 回放显式候选，保持给定顺序，top_k 不参与选择 |

`metric_directions=None` 是未提供覆盖，`{}` 是无覆盖项；都保留历史。部分覆盖不丢失其他方向。
结果的 `metric_directions` 及 replay artifact 记录实际方向。均值排序不是普适最优模型选择，
相同均值的默认排序不作为外部并列政策。

外部流程先取得候选的真实 TEST 评估，声明方向、候选数量、资格条件和并列处理，再生成 trial_nums。
下面是**外部表约定**，不是新增的 MARS history 字段；缺少 TEST 结果时必须先评估候选，
不能将 val 重命名为 test。示例无需模型 SDK、LLM 或重训练：

```python
--8<-- "docs/snippets/external_candidate_selection.py"
```

已有 `runner`、`tuning_result` 和 `df` 时调用现有接口：

```python
replay = runner.replay(
    tuning_result, df, trial_nums=trial_nums,
    metric_directions={"ks": "maximize"}, retrain=False,
)
```

`retrain=False` 仅适用于 tune 已保留这些模型的情况，否则使用 `retrain=True`。
最终选择在外部业务代码／Agent 中完成。预先配置实际 TEST／OOT 切片、指标方向、稳定性约束、
客群表现、复杂度及成本权衡；资格阈值由业务提供，本例只读显式 eligible 标记。
读取 `replay.reports` 的真实评估表后保存最终记录，例如：

```python
final_record = {
    "candidate_trials": trial_nums,
    "selection_policy": {"candidate_slice": "TEST", "metric": "ks",
                         "direction": "maximize", "candidate_count": 10,
                         "tie_policy": "trial_num ascending"},
    "final_trial": chosen_trial,  # 外部业务代码根据已计算证据决定
    "evidence": evidence_references,  # 实际表、样本切片、指标及约束检查记录
    "reason": selection_reason,
}
```

`chosen_trial`、证据和理由由外部流程产生，上段是记录结构示意，不能替代决策。
用户可按业务工程需要使用 OOT；MARS 不强制 OOT 权限政策，也不自动依据 OOT 无限扩大搜索。

## 常见失败

- 缺少 `ml,tuning` extra：按本页安装命令补齐模型和 Optuna 依赖。
- 时间切分没有足够的 train/val/oot 样本：扩大确定性示例或调整切分比例。
- Pipeline 提供 `split_ratios` 却没有 `time_col`：两者必须同时配置。
- 升级后读取旧 artifact 失败：按[稳定性与兼容性](../project/stability.md)固定版本并检查 Release Notes。

## 下一步

- 查看完整端到端流程：[LightGBM 建模与监控示例](../demos/lgb-modeling-monitoring.ipynb)。
- 对模型分和入模特征做周期监控：[特征与模型监控](monitoring.md)。
- 查询全部结果对象：[Modeling / Pipeline API](../reference/modeling.md)。
