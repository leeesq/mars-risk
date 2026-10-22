---
description: 使用统计、线性和重要性选择器缩小候选特征范围。
---

# 特征筛选

先看[案例 3：为什么保留这个，删除另一个？](../demos/selection-correlation.md)，
用真实决策和[相关性证据](../demos/selection-correlation.md#correlation)复核取舍。

## 适用场景

用特征筛选从宽表中选出适合建模和监控的候选字段。MARS 提供统计、线性和重要性三类选择器，
可以独立使用，也可以组合到 Pipeline 中。

## 统计筛选完整调用

```python
--8<-- "docs/snippets/feature_selection.py"
```

示例关闭了 PSI 和 RC，因为它只做静态筛选。默认 `psi_thr` 和 `rc_thr` 已启用；使用默认值时必须
提供 `group_col` 或 `time_col`。

## 选择器职责

| 选择器 | 输入风格 | 主要用途 |
| --- | --- | --- |
| `MarsStatsSelector` | `fit(df, target=...)` | 质量、IV/Lift、PSI、RC、相关性和黑白名单 |
| `MarsLinearSelector` | `fit(X, y)` | L1、相关性和线性模型筛选 |
| `MarsImportanceSelector` | `fit(X, y)` 或重要性表 | 模型原生重要性、SHAP 或已有重要性结果 |

`MarsImportanceSelector.method` 只支持 `"importance"` 和 `"shap"`。
`"rfe"` 和 `"sfm"` 未实现，构造时会立即报错。

## 基准样本

`MarsStatsSelector.fit(..., benchmark_df=...)` 用基准数据拟合粗筛和精筛分箱，并提供 PSI 的参考分布。
监督分箱只使用拟合数据中 target 非空的已表现样本；提供基准表时，拟合数据就是 `benchmark_df`，否则使用 `df`。
评估当前数据时，质量、缺失率、分布和 PSI 使用完整 `df`；IV、Lift、RC 和 WOE 相关性只使用 `df` 的已表现样本。
当前数据仍须包含至少两个有效 target 类别。
最新未表现样本的 target 为空，不会稀释标签类指标或 WOE 相关性。

Selector 不长期保存原始基准样本。调用 `get_binning_report()` 时，需要再次传入同一
`benchmark_df` 才能复现基准口径。

## 宽表执行与批宽

`batch_size` 是特征列批宽，默认和 `None` 都取 100，必须为正整数。
Quality 将它传给画像的 `overview_batch_size`；Rough Performance、Fine 和 Stability
PSI/RC 使用同一批宽。Rough Fit 和 Fine Fit 仍按算法学习规则，不能将拟合总体误认为行批样本。
已表现标签掩码只决定监督统计人口。无监督分箱和分布统计继续使用完整拟合人口。

表现统计先投影本批特征，再过滤已表现标签、转换分箱并聚合。benchmark 的规则校验只读取 lazy
计划的 schema；expected distribution 在当前批内聚合。原始列的投影和 lazy 计划可以共享输入
buffer，不等于另复制一份宽表。输入表、拟合所需缓存、规则及保留的报告仍有成本。
`n_jobs` 继续控制分箱任务，Polars 和 BLAS 线程由各自环境配置控制。

全局 WOE Pearson 使用 Stage 3 已拟合映射，只将 WOE null 填为零。每个内部行块最多 4,096 行，
且按特征数缩小，使一份 Float64 行块通常不超过 8 MiB（至少一行）。内部行块独立于列批宽。
行块内再按 `batch_size` 转换，使用 Float64 中心矩累计完整候选矩阵，不保存完整人口的 WOE 表。
时间复杂度为 O(NP²)，中心矩及完整报告仍为 O(P²)。内存另包含当前块、列批、规则、原始数据和报告。
批宽既不是内存硬上限，也不是行数或并发数。

累计使用首行平移和 Chan 中心矩合并，避免大均值小方差下的平方和相减。结果按 NumPy 实数
Pearson 的方式限制在 [-1, 1]，常量及不可用对角线保留不可用。浮点舍入范围内接近阈值的 pair
使用两列、完整监督人口的 NumPy `corrcoef` 复核。该复核可能增加时间及 O(N×2) 工作内存，
两列复核独立于列批宽，因此 `batch_size=1` 时也会生成这一对列。
复核不会改阈值或严格 `abs(corr) > corr_thr` 的规则。IV 降序、特征 ID 升序、白名单和最终原候选顺序不变。
完整低相关 pair、signed 值、实际决策和快照查询均保留。

容量记录及显式本地冻结输入入口见 [M12 上游验收](../performance/m12-upstream.md)。

## 输出

| 字段 | 用途 |
| --- | --- |
| `selected_features_` | 最终保留的特征名列表 |
| `get_report()` | 每个特征的选择结果和指标记录 |
| `get_binning_report()` | 对保留特征生成结构化分箱评估报告 |

`white_list` 保护必须保留的业务特征；`black_list` 排除泄漏、合规受限或线上不可用字段。黑名单优先
于自动指标规则。

三个 selector 的 `transform()` 默认要求输入包含全部已选特征，缺列时会报错。
调用方已处理 schema drift、确实允许缺列时，再显式传入 `on_missing="warn"` 或 `"ignore"`。
报告缺少必需指标列时，筛选会终止；某个特征的指标缺行、为空或非有限值时，
选择器会淘汰该特征，并在决策报告中记录 `metric_unavailable`。

## 常见失败

- 静态筛选没有分组列却保留默认 PSI/RC：显式设置 `psi_thr=None, rc_thr=None`。
- `importance_table` 未定义或列结构不明：先由 Modeling 结果读取受支持的重要性表，再传给选择器。
- 筛选后没有特征：读取 `get_report()`，调整最早导致全量淘汰的阈值或名单规则。

## 下一步

- 将筛选器组合进建模流程：[Modeling / Pipeline](modeling-pipeline.md)。
- 对最终特征生成评估报告：[分箱与风险评估](binning-risk-evaluation.md)。
- 查询完整参数：[Feature API](../reference/feature.md)。
