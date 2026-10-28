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

## 分客群统计筛选

`segment_col` 放在 `fit()` 的本次数据上下文中。它自动排除在默认候选特征之外，
不能同时声明为特征、目标、分组或时间列。原有 `group_col` 仍优先于时间分组。

| `selection_scope` | `binning_scope` | 行为 |
| --- | --- | --- |
| `global` | `shared` | 保留整体筛选；客群列只登记角色，不生成客群名单或报告 |
| `per_segment` | `shared` | All 和各客群独立筛选，分阶段共享整体拟合规则及转换映射 |
| `per_segment` | `per_segment` | All 和各客群分别拟合、分别筛选 |
| `global` | `per_segment` | 入口拒绝，尚未定义多个客群证据如何合成整体名单 |

两个策略参数保存在构造函数中。未提供 `segment_col` 时，只能使用默认的
`global + shared`，原有表结构和整体名单保持不变。

```python
import polars as pl
from mars.feature import MarsStatsSelector

df = pl.DataFrame([
    {
        "customer_segment": segment,
        "dataset": "Train" if i < 40 else "Val",
        "target": i % 2,
        "x": float(1 - i % 2 if segment == "seg2" else i % 2),
        "noise": float(i % 5),
    }
    for segment in ("seg1", "seg2", "seg3")
    for i in range(80)
])
benchmark = df.filter(pl.col("dataset") == "Train")
selector = MarsStatsSelector(
    selection_scope="per_segment",
    binning_scope="shared",  # 改为 per_segment 可分别拟合各客群。
    skip_fine_scan=True,
    psi_thr=None,
    rc_thr=None,
    corr_thr=None,
    n_jobs=1,
)
selector.fit(
    df,
    target="target",
    features=["x", "noise"],
    segment_col="customer_segment",
    group_col="dataset",
    benchmark_df=benchmark,
)

print(selector.selected_features_)  # 整体名单。
print(selector.scopes_)  # 原始客群身份、稳定 scope_id 和状态。
print(selector.selected_features_by_segment_)  # 以 scope_id 索引各份名单。
decisions = selector.get_report(segments=["seg2"], include_all=False)
seg2 = selector.transform(df, segments=["seg2"], include_all=False)
report = selector.get_binning_report(
    df,
    benchmark_df=benchmark,
    segments=["seg2"],
    include_all=False,
)
summary = report.get_table("summary")
```

All 从完整人口独立计算。Total 表示当前 scope 跨原分组的汇总。
各客群从同一份原始候选集开始，整体质量或 IV 已淘汰的字段仍可在某个客群入选。
共享模式按阶段覆盖各客群所需候选并集，整体最终名单的剪枝不会删除其他客群的规则。
共享切点适合在相同区间比较客群；独立切点描述各自人口，箱号相同不代表区间相同。

`selected_features_`、默认 `transform()`、`get_report()` 和漏斗摘要始终使用 All。
`selected_features_by_segment_` 通过目录中的 `scope_id` 区分业务原值，包含独立的 All 身份。
真实字符串 `All`、`Missing`、空字符串和缺失客群不会与整体身份混淆。
空列表表示合法空名单；`None` 表示客群未计算或失败，具体原因保存在 `scopes_`。
监督拟合的单类别人口、缺少拟合参考和没有已表现标签分别保留状态，不借用整体名单。

`transform()` 显式选择客群后，仅返回该客群的行，保持原始相对顺序，
`keep_target=True` 时目标列使用同一批行。`fit_transform()` 使用相同选择规则。
一次只能选择一个 scope，指定业务客群时须传 `include_all=False`。
多份名单不会自动拼成并集、交集或按行路由的宽表。

`get_report()`、`show_summary()`、`get_binning_report()` 和 `get_correlation_report()`
都接受同一组 `segments`、`include_all` 参数。相关报告保留当前 scope 筛选时的完整矩阵和事件，
`get_correlation_matrix()`、`get_related_features()` 及矩阵展示只消费这份单 scope 证据。
生成风险报告时仍须重传拟合使用的 benchmark。报告快照保存统计和规则身份；
固定转换需要另存正式 Binner 规则，两类产物的用途见[保存与恢复](external-agents.md)。

PSI 的 expected 人口独立于切点共享方式。seg2 只使用 benchmark 的 seg2，
缺少该客群参考时 PSI 保留 null 和未计算原因，不回退整体分布。
没有显式 benchmark 时，各客群使用整体排序后的同一首组；缺少该组也不改用自己的下一期。
均值、缺失率等质量统计继续使用当前 scope 的完整人口，标签类指标只使用已表现人口。

共享转换映射在整体拟合人口的已表现标签上学习。粗筛使用无标签的无监督 benchmark 时，
边界仍由 benchmark 学习，公共 WOE 映射沿粗筛合同在整体 `df` 的已表现人口学习一次。
实际人口记录在相关报告的 `woe_mapping_population` 参数中。
报告中的客群统计 WOE 与用于转换的已学映射分别保存，不因某个客群更新或剪枝而串值。

客群按顺序执行并释放样本缓存。当前 shared 粗筛会额外预评估一次，
用于确定精筛的候选并集；各 scope 的统计转换仍各自执行。
这会增加计算成本，尚未实现每个特征批只转换一次后联合聚合的优化。

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
