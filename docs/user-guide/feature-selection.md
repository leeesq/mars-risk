---
description: 使用统计、线性和重要性选择器缩小候选特征范围。
---

# 特征筛选

## 适用场景

特征筛选用于将宽表候选特征压缩成可建模、可监控的集合。MARS 提供统计筛选、线性筛选和重要性
筛选三类对象；它们可以独立使用，也可以进入 Pipeline。

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

## 基准样本

`MarsStatsSelector.fit(..., benchmark_df=...)` 使用基准数据拟合粗筛和精筛分箱，并构造 PSI expected
distribution。选择器采用双样本口径：质量、缺失率、分布和 PSI 使用完整 `df`；监督分箱、IV、
Lift、RC 和 WOE 相关性只使用 `df` 中 target 非空的已表现样本。当前数据仍必须包含至少两个
有效 target 类别；target 为空的最新未表现样本不会稀释标签类指标或 WOE 相关性。

Selector 不长期保存原始基准样本。调用 `get_binning_report()` 时，需要再次传入同一
`benchmark_df` 才能复现基准口径。

## 输出

| 字段 | 用途 |
| --- | --- |
| `selected_features_` | 最终保留的特征名列表 |
| `get_report()` | 每个特征的选择结果和指标记录 |
| `get_binning_report()` | 对保留特征生成结构化分箱评估报告 |

`white_list` 保护必须保留的业务特征；`black_list` 排除泄漏、合规受限或线上不可用字段。黑名单优先
于自动指标规则。

## 常见失败

- 静态筛选没有分组列却保留默认 PSI/RC：显式设置 `psi_thr=None, rc_thr=None`。
- `importance_table` 未定义或列结构不明：先由 Modeling 结果读取受支持的重要性表，再传给选择器。
- 筛选后没有特征：读取 `get_report()`，调整最早导致全量淘汰的阈值或名单规则。

## 下一步

- 将筛选器组合进建模流程：[Modeling / Pipeline](modeling-pipeline.md)。
- 对最终特征生成评估报告：[分箱与风险评估](binning-risk-evaluation.md)。
- 查询完整参数：[Feature API](../reference/feature.md)。
