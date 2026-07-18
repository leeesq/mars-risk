---
description: 使用 MarsDataProfiler 检查宽表的数据质量、分布和分组漂移。
---

# 数据画像

## 适用场景

在分箱或建模前使用数据画像回答以下问题：字段是否可用，缺失和特殊值是否异常，分布是否稳定，
不同时间或业务分组之间是否发生漂移。

## 前置条件

- 输入是 Pandas 或 Polars DataFrame。
- `features` 只包含需要画像的业务特征，不包含 target、分组和日期列。
- 业务缺失码和特殊值需要显式配置，例如 `-999`。

## 完整调用

```python
--8<-- "docs/snippets/data_profiling.py"
```

`MarsDataProfiler` 的构造函数保存可复用的缺失值等策略；`generate_profile()` 接收本次数据、特征、
分组和指标。

## 输出

| 字段 | 用途 |
| --- | --- |
| `overview_table` | 特征级质量和统计汇总 |
| `dq_tables` | 缺失、零值等数据质量趋势 |
| `stats_tables` | 均值、分位数等统计趋势 |
| `get_profile_data()` | 以结构化对象返回 overview、DQ 和统计趋势 |

## 分组与日期

已有月份、渠道或客群列时传 `group_col`。只有原始日期时，传 `time_col` 与 `time_grain` 生成分组。
同时存在时，`group_col` 决定面板分组，`time_col` 保留日期语义。

## PSI 口径

画像 PSI 会先对数值或类别特征分箱，再比较各分组的分布。缺失率通常单独观察，因此默认不把缺失
箱和特殊值箱纳入 PSI。需要复现其他口径时显式设置 `psi_include_missing` 和
`psi_include_special`。

数值特征可通过 `psi_merge_small_bins` 和 `psi_min_bin_size` 控制小箱合并；类别特征通过
`psi_n_bins` 控制保留的 Top-K 类别，其余类别进入 Other 箱。

## 常见失败

- `features` 包含不存在的列：先固定本次特征清单，不要依赖全表自动推断。
- 业务缺失码仍进入均值或分布：在构造 `MarsDataProfiler` 时传入 `missing_values`。
- 期望时间趋势但只提供月份字符串：需要原始日期语义时同时传入 `time_col`。

## 下一步

- 对可用特征计算 IV、KS 和风险趋势：[分箱与风险评估](binning-risk-evaluation.md)。
- 用质量、区分度和稳定性规则压缩特征集合：[特征筛选](feature-selection.md)。
- 查询全部参数：[Analysis API](../reference/analysis.md)。
