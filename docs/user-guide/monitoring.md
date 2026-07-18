---
description: 使用固定或基准期分箱规则监控特征分布、模型分、表现覆盖率和报警摘要。
---

# 特征与模型监控

!!! warning "Experimental"

    Monitoring 的 report 字段、target 校验和报警结果仍可能调整。受控生产使用应固定
    `mars-risk==0.0.24`，并为依赖的 report 字段和报警结果增加契约测试。

## 适用场景

- 前端监控：申请入口、渠道、产品或地区等模型前特征的分布和缺失变化。
- 后端监控：模型输出分布、target 表现覆盖率和已表现样本风险指标。

调用方负责按监控周期提供当前数据和基准数据，并保存 report；MARS 不负责调度任务或发送通知。

## 完整调用

下面的示例使用带标签的 `baseline_df` 拟合规则，监控尚未表现的 `current_df`：

```python
--8<-- "docs/snippets/monitoring.py"
```

## Target 规则

target 的有效非空值只能是 `0`、`1`、`True` 或 `False`；空值表示尚未表现。字符串、`-1`、`2`
或其他非空类别会抛出 `ValueError`，调用方应在监控前完成清洗。

`target=None` 时只输出无标签分布监控。传入 target 时，PSI、缺失率和分箱占比使用全量样本；
坏账率、IV、KS、AUC 和 Lift 只使用已表现样本。

## 规则与基准

显式 `binner` 优先于 `benchmark_df`。基准样本负责提供分箱规则和 PSI expected distribution；
当前样本不需要已经充分表现。监督分箱时，基准 target 必须至少包含两个有效类别。

## 输出

| 字段 | 含义 |
| --- | --- |
| `summary_table` | 特征级监控汇总 |
| `detail_table` | 分箱明细 |
| `trend_tables` | PSI、缺失率、坏账率等趋势 |
| `bin_stat_table` | 每个特征和分箱的统计量 |
| `bin_stat_trend_tables` | 分箱统计量随分组变化的趋势 |
| `target_observation_table` | 已表现、未表现样本和观察坏账率 |
| `metadata` | 规则来源、趋势顺序和 PSI 口径 |

`trend_column_order="asc"` 从早到晚排列趋势列，`"desc"` 让最新分组靠前；`Total` 始终位于最后。

## 报警摘要

`generate_monitoring_alert()` 从已有 report 中读取指标并生成文本摘要。缺少某类表或字段时跳过对应
检查。调用方需要将摘要接入自己的阈值审批、通知和处置流程。

## 常见失败

- 当前期无标签却使用监督分箱且没有基准数据：提供带标签 `benchmark_df` 或改用 native 无监督分箱。
- 基准数据缺少监控特征：基准表必须覆盖全部 active features 和可选权重列。
- 把 target 字符串当二分类值：在调用前转换成整数、布尔值或空值。

## 下一步

- 理解基准期、当前期与 target 表现状态：[数据角色与运行边界](../concepts/data-and-runs.md)。
- 导出监控结果：[报告与评分卡](reports-and-exports.md)。
- 查询精确签名：[Monitoring API](../reference/monitoring.md)。
