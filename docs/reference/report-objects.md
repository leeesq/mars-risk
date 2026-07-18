---
description: MARS 结构化 report 和 Modeling 结果对象的字段索引。
---

# Report 对象

Report 保存多粒度结构化数据；导出文件只是可选呈现。概念说明见
[Report 与 Artifact](../concepts/reports-and-artifacts.md)。

| 对象 | 状态 | 常用字段 |
| --- | --- | --- |
| `MarsProfileReport` | Stable | `overview_table`、`dq_tables`、`stats_tables` |
| `MarsBinningReport` | Stable | `summary_table`、`detail_table`、`trend_tables` |
| `MarsMonitoringReport` | Experimental | 汇总、分箱统计、表现覆盖率、`metadata` |
| `MarsScorecard` | Experimental | `points_table`、评分刻度参数和 SQL 导出 |
| `MarsPipelineResult` | Experimental | active features、step 结果、建模结果、`metadata` |
| `MarsModelTuningResult` | Experimental | best model、history、retained models、importance、artifact |
| `MarsModelReplayResult` | Experimental | ranking、leaderboard、models、scored data、reports |
| `MarsModelingReport` | Experimental | `summary_table`、`detail_tables`、`trend_tables`、`metadata` |

精确属性和类型以对应模块 Reference 为准。
