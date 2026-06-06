# Report 对象

MARS 的 report 对象不只是导出文件，也保留多粒度结构化数据，便于复盘、看板接入和 Agent 二次加工。本页说明常用数据结构；完整签名和 docstring 请进入对应模块 API 页面查看。

## 对象地图

| 对象 | 来源模块 | 常用场景 | API 页面 |
| --- | --- | --- | --- |
| `MarsProfileReport` | 数据画像 | 数据质量、统计分布、PSI 和趋势表 | [Analysis](analysis.md) |
| `MarsEvaluationReport` | 分箱评估 | IV、KS、AUC、Lift、分箱明细和趋势 | [Analysis](analysis.md) |
| `MarsMonitoringReport` | 监控 | 特征/模型监控、target 表现覆盖率和报警摘要 | [Monitoring](monitoring.md) |
| `MarsModelTuningResult` | Modeling Pipeline | 调参结果、最优模型、重要性表和 artifact 元数据 | [Modeling Pipeline](modeling.md) |
| `MarsModelReplayResult` | Modeling Pipeline | Top-K replay、leaderboard、打分数据和评估报告 | [Modeling Pipeline](modeling.md) |
| `MarsModelingReport` | Modeling Pipeline | train/val/oot 或业务切片的建模评估报告 | [Modeling Pipeline](modeling.md) |
| `MarsScorecard` | Scoring | 评分卡映射、分数规则和 SQL 导出 | [Scoring](scoring.md) |

## 常用字段

| 对象 | 常用字段 |
| --- | --- |
| `MarsProfileReport` | `overview_table`、`dq_tables`、`stats_tables`、`get_profile_data()` |
| `MarsEvaluationReport` | `summary_table`、`detail_table`、`trend_tables`、`missing_by_day_table` |
| `MarsMonitoringReport` | `summary_table`、`detail_table`、`trend_tables`、`bin_stat_table`、`bin_stat_trend_tables`、`target_observation_table`、`metadata` |
| `MarsModelTuningResult` | `best_model`、`best_params`、`leaderboard`、`importance_table`、`metadata` |
| `MarsModelReplayResult` | `leaderboard`、`models`、`predictions`、`evaluation_report`、`importance_table` |
| `MarsModelingReport` | `summary_table`、`detail_tables`、`trend_tables`、`metadata` |
| `MarsScorecard` | `score_table`、`binning_rules`、`pdo`、`base_score` |

## 二次加工

这些对象中的表格通常可以继续用于：

- 生成内部看板或补充业务报表。
- 对特征、分组、月份或模型版本做定制化复盘。
- 交给 Agent 基于明细表做摘要、筛选、解释和报告重排。
- 将同一套结构化结果继续接入 Excel/HTML 导出链路。
