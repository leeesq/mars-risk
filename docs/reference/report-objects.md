---
description: MARS 结构化 report 和 Modeling 结果对象的字段索引。
---

# Report 对象

Report 保存汇总、明细和趋势等结构化统计结果，你可以直接查询这些结果，也可以导出文件。概念说明见
[Report 与 Artifact](../concepts/reports-and-artifacts.md)。

| 对象 | 状态 | 常用字段或查询表 |
| --- | --- | --- |
| `MarsProfileReport` | Stable | `overview_table`、`dq_tables`、`stats_tables` |
| `MarsBinningReport` | Stable | `summary_table`、`detail_table`、`trend_tables` |
| `CorrelationReport` | Stable | 查询表 `features`、`pairs`、`correlation_decisions` 和 `selection`；[相关性指南](../user-guide/correlation-and-score-cross.md#相关性报告) |
| `ScoreCrossReport`（`mars.analysis`） | Stable | 查询表 `cells`、`row_summary`、`column_summary`、`overall` 和 `bins`；[交叉指南](../user-guide/correlation-and-score-cross.md#固定分段交叉) |
| `MarsRuleReport` | Experimental | `summary_table`、`detail_tables` 和 `metadata`；支持公共 Report 查询、保存和按成员特征筛选规则 |
| `MarsMonitoringReport` | Experimental | `summary_table`、`detail_table`、`bin_stat_table`、`target_observation_table` 和 `metadata` |
| `MarsScorecard` | Experimental | `points_table`、`base_points`、`factor`、`offset` 等评分刻度参数；支持 SQL 导出 |
| `MarsPipelineResult` | Experimental | `active_features`、`step_results`、`modeling_result` 和 `metadata` |
| `MarsModelTuningResult` | Experimental | `best_model`、`history_table`、`retained_models`、`importance_table` 和 `artifact_path` |
| `MarsModelReplayResult` | Experimental | `ranking_table`、`leaderboard_table`、`models`、`scored_df` 和 `reports` |
| `MarsModelingReport` | Experimental | `summary_table`、`detail_tables` 和 `metadata` |

相关性和模型分交叉报告的表名用于 `get_table()` 等查询方法，其余条目列出对象属性。完整属性和类型以对应模块 Reference 为准。

原始规则报告与加载后的 `ReportSnapshot` 都支持 `describe()`、`get_table()`、`query_page()`、
`search_features()`、`get_feature()`、`to_ai_context()` 和 `save()`。
加载后得到通用统计快照，报告目录会保留规则与成员特征的关联；快照不会恢复 RuleSet 或赋予部署权限。
示例见[规则报告与外部 Agent](../user-guide/rule-reports-and-agents.md)，
完整签名见 [Rule API](rule.md)。
