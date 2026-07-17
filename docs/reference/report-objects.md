# Report 对象

report 对象保存多粒度结构化数据。本页列出常用字段；完整签名和 docstring 请进入对应模块 API 页面查看。

## 对象地图

| 对象 | 来源模块 | 常用场景 | API 页面 |
| --- | --- | --- | --- |
| `MarsProfileReport` | 数据画像 | 数据质量、统计分布、PSI 和趋势表 | [Reporting](reporting.md) |
| `MarsBinningReport` | 分箱评估 | IV、KS、AUC、Lift、分箱明细和趋势 | [Reporting](reporting.md) |
| `MarsMonitoringReport` | 监控 | 特征/模型监控、target 表现覆盖率和报警摘要 | [Monitoring](monitoring.md) |
| `MarsPipelineResult` | Pipeline 编排 | 每个 step 的输入输出、特征映射、最终特征和建模结果 | [Modeling / Pipeline](modeling.md) |
| `MarsModelTuningResult` | Modeling 建模 | 调参结果、最优模型、retained models、重要性表和 artifact 元数据 | [Modeling / Pipeline](modeling.md) |
| `MarsModelReplayResult` | Modeling 建模 | Top-K / 指定 trial replay、leaderboard、打分数据和评估报告 | [Modeling / Pipeline](modeling.md) |
| `MarsModelingReport` | Modeling 建模 | train/val/oot 或业务切片的建模评估报告 | [Modeling / Pipeline](modeling.md) |
| `MarsScorecard` | Scoring | 评分卡映射、分数规则和 SQL 导出 | [Scoring](scoring.md) |

## 常用字段

| 对象 | 常用字段 |
| --- | --- |
| `MarsProfileReport` | `overview_table`、`dq_tables`、`stats_tables`、`get_profile_data()` |
| `MarsBinningReport` | `summary_table`、`detail_table`、`trend_tables`、`missing_by_day_table` |
| `MarsMonitoringReport` | `summary_table`、`detail_table`、`trend_tables`、`bin_stat_table`、`bin_stat_trend_tables`、`target_observation_table`、`metadata` |
| `MarsPipelineResult` | `active_features`、`selected_features`、`feature_map`、`step_results`、`modeling_result`、`metadata` |
| `MarsModelTuningResult` | `best_model`、`best_params`、`history_table`、`retained_models`、`retained_model_table`、`importance_table`、`importance_tables`、`artifact_path`、`run_id`、`metadata` |
| `MarsModelReplayResult` | `ranking_table`、`leaderboard_table`、`models`、`scored_df`、`reports`、`importance_tables`、`diagnostic_tables` |
| `MarsModelingReport` | `summary_table`、`detail_tables`、`trend_tables`、`metadata` |
| `MarsScorecard` | `score_table`、`binning_rules`、`pdo`、`base_score` |

## 二次加工

这些对象中的表格可继续用于：

- 按特征、分组、月份或模型版本筛选与复盘。
- 将 `summary_table`、`detail_table`、`trend_tables` 和 `metadata` 交给 Agent 做摘要、筛选、
  解释和报告重排；调用方负责选择模型、提供上下文并校验结果。
- 使用 `write_excel()` 或 `write_html()` 导出结果。
