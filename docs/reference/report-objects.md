# Report 对象

MARS 的 report 对象不只是导出文件，也保留多粒度结构化数据，便于复盘、看板接入和 Agent 二次加工。

## MarsProfileReport

::: mars.analysis.MarsProfileReport

常用字段：

- `overview_table`
- `dq_tables`
- `stats_tables`
- `get_profile_data()`

## MarsEvaluationReport

::: mars.analysis.MarsEvaluationReport

常用字段：

- `summary_table`
- `detail_table`
- `trend_tables`
- `missing_by_day_table`

## MarsMonitoringReport

::: mars.monitoring.MarsMonitoringReport

常用字段：

- `summary_table`
- `detail_table`
- `trend_tables`
- `bin_stat_table`
- `bin_stat_trend_tables`
- `target_observation_table`
- `metadata`

## Modeling Results

::: mars.modeling.results.MarsModelTuningResult

::: mars.modeling.results.MarsModelReplayResult

::: mars.modeling.report.MarsModelingReport

## MarsScorecard

::: mars.scoring.MarsScorecard

