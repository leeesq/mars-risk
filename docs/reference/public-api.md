# 公开 API

本页使用 `mkdocstrings` 从源码 docstring 生成主要 public API 引用。自然语言说明以中文为主，API 名、参数名、类型名和生态术语保持英文。

## Analysis

::: mars.analysis.MarsDataProfiler

::: mars.analysis.profile_stats

::: mars.analysis.MarsBinEvaluator

::: mars.analysis.profile_risk

::: mars.analysis.MarsRiskProfile

## Feature

::: mars.feature.MarsNativeBinner

::: mars.feature.MarsOptimalBinner

::: mars.feature.MarsStatsSelector

::: mars.feature.MarsLinearSelector

::: mars.feature.MarsImportanceSelector

## Monitoring

::: mars.monitoring.MarsMonitor

::: mars.monitoring.MarsMonitoringAlertConfig

::: mars.monitoring.MarsMonitoringAlerter

::: mars.monitoring.generate_monitoring_alert

## Modeling Pipeline

!!! warning "快速迭代模块"
    Modeling Pipeline 仍在快速迭代中，接口约定、结果对象和调参参数后续可能发生较大变化。

::: mars.modeling.MarsModelingSession

::: mars.modeling.tuning.MarsModelTuner

::: mars.modeling.tuning.MarsModelReplayRunner

## Scoring

::: mars.scoring.MarsScorecard

::: mars.scoring.build_scorecard

