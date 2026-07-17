# Analysis

数据画像、分箱评估和高层风险画像入口。

先按任务阅读：[数据画像](../user-guide/data-profiling.md)用于质量与分布检查；
[分箱与风险评估](../user-guide/binning-risk-evaluation.md)说明建箱、benchmark、趋势图和导出。

!!! tip "选择入口"

    `profile_risk()` 适合一次性高层工作流；需要传入或复用固定 `binner` 时使用
    `MarsBinEvaluator.evaluate()`。风险趋势图需要有效 `time_col`，而 `profile_risk()` 不接收
    显式 `binner`。

## 数据画像

::: mars.analysis.MarsDataProfiler

::: mars.analysis.profile_stats

## 分箱评估

::: mars.analysis.MarsBinEvaluator

::: mars.analysis.profile_risk

::: mars.analysis.MarsRiskProfile
