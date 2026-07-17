# Feature

特征分箱、特征筛选和底层 `fit` / `transform` 对象。

本页是精确签名索引。先读[分箱与风险评估](../user-guide/binning-risk-evaluation.md)了解
`native`、`lite_opt`、`optimal` 的选择与报告链路，再读[特征筛选](../user-guide/feature-selection.md)
了解筛选器的组合方式。

!!! note "底层接口"

    Binner 使用 `fit(X, y)` / `transform(X)`，由调用方持有规则；高层风险评估使用 `df, target`
    并负责构造 report。不要将高层 `target` 参数与底层 `y` 混用。

## Binner

所有分箱器都继承 `MarsBinnerBase`，因此共享 `fit_transform`、`transform`、
`profile_bin_performance`、`to_dict/from_dict`、`prune` 等规则转换、评估和序列化能力。

::: mars.feature.MarsBinnerBase

::: mars.feature.MarsNativeBinner

::: mars.feature.MarsLiteOptBinner

::: mars.feature.MarsOptimalBinner

## Selector

::: mars.feature.MarsStatsSelector

::: mars.feature.MarsLinearSelector

::: mars.feature.MarsImportanceSelector
