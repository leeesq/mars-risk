# Feature

特征分箱、特征筛选和底层 `fit` / `transform` 对象。

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
