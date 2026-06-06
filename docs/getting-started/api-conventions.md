# 核心 API 约定

MARS 的 API 设计遵循一个核心原则：构造函数保存稳定策略，方法入参保存本次运行上下文。这样同一个对象可以复用到不同数据集、不同特征范围和不同分组切片。

## 高层工作流：`df, target`

高层风控工作流处理完整业务表，目标变量是表中的列名，所以使用 `df, target`：

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    group_col="month",
)
```

适用对象包括：

- `MarsDataProfiler.generate_profile(df, ...)`
- `MarsBinEvaluator.evaluate(df, target=..., ...)`
- `profile_risk(df, target=..., ...)`
- `MarsStatsSelector.fit(df, target=..., ...)`
- `MarsMonitor.monitor(df, target=..., ...)`

## 底层算法对象：`X, y`

底层算法对象接近 sklearn 风格，处理特征矩阵和标签向量，所以使用 `X, y`：

```python
from mars.feature import MarsNativeBinner

binner = MarsNativeBinner(method="quantile", n_bins=5)
binner.fit(X, y, cat_features=["segment"])
X_woe = binner.transform(X, return_type="woe")
```

适用对象包括：

- `MarsNativeBinner.fit(X, y=None, ...)`
- `MarsOptimalBinner.fit(X, y, ...)`
- `MarsLinearSelector.fit(X, y, ...)`
- `MarsImportanceSelector.fit(X, y=None, ...)`

## 构造函数与方法入参

| 位置 | 放什么 | 示例 |
| --- | --- | --- |
| 构造函数 | 稳定策略、阈值、模型规格 | `missing_thr`、`iv_thr`、`model_type`、`seed` |
| 方法入参 | 数据、列名、特征范围、分组、时间、输出路径 | `df`、`target`、`features`、`group_col`、`time_col` |

不要把本次数据集、特征范围或目标列放在构造函数里。这样可以避免对象状态和方法入参冲突。

## 分组、时间和样本切片命名

| 参数 | 语义 |
| --- | --- |
| `group_col` | 已经存在的分组列，例如 `month`、`channel`、`product` |
| `time_col` | 原始日期列，例如 `apply_dt` |
| `time_grain` | 时间聚合粒度，例如 `"day"`、`"week"`、`"month"`、`"7d"` |
| `dataset_flag_col` | 建模样本切片列，例如 train/val/oot |

如果已经有 `month` 这样的分组列，直接传 `group_col="month"`。如果只有原始日期列，传 `time_col="apply_dt"` 和 `time_grain="month"`。

## report 对象

各模块 report 不只是导出文件，也保存多粒度结构化数据。

常见字段包括：

- `summary_table`：特征级或模型级汇总。
- `detail_table` / `detail_tables`：分箱、指标或切片明细。
- `trend_tables`：按时间或分组展开的趋势宽表。
- `metadata` / `report_meta`：运行上下文、参数和产出元数据。

这些表可以继续用于特征复盘、监控规则定制、内部看板接入，或借助 Agent 做定制化摘要和报告重排。

