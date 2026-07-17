---
description: MARS 高层工作流、分箱规则来源、时间分组和 report 对象的核心 API 契约。
---

# 核心 API 约定

MARS 把稳定策略放在构造函数，把本次运行的数据与列名放在方法参数中。高层工作流使用
`df, target`；底层分箱器和选择器使用更接近 sklearn 的 `X, y`。

## 入口怎么选

| 目标 | 推荐入口 | 规则来源 |
| --- | --- | --- |
| 一次性风险评估 | `profile_risk(df, ...)` | 自动按高层参数构建 |
| 复用或传入已有分箱器 | `MarsBinEvaluator.evaluate(df, binner=...)` | 显式 `binner` |
| 特征/模型监控 | `MarsMonitor.monitor(df, ...)` | 显式 `binner`、benchmark 或当前期 |
| 底层分箱转换 | `Mars*Binner.fit(X, y)` / `transform(X)` | 调用方自行管理 |

`profile_risk()` 是便捷编排入口，**不接受** `binner` 参数。需要固定规则时，使用
`MarsBinEvaluator.evaluate()`；多 target 的 `profile_risk()` 会按首个 target 拟合一次，并复用
该规则处理后续 target。

## 构造函数与方法参数

| 位置 | 放什么 | 示例 |
| --- | --- | --- |
| 构造函数 | 稳定策略、阈值、模型规格 | `missing_thr`、`iv_thr`、`model_type`、`seed` |
| 方法参数 | 数据、列名、特征范围、分组、日期和输出路径 | `df`、`target`、`features`、`group_col`、`time_col` |

这样同一个对象可复用于不同月份、客群或样本切片，而不会把上一轮数据上下文保存在实例状态中。

## 分组、日期与趋势图

| 参数 | 职责 | 何时生效 |
| --- | --- | --- |
| `group_col` | 已有分组，例如 `month`、`channel`、`segment` | 最高优先级，决定报表与图表面板分组 |
| `time_col` | 原始日期，例如 `apply_dt` | 风险趋势图的唯一时间范围来源；也可配合粒度生成分组 |
| `time_grain` | `"day"`、`"week"`、`"month"`、`"7d"` 等时间聚合粒度 | 仅未传 `group_col` 时根据 `time_col` 生成时间分组 |
| `dataset_flag_col` | train/val/oot 等建模样本切片 | 只服务 Modeling，不等同于趋势分组 |

例如已有 `month` 时传 `group_col="month", time_col="apply_dt"`。趋势图按 `month` 展开，
但左上角始终显示 `apply_dt` 的真实最小/最大日期，精确到日。只有日期没有现成分组时，才传
`time_col="apply_dt", time_grain="month"`。

## `benchmark_df` 与分箱规则

`benchmark_df` 是基准期样本，不是会被并入当前期明细或 Total 的额外数据。它会提供 PSI 的
expected distribution，并可在当前期未表现时提供监督分箱标签。

| API | 规则来源优先级 |
| --- | --- |
| `MarsBinEvaluator.evaluate` | 显式 `binner` → `benchmark_df` → 当前 `df` |
| `MarsMonitor.monitor` | 显式 `binner` → `benchmark_df` → 当前 `df` |
| `profile_risk` | `benchmark_df` → 当前 `df`；该入口没有 `binner` 参数 |

监督分箱需要拟合期至少有两个有效 target 类别。若当前期 target 缺列或全空，但 benchmark 有效，
仍可用 benchmark 监督建箱，当前期输出保持无标签模式。默认 RC 基准仍是当前期 Total；只有
`risk_corr_baseline="benchmark"` 才使用 benchmark 坏率。

## report 是结构化结果，不只是文件

| 字段 | 用途 |
| --- | --- |
| `summary_table` | 特征或模型级汇总、排序和首页摘要 |
| `detail_table` / `detail_tables` | 分箱、指标和分组明细 |
| `trend_tables` | 按时间或分组展开的宽表趋势 |
| `metadata` / `report_meta` | 运行上下文、指标口径和输出元数据 |

查看[Report 对象](../reference/report-objects.md)了解各类 report 的具体字段；查看
[报告导出与二次加工](../user-guide/reports-and-exports.md)了解 HTML、Excel 和图表资产的交付方式。
