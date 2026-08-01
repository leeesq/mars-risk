---
description: MARS 高层工作流、底层 estimator、运行参数和结构化返回对象的调用约定。
---

# 核心 API 约定

MARS 将稳定策略放在构造函数，将本次运行的数据和列名放在方法参数。高层工作流操作完整业务表，
使用 `df, target`；底层分箱器和 sklearn 风格选择器使用 `X, y`。

## 高层工作流

| 任务 | 入口 | 规则来源 |
| --- | --- | --- |
| 自动建箱并评估 | `profile_risk(df, ...)` | `benchmark_df` 或当前 `df` |
| 复用固定规则评估 | `MarsBinEvaluator.evaluate(df, binner=...)` | 显式 `binner` |
| 特征或模型监控 | `MarsMonitor.monitor(df, ...)` | `binner`、`benchmark_df` 或当前 `df` |
| 数据画像 | `MarsDataProfiler.generate_profile(df, ...)` | `benchmark_df` 或当前 `df` |

`profile_risk()` 不接受 `binner` 参数。需要固定或外部恢复的规则时，使用
`MarsBinEvaluator.evaluate()`。

## 构造参数与运行参数

| 位置 | 内容 | 示例 |
| --- | --- | --- |
| 构造函数 | 可跨运行复用的策略、阈值和模型规格 | `missing_thr`、`iv_thr`、`model_type`、`seed` |
| 方法参数 | 本次数据、列名、分组、日期、特征范围和输出路径 | `df`、`target`、`group_col`、`time_col` |

同一个对象可以服务多个数据切片，而不会依赖上一次调用留下的隐式数据上下文。

Stable API 默认 fail-closed：缺列、缺少必需指标、空报告和导出失败均抛出明确异常。
逐特征宽表任务只在诊断结构中记录了失败特征时允许部分成功；零可用结果会终止。

## 结构化结果

工作流优先返回 report 或结果对象，而不是只写文件。调用方可以先读取表格，再决定是否导出：

| 字段 | 常见用途 |
| --- | --- |
| `summary_table` | 特征或模型级汇总、排序和筛选 |
| `detail_table` / `detail_tables` | 分箱、指标或样本切片明细 |
| `trend_tables` | 按时间或业务分组展开的趋势 |
| `metadata` / `report_meta` | 指标口径、列角色和输出元数据 |

完整数据角色见[数据角色与运行边界](../concepts/data-and-runs.md)，report 生命周期见
[Report 与 Artifact](../concepts/reports-and-artifacts.md)。
