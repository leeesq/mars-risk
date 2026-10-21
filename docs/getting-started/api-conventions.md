---
description: MARS 高层工作流、底层 estimator、运行参数和结构化返回对象的调用约定。
---

# 核心 API 约定

MARS 的构造函数保存可复用策略，方法参数接收本次运行的数据和列名。
高层工作流操作完整业务表，使用 `df, target`；底层分箱器和 sklearn 风格选择器使用 `X, y`。

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

你可以用同一个对象处理多个数据切片，每次调用都明确传入本次数据和列配置。

Stable API 默认采用 fail-closed 策略：缺列、缺少必需指标、空报告或导出失败时，接口会抛出明确异常。
逐特征处理宽表时，只有在诊断中记录失败特征，任务才允许部分成功；没有可用结果时，任务会终止。

## 结构化结果

分析人员和外部 Agent 可以查询同一个公共 Report。如何读取表目录、筛选行、选择列、分页、
核对实际参数与状态，以及保存和加载 `.marsreport`，见[外部 Agent 指南](../user-guide/external-agents.md)。
Reporting 属于当前核心；监控、建模（含 Pipeline）、评分卡暂停功能迭代并直接适配核心接口。

工作流优先返回 report 或结果对象。你可以先读取表格，再决定是否导出：

| 字段 | 常见用途 |
| --- | --- |
| `summary_table` | 特征或模型级汇总、排序和筛选 |
| `detail_table` / `detail_tables` | 分箱、指标或样本切片明细 |
| `trend_tables` | 按时间或业务分组展开的趋势 |
| `metadata` / `report_meta` | 指标口径、列角色和输出元数据 |

完整数据角色见[数据角色与运行边界](../concepts/data-and-runs.md)，report 生命周期见
[Report 与 Artifact](../concepts/reports-and-artifacts.md)。
