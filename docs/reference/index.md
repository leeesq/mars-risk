---
description: MARS 公开 Python API 的模块索引和稳定性状态。
---

# API Reference

本节由源码公开 docstring 生成，用于查询精确签名、参数、返回值和异常。完成具体任务时先阅读
[使用指南](../user-guide/data-profiling.md)。

| 模块 | 状态 | 内容 |
| --- | --- | --- |
| [Analysis](analysis.md) | Stable | 画像、高层风险评估和 evaluator |
| [Feature](feature.md) | Stable | Binner 与 selector |
| [Rule](rule.md) | Experimental | DSL、生成器、评估、挖掘、RuleSet 与报告 |
| [Monitoring](monitoring.md) | Experimental | 监控 report 和报警入口 |
| [Reporting](reporting.md) | Stable | Report 与 HTML 渲染结果 |
| [Scoring](scoring.md) | Experimental | 评分卡与 SQL |
| [Modeling / Pipeline](modeling.md) | Experimental | 切分、调参、replay、评估、预测和编排 |

公开模块 `__all__` 与本节覆盖范围由文档测试自动核对。
