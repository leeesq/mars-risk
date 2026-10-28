---
description: MARS 公开 Python API 的模块索引和稳定性状态。
---

# API Reference

本节从源码中的公开 docstring 生成 API 文档，用于查询准确的签名、参数、返回值和异常。需要完成具体任务时，先阅读
[使用指南](../user-guide/data-profiling.md)。

| 模块 | 状态 | 内容 |
| --- | --- | --- |
| [Analysis](analysis.md) | Stable | 数据画像、风险评估和模型分交叉 |
| [Feature](feature.md) | Stable | 分箱器和特征选择器 |
| [Rule](rule.md) | Experimental | DSL、生成器、评估、挖掘、RuleSet 与报告 |
| [Monitoring](monitoring.md) | Experimental | 监控报告和报警入口 |
| [Agent](agent.md) | Experimental | Python 3.10+；风控工具调用、会话和报告证据 |
| [Reporting](reporting.md) | Stable | Report 与 HTML 渲染结果 |
| [Scoring](scoring.md) | Experimental | 评分卡与 SQL |
| [Modeling / Pipeline](modeling.md) | Experimental | 切分、调参、replay、评估、预测和编排 |

文档测试会核对公开模块的 `__all__`，确保本节引用了所有公开名称。

风险画像、数据画像与 StatsSelector 的客群参数、规则及查询合同见[分客群分析](../user-guide/segmented-analysis.md)。
参数精确顺序以各模块生成的签名为准。
