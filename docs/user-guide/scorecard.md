---
description: Experimental 评分卡保留现有转换和部署能力，暂时停止功能迭代。
---

# 评分卡

Scoring 为 **Experimental**，暂时停止功能迭代。现有评分映射、SQL 与使用文档继续保留，
我们继续处理必要的正确性问题、运行修复和上游适配。暂停说明开发投入，接口成熟度仍为 Experimental。

已有分箱规则与逻辑回归系数的转换示例仍在
[报告指南的评分卡章节](reports-and-exports.md#4)，原入口继续有效。
精确 API 见[Scoring Reference](../reference/scoring.md)。

你可以结合编程型 AI、MARS 核心分析报告和自己的模型工具，构建业务评分流程。
核心接口变化时，Scoring 直接修改调用代码；核心代码不为 Scoring 保留旧字段、转发接口或新旧两套实现。完整规则见
[稳定性与兼容性](../project/stability.md#暂停模块的下游适配)。
