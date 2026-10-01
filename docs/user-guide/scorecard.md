---
description: Experimental 评分卡保留现有转换和部署能力，暂时停止功能迭代。
---

# 评分卡

Scoring 为 **Experimental**，暂时停止功能迭代。现有评分映射、SQL 与使用文档继续保留，
必要的正确性、运行修复及上游适配继续处理。暂停说明开发投入，不表示接口已 Stable。

已有分箱规则与逻辑回归系数的转换示例仍在
[报告指南的评分卡章节](reports-and-exports.md#4)，原入口继续有效。
精确 API 见[Scoring Reference](../reference/scoring.md)。

可以借助编程型 AI、MARS 核心分析与报告能力，以及自己的模型工具构建业务评分流程。
Scoring 直接适配核心接口变化，上游不为其保留旧字段、转发壳或双轨实现；完整规则见
[稳定性与兼容性](../project/stability.md#暂停模块的下游适配)。
