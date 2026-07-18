---
description: MARS 安装、入口选择、target、基准数据和 Experimental 模块的常见问题。
---

# FAQ

## MARS 和 toad / optbinning 是替代关系吗？

不是。MARS 提供 Polars-first 的画像、评估、筛选、监控和结构化报告工作流；需要经典评分卡工具链
或单独的数学规划分箱时，可以直接使用对应项目。MARS 的 `MarsOptimalBinner` 基于 optbinning
能力构建规则，并将规则接入其他 MARS 工作流。

## `profile_risk()` 返回什么？

返回 `MarsRiskProfile(report, binner, targets, metadata)`。需要显式复用已有规则时，使用
`MarsBinEvaluator.evaluate(..., binner=...)`。

## `benchmark_df` 会并入当前数据吗？

不会。它用于拟合基准规则和提供 PSI expected distribution，不进入当前 `df` 的 Total 或趋势分组。

## 为什么高层 API 用 `target`，底层算法用 `y`？

高层 API 面向包含列名的完整业务表；底层 estimator 面向特征矩阵与标签向量。

## 监控 target 可以传字符串 `"0"` 或 `"1"` 吗？

不可以。只接受整数/布尔二分类值和空值。调用前应完成类型清洗。

## Experimental 模块可以用于生产吗？

Monitoring、Modeling、Pipeline 和 Scoring 是 Experimental。它们可以用于实验和受控生产流程，
但必须固定精确版本，并为 report 字段、报警结果、评分映射、生成 SQL、step 契约、replay 和
artifact 路径增加契约测试。

## `artifact_dir=None` 会写文件吗？

不会。调参结果只保存在返回对象中。指定目录时才会创建独立运行目录。
