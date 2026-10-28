---
description: MARS 安装、入口选择、target、基准数据和 Experimental 模块的常见问题。
---

# FAQ

## MARS 和 toad / optbinning 是替代关系吗？

MARS 主要使用 Polars 计算画像、分箱评估、特征筛选和监控指标，并返回结构化报告。
如果你需要经典评分卡工具链，或单独使用数学规划分箱，可以选择 toad 或 optbinning。
MARS 的 `MarsOptimalBinner` 使用 optbinning 拟合分箱规则，供其他 MARS 分析入口复用。

## `profile_risk()` 返回什么？

返回 `MarsRiskProfile(report, binner, targets, metadata)`，分别保存报告、分箱器、目标列和本次运行元数据。
如果你要复用已有分箱规则，使用
`MarsBinEvaluator.evaluate(..., binner=...)`。

## `benchmark_df` 会并入当前数据吗？

不会。MARS 用 `benchmark_df` 拟合基准分箱规则，并计算 PSI 的参考分布（expected distribution）。
基准样本不计入当前 `df` 的 Total 或趋势分组。

## 共享分箱是否让所有客群使用整体 PSI 基准？

共享的是固定规则。PSI 的 expected 来自 benchmark 中对应客群；缺客群参考会记录未计算，
不会借用整体分布。All 与当前 scope 的 Total、独立名单和单 scope Excel 用法见[分客群分析](user-guide/segmented-analysis.md)。

## 为什么高层 API 用 `target`，底层算法用 `y`？

高层 API 接收完整业务表，用 `target` 指定标签列名。底层 estimator 分别接收特征矩阵 `X` 和标签向量 `y`。

## 监控 target 可以传字符串 `"0"` 或 `"1"` 吗？

可以。监控接受整数 `0/1`、浮点数 `0.0/1.0`、布尔值，以及字符串 `"0"`、`"1"`、`"true"`、`"false"`、`"True"` 和 `"False"`。
null、NaN 和空字符串表示未观测；其他字符串和 -1 等非法值会报错。MARS 只转换这些明确支持的编码，其他业务标签请在调用前清洗。

## Experimental 模块可以用于生产吗？

Monitoring、Modeling、Pipeline 和 Scoring 是 Experimental。它们可以用于实验和受控生产流程，
但必须固定精确版本，并为依赖的 report 字段、报警结果、评分映射、生成的 SQL、step 的输入输出约定、replay 行为和
artifact 路径增加契约测试。

## `artifact_dir=None` 会写文件吗？

调参时不会。`MarsModelTuner.tune()` 将结果保存在返回对象中；指定 `artifact_dir` 时才会创建独立运行目录并写入调参产物。
