# FAQ

## MARS 和 toad 有什么区别？

toad 是成熟的传统评分卡工具，强在经典评分卡链路、社区沉淀和易用性。MARS 更强调 Polars-first、宽表性能、结构化 report、特征/模型监控、Excel/HTML 交付和 Modeling Pipeline 串联。

## MARS 和 optbinning 有什么区别？

optbinning 是优秀的最优分箱算法库。MARS 使用 `MarsOptimalBinner` 提供面向宽表风控流程的封装，并把分箱结果接入风险评估、特征筛选、监控和报表链路。

## `profile_risk()` 返回什么？

返回 `MarsRiskProfile(report, binner, targets, metadata)`。其中 `report` 是风险评估报告，`binner` 是本次拟合或复用的分箱器，`targets` 是目标列列表，`metadata` 保存运行上下文。

## 为什么高层 API 用 `target`，底层算法用 `y`？

高层 API 面向完整业务表，目标变量是表中的列名，所以使用 `df, target`。底层算法对象面向特征矩阵和标签向量，所以使用 `X, y`。

## `MarsMonitor` 是完整模型监控平台吗？

不是。`mars.monitoring` 是通用监控指标计算层。它计算 PSI、缺失率、分箱占比、分箱统计量、target 表现覆盖率和默认报警摘要。调度、模型版本、阈值策略、看板和业务处置流程由使用者定义。

## 监控 target 可以传字符串 `"0"` 或 `"1"` 吗？

不可以。监控模块只接受 `0`、`1`、`True`、`False` 和空值。字符串、`-1`、`2`、`"pending"` 等非空异常值会直接抛出 `ValueError`。

## Modeling Pipeline 稳定吗？

Modeling Pipeline 仍在快速迭代中，可能不稳定。后续接口约定、结果对象和调参参数都可能发生较大变动。生产流程建议固定版本，并在升级前检查返回对象和字段名称。

## `MarsModelTuner.tune(artifact_dir=None)` 会写文件吗？

不会。`artifact_dir=None` 时调参历史、模型和元信息只保存在返回对象中。默认会在 `modeling_artifacts/` 下为每次调参创建独立运行目录，写入 `history.csv`、`run_config.json`、`metadata.json`、特征重要性和已保留模型，不会覆盖旧运行。
