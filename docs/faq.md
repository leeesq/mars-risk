# FAQ

## MARS 和 toad 有什么区别？

toad 提供经典评分卡工具链。需要以 Polars 宽表运行画像、分箱评估、监控或 HTML 报告时，可从
MARS 的[任务导航](index.md)选择对应入口；只需要 toad 已覆盖的评分卡能力时，直接使用 toad 即可。

## MARS 和 optbinning 有什么区别？

需要数学规划最优分箱时，使用 `MarsOptimalBinner`；该规则可继续传入风险评估、特征筛选、监控和
报告流程。若不使用数学规划求解器，可选择 `MarsLiteOptBinner` 进行轻量监督式分箱。

## `profile_risk()` 返回什么？

返回 `MarsRiskProfile(report, binner, targets, metadata)`。其中 `report` 是风险评估报告，`binner` 是本次自动拟合出的分箱器，`targets` 是目标列列表，`metadata` 保存运行上下文。如需显式复用已有分箱器，请使用 `MarsBinEvaluator.evaluate(..., binner=...)`。

## 为什么高层 API 用 `target`，底层算法用 `y`？

高层 API 面向完整业务表，目标变量是表中的列名，所以使用 `df, target`。底层算法对象面向特征矩阵和标签向量，所以使用 `X, y`。

## 如何把 `MarsMonitor` 接入周期监控？

每个监控周期调用 `MarsMonitor.monitor()`，读取 PSI、缺失率、分箱占比、表现覆盖率和报警摘要。
调用方需要按自身运行环境安排触发频率、保存 report、配置阈值并处理报警结果。

## 监控 target 可以传字符串 `"0"` 或 `"1"` 吗？

不可以。监控模块只接受 `0`、`1`、`True`、`False` 和空值。字符串、`-1`、`2`、`"pending"` 等非空异常值会直接抛出 `ValueError`。

## Modeling 和 Pipeline 稳定吗？

Modeling 建模和 Pipeline 编排的接口、结果对象和调参参数仍可能变化。生产流程请固定依赖版本；
升级前在测试环境核对所依赖的返回字段和产物路径。

## `MarsModelTuner.tune(artifact_dir=None)` 会写文件吗？

不会。`artifact_dir=None` 时调参历史、模型和元信息只保存在返回对象中。默认会在 `modeling_artifacts/` 下为每次调参创建独立运行目录，写入 `history.csv`、`run_config.json`、`metadata.json`、特征重要性和已保留模型，不会覆盖旧运行。
