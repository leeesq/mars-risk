---
description: 解释 MARS report 对象、导出文件与 Modeling artifact 的职责和生命周期。
---

# Report 与 Artifact

MARS 将计算结果分成内存中的结构化对象和可选的持久化产物。两者用途不同。

## Report

Report 保存汇总表、明细表、趋势表和元数据。它适合在 Python 中继续筛选、排序、复盘或组合，
也可以按需导出 Excel、HTML 和图表资产。

导出不是读取结果的前置步骤。对自动化流程，优先消费公共 `describe/get_table/query_page`
接口；画像、分箱、相关性及模型分交叉报告可通过 `save/load_report` 保存为含 JSON 清单及 Parquet 完整统计表的
单个 `.marsreport` 文件。恢复的 `ReportSnapshot` 不需要原始数据和分析器，可被外部 Agent
直接查询，也可登记到 `MarsAgentSession`。持久报告标识在恢复后保持不变，AI 上下文是受预算
限制的摘要。业务元数据、实际参数及计算状态的完整说明见
[外部 Agent 指南](../user-guide/external-agents.md)。对人工交付，再调用
`write_excel()` 或 `write_html()`。

`to_ai_context()` 的 `evidence.query` 表示预算内实际展示的查询范围，保存加载后可直接精确回放。
数值投影省略特征列时，`identities` 按行保留既有身份，业务元数据只关联当前页及必要关系端点。
字符预算覆盖这些身份、有效查询、说明与完整 JSON；最小必要信封无法容纳时明确报错。
同名 `status` 按真实报告与表解释，通用计算状态、模型分交叉风险状态和字段比较状态分别定义。

## HTML 与 Excel

Excel 适合归档和人工筛选。HTML 适合大量特征的搜索、图表浏览和离线分享。大报告的 HTML 可将
图片写入同级资产目录并按需加载；单文件交付则使用内嵌模式。

## Modeling Artifact

Modeling 的 artifact 保存调参历史、运行配置、模型、重要性和元数据。`artifact_dir=None` 表示完全
不落盘；指定目录时，每次运行创建独立子目录，避免覆盖旧实验。

Artifact 是实验复现和模型交付材料，不等同于 report。Report 回答“本次结果如何”，artifact
回答“本次模型如何产生并如何恢复”。

## Binner JSON Artifact

已拟合分箱规则的正式跨版本格式是 `schema_version=1` JSON artifact，通过
`save_json()` 和 `MarsBinnerBase.load_json()` 写入、恢复。它保存构造配置、分箱规则、
WOE、趋势状态和拟合诊断，不保存训练数据缓存。

Pickle/joblib 可用于同一受控 Python 环境中的便利存储，但不是分箱规则的跨版本承诺。
旧 `{params, state}` dict/JSON 不兼容 0.0.26 artifact，不应为它建立自动降级路径。

## 生命周期建议

- Notebook 探索可以只保留返回对象。
- 周期监控应由调用方保存 report 表或导出结果，并记录运行批次。
- 建模实验需要复现时应保留 artifact、依赖版本和输入数据快照标识。
- 任何下游解释或自动化处理都应以结构化字段和明确业务口径为输入。
