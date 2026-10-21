---
description: 解释 MARS report 对象、导出文件与 Modeling artifact 的职责和生命周期。
---

# Report 与 Artifact

MARS 将计算结果分成内存中的结构化对象和可选的持久化产物。两者用途不同。

## Report

Report 保存汇总表、明细表、趋势表和元数据。你可以在 Python 中继续筛选、排序或组合分析结果，
也可以按需导出 Excel、HTML 和图表文件。

读取结果无需先导出文件。自动化流程优先调用公共 `describe()`、`get_table()` 和 `query_page()` 接口。
画像、分箱、相关性和模型分交叉报告可以通过 `save()` 保存为单个 `.marsreport` 文件，
其中包含 JSON 清单和 Parquet 完整统计表。

调用 `load_report()` 后得到 `ReportSnapshot`。外部 Agent 无需原始数据和分析器，就能查询快照；
也可以将快照登记到 `MarsAgentSession`。加载后，持久报告标识保持不变。
AI 上下文只提供预算范围内的摘要；业务元数据、实际参数和计算状态见
[外部 Agent 指南](../user-guide/external-agents.md)。需要向人交付文件时，再调用 `write_excel()` 或 `write_html()`。

`to_ai_context()` 的 `evidence.query` 记录实际展示的行、列和查询条件，保存并加载后仍可直接回放。
只选择数值列、未输出特征列时，`identities` 会按行保留特征身份。
业务元数据仅包含当前页涉及的实体，以及解释关系所需的端点。
字符预算包含身份信息、实际查询、说明及 JSON 的其他内容；必要信息超过预算时，接口会报错。

解释 `status` 时，先确认报告类型和表名。通用计算状态、模型分交叉的风险状态和字段比较状态各有定义。

## HTML 与 Excel

Excel 适合归档和人工筛选。HTML 适合大量特征的搜索、图表浏览和离线分享。大报告的 HTML 可将
图片写入同级资产目录并按需加载；单文件交付则使用内嵌模式。

## Modeling Artifact

Modeling 的 artifact 保存调参历史、运行配置、模型、重要性和元数据。`artifact_dir=None` 表示完全
不落盘；指定目录时，每次运行创建独立子目录，避免覆盖旧实验。

Artifact 用于复现实验和交付模型。Report 回答“本次结果如何”，artifact 回答“本次模型如何产生和恢复”。

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
