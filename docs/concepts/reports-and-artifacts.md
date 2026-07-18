---
description: 解释 MARS report 对象、导出文件与 Modeling artifact 的职责和生命周期。
---

# Report 与 Artifact

MARS 将计算结果分成内存中的结构化对象和可选的持久化产物。两者用途不同。

## Report

Report 保存汇总表、明细表、趋势表和元数据。它适合在 Python 中继续筛选、排序、复盘或组合，
也可以按需导出 Excel、HTML 和图表资产。

导出不是读取结果的前置步骤。对自动化流程，优先消费 report 字段；对人工交付，再调用
`write_excel()` 或 `write_html()`。

## HTML 与 Excel

Excel 适合归档和人工筛选。HTML 适合大量特征的搜索、图表浏览和离线分享。大报告的 HTML 可将
图片写入同级资产目录并按需加载；单文件交付则使用内嵌模式。

## Modeling Artifact

Modeling 的 artifact 保存调参历史、运行配置、模型、重要性和元数据。`artifact_dir=None` 表示完全
不落盘；指定目录时，每次运行创建独立子目录，避免覆盖旧实验。

Artifact 是实验复现和模型交付材料，不等同于 report。Report 回答“本次结果如何”，artifact
回答“本次模型如何产生并如何恢复”。

## 生命周期建议

- Notebook 探索可以只保留返回对象。
- 周期监控应由调用方保存 report 表或导出结果，并记录运行批次。
- 建模实验需要复现时应保留 artifact、依赖版本和输入数据快照标识。
- 任何下游解释或自动化处理都应以结构化字段和明确业务口径为输入。
