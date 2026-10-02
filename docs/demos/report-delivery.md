---
description: 将同批公共报告以 HTML、静态 Excel、快照、有限 JSON 和 TXT 交给不同使用者。
---

# 7 · 同一次分析，如何交付给不同使用者？

**任务：按读者能力选格式，并让关键值与同一份来源报告对齐。**
本页复用前六例的交叉／规则报告，不再复制一套计算代码。

<div class="mars-cases" markdown="1">

## 真实交付预览

[真实交付核对结果](../assets/cases/case-7.json) · [共享下载包](../assets/cases/cases.zip) ·
[文件大小和 SHA-256](../assets/cases/manifest.json)

--8<-- "docs/assets/cases/previews.txt:case7"

案例 4、6、7 复用同一持久报告；实际 Excel 公共表为 000_cells、001_row_summary、
002_column_summary、003_overall、004_bins，加上三张元数据页。
首行 discovery/202601/bad30 的 b3/b0 在静态 Excel 和快照查询中逐项核对，完整值见下方有限 JSON。

HTML 适合人工探索，静态 Excel 适合阅读归档，快照适合继续工具查询。
它们共享关键值与来源，但交互、计算和可恢复能力并不等价。

## 实际格式与用途

| 格式 | 真实文件 | 用途与边界 |
| --- | --- | --- |
| HTML | [交叉](../assets/cases/score-cross.html)／[规则](../assets/cases/rules.html) | 已有人工展示与实际交互；离线／资源行为按本轮导出验证，文档站主题不等于离线报告主题 |
| Excel | [交叉](../assets/cases/score-cross.xlsx)／[规则](../assets/cases/rules.xlsx) | 快照公共表的当前静态数值；不依赖旧透视缓存，不提供 HTML 同等交互或计算 |
| .marsreport | [交叉](../assets/cases/score-cross.marsreport)／[规则](../assets/cases/rules.marsreport) | 完整公共结果、保存加载、查询和实际支持的专用回放；不恢复任意分析对象或部署权限 |
| 有限 JSON | [查询证据](../assets/cases/case-4.json)／[预算上下文](../assets/cases/agent-context.json) | 真实目录、状态、参数和有限证据；字符预算有省略记录，不能替代完整快照 |
| TXT 提示材料 | [外部 Agent 任务](../assets/cases/external-agent-task.txt) | 指引发现表、查询、引用与承认不足；不是模型实际回答，也不自带 Python 环境 |

旧分箱 Excel 透视模板可能需要原生 Excel 刷新并保存；本轮交付使用快照静态导出，
读取工作表当前值而不是旧缓存。原生 Excel 的外观验收是否完成，以本轮记录为准。

=== "人工阅读"

    阅读者从 HTML 的真实格子、状态和规则详情入手，用静态 Excel 核对同表关键数值。
    Excel 是当前数值交付，不能点击后执行新策略或假定公式会重算原分析。
    公开包的内容、文件大小、来源和哈希见 manifest，不依赖会过期的 Actions artifact 链接。

=== "Agent 消费"

    **具体问题：如何证明人工交付与机器证据属于同一次分析？**
    用 `.marsreport` 的持久身份、公共查询 reference、同批 manifest 和实际工作表值对齐。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/score-cross.marsreport")
    print(report.report_id)
    page = report.query_page("cells", limit=1)
    print(page["reference"], page["data"])
    report.write_excel("output/task-cases/score-cross-static.xlsx")
    ```

    ??? example "完整有限 JSON：实际 Excel 首行对齐查询（格式目录见完整证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-7.json"
        ```

    **示例回答（人工依据确定性证据整理）：**公共表关键值在静态 Excel 与快照查询中对应，
    HTML／JSON 引用同批报告身份，下载文件路径和哈希可查。预算 JSON 只代表实际返回内容，
    不能把目录或表中的省略部分当作完整证据。

    **无法回答的追问：交付文件能否直接上线部署规则？**报告与 TXT 是证据和提示材料，
    不恢复 RuleSet 部署资格，也没有业务批准、模型训练或上线操作授权。

## 导出与运行入口

先执行[案例 4](score-cross.md)或索引全量命令，生成交叉快照、policy 快照、
`case-4.json` 与交叉 Excel；下面的 `--case 7` 复用这些已存在的产物。
案例 1／2／3 的单例命令不满足此前置。导出调用来自已测试共享源码：

```bash
python docs/snippets/task_cases.py --case 7 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：导出和交付"

    ```python
    --8<-- "docs/snippets/task_cases.py:delivery"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[有限 JSON](../assets/cases/case-7.json) · [全部文件来源、大小和哈希](../assets/cases/manifest.json)

共享包是预生成结果与复现材料；不包含用户真实宽表或可部署任意对象。
HTML 查看地址通过站点静态资源打开；仓库的 HTML 源码链接不是浏览器展示。
更新报告时应一起更新预览、JSON 与包，CI 检查不替代实际浏览器／下载验收。

[报告导出详细参考](../user-guide/reports-and-exports.md) · [外部 Agent 指南](../user-guide/external-agents.md) ·
[回到案例索引](index.md) · [本轮验收记录](../project/task-cases-validation.md)

</div>
