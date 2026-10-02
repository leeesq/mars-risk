---
description: 新进程仅加载已有快照，验证目录、身份、分页、预算和不足回答。
---

# 6 · 分析完成后，如何保存并继续查询？

**任务：结束生成进程后，仅拿报告文件继续读取证据。**
本例复用交叉快照及关联 policy，不重新分析原宽表，不保存七份相同报告。

<div class="mars-cases" markdown="1">

## 真实结果预览

[独立加载查询与状态](../assets/cases/case-6.json) · [实际预算上下文](../assets/cases/agent-context.json)

--8<-- "docs/assets/cases/previews.txt:case6"

第一页还有一个有效零：discovery/202601/bad30 的 b3/b1 坏率为 **0.0**，状态 **valid**。
它与 observation/202607/late60 的 unobserved 不同，不能合并为“没有风险”。

快照恢复后仍能发现表、查询、分页和生成有限 JSON。`report_id` 与同批源报告一致，
但没有凭空新增 `query_id` 或跨运行身份恒定保证。

## 保存与消费边界

`.marsreport` 保存 manifest 与 Parquet 证据，恢复返回 `ReportSnapshot`。
目录、schema、关键数值、元数据和持久报告身份可继续核对；它不会自动恢复分析器、原始宽表、
任意方法、模型或可部署 RuleSet。
公共 `describe()`、`get_table()`、`query_page()`、`to_ai_context()` 查询已有表，不调用 LLM。
交叉专用回放只在当前聚合证据支持的范围内成立。

=== "人工阅读"

    从 `describe()` 找真实表名与粒度，再按问题筛选、排序、投影和分页。
    通过 `next_offset` 请求下一页，空结果保留零行；非法表名、字段或预算应显示真实异常。
    不要把查询裁剪前的行数或引用冒充预算 JSON 的完整内容。

    [查看新进程查询结果](../assets/cases/case-6.json) · [查看字符预算上下文](../assets/cases/agent-context.json)

=== "Agent 消费"

    **具体问题：只给这份快照，能继续读交叉格子下一页并引用原报告吗？**
    以下是独立消费进程的公共入口；无原始数据和 API Key。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/score-cross.marsreport")
    print(report.describe())
    scope = {"group": "discovery", "period": "202601", "target": "bad30"}
    columns = ["x_bin", "y_bin", "sample_count", "bad_rate", "status"]
    first = report.query_page("cells", filters=scope, columns=columns, limit=3)
    if first["next_offset"] is not None:
        second = report.query_page("cells", filters=scope, columns=columns, offset=first["next_offset"], limit=3)
        print(second["reference"], second["data"])
    context = report.to_ai_context(
        queries={"cells": {"filters": scope, "columns": ["x_bin", "y_bin", "sample_count", "bad_rate", "status"], "limit": 40}},
        max_chars=9000,
    )
    print(context)
    ```

    ??? example "完整有限 JSON：实际第一页查询（预算／下一页／错误见完整证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-6.json"
        ```

    **示例回答（人工依据确定性证据整理）：**同批 `report_id` 保持，scope 的 42 行先返回 3 行，
    `next_offset=3` 后续已实际查询。9,000 字符预算生成 8,929 字符，cells 请求 40 行但只保留 30 行；
    继续读取应从裁剪后 `continue_at.offset=30` 开始，不用裁剪前 offset 40 伪装完整证据。

    **无法回答的追问：加载后能否换目标、重分箱或对新样本执行规则？**
    快照只有已计算证据，缺原始宽表与拟合状态；需要原分析入口及相应数据重新分析。

## 最短独立消费入口

先运行[案例 4](score-cross.md)或索引的全量命令，生成 `score-cross.marsreport`、
`policy.marsreport` 和 `case-4.json`。案例 1／2／3 的单例命令不会生成这些前置文件。
已有这些共享产物时结束生成进程，直接执行下面的独立消费命令：

```bash
python docs/snippets/task_cases.py --phase consume --output-dir output/task-cases
```

??? example "已测试共享源码：只加载和查询"

    ```python
    --8<-- "docs/snippets/task_cases.py:restore"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[policy 快照](../assets/cases/policy.marsreport) · [交叉快照](../assets/cases/score-cross.marsreport) ·
[有限 JSON](../assets/cases/case-6.json) · [来源与哈希](../assets/cases/manifest.json)

报告文件可能包含原分析的特征名、参数、业务说明与聚合敏感信息；仅加载可信输入，
公开前核对内容。格式版本与源码版本边界按[公共报告指南](../user-guide/reports-and-exports.md#snapshot-format)处理。
普通对话模型还需要能运行 Python 的工具环境；上传文件不等于自动恢复全部分析能力。

[下一步：同次分析的多种交付](report-delivery.md) ·
[外部 Agent 详细参考](../user-guide/external-agents.md) · [案例索引](index.md)

</div>
