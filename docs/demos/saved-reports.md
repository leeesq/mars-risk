---
description: 在新进程加载 Home Credit 2018 报告快照，验证分页、字符预算和可重放的证据引用。
---

# 6 · 分析完成后，如何保存并继续查询？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/binning.html) · [有限查询与结论](../assets/cases/case-6.json)

--8<-- "docs/assets/cases/previews.txt:case6"

在新进程中只读取 `binning.marsreport`。先用 `describe()` 查看表目录、单位、计算状态和业务元数据，再按问题分页查询。
每页记录 `total_rows`、`returned_rows`、`omitted_rows`、`next_offset` 和 `reference`。`to_ai_context()` 的 `max_chars` 按字符计数，不按 token 计数。

比较 `1val` 与 `4_oot_undue` 时，先核对标签覆盖率。无标签组的监督指标处于未观测状态，不能把 null 解释为零风险。
本例直接调用公共 API，没有调用 LLM。外部 Agent 可以用同样的接口继续取证，查询无需内置会话或模型 SDK。

加载快照后得到 ReportSnapshot，其中保存已计算的统计结果和元数据。需要新增分析维度、查看原始记录或重新拟合时，必须读取原数据重新计算。

[实际字符预算 JSON](../assets/cases/agent-context.json)

=== "人工分析"

    下面的代码加载已有报告并验证查询结果，不重新准备宽表或拟合分箱。

    ```python
    --8<-- "docs/snippets/task_cases.py:restore"
    ```

=== "Agent 消费"

    在新进程中加载快照，先查看报告目录，再查询已保存的统计结果。下方 JSON 包含查询结果；按其中的 `reference` 可以再次查询同一证据。

    ```python
    from mars.reporting import load_report

    report = load_report("binning.marsreport")
    print(report.describe())
    page = report.query_page("summary", features="EXT_SOURCE_2", limit=3)
    print(page["reference"], page["data"])
    ```

    ??? example "实际查询结果与分页引用"

        ```json
        --8<-- "docs/assets/cases/query-6.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 6 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
