---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 6 · 分析完成后，如何保存并继续查询？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/binning.html) · [有限查询与结论](../assets/cases/case-6.json)

--8<-- "docs/assets/cases/previews.txt:case6"

新进程只读取 binning.marsreport。从 describe() 找到目录、单位、计算状态和业务元数据，再按问题请求有限页。
分页结果记录 total_rows、returned_rows、omitted_rows、next_offset 和 reference。to_ai_context 的 max_chars 是字符预算，不是 token。
比较 1val 与 4_oot_undue 时先核对标签覆盖。无标签指标是未观测，不能把 null 翻译为零风险。
这些查询是确定性公共 API 消费，没有调用 LLM。外部 Agent 可按同样接口继续取证，不依赖内置会话或模型 SDK。
快照只保存已计算统计；新增维度、原始记录或重新拟合需要原数据。恢复对象为 ReportSnapshot。

[实际字符预算 JSON](../assets/cases/agent-context.json)

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:restore"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

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

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
