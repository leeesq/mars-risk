---
description: 追踪 Home Credit 2018 候选规则的来源和筛选原因，再用独立样本复核固定规则。
---

# 5 · 候选规则如何变成可以审查的证据？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/rules.html) · [有限查询与结论](../assets/cases/case-5.json)

--8<-- "docs/assets/cases/previews.txt:case5"

本例在 `0train` 发现候选，在 `1val` 验证或筛选，再用 `2test` 和 `3oot` 独立复核固定规则。`1val` 参与过选择，不能作为最终的独立效果证明。
从交叉格子提取的种子候选与自动组合生成的候选，分别保存候选数、预算和审计记录。报告按实际原因解释开发阶段和验证阶段的淘汰结果。

规则筛选使用 `MarsRuleMiningSpec.production()`，但本例没有执行真实时间验证；模拟的 `APPLY_DATE` 不能作为时间验证或上线的依据。Rule 仍为 Experimental。
`evaluation.sample_count` 表示已观测人数，`total_count` 和 `observed_count` 分别记录总样本数和标签已观测人数。标签为 null 的样本不当作好样本。

MarsRuleReport 保存统计证据。RuleSet 可以判断新样本是否命中规则，也可以按资格要求导出 SQL 片段。本例不部署策略。

[自动生成报告](../assets/cases/auto-rules.html) · [生成预算与审计](../assets/cases/rule-generation.json) · [固定规则独立评估快照](../assets/cases/rules-independent.marsreport) · [外部 Agent 规则任务](../assets/cases/external-agent-rule-task.txt)

=== "人工分析"

    查看候选审计和独立评估结果，区分筛选样本与独立复核样本。下面的代码调用共享数据准备函数，复用同一宽表和固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:rules"
    ```

=== "Agent 消费"

    在新进程中加载快照，先查看报告目录，再查询已保存的统计结果。下方 JSON 包含查询结果；按其中的 `reference` 可以再次查询同一证据。

    ```python
    from mars.reporting import load_report

    report = load_report("rules.marsreport")
    print(report.describe())
    page = report.query_page("candidates", features="EXT_SOURCE_2", limit=3)
    print(page["reference"], page["data"])
    ```

    ??? example "实际查询结果与分页引用"

        ```json
        --8<-- "docs/assets/cases/query-5.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 5 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
