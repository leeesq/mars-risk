---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 5 · 候选规则如何变成可以审查的证据？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/rules.html) · [有限查询与结论](../assets/cases/case-5.json)

--8<-- "docs/assets/cases/previews.txt:case5"

0train 发现候选，1val 验证或筛选；固定规则再交给 2test 与 3oot 独立复核。1val 参与选择后不作为最终独立效果证明。
种子格子候选和自动组合生成分别保存候选数、预算与审计，开发淘汰和验证淘汰按实际原因展示。
使用探索或随机验证的适当配置，模拟 APPLY_DATE 不构成真实时间验证或上线依据。Rule 仍为 Experimental。
evaluation 的 sample_count 沿用已观测人数，total_count/observed_count 明确总样本与标签覆盖；null 不当作好样本。
MarsRuleReport 保存统计证据；RuleSet 才能对新样本命中或执行受门禁控制的 SQL。这里不部署策略。

[自动生成报告](../assets/cases/auto-rules.html) · [生成预算与审计](../assets/cases/rule-generation.json) · [固定规则独立评估快照](../assets/cases/rules-independent.marsreport) · [外部 Agent 规则任务](../assets/cases/external-agent-rule-task.txt)

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:rules"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

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

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
