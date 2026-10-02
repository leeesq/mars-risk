---
description: 18,000 行合成案例中，连接发现格子、候选审计和独立验证的规则证据。
---

# 5 · 从候选规则走到可以审查的证据

**任务：把发现期的高风险区域变成候选，并核对哪些通过独立验证、哪些未通过。**
本例复用共享的 18,000 行合成数据、规则对象、候选审计和公共报告。
Rule 为 **Experimental**；候选入选不代表生产策略批准、部署资格或真实收益。

<div class="mars-cases" markdown="1">

## 真实结果预览

[打开实际规则报告](../assets/cases/rules.html) · [完整候选与验证 Excel](../assets/cases/rules.xlsx) ·
[机器可查询证据](../assets/cases/case-5.json)

--8<-- "docs/assets/cases/previews.txt:case5"

格子候选的区域名称、保留状态与独立验证指标在上表对应；完整稳定 `rule_id` 与表达式保留在机器证据。
正常箱规则按实际分段排除 missing、special 与 ±inf，逐个 sample_id 核对成员相同，
不会把交叉的 invalid 分偷偷放进最高正常箱。人数与事件率按规则报告的有效标签口径解释。

下方的自动组合生成器另外生成最多 24 个候选，并使用同样的 production 门禁及独立验证。
它先显式排除非法／缺失模型分，剔除人数与生成来源一起记录，不能与格子种子报告混为同一样本范围。
[查看自动挖掘报告](../assets/cases/auto-rules.html) ·
[下载自动挖掘 Excel](../assets/cases/auto-rules.xlsx) · [生成器证据](../assets/cases/rule-generation.json)

发现门槛与独立验证门槛分开执行。开发淘汰不是验证失败；信息不足也不是有效规则。
候选来源关联发现格子，手工压力候选明确标记没有格子来源，不虚构起源。

## 数据、审计与验证

发现／参考集为 2026 年 1–3 月 6,000 个 ID，验证为 4–6 月另外 6,000 个 ID；两者不共享样本行。
正常箱分段固定后选候选，验证不重新挑选。观察分区不反复选规则，缺标签时不声称规则效果。
候选的 expression、rule_id、生成来源、轮次、状态、筛选阶段和真实理由来自审计表。

`evaluation` 粒度是 dataset / rule_id / target / slice / group；group 可为 hit、miss、total。
规则的 `sample_count` 是当前 target 的有效表现人数，与交叉的全样本同名字段不同。
coverage 的分母是同 dataset、target、slice 的总体有效表现人数；
事件率和覆盖率为比例，Lift 的基准是同范围总体事件率。多目标、时间切片和客户／金额不能简单相加。

=== "人工阅读"

    已完成模型分域检查的 `train_df` 与 `validation_df` 是独立样本。换成自己的数据即可自动生成候选，
    不需要先写出想保留的规则：

    ```python
    from mars.rule import MarsCombinationRuleGenerator, MarsRuleMiningSpec, mine_rules

    result = mine_rules(
        train_df, target="bad30", validation_df=validation_df,
        features=["main_score", "aux_score"],
        generators=[MarsCombinationRuleGenerator(n_bins=4, max_candidates=24, random_state=42)],
        time_col="application_date", time_grain="month",
        spec=MarsRuleMiningSpec.production(max_candidates=24, top_k=5),
    )
    result.to_report().write_html("automatic-rules.html")
    ```

    [已测试的完整自动生成源码](../snippets/task_cases.py)包括输入域检查、快照和生成来源。
    先看发现摘要，再看 `candidates` 的筛选阶段与原因，最后看最终规则在独立验证的表现。
    规则、规范条件、发现格子和验证行的关联用于复核，不能只交付自然语言总结。
    HTML 候选预览有真实限额；完整审计在 Excel、快照与公共分页查询中。

    [规则 HTML](../assets/cases/rules.html) · [完整静态审计](../assets/cases/rules.xlsx)

=== "Agent 消费"

    **具体问题：这条规则为什么保留，开发和验证分别有什么证据？**
    查询已存在的候选与验证表，不重新生成规则，不使用 LLM Key。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/rules.marsreport")
    print(report.describe())
    summary = report.query_page("summary", limit=1)
    audit = report.query_page("candidates", limit=10)
    validation = report.query_page(
        "evaluation",
        filters={"dataset": "validation", "target": "bad30", "group": "hit"},
        limit=10,
    )
    print(audit["reference"], validation["reference"])
    ```

    ??? example "完整有限 JSON：实际挖掘 summary 查询（完整审计与验证见证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-5.json"
        ```

    **示例回答（人工依据确定性证据整理）：**`summary` 显示 production 的 5 个候选里 2 个入选；
    审计中 2 个在 `candidate_filter` 淘汰，1 个在 `validation_filter` 淘汰。
    每条入选规则的 validation/bad30/__overall__/hit 有效人数和事件数按上表核对，
    引用同一 rule_id 与查询范围；自动生成报告单独加载 `auto-rules.marsreport`。
    发现 Lift 不等于验证结果；未保留候选按真实审计解释。

    **无法回答的追问：returning 客群或未执行高级分析里的规则表现如何？**
    本次报告缺该统计维度或相关表，需要明确补充计算；不能从整体表现推断客群，
    也不能把未执行的 interaction／bootstrap 当成空结果或 0。

## 最短运行入口

```bash
python docs/snippets/task_cases.py --case 5 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：规则发现与验证"

    ```python
    --8<-- "docs/snippets/task_cases.py:rules"
    ```

??? example "已测试共享源码：真实组合生成器"

    ```python
    --8<-- "docs/snippets/task_cases.py:automatic_rules"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[规则快照](../assets/cases/rules.marsreport) · [有限 JSON](../assets/cases/case-5.json) ·
[外部 Agent 规则 TXT 任务](../assets/cases/external-agent-rule-task.txt) ·
[同任务 Markdown](../assets/cases/external-agent-task.md) · [来源与哈希](../assets/cases/manifest.json)

默认流程是本地确定性工具消费。TXT 教外部 Agent 如何发现表、检查状态、引用证据和承认不足，
不是已执行的 LLM 结果。实际模型试验须单独记录，不能把脚本成功冒充 Agent 自主结论。
加载 ReportSnapshot 不重建可部署 RuleSet，不能对新数据 transform 或跳过既有 SQL 导出门禁。
仅当本次真实资格支持时才单独提供 RuleSet 定义，部署仍由调用者负责。

[下一步：保存后继续查询](saved-reports.md) ·
[规则报告与外部 Agent 详细参考](../user-guide/rule-reports-and-agents.md) ·
[规则参数指南](../user-guide/rule-mining.md) · [案例索引](index.md)

</div>
