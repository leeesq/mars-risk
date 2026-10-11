---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 3 · 为什么保留这个特征，删除另一个？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/correlation.html) · [有限查询与结论](../assets/cases/case-3.json)

--8<-- "docs/assets/cases/previews.txt:case3"

开发筛选及相关性拟合只使用 0train。1val 的使用范围写入上下文；2test、3oot 不参与调整阈值。
Stats 的 WOE Pearson 与 Linear 的原始数值 Spearman 分开保存，不用原始值符号解释 WOE 决策。
get_report() 返回真实决策 DataFrame，get_correlation_report() 返回对应相关性报告。
下方展示实际关系与决策；没有发生相关性淘汰时，保留未触发结论，不人为造冗余字段或删项。
候选 fcols、模块支持的候选子集和真正参与相关计算的字段分别登记，未计算关系不补数。

## 相关性证据 { #correlation }

[原始值快照](../assets/cases/raw-correlation.marsreport) · [WOE 快照](../assets/cases/woe-correlation.marsreport) · [实际筛选决策](../assets/cases/selection.json) · [相关性 HTML](../assets/cases/correlation.html)

先查 pairs 的真实端点、方法与相关值，再下钻 correlation_decisions 和局部矩阵。缺失处理、有效样本与阈值读取 describe() 中的参数，不能合并两种表示的统计。

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:selection"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

    ```python
    from mars.reporting import load_report

    report = load_report("raw-correlation.marsreport")
    print(report.describe())
    page = report.query_page("pairs", features="AMT_INCOME_TOTAL", limit=3)
    print(page["reference"], page["data"])
    ```

    ??? example "实际查询结果与分页引用"

        ```json
        --8<-- "docs/assets/cases/query-3.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 3 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
