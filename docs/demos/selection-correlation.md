---
description: 查看 Home Credit 2018 特征保留与删除的原因，分别核对原始值和 WOE 相关性证据。
---

# 3 · 为什么保留这个特征，删除另一个？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/correlation.html) · [有限查询与结论](../assets/cases/case-3.json)

--8<-- "docs/assets/cases/previews.txt:case3"

本例只用 `0train` 筛选特征和拟合相关性。`1val`、`2test` 和 `3oot` 都不参与筛选或调整阈值，使用范围写入报告上下文。
Stats 保存 WOE 值的 Pearson 相关性，Linear 保存原始数值的 Spearman 相关性。两种结果分别读取，不能用原始值相关系数的正负解释 WOE 筛选决策。

`get_report()` 返回记录筛选决策的 DataFrame，`get_correlation_report()` 返回对应的相关性报告。
下方展示实际发生的关系和决策。如果没有特征因相关性被淘汰，结果会保留“未触发”的结论，不额外制造冗余字段或删除记录。
报告分别登记候选字段 `fcols`、模块支持的候选子集，以及实际参与相关性计算的字段；未计算的关系不补填数值。

## 相关性证据 { #correlation }

[原始值快照](../assets/cases/raw-correlation.marsreport) · [WOE 快照](../assets/cases/woe-correlation.marsreport) · [实际筛选决策](../assets/cases/selection.json) · [相关性 HTML](../assets/cases/correlation.html)

先查 `pairs`，确认每对特征、计算方法和相关系数，再查看 `correlation_decisions` 和局部矩阵。
用 `describe()` 核对缺失处理、有效样本和阈值。原始值与 WOE 值采用不同表示，统计结果不能混用。

=== "人工分析"

    查看筛选决策表和相关性矩阵，分别解释原始值与 WOE 结果。下面的代码调用共享数据准备函数，复用同一宽表和固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:selection"
    ```

=== "Agent 消费"

    在新进程中加载快照，先查看报告目录，再查询已保存的统计结果。下方 JSON 包含查询结果；按其中的 `reference` 可以再次查询同一证据。

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

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
