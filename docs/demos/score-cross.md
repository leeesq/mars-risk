---
description: 交叉比较 Home Credit 2018 两项外部来源分的风险，核对格子人数、有效标签与组合条件。
---

# 4 · 两项外部来源分如何共同区分风险？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/score-cross.html) · [有限查询与结论](../assets/cases/case-4.json)

--8<-- "docs/assets/cases/previews.txt:case4"

X 轴使用 `EXT_SOURCE_2`，Y 轴使用 `EXT_SOURCE_3`。这两个字段是公开的外部来源分；本例没有训练模型，也不能把分数视为校准后的违约概率。
两轴切点只在 `0train` 拟合。分数方向结合官方说明和开发样本中的观察结果登记，相关关系不能解释为因果关系。

报告保留小数切点的精度、Missing 箱、Bad Rate、人数、Lift 和组合条件。
`sample_count` 是全样本数，`observed_sample_count` 是有效标签人数；Bad Rate 以有效标签人数为分母。
`4_oot_undue` 的人数和分布完整保留，监督指标标记为未观测。格子的颜色或单格 Lift 不能作为部署资格的证明。

=== "人工分析"

    查看交叉矩阵和格子明细，按有效标签人数解释 Bad Rate。下面的代码调用共享数据准备函数，复用同一宽表和固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:score_cross"
    ```

=== "Agent 消费"

    在新进程中加载快照，先查看报告目录，再查询已保存的统计结果。下方 JSON 包含查询结果；按其中的 `reference` 可以再次查询同一证据。

    ```python
    from mars.reporting import load_report

    report = load_report("score-cross.marsreport")
    print(report.describe())
    page = report.query_page("cells", features="EXT_SOURCE_2", limit=3)
    print(page["reference"], page["data"])
    ```

    ??? example "实际查询结果与分页引用"

        ```json
        --8<-- "docs/assets/cases/query-4.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 4 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
