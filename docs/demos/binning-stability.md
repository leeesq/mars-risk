---
description: 用固定分箱比较 Home Credit 2018 各分区的风险与稳定性，查看原生趋势图并查询统计证据。
---

# 2 · 哪些特征有区分度，而且足够稳定？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/binning.html) · [有限查询与结论](../assets/cases/case-2.json)

--8<-- "docs/assets/cases/previews.txt:case2"

本例只在 `0train` 拟合分箱，并以它作为 PSI 的参考分布。五个分区复用同一套分箱边界。
下方主图展示 `EXT_SOURCE_2` 的 MARS 原生趋势图。你还可以在报告中查询 `EXT_SOURCE_3`、`AMT_INCOME_TOTAL` 和 `DAYS_BIRTH`。
manifest 分别记录全部候选字段 `fcols` 和页面展示的字段，图表只展示其中一部分结果。

无标签组仍保留各箱占比和适用的 PSI。解释 IV、KS、AUC 和已观测 Bad Rate 时，应先查看报告中的未观测状态。
总人数、已观测人数和未观测人数以 `count`、`observed_count` 等报告字段为准。本例不设权重；KS 的范围为 0～100，Bad Rate 的范围为 0～1，IV 和 PSI 无量纲。

按模拟月份统计的趋势另存为 `binning-time.marsreport`。模拟时间上的差异不能证明真实业务的时间外（OOT）表现。

[![EXT_SOURCE_2 原生分箱图，TARGET 为官方标签，APPLY_DATE 为模拟日期](../assets/cases/binning-native-home-credit.png)](../assets/cases/binning-native-home-credit.svg)

[下载 300 dpi PNG](../assets/cases/binning-native-home-credit.png) · [查看 SVG](../assets/cases/binning-native-home-credit.svg) · [原生图参数与查询来源](../assets/cases/binning-native-evidence.json)

图由 `MarsBinningReport.save_risk_trend_images()` 直接导出，保留原生布局、坐标和标签。`AMT_CREDIT` 是贷款金额，沿用公开数据的原始金额单位，不能解释为利润或损失。

=== "人工分析"

    查看分箱表和趋势图，按各指标的实际分母解释结果。下面的代码调用共享数据准备函数，复用同一宽表和固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:binning"
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
        --8<-- "docs/assets/cases/query-2.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 2 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
