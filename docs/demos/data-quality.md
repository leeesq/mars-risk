---
description: 检查 Home Credit 2018 数据的缺失值、标签覆盖和分区差异，再从报告中查询统计证据。
---

# 1 · 这份数据能直接用吗？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/profile.html) · [有限查询与结论](../assets/cases/case-1.json)

--8<-- "docs/assets/cases/previews.txt:case1"

画像覆盖全部候选字段。先检查字段缺失情况和外部来源分的覆盖率，找到需要解释的字段，再比较五个分区的分布。
`DAYS_EMPLOYED` 的异常占位值按这个字段的规则登记，数据中仍保留原值。这条规则不用于其他字段。

July 的差异同时包含原比赛 train/test 的数据来源差异，不能仅凭模拟日期判断生产数据发生了漂移。
PSI 以 `0train` 为参考分布；是否计入缺失值和特殊值，以报告记录的实际参数为准。

=== "人工分析"

    查看画像表和图，按各指标的实际分母解释结果。下面的代码调用共享数据准备函数，复用同一宽表和固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:profile"
    ```

=== "Agent 消费"

    在新进程中加载快照，先查看报告目录，再查询已保存的统计结果。下方 JSON 包含查询结果；按其中的 `reference` 可以再次查询同一证据。

    ```python
    from mars.reporting import load_report

    report = load_report("profile.marsreport")
    print(report.describe())
    page = report.query_page("overview", features="EXT_SOURCE_1", limit=3)
    print(page["reference"], page["data"])
    ```

    ??? example "实际查询结果与分页引用"

        ```json
        --8<-- "docs/assets/cases/query-1.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 1 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
