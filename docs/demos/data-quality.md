---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 1 · 这份数据能直接用吗？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/profile.html) · [有限查询与结论](../assets/cases/case-1.json)

--8<-- "docs/assets/cases/previews.txt:case1"

画像覆盖完整候选字段。先从缺失与外部分覆盖率定位需要解释的字段，再比较五分区的分布。
DAYS_EMPLOYED 的异常占位值只按该字段的实际规则登记，原值保留，不把它套到其他字段。
July 的差异同时包含原比赛 train/test 来源差异，不能仅凭模拟日期称为生产漂移。
PSI 使用 0train 为参考，缺失与特殊值是否纳入由实际参数记录。

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:profile"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

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

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
