---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 4 · 两项外部来源分如何共同区分风险？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/score-cross.html) · [有限查询与结论](../assets/cases/case-4.json)

--8<-- "docs/assets/cases/previews.txt:case4"

X 是 EXT_SOURCE_2，Y 是 EXT_SOURCE_3。它们是公开的外部来源分，没有在本例训练模型，也不能视为校准违约概率。
两轴切点只在 0train 拟合；方向结合官方说明与开发样本观察登记，关联不解释为因果。
保留小数切点精度、Missing 箱、Bad Rate、人数、Lift 和组合条件。
sample_count 是全样本数，observed_sample_count 是有效标签人数；Bad Rate 分母使用后者。
4_oot_undue 的人数与分布完整保留，监督指标为实际未观测状态。颜色或单格 Lift 不等于部署资格。

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:score_cross"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

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

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
