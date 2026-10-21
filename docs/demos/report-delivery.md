---
description: 为 Home Credit 2018 分析结果选择 HTML、Excel、快照或查询摘要，并核对交付文件中的关键值。
---

# 7 · 同一次分析，如何交付给不同使用者？

<div class="mars-cases" markdown="1">

本例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/binning.html) · [有限查询与结论](../assets/cases/case-7.json)

--8<-- "docs/assets/cases/previews.txt:case7"

按使用方式选择格式：用 HTML 交互查看，用 Excel 读取当前静态公共表，用 `.marsreport` 保存目录、参数和报告查询身份。只需部分证据时，可以导出限定范围的 JSON。
同批工作簿的关键值已经逐项与快照核对。Excel 不执行 HTML 中的交互，也不依赖旧透视缓存。

默认特征查询依次返回身份列 `feature → display_name → dtype → data_source`；显式指定 `columns` 时，结果严格按请求选择列。
案例 JSON、报告目录和 AI 上下文均由同批查询生成。manifest 的 `source_commit` 记录生成时已知的 Git 基线，代码指纹记录实际用于生成的工作树代码。

`cases.zip` 只包含允许公开的统计产物和源码，不包含个人写作 Skill、原始 CSV、清洗宽表或样本 ID 明细。
没有原数据时，仍可加载现成快照并按引用重放查询。需要完整复算时，先自行取得三个官方文件，再向脚本传入文件路径。

| 读者用途 | 文件 |
| --- | --- |
| 原生分箱与质量 | [分箱 HTML](../assets/cases/binning.html) / [画像 HTML](../assets/cases/profile.html) |
| 当前静态数值 | [分箱 Excel](../assets/cases/binning.xlsx) / [规则 Excel](../assets/cases/rules.xlsx) |
| 新进程继续查询 | [分箱快照](../assets/cases/binning.marsreport) / [规则快照](../assets/cases/rules.marsreport) |
| 路径、哈希与字典 | [manifest](../assets/cases/manifest.json) / [业务字段字典](../assets/cases/feature-metadata.json) |
| 完整允许产物与源码 | [cases.zip](../assets/cases/cases.zip) / [无 CSV 外部任务](../assets/cases/external-agent-task.txt) |

=== "人工分析"

    下面的代码读取同批 Excel 和快照，核对关键值，并生成外部 Agent 任务说明。

    ```python
    --8<-- "docs/snippets/task_cases.py:delivery"
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
        --8<-- "docs/assets/cases/query-7.json"
        ```

## 运行与下载

```bash
python docs/snippets/task_cases.py --case 7 --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

首次运行时，用 `--case all` 生成全部关联证据。缺少输入文件时，脚本会报错，不会自动改用合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
