---
description: Home Credit 2018 全量公开案例，保存统计与有限查询共同支持人工阅读和外部 Agent。
---

# 7 · 同一次分析，如何交付给不同使用者？

<div class="mars-cases" markdown="1">

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

## 本批实际结果

[打开报告](../assets/cases/binning.html) · [有限查询与结论](../assets/cases/case-7.json)

--8<-- "docs/assets/cases/previews.txt:case7"

HTML 供人工探索，Excel 交付当前静态公共表，.marsreport 保留目录、参数与查询身份，有限 JSON 提供有范围的证据。
同批工作簿关键值逐项对快照核验。Excel 不执行 HTML 交互，也不依赖旧透视缓存。
cases.zip 只包含明确允许的统计产物与源码，个人写作 Skill、原始 CSV、清洗宽表、样本 ID 明细均不进入公共材料。
无原数据可加载现成快照并重放查询；完整复算须自行取得三个官方文件并传路径。

| 读者用途 | 文件 |
| --- | --- |
| 原生分箱与质量 | [分箱 HTML](../assets/cases/binning.html) / [画像 HTML](../assets/cases/profile.html) |
| 当前静态数值 | [分箱 Excel](../assets/cases/binning.xlsx) / [规则 Excel](../assets/cases/rules.xlsx) |
| 新进程继续查询 | [分箱快照](../assets/cases/binning.marsreport) / [规则快照](../assets/cases/rules.marsreport) |
| 路径、哈希与字典 | [manifest](../assets/cases/manifest.json) / [业务字段字典](../assets/cases/feature-metadata.json) |
| 完整允许产物与源码 | [cases.zip](../assets/cases/cases.zip) / [无 CSV 外部任务](../assets/cases/external-agent-task.txt) |

=== "人工分析"

    从报告查看真实表与图，按实际分母判断。完整可运行代码使用共享准备入口，复用同一宽表与固定参考样本。

    ```python
    --8<-- "docs/snippets/task_cases.py:delivery"
    ```

=== "Agent 消费"

    新进程按目录读取已保存统计；实际有限 JSON 中的引用可重放。

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

首次生成全部关联证据用 --case all。文件缺失会明确失败，正式入口不回退合成数据。
[数据下载与输入准备](../user-guide/home-credit-data.md) · [完整共享源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) · [来源和哈希](../assets/cases/manifest.json) · [案例索引](index.md)

</div>
