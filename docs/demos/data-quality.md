---
description: 用真实画像和明确标注的前置检查判断数据是否支持下一步分析。
---

# 1 · 这份数据能直接用吗？

**任务：先确认样本、标签和字段能否支持分析，再讨论清洗或效果。**
本例读取共享合成数据的质量画像，并把 schema 差异／未见类别标为案例侧前置检查。

<div class="mars-cases" markdown="1">

## 真实结果预览

[打开真实画像 HTML](../assets/cases/profile.html) · [完整机器证据](../assets/cases/case-1.json)

--8<-- "docs/assets/cases/previews.txt:case1"

`sparse` 缺失率 **98%**；概率前置检查发现 **177** 行大于 1；观察期新类别为 `branch`。
schema 侧检查明确模拟 `income_copy` 缺列、`main_score` 从 Float64 变 String。
以上人数与检查来自[同批结果摘要](../assets/cases/summary.json)，质量字段来自 `overview`。

这份数据能支持有标签分区的分析；观察期 `late60` 全未表现，只能支持该目标的分布检查。
质量问题需要解释和后续处理，本例不执行自动清洗，也不自动批准建模。

## 数据与口径

共享数据共 18,000 行，发现／参考、验证、观察各 6,000 个独立 ID。
两个模拟目标 `bad30`、`late60` 分别统计有效标签；全量样本数不能当作每个目标的坏率分母。
标签缺失表示未表现，不补成 0。

画像的缺失／特殊值、统计与比较结果来自 `MarsProfileReport`。
原始字段集合、dtype 差异和参考期未出现的类别，由共享脚本的简单前置检查给出。
后者是案例材料，不是新的 MARS 核心报告或自动决策 API。
PSI 的参考、比较分区以及 missing/special 是否纳入，以报告实际参数为准。

=== "人工阅读"

    用你的 `df` 与参考样本 `reference_df` 替换共享合成数据，按实际字段名调整 features：

    ```python
    from mars.analysis import profile_stats

    report = profile_stats(
        df, features=["income", "channel"], categorical_features=["channel"],
        metrics=["missing", "mean", "psi"], benchmark_df=reference_df,
        group_col="dataset", special_values=[-999.0],
    )
    print(report.get_table("overview"))
    report.write_html("quality.html")
    ```

    先读总样本与有效标签，再定位缺失、特殊值和分布变化。
    schema 不同要先核对字段含义与类型，未见类别需要确认参考分箱如何接收它。
    缺失率上升与标签缺失是不同问题，不能直接解释为模型效果下降。

    [查看画像](../assets/cases/profile.html) · [下载当前值静态 Excel](../assets/cases/profile.xlsx)

=== "Agent 消费"

    **具体问题：先检查哪些字段，证据是什么？**
    本地确定性消费先调用 `describe()` 查实际目录，再用 `query_page()` 按表、字段和预算查询。
    表名、状态、有效查询参数与 `report_id` 全部保存在下列真实 JSON 中。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/profile.marsreport")
    print(report.describe())
    page = report.query_page("overview", limit=8)
    print(page["reference"], page["data"])
    ```

    ??? example "完整有限 JSON：实际 overview 查询（目录与前置检查见完整证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-1.json"
        ```

    **示例回答（人工依据确定性证据整理）：**先检查 `sparse`：`overview` 显示缺失率 0.98；
    再核对观察期 `late60` 的 0 个有效标签和前置检查发现的 `branch`。
    `aux_score` 的 mean 为公共非有限编码 inf，应结合输入非法分检查，不能改成 0。
    查询引用可重放到同一份快照。
    schema／未见类别检查来源是共享脚本的前置检查，不能引用成公共画像表。

    **无法回答的追问：观察期 late60 模型效果是否下降？**该目标缺有效表现标签，
    分布证据只能说明分布变化；需要等标签成熟或另做同目标、同样本口径的效果评估。

## 最短运行入口

先按[共享安装](index.md#run)与[数据字典](index.md#data-dictionary)准备，在仓库根目录运行：

```bash
python docs/snippets/task_cases.py --case 1 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：画像阶段"

    ```python
    --8<-- "docs/snippets/task_cases.py:profile"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[画像快照](../assets/cases/profile.marsreport) · [有限 JSON](../assets/cases/case-1.json) ·
[来源、参数和哈希](../assets/cases/manifest.json)

共享包提供预生成结果与复现材料；原始宽表由固定 seed 重建，不把用户业务记录写入公开文件。
快照保留可查询结果，不恢复原始宽表、画像器或自动清洗流程。
阈值判断是分析提示，不代表字段可以删除，也不证明业务因果。

[下一步：固定分箱比较区分度与稳定性](binning-stability.md) ·
[画像参数指南](../user-guide/data-profiling.md) · [案例索引](index.md)

</div>
