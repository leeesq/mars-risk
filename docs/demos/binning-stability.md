---
description: 参考分箱固定后，在独立分区和多目标上复核风险、IV、KS 与 PSI。
---

# 2 · 哪些特征有区分度，而且足够稳定？

**任务：同时查看区分度和分布稳定性，避免只按一个开发表现指标排名。**
本例只在发现／参考集拟合分箱，验证集与观察集复用原分箱，不重新拟合后声称稳定。

<div class="mars-cases" markdown="1">

## 真实结果预览

[打开分箱 HTML](../assets/cases/binning.html) · [当前值 Excel](../assets/cases/binning.xlsx) ·
[查询证据](../assets/cases/case-2.json)

--8<-- "docs/assets/cases/previews.txt:case2"

来源为同批快照的 `trend.iv`、`trend.ks`、`trend.psi`。
辅助分的 bad30 开发 KS 59.6030、验证 KS 57.2376，区分度仍有证据；
收入和渠道的观察 PSI 分别 1.0859、1.6354，表现为强分布变化，不能视为风险提升。
Total 汇总与两个目标的完整值在[真实查询证据](../assets/cases/case-2.json)中分别保存。

[![MARS 原生主分分箱风险趋势：固定箱件数与金额风险、分区占比及真实日期范围](../assets/cases/binning-native-main-score.png)](../assets/cases/binning.html)

[下载 300 dpi PNG](../assets/cases/binning-native-main-score.png) ·
[查看清晰 SVG](../assets/cases/binning-native-main-score.svg) ·
[原生图查询来源](../assets/cases/binning-native-evidence.json)

这张图直接由 `MarsBinningReport.save_risk_trend_images()` 导出，保留原有布局。
上方查看固定分箱的件数与金额风险，下方查看各箱在三个样本分区的风险和占比。
日期来自 `application_date`；本图按分区比较，不把三个分区说成九个月风险序列。
每月只有一个模拟申请日期，共九个实际日期，不将其展示成逐日连续业务轨迹。

区分度和稳定性需要分别判断。两个目标、不同分区和特殊箱的统计不应合并为单一“最优特征”结论。
本页的具体对比读取实际 `summary` 与 `detail`，来源可用同一快照重放。

## 数据与指标口径

`bad30` 与 `late60` 是分别生成的合成目标；有效标签人数分别统计。
分布统计使用全量样本；本例配置分析权重，坏率使用每个 target 的有效表现权重分母，
原始有效标签人数见[关联的数据质量案例](data-quality.md)。缺失标签不算好样本。
本次 `detail.count`、`observed_count`、`bad` 均为权重和，单位是 `weight_sum`，不能当作整数人数。
坏率是 0–1 比例，KS 是 0–100，IV、PSI 无量纲。
PSI 的参考样本、权重与缺失／特殊箱范围以本次报告 `parameters` 为准。

missing、special、unseen 先按实际箱定义定位，再确认是否进入风险和 PSI 分母。
验证期新类别不在验证集重新学习箱边界；箱的处理规则见同批 `detail` 与元数据。
观察期 `late60` 无有效标签，仍可有适用的分布 PSI；该目标 IV／KS 等标签指标不能因此补算或填 0。

实际收入缺失进入 `bin_index=-1 / Missing`，`-999` 进入 `-3 / Special_-999`；
它们保留箱风险，但本次 `psi_include_missing=false`、`psi_include_special=false`，PSI 明细为 null。
参考期未见的 `branch` 进入渠道的 `-2 / Other`，该箱有真实 PSI contribution，未伪装成缺失。
案例 2 与案例 4 分别复用所属公共入口的参考规则，箱号与表达式以各自报告定义为准。

=== "人工阅读"

    `df` 是待评估样本，`reference_df` 是独立参考样本；日期、分区、标签和金额列按自己的数据替换：

    ```python
    from mars.analysis import profile_risk

    report = profile_risk(
        df, target=["bad30", "late60"], features=["main_score", "aux_score", "income"],
        benchmark_df=reference_df, method="quantile", n_bins=4,
        group_col="dataset", time_col="application_date", time_grain="month",
        weights_col="weight", amount_col="amount", special_values=[-999.0],
    ).report
    report.write_html("binning.html", chart_embed_mode="inline")
    report.save_risk_trend_images("charts", features="main_score", target="bad30", image_format="png", dpi=300)
    ```

    分区指标读取 `trend.iv`、`trend.ks`、`trend.psi`；`summary` 的 IV／KS 是 Total 汇总。
    再下钻 `detail` 核对箱风险；本次 `count`／`observed_count` 是权重和，不能当作整数人数。
    单独比较 PSI；开发区分度高与验证分布稳定是两项证据。
    经验阈值用于提示检查，不能决定业务价值、利润、通过率或因果方向。

    [查看箱风险与指标](../assets/cases/binning.html) · [下载静态统计](../assets/cases/binning.xlsx)

=== "Agent 消费"

    **具体问题：哪些字段的开发表现与验证稳定性需要一起复核？**
    先发现实际目录，再读取主目标的分区趋势；`summary` 的 Total 不能代替开发表现。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/binning.marsreport")
    print(report.describe())
    page = report.query_page("trend.ks", features=["main_score", "aux_score", "income", "channel"], limit=4)
    print(page["reference"], page["data"])
    ```

    ??? example "完整有限 JSON：实际主目标 KS 趋势查询（IV、PSI 与箱风险见完整证据）"

        ```json
        --8<-- "docs/assets/cases/query-2.json"
        ```

    **示例回答（人工依据确定性证据整理）：**上方分区指标支持先保留辅助分的区分度证据，
    再重点复核收入与渠道的分布变化。读取 `trend.ks` 比较开发与验证，读取 `trend.psi` 检查漂移，
    不把 Total 的 IV/KS 改写成开发指标，也不把不同目标或重新拟合结果混在一起。
    本次箱范围、状态与分母见上述实际查询引用。

    **无法回答的追问：高 IV 是否意味着真实业务收益？**合成标签和统计关联没有成本、
    收益或因果证据；需要真实业务目标和独立验证，不能从 IV 推导利润。

## 最短运行入口

```bash
python docs/snippets/task_cases.py --case 2 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：固定参考分箱与评估"

    ```python
    --8<-- "docs/snippets/task_cases.py:binning"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[分箱快照](../assets/cases/binning.marsreport) · [有限 JSON](../assets/cases/case-2.json) ·
[来源与哈希](../assets/cases/manifest.json)

本例 Excel 是快照逐公共表导出的当前值，不依赖旧透视模板缓存。
HTML、Excel 与 JSON 从同一报告导出，但静态 Excel 不提供 HTML 同等交互。
箱风险不等于可部署规则，稳定不等于有效；无效或未计算指标按状态解释。

[下一步：筛选与相关冗余](selection-correlation.md) ·
[分箱参数指南](../user-guide/binning-risk-evaluation.md) · [案例索引](index.md)

</div>
