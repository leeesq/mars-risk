---
description: 固定主模型等级，复核辅助分梯度、格子状态及已有聚合规则回放。
---

# 4 · 主模型同等级内，辅助分还能进一步区分风险吗？

**任务：固定 X 风险层，比较 Y 的风险差异，再审查候选区域。**
这里的两个分数由合成数据直接生成；本例没有训练“新模型”，也不声称辅助分整体更优。

<div class="mars-cases" markdown="1">

## 真实结果预览

[打开真实交互报告](../assets/cases/score-cross.html) · [下载格子 Excel](../assets/cases/score-cross.xlsx) ·
[查询格子证据](../assets/cases/case-4.json)

--8<-- "docs/assets/cases/previews.txt:case4"

默认范围是 **discovery / 202601 / bad30 / X=b1**。Y 从 b0 到 b3 的加权坏率为
1.47%、3.92%、11.67%、56.72%；同 X 行基线 19.20%，b3 高出 **37.51 pp**。
该格全量 122 人、有效标签 117 人，加权坏率分母 119.3161；它只是进入独立验证的候选证据。

同一 X 等级内的 Y 梯度提供候选证据；候选是否通过独立验证，继续看[案例 5](rule-evidence.md)。
颜色、对角线或单格高 Lift 都不足以判断新旧模型整体效果。

## 方向、范围与分母

X=`main_score`（主模型，高分低风险），Y=`aux_score`（辅助分，高分高风险）。
四段正常箱只在发现／参考期确定，验证与观察复用固定边界；`risk_rank` 按实际方向记录风险排序，
不能仅凭 `b0`／`b3` 名称猜风险。目标是 `bad30` 或 `late60`，group 是样本分区，period 是实际周期。

`sample_count` 为全样本人数，`observed_sample_count` 才是当前目标有效标签人数。
本例配置 `weight`；坏率 = 有效事件权重 / `observed_weight_sum`，有效标签人数另行保留。
Δ = 当前格坏率 − 同一 X 行基线，比例差乘 100 显示为 pp。
Lift = 当前格坏率 / 同范围整体坏率；整体分母无效或整体坏率为 0 时按实际状态表达。
权重、金额与币种只按本次配置解释，合成金额不是利润或损失。

行／列边际从计数加总重算，并包含另一轴的特殊箱；不能平均屏幕上可见格子的坏率。
0 是有效数值，空格、低样本、未观测、无有效分母与未定义是不同状态。
双向梯度同时检查固定 X 内的 Y 和固定 Y 内的 X，不能只选一条方向制造结论。

=== "人工阅读"

    用自己的开发样本确定分段，再将边界复用于待评估样本；分数方向按实际模型声明：

    ```python
    from mars.analysis import cross_scores, get_score_bin_definitions

    directions = {"main_score": "lower_risk", "aux_score": "higher_risk"}
    reference = cross_scores(train_df, score_x="main_score", score_y="aux_score", targets=["bad30"], score_directions=directions, n_bins=4)
    report = cross_scores(
        df, score_x="main_score", score_y="aux_score", targets=["bad30"],
        score_directions=directions, bin_definitions=get_score_bin_definitions(reference),
        group_col="dataset", time_col="application_date", time_grain="month",
    )
    report.write_html("score-cross.html")
    ```

    打开交互 HTML，切换实际 target／group／period，选中格子后核对全样本数、表现分母、
    绝对坏率、Δ pp、Lift 与状态，再查看双向梯度和基线。
    使用页面已有的选格、规则表达式、应用／恢复、policy 查看和证据复制能力。
    页面是否提供某个按钮，以实际报告为准；文档不添加假的上传、下载或主题按钮。

    网页表达式针对正常箱区域。Python policy 能处理更广的箱选择与特殊箱规则，
    聚合快照回放仍只支持已经保存的维度和固定分箱，不访问原始个体。

    [打开交互报告](../assets/cases/score-cross.html) · [已有 policy 静态证据](../assets/cases/policy.xlsx)

=== "Agent 消费"

    **具体问题：同一主模型 `b1` 内，辅助分哪些格子风险更高，分母够吗？**
    查询同一份 ScoreCrossReport 的 `cells`，不能从截图猜数字。

    ```python
    from mars.reporting import load_report

    report = load_report("output/task-cases/score-cross.marsreport")
    print(report.describe())
    page = report.query_page(
        "cells",
        filters={"target": "bad30", "group": "discovery", "period": "202601", "x_bin": "b1"},
        limit=10,
    )
    print(page["reference"], page["data"])
    ```

    ??? example "完整有限 JSON：实际 b1 格子查询（状态与 policy 见完整证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-4.json"
        ```

    **示例回答（人工依据确定性证据整理）：**`cells` 在 discovery/202601/bad30 的 b1/b3
    返回加权坏率 56.7155%、同 X 行基线 19.2019%、Δ +37.5136 pp、整体基准 Lift 3.5496。
    全量 122 人、有效标签 117 人；实际坏率分母是 `observed_weight_sum=119.3161`。
    低样本或未观测格子需保留状态，不将 null 替换成 0。

    **无法回答的追问：辅助模型整体比主模型更好吗？**交叉格子的局部风险差异不能替代
    同样本、同目标、同指标的整体模型评估；本次也没有真实业务成本、收益或部署证据。

## 最短运行入口

```bash
python docs/snippets/task_cases.py --case 4 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：固定分段、交叉与回放"

    ```python
    --8<-- "docs/snippets/task_cases.py:score_cross"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[交叉快照](../assets/cases/score-cross.marsreport) · [policy 快照](../assets/cases/policy.marsreport) ·
[有限 JSON](../assets/cases/case-4.json) · [来源与哈希](../assets/cases/manifest.json)

报告的离线交互与文档站的亮／暗主题是两套展示；离线 HTML 配色以真实导出为准。
保存后回放读取聚合证据，不能改变原样本、重拟合分段或计算任意新维度。
剪贴板权限和 `file://` 限制可能影响复制；失败应读页面真实提示并使用 JSON 文件。

[下一步：候选规则独立验证](rule-evidence.md) ·
[完整参数指南](../user-guide/correlation-and-score-cross.md#固定分段交叉) ·
[保留的 Score Cross Notebook](correlation_and_score_cross.ipynb) · [案例索引](index.md)

</div>
