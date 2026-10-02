---
description: 从真实筛选决策复核保留／删除；分别检查 raw 与目标感知 WOE 相关性。
---

# 3 · 为什么保留这个特征，删除另一个？

**任务：把“删掉冗余特征”变成可复核的候选、步骤、阈值和取舍证据。**
本例的冗余字段来自同一套共享数据；筛选决策来自 selector 实际输出，不根据相关矩阵重编历史。

<div class="mars-cases" markdown="1">

## 真实结果预览

[筛选决策 JSON](../assets/cases/selection.json) · [筛选 Excel](../assets/cases/selection.xlsx) ·
[相关性 HTML](../assets/cases/correlation.html) · [Agent 查询证据](../assets/cases/case-3.json)

--8<-- "docs/assets/cases/previews.txt:case3"

Stats 使用完整门禁筛选强辅助分、主分及冗余／质量／模拟漂移字段；实际保留以表中结果为准。
Linear 在单独的 raw 相关池保留 `main_score`、`income`。
但 raw 的 `main_score/score_inverse` 是 **−1**，`income/income_copy` 是 **+1**；
不能把 WOE 的正相关改写成原始值的关系。来源为[实际决策](../assets/cases/selection.json)与同批相关表。

相关正负号说明原始关系方向，筛选的绝对相关阈值只说明强度。
保留／删除要结合真实决策链；仅看到一对高相关，不能推断哪个字段被保留或为什么。

## 数据与口径

筛选使用发现／参考样本及明确的主目标，验证样本不参与重新挑选。
候选、每步 active 集合、保留／删除、实际阈值与已有原因读取真实决策输出。
若决策表没有业务原因字段，就只呈现已有证据，不能补成完整业务解释。

本次 Stats 使用缺失阈值 0.9、粗筛及精筛 IV 0.01／Lift 1.2、PSI 0.25、RC 0.5、
WOE 相关阈值 0.8，保留粗筛、精筛及稳定性阶段。
`aux_drift` 是辅助分加上逐月固定平移的演示字段：即使有区分度，也要经过稳定性筛选。
非有限模型分在本次筛选配置为缺失，交叉中仍独立标为 invalid；两种任务的参数分别保存。
原始相关性只演示冗余关系，不能用其保留的 `income` 替代 Stats 的统计筛选结果。

## 相关性：先确认表示空间 { #correlation }

`raw-correlation.marsreport` 是原始数值表示；`woe-correlation.marsreport` 是目标感知 WOE 表示。
两份矩阵对应不同表示，不能视为同一矩阵或用 raw 的符号解释 WOE 的筛选轨迹。
绝对值阈值与显示正负号分别说明；无效相关保持实际状态，不填 0 或对角 1。

raw 采用候选池与目标完整行删除；WOE 采用 bad30 有效标签并把 WOE null 填 0。
具体行数和阈值在上方真实预览中分别保留，不把两个空间当成相同统计。

=== "人工阅读"

    用自己的开发样本 `train_df`，替换候选字段；日期用于稳定性检查，不从验证集重新选字段：

    ```python
    from mars.feature import MarsStatsSelector

    selector = MarsStatsSelector(
        psi_thr=0.25, rc_thr=0.5, corr_thr=0.8, n_jobs=1,
        missing_values=[float("inf"), float("-inf")], special_values=[-999.0],
    )
    selector.fit(
        train_df, target="bad30", features=["main_score", "aux_score", "income"],
        time_col="application_date", time_grain="month",
    )
    print(selector.selected_features_)
    print(selector.get_report())
    ```

    先看真实候选与步骤，再查看相关冗余对和实际留下的字段。
    如负相关字段因 `abs(correlation)` 超阈值被处理，保留负号，说明阈值只取绝对强度。
    需要业务取舍时，还要增加字段成本、可用性与稳定性证据。

    [查看实际决策](../assets/cases/selection.xlsx) · [检查相关性](../assets/cases/correlation.html)

=== "Agent 消费"

    **具体问题：这个字段在哪一步被删除，相关证据是什么？**
    从真实决策文件与同批 CorrelationReport 读取已有记录；不用系数大小替代决策。

    ```python
    from mars.reporting import load_report

    raw = load_report("output/task-cases/raw-correlation.marsreport")
    woe = load_report("output/task-cases/woe-correlation.marsreport")
    print(raw.describe())
    print(woe.describe())
    page = raw.query_page("correlation_decisions", limit=8)
    print(page["reference"], page["data"])
    ```

    ??? example "完整有限 JSON：raw 决策查询（WOE 与完整目录见证据文件）"

        ```json
        --8<-- "docs/assets/cases/query-3.json"
        ```

    **示例回答（人工依据确定性证据整理）：**raw 的 `correlation_decisions` 记录
    `score_inverse` 被 `main_score` 触发删除：有符号相关 −1，绝对值 1，阈值 `>=0.95`。
    主目标强度相等时实际 tie_rule 保留左侧候选；原始值负相关与绝对强度门槛并不矛盾。
    未记录的业务原因保持未知，不能补写成“因为这个字段成本更低”。

    **无法回答的追问：保留字段上线后一定更稳吗？**筛选只覆盖本次候选和样本范围，
    缺少未来表现、上线成本与真实业务稳定性证据，需要独立观察和业务约束。

## 最短运行入口

```bash
python docs/snippets/task_cases.py --case 3 --output-dir output/task-cases --rows 18000 --seed 20261001
```

??? example "已测试共享源码：筛选和相关性"

    ```python
    --8<-- "docs/snippets/task_cases.py:selection"
    ```

## 下载、复现与边界

[完整源码](../snippets/task_cases.py) · [共享包](../assets/cases/cases.zip) ·
[raw 快照](../assets/cases/raw-correlation.marsreport) · [WOE 快照](../assets/cases/woe-correlation.marsreport) ·
[有限 JSON](../assets/cases/case-3.json) · [来源与哈希](../assets/cases/manifest.json)

相关矩阵、审计和 JSON 读取同批计算结果；没有第二套 Agent 专用算法。
CorrelationReport 快照保存证据，不能恢复 selector 重新处理任意新数据。
不同表示空间、缺失策略、目标和候选范围必须分开解释。

[下一步：同等级内辅助分交叉](score-cross.md) ·
[筛选参数指南](../user-guide/feature-selection.md) ·
[相关性指南](../user-guide/correlation-and-score-cross.md#相关性报告) · [案例索引](index.md)

</div>
