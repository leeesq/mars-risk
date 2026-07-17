---
description: 用一条可运行的风险评估和 HTML 报告链路快速开始使用 MARS。
---

# 10 分钟 Quickstart

本页只走一条高频链路：准备带日期和标签的样本，完成风险评估，再导出可检索 HTML。
画像、分箱器细节、特征筛选、建模和监控分别进入对应的用户指南。

## 1. 准备数据

`time_col` 保留原始日期；`month` 是已有分组列。两者同时存在时，`month` 用于面板分组，
`apply_dt` 用于趋势图左上角的真实时间范围。

```python
import polars as pl

df = pl.DataFrame(
    {
        "apply_dt": [
            "2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24",
            "2024-02-03", "2024-02-10", "2024-02-17", "2024-02-24",
            "2024-03-03", "2024-03-10", "2024-03-17", "2024-03-24",
        ],
        "month": [
            "2024-01", "2024-01", "2024-01", "2024-01",
            "2024-02", "2024-02", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-03", "2024-03",
        ],
        "income": [
            3200, 3600, -999, None, 3300, 4200, -999, 5800, 3400, 4300, None, 6100,
        ],
        "utilization": [
            0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58, 0.16, 0.31, 0.56, 0.63,
        ],
        "segment": [
            "new", "repeat", "vip", "vip", "new", "repeat",
            "vip", "vip", "new", "repeat", "vip", "vip",
        ],
        "target": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    }
)
```

## 2. 评估风险并保留分箱规则

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    time_col="apply_dt",
    binning_type="native",
    method="quantile",
    n_bins=4,
    missing_values=[-999],
    special_values=[-999],
    psi_include_missing=False,
    psi_include_special=False,
)

report = risk_profile.report
binner = risk_profile.binner
summary = report.summary_table
```

`report` 保存汇总、分箱明细、趋势表和导出方法；`binner` 可以在后续流程中复用相同分箱规则。

## 3. 查看趋势并导出 HTML

```python
report.plot_risk_trends(max_plots=5)

report.write_html(
    "risk_report.html",
    report_name="March risk review",
    max_plots=500,
    chart_embed_mode="auto",
)
```

`write_html()` 每个 target 最多绘制 500 个特征。图表超过 50 张时，`auto` 会在 HTML 同级
生成 `<报告名>_assets/` 图片目录，并在 Charts 视图中懒加载；小报告仍是单文件。全局搜索可从
Summary 或 Charts 直达特征趋势图。

!!! note "趋势图契约"

    `plot_risk_trends()` 和 HTML 内的图表都需要有效 `time_col`。左上角范围仅使用原始日期的
    最小/最大有效值，显示为 `YYYY-MM-DD`；`group_col` 不能代替日期来源。

## 接下来读什么

- 需要五月建箱、六月评估或显式复用分箱器：读[分箱与风险评估](../user-guide/binning-risk-evaluation.md)。
- 当前期没有成熟标签：读[特征/模型监控](../user-guide/monitoring.md)。
- 需要画像、筛选、建模或 Pipeline：从[首页任务导航](../index.md)选择对应路径。
- 需要精确参数、默认值和异常：查看[API Reference](../reference/index.md)。
