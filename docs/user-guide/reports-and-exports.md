---
description: 导出 MARS 的 Excel、可检索 HTML、风险趋势图和结构化 report 数据。
---

# 报告导出与二次加工

<div align="center">
  <img src="../assets/mars-report-flow.svg" alt="Report 对象输出 summary、detail、trend 和 metadata，并支持 Excel、HTML 和 Agent 二次加工" width="920">
</div>

report 保存用于导出和二次处理的结构化数据。先读取表对象完成筛选或复盘；需要分享时再导出
Excel 或 HTML。

## 常用 report 与表

| report | 来源 | 高价值字段 |
| --- | --- | --- |
| `MarsProfileReport` | 数据画像 | `overview_table`、`dq_tables`、`stats_tables` |
| `MarsBinningReport` | 分箱评估 | `summary_table`、`detail_table`、`trend_tables`、`missing_by_day_table` |
| `MarsMonitoringReport` | 监控 | 监控汇总、分箱统计、表现覆盖率和报警元数据 |
| `MarsModelingReport` | 建模评估 | 多样本切片的汇总、明细、趋势和元数据 |

```python
summary = eval_report.summary_table
detail = eval_report.detail_table
trends = eval_report.trend_tables
metadata = eval_report.report_meta
```

## Excel

```python
profile_report.write_excel("mars_profile.xlsx")
eval_report.write_excel("mars_evaluation.xlsx", engine="openpyxl")
```

Excel 适合归档、人工筛选和固定格式交付；需要针对大量特征即时检索时，优先使用 HTML。

需要由 Agent 生成摘要、筛选说明或重排报告时，可传入 `summary`、`detail`、`trends` 和 `metadata`。
MARS 不会调用 Agent，也不会校验 Agent 输出；调用方负责选择模型、补充业务上下文和复核结果。

## 可检索 HTML

```python
eval_report.write_html(
    "mars_evaluation.html",
    report_name="June feature review",
    max_plots=500,
    chart_embed_mode="auto",
)
```

HTML 是单文件入口，内部通过按钮和 URL hash 切换以下视图：Overview、Summary、Missing By Day、
Trend Tables、Grouped Pivot 和 Charts。刷新、前进和后退会保留当前视图；全局搜索会检索 Summary
和 Charts，并可直接打开对应 target 的特征趋势图。

### 图表数量与资产模式

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `chart_embed_mode="auto"` | 不超过 50 张图时内嵌；超过时写入同级 `<html_stem>_assets/` 并懒加载 | 默认推荐 |
| `chart_embed_mode="inline"` | 所有图 Base64 内嵌为单文件 | 少量图、单文件必须可离线转发 |
| `chart_embed_mode="asset"` | 强制生成图片资产目录并用相对路径引用 | 大报告、长期归档或快速打开 |

`max_plots=500` 的上限按每个 target 单独计算。资产模式下，图片初始保存在 `data-src`，只有进入
Charts 视图或滚动到可见区域才会解码，避免浏览器同时加载数百张图片。

### Missing By Day

当评估期传入有效 `time_col` 且生成了按日缺失率表时，HTML 会显示独立的 Missing By Day 视图。
没有有效日期或该表为空时，页面会说明原因；不会把 `group_col` 误当成日期来源。设置
`include_trends=False` 会关闭整体趋势区域及该视图。

!!! warning "风险趋势图需要日期"

    `group_col` 决定面板分组，不能提供趋势图的时间范围。调用风险趋势图或包含 Charts 的 HTML
    前，必须在评估时传入有效 `time_col`。日期时间值会截断显示为 `YYYY-MM-DD`。

## 单独复用风险趋势图

```python
figures = eval_report.build_risk_trend_figures(features=["income"])

fragment = eval_report.render_risk_trends_html(
    features=["income"],
    image_format="svg",
    embed_mode="inline",
)

asset_fragment = eval_report.render_risk_trends_html(
    features=["income"],
    image_format="svg",
    embed_mode="asset",
    output_dir="report/assets",
    relative_to="report",
)
```

这些接口适用于 Notebook、外部 HTML 模板或多个报告组合。`fragment.html` 只包含 HTML 片段；
资产模式会在 `asset_fragment.assets` 中返回已写出的图片路径。

## 评分卡与 SQL

```python
from mars.scoring import build_scorecard

scorecard = build_scorecard(
    binner,
    coefficients={"income": 0.25, "utilization": 0.60},
    intercept=-1.2,
    pdo=20,
    base_score=600,
    base_odds=50,
)

sql = scorecard.generate_sql(
    features=["income", "utilization"],
    table_prefix="t",
    score_name="score",
)
```
