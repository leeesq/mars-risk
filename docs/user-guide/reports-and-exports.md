# 报表导出与二次加工

<div align="center">
  <img src="../assets/mars-report-flow.svg" alt="Report 对象输出 summary、detail、trend 和 metadata，并支持 Excel/HTML、看板和 Agent 二次加工" width="920">
</div>

MARS 的 report 对象同时承担两类职责：一是导出 Excel/HTML 报表，二是保存多粒度结构化数据，方便继续加工。

## 常见 report

| report | 来源 |
| --- | --- |
| `MarsProfileReport` | 数据画像 |
| `MarsBinningReport` | 分箱评估 |
| `MarsModelingReport` | 建模评估 |
| `MarsMonitoringReport` | 特征/模型监控 |

## 结构化数据

不同 report 的字段略有差异，常见字段包括：

- `overview_table`：画像总览表，主要出现在 `MarsProfileReport`。
- `summary_table`：汇总表。
- `detail_table` / `detail_tables`：明细表。
- `trend_tables`：趋势宽表。
- `dq_tables` / `stats_tables`：画像的数据质量和统计趋势表。
- `metadata` / `report_meta`：运行上下文。

```python
summary = eval_report.summary_table
detail = eval_report.detail_table
trends = eval_report.trend_tables
```

风险趋势图的绘图契约要求评估时传入 `time_col`。`group_col` 仍然优先控制面板分组，但图中左上角时间范围只使用 `time_col` 的有效最小值和最大值；没有 `group_col` 时才使用 `time_grain` 生成时间分组。

HTML 风险报告默认按每个 target 最多绘制 500 个特征，超过 50 张图时自动使用同级图片资产和懒加载。报告页面可通过导航按钮切换，且全局特征搜索可以直接打开对应趋势图。

这些结构化表适合继续做特征复盘、监控规则定制、内部看板接入，也可以交给 Agent 基于明细表进行摘要、筛选、解释和报告重排。

## Excel/HTML 导出

```python
profile_report.write_excel("mars_profile.xlsx")
eval_report.write_excel("mars_evaluation.xlsx", engine="openpyxl")
eval_report.write_html("mars_evaluation.html")
```

基础安装已经包含 Excel/HTML 报表导出和绘图报告依赖。

## 风险趋势图组件

`MarsBinningReport` 支持把风险趋势图单独作为组件取出，供 Notebook、外部模型报告或多文件 HTML 报告复用：

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

`fragment.html` 是 HTML 片段，不包含完整的 `html` 或 `body` 标签；asset 模式会在 `asset_fragment.assets` 中返回写出的图片路径。

## 评分卡与 SQL

评分卡链路支持从逻辑回归模型和 WOE 分箱结果生成分数映射，并导出 SQL 规则。

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
