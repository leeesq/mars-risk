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

这些结构化表适合继续做特征复盘、监控规则定制、内部看板接入，也可以交给 Agent 基于明细表进行摘要、筛选、解释和报告重排。

## Excel/HTML 导出

```python
profile_report.write_excel("mars_profile.xlsx")
eval_report.write_excel("mars_evaluation.xlsx", engine="openpyxl")
eval_report.write_html("mars_evaluation.html")
```

基础安装已经包含 Excel/HTML 报表导出和绘图报告依赖。

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
