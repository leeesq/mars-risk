"""README 与首页共享的最小分析、人工交付及外部 Agent 查询链路。"""

import polars as pl

from mars.analysis import profile_risk
from mars.reporting import load_report

df = pl.DataFrame(
    {
        "date": ["2026-01-01"] * 4 + ["2026-02-01"] * 4,
        "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
        "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
        "target": [1, 1, 0, 0, 1, 1, 0, 0],
    }
).with_columns(pl.col("date").str.to_date())
# --8<-- [start:analysis]
report = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    time_col="date",
    method="quantile",
    n_bins=4,
).report
top_features = report.get_table("summary", sort_by="iv", descending=True, limit=10)
report.show_summary(limit=10)
report.write_html("risk_report.html", chart_embed_mode="inline")
report.save("risk_report.marsreport")
# --8<-- [end:analysis]

# --8<-- [start:agent]
restored = load_report("risk_report.marsreport")
description = restored.describe()
page = restored.query_page("summary", limit=10)
context_json = restored.to_ai_context(tables=["summary"], max_chars=16000)
# --8<-- [end:agent]
