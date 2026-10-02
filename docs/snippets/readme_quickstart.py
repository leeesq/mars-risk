"""中英文 README 共用的独立最小分析示例。"""

# --8<-- [start:quickstart]
import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame({
    "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
    "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
    "target": [1, 1, 0, 0, 1, 1, 0, 0],
})
report = profile_risk(
    df, target="target", features=["income", "utilization"],
    method="quantile", n_bins=4,
).report
print(report.get_table("summary", sort_by="iv", descending=True, limit=2))
report.write_html("risk_report.html", include_charts=False)
report.save("risk_report.marsreport")
# --8<-- [end:quickstart]
