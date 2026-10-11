"""中英文 README 共用的 Home Credit 最小分析示例。"""

# --8<-- [start:quickstart]
from home_credit_data import prepare_home_credit_data

from mars.analysis import profile_risk

inputs = prepare_home_credit_data()
report = profile_risk(
    inputs.df, target="TARGET", features=["EXT_SOURCE_2", "EXT_SOURCE_3"],
    benchmark_df=inputs.benchmark_df, group_col="dataset_flag", time_col="APPLY_DATE",
    amount_col="AMT_CREDIT", method="quantile", n_bins=4,
    feature_metadata=inputs.feature_metadata, business_context=inputs.business_context,
).report
report.save_risk_trend_images(
    "risk_charts", features="EXT_SOURCE_2", target="TARGET", image_format="png", dpi=300,
)
report.write_html("risk_report.html", chart_embed_mode="inline")
report.save("risk_report.marsreport")
print(report.query_page("summary", features="EXT_SOURCE_2", limit=1)["data"])
# --8<-- [end:quickstart]
