"""报告查询与 AI 上下文示例，不调用模型，不读写原始数据文件。"""

import json

import polars as pl

from mars.analysis import MarsBinEvaluator, profile_stats

data = pl.DataFrame(
    {
        "income": [2000.0, 3000.0, 4000.0, None, 5000.0, 6000.0, 7000.0, 8000.0],
        "target": [1, 1, 0, None, 1, 0, 0, 0],
        "month": ["2026-01"] * 4 + ["2026-02"] * 4,
    }
)
result = MarsBinEvaluator(binner_params={"n_bins": 3}).evaluate(
    data,
    target="target",
    features=["income"],
    group_col="month",
    batch_size=1,
    feature_data_source={"application": ["income"]},
)
report = result.report
description = report.describe()
top_features = report.get_table("summary", sort_by="iv", descending=True, limit=10)
bins = report.get_table(
    "detail", features="income", filters={"bin_index": {"op": "ge", "value": 0}}
)
context_json = report.to_ai_context(features="income", tables=["summary"], max_chars=16000)
assert json.loads(context_json)["evidence"][0]["reference"] == "summary"
for evidence in json.loads(context_json)["evidence"]:
    replayed = report.get_table(evidence["reference"], **evidence["query"])
    assert replayed.to_dicts() == evidence["rows"]
feature_data = report.get_feature("income")
summary_view = report.show_summary(sources="application", columns=["feature", "iv", "ks"], limit=10)

profile = profile_stats(data, features=["income"], metrics=["missing", "mean"], group_col="month")
overview = profile.get_table("overview", columns=["feature", "missing_rate", "mean"], limit=10)
profile_context = profile.to_ai_context(tables=["dq.missing"], features="income")
numeric_context = json.loads(profile.to_ai_context(columns=["mean"], limit=1))
assert numeric_context["evidence"][0]["identities"] == [{"feature": "income"}]
assert set(numeric_context["description"]["feature_metadata"]) == {"income"}
