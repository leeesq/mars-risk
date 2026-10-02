"""静态 Excel 交付与比较表 Notebook 展示的最短完整示例。"""

from pathlib import Path

import polars as pl
from openpyxl import load_workbook

from mars.analysis import profile_risk, profile_stats
from mars.reporting import load_report, snapshot_report

data = pl.DataFrame({
    "income": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "bad": [0, 0, None, 0, 1, 1, None, 1],
})
report = profile_risk(data, target="bad", n_bins=2).report

# 静态工作簿直接写出当前统计值，不需要 Excel 刷新透视缓存。
snapshot = snapshot_report(report)
snapshot.write_excel("risk_static.xlsx")
workbook = load_workbook("risk_static.xlsx", data_only=True)
assert workbook["000_summary"]["A1"].value == "feature"
workbook.close()

# 只有 marsreport 文件也能导出；可将此步骤放到另一 Python 进程。
archive = Path("risk.marsreport")
report.save(archive)
restored = load_report(archive)
restored.write_excel("risk_restored_static.xlsx")

current = pl.DataFrame({"category": ["甲", "新类别", None], "number": [1, 2, 3]})
benchmark = pl.DataFrame({"category": ["甲", "乙"], "number": ["one", "two"]})
comparison = profile_stats(
    current, benchmark_df=benchmark, metrics=["schema", "unseen"],
    categorical_features=["number"],
)

# Notebook 中可直接显示 Styler；这里实际渲染，确保文本状态保持原样。
schema = comparison.show_trend("schema")
unseen = comparison.show_trend("unseen", sort_by="feature", sort_ascending=True)
assert "incompatible_change" in schema.to_html()
assert "incompatible_dtype" in unseen.to_html()
comparison.write_html("comparison.html")
