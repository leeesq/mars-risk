import polars as pl

from mars.analysis import MarsDataProfiler

df = pl.DataFrame(
    {
        "month": ["2026-01"] * 4 + ["2026-02"] * 4,
        "income": [3200, 3600, -999, None, 3300, 4200, -999, 5800],
        "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58],
        "segment": ["new", "repeat", "vip", "vip"] * 2,
    }
)

profiler = MarsDataProfiler(missing_values=[-999])
report = profiler.generate_profile(
    df,
    features=["income", "utilization", "segment"],
    group_col="month",
    metrics=["missing", "zeros", "mean", "psi"],
    enable_sparkline=False,
    psi_include_missing=False,
    psi_include_special=False,
)

overview = report.overview_table
profile_data = report.get_profile_data()
dq_trends = profile_data.dq_trends
