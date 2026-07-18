import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame(
    {
        "apply_dt": [
            "2026-01-03",
            "2026-01-10",
            "2026-01-17",
            "2026-01-24",
            "2026-02-03",
            "2026-02-10",
            "2026-02-17",
            "2026-02-24",
            "2026-03-03",
            "2026-03-10",
            "2026-03-17",
            "2026-03-24",
        ],
        "month": ["2026-01"] * 4 + ["2026-02"] * 4 + ["2026-03"] * 4,
        "income": [
            3200,
            3600,
            -999,
            None,
            3300,
            4200,
            -999,
            5800,
            3400,
            4300,
            None,
            6100,
        ],
        "utilization": [
            0.12,
            0.18,
            0.52,
            0.61,
            0.14,
            0.29,
            0.54,
            0.58,
            0.16,
            0.31,
            0.56,
            0.63,
        ],
        "segment": [
            "new",
            "repeat",
            "vip",
            "vip",
            "new",
            "repeat",
            "vip",
            "vip",
            "new",
            "repeat",
            "vip",
            "vip",
        ],
        "target": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    }
)

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
summary = report.summary_table
binner = risk_profile.binner
