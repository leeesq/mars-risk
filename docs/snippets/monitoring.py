import polars as pl

from mars.monitoring import MarsMonitor, generate_monitoring_alert

baseline_df = pl.DataFrame(
    {
        "model_score": [0.08, 0.16, 0.27, 0.39, 0.53, 0.66, 0.79, 0.91],
        "income": [7200, 6500, 5900, 5100, 4400, 3800, 3200, 2700],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    }
)

current_df = pl.DataFrame(
    {
        "period": ["2026-04"] * 4 + ["2026-05"] * 4,
        "model_score": [0.12, 0.24, 0.46, 0.72, 0.18, 0.35, 0.61, 0.86],
        "income": [6900, 6100, 4700, 3300, 6600, 5400, 4100, 2900],
        "target": [None] * 8,
    }
)

report = MarsMonitor(
    binner_params={"method": "cart", "n_bins": 4},
    psi_include_missing=False,
).monitor(
    current_df,
    features=["model_score", "income"],
    target="target",
    benchmark_df=baseline_df,
    group_col="period",
    trend_column_order="asc",
)

alert_text = generate_monitoring_alert(
    report,
    score_key="model_score",
    model_features=["income"],
)
