import polars as pl

from mars.analysis import MarsBinEvaluator

baseline_df = pl.DataFrame(
    {
        "income": [2600, 3100, 3500, 3900, 4500, 5200, 6100, 7200],
        "utilization": [0.72, 0.64, 0.55, 0.48, 0.36, 0.28, 0.19, 0.11],
        "target": [1, 1, 1, 0, 1, 0, 0, 0],
    }
)

current_df = pl.DataFrame(
    {
        "apply_dt": [
            "2026-04-03",
            "2026-04-10",
            "2026-04-17",
            "2026-04-24",
            "2026-05-03",
            "2026-05-10",
            "2026-05-17",
            "2026-05-24",
        ],
        "period": ["2026-04"] * 4 + ["2026-05"] * 4,
        "income": [2800, 3400, 4100, 5600, 3000, 3700, 4900, 6800],
        "utilization": [0.69, 0.58, 0.41, 0.24, 0.66, 0.51, 0.33, 0.16],
        "target": [None] * 8,
    }
)

evaluator = MarsBinEvaluator(
    binning_type="native",
    binner_params={"method": "cart", "n_bins": 4},
)
risk_profile = evaluator.evaluate(
    current_df,
    target="target",
    features=["income", "utilization"],
    benchmark_df=baseline_df,
    group_col="period",
    time_col="apply_dt",
)

report = risk_profile.report
fitted_binner = risk_profile.binner
