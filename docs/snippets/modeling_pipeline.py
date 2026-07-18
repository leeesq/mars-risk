from datetime import date, timedelta

import numpy as np
import polars as pl

from mars.pipeline import MarsModelingPipeline, MarsModelingStep

rng = np.random.default_rng(1206)
row_count = 180
income = rng.normal(5200, 1300, row_count)
utilization = rng.uniform(0.05, 0.95, row_count)
logit = -0.00045 * (income - 5200) + 3.2 * (utilization - 0.5)
probability = 1.0 / (1.0 + np.exp(-logit))

development_df = pl.DataFrame(
    {
        "apply_dt": [date(2025, 1, 1) + timedelta(days=index) for index in range(row_count)],
        "income": income,
        "utilization": utilization,
        "target": rng.binomial(1, probability),
    }
)

pipeline = MarsModelingPipeline(
    target="target",
    features=["income", "utilization"],
    steps=[
        MarsModelingStep(
            name="modeling",
            model_type="lgb",
            time_col="apply_dt",
            split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
            tune_params={
                "n_trials": 1,
                "startup_trials": 1,
                "num_boost_round": 20,
                "early_stopping_rounds": 5,
                "artifact_dir": None,
            },
        )
    ],
)

pipeline_result = pipeline.fit(development_df)
scored_df = pipeline.predict(development_df, pred_col="model_score")
