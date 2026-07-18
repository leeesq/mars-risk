import polars as pl

from mars.feature import MarsNativeBinner
from mars.scoring import build_scorecard

X = pl.DataFrame(
    {
        "income": [2600, 3100, 3500, 3900, 4500, 5200, 6100, 7200],
        "utilization": [0.72, 0.64, 0.55, 0.48, 0.36, 0.28, 0.19, 0.11],
    }
)
y = pl.Series("target", [1, 1, 1, 0, 1, 0, 0, 0])

binner = MarsNativeBinner(method="quantile", n_bins=4).fit(
    X,
    y,
    features=["income", "utilization"],
)
scorecard = build_scorecard(
    binner,
    coefficients={"income": -0.35, "utilization": 0.70},
    intercept=-1.2,
    pdo=20,
    base_score=600,
    base_odds=50,
)

points_table = scorecard.points_table
sql = scorecard.generate_sql(
    features=["income", "utilization"],
    table_prefix="applications",
    score_name="credit_score",
)
