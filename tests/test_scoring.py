from pathlib import Path

import pandas as pd
import polars as pl

from mars.feature import MarsNativeBinner
from mars.scoring import MarsScorecard, build_scorecard


def test_build_scorecard_generates_points_and_sql(sample_credit_df):
    features = ["income", "utilization"]
    X = sample_credit_df.select(features)
    y = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        features=features,
        method="quantile",
        n_bins=3,
        special_values=[-999],
    )
    binner.fit(X, y)

    scorecard = build_scorecard(
        binner=binner,
        coefficients={"income": -0.35, "utilization": 0.82},
        intercept=-1.1,
        pdo=50,
        base_score=600,
        base_odds=20,
    )

    assert isinstance(scorecard, MarsScorecard)
    points_table = scorecard.points_table
    if isinstance(points_table, pd.DataFrame):
        features_found = set(points_table["feature"])
    else:
        features_found = set(points_table["feature"].to_list())
    assert features_found == set(features)
    sql = scorecard.generate_sql(score_name="credit_score")
    assert "income_points" in sql
    assert "utilization_points" in sql
    assert "AS credit_score" in sql


def test_scorecard_to_integer_rebalances_base_points(sample_credit_df):
    features = ["income", "utilization"]
    X = sample_credit_df.select(features)
    y = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        features=features,
        method="quantile",
        n_bins=3,
        special_values=[-999],
    )
    binner.fit(X, y)

    scorecard = build_scorecard(
        binner=binner,
        coefficients={"income": -0.35, "utilization": 0.82},
        intercept=-1.1,
        pdo=50,
        base_score=600,
        base_odds=20,
    )
    int_scorecard = scorecard.to_integer()

    original_points = scorecard.points_table["points"].to_list()
    integer_points = int_scorecard.points_table["points"].to_list()
    original_total = round(scorecard.base_points + sum(original_points), 0)
    integer_total = int_scorecard.base_points + sum(integer_points)

    assert isinstance(int_scorecard.points_table, pl.DataFrame)
    assert float(int_scorecard.base_points).is_integer()
    assert all(float(point).is_integer() for point in integer_points)
    assert integer_total == original_total
    assert {"points_original", "points_round_error"}.issubset(int_scorecard.points_table.columns)
    sql = int_scorecard.generate_sql(score_name="credit_score")
    assert f"{int_scorecard.base_points}.000000" not in sql


def test_scorecard_to_integer_preserves_pandas_points_table(sample_credit_pd):
    features = ["income"]
    X = sample_credit_pd[features]
    y = sample_credit_pd["target"]

    binner = MarsNativeBinner(
        features=features,
        method="quantile",
        n_bins=3,
        special_values=[-999],
    ).set_output("pandas")
    binner.fit(X, y)

    scorecard = build_scorecard(
        binner=binner,
        coefficients={"income": -0.35},
        intercept=-1.1,
        pdo=50,
        base_score=600,
        base_odds=20,
    )
    int_scorecard = scorecard.to_integer(rebalance=False)

    assert isinstance(scorecard.points_table, pd.DataFrame)
    assert isinstance(int_scorecard.points_table, pd.DataFrame)
    assert int_scorecard.base_points == round(scorecard.base_points, 0)


def test_scorecard_can_write_csv_and_excel(sample_credit_df):
    features = ["income"]
    X = sample_credit_df.select(features)
    y = sample_credit_df.get_column("target")

    binner = MarsNativeBinner(
        features=features,
        method="quantile",
        n_bins=3,
        special_values=[-999],
    )
    binner.fit(X, y)

    scorecard = build_scorecard(
        binner=binner,
        coefficients={"income": -0.35},
        intercept=-1.1,
        pdo=50,
        base_score=600,
        base_odds=20,
    )

    artifacts_dir = Path(__file__).resolve().parent / "_artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    csv_path = artifacts_dir / "scorecard.csv"
    xlsx_path = artifacts_dir / "scorecard.xlsx"

    try:
        scorecard.write_csv(str(csv_path))
        scorecard.write_excel(str(xlsx_path))
        assert csv_path.exists()
        assert xlsx_path.exists()
    finally:
        for path in [csv_path, xlsx_path]:
            if path.exists():
                path.unlink()
        if artifacts_dir.exists() and not any(artifacts_dir.iterdir()):
            artifacts_dir.rmdir()
