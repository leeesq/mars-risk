from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mars.analysis import MarsDataProfiler, profile_risk
from mars.feature import MarsNativeBinner


def make_dataset(rows: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = np.array(["2024-01", "2024-02", "2024-03", "2024-04"])

    income = rng.normal(5000, 1300, rows).round(0)
    utilization = rng.uniform(0.02, 0.95, rows).round(4)
    age = rng.integers(21, 60, rows)
    segment = rng.choice(["new", "repeat", "vip"], size=rows, p=[0.45, 0.4, 0.15])
    month = rng.choice(months, size=rows)

    raw_score = 0.0012 * (6200 - income) + 2.4 * utilization + 0.018 * (32 - age)
    raw_score += np.where(segment == "vip", -0.35, 0.0)
    raw_score += rng.normal(0, 0.18, rows)
    target = (raw_score > np.quantile(raw_score, 0.55)).astype(int)

    income = income.astype(object)
    income[::17] = -999
    income[::29] = None

    return pl.DataFrame(
        {
            "month": month.tolist(),
            "age": age.tolist(),
            "income": income.tolist(),
            "utilization": utilization.tolist(),
            "segment": segment.tolist(),
            "target": target.tolist(),
        }
    )


def time_call(label: str, fn, repeats: int) -> None:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    avg = statistics.mean(timings)
    best = min(timings)
    worst = max(timings)
    print(f"{label:<24} avg={avg:.4f}s  best={best:.4f}s  worst={worst:.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight MARS benchmarks on synthetic data.")
    parser.add_argument("--rows", type=int, default=20000, help="Number of synthetic rows to generate.")
    parser.add_argument("--repeats", type=int, default=3, help="How many times to repeat each benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = make_dataset(args.rows, args.seed)
    X = df.select(["age", "income", "utilization", "segment"])
    y = df.get_column("target")

    def run_binner() -> None:
        binner = MarsNativeBinner(
            method="quantile",
            n_bins=8,
            cat_features=["segment"],
            special_values=[-999],
        )
        binner.fit_transform(X, y)

    def run_profiler() -> None:
        profiler = MarsDataProfiler(
            df,
            missing_values=[-999],
        )
        profiler.generate_profile(
            profile_by="month",
            config_overrides={
                "enable_sparkline": False,
                "dq_metrics": ["missing", "zeros"],
                "stat_metrics": ["mean", "psi"],
            },
        )

    def run_profile_risk() -> None:
        profile_risk(
            df,
            target="target",
            features=["age", "income", "utilization", "segment"],
            profile_by="month",
            binning_type="native",
            n_bins=8,
            binner_kwargs={"method": "quantile"},
            plot=False,
        )

    print(f"Rows: {args.rows}")
    print(f"Repeats: {args.repeats}")
    print("")
    time_call("MarsNativeBinner", run_binner, args.repeats)
    time_call("MarsDataProfiler", run_profiler, args.repeats)
    time_call("profile_risk", run_profile_risk, args.repeats)


if __name__ == "__main__":
    main()
