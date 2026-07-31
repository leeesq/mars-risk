from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _daily_datetimes(start: str, periods: int) -> list[datetime]:
    """生成测试用的日频时间，避免 CI 上 pandas date_range 的 C 扩展段错误。"""
    start_dt = datetime.fromisoformat(start)
    return [start_dt + timedelta(days=idx) for idx in range(periods)]


@pytest.fixture
def sample_credit_df() -> pl.DataFrame:
    data = {
        "month": [
            "2024-01", "2024-01", "2024-01", "2024-01", "2024-01", "2024-01", "2024-01", "2024-01",
            "2024-02", "2024-02", "2024-02", "2024-02", "2024-02", "2024-02", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-03", "2024-03", "2024-03", "2024-03", "2024-03", "2024-03",
        ],
        "age": [
            24, 29, 31, 35, 40, 27, 45, 33,
            26, 30, 38, 41, 36, 28, 47, 34,
            25, 32, 39, 43, 37, 29, 48, 35,
        ],
        "income": [
            3200, 3600, 4100, -999, None, 5200, 6100, 4500,
            3300, 3700, 4200, -999, 5800, None, 6400, 4700,
            3400, 3900, 4300, -999, 6000, None, 6800, 4900,
        ],
        "utilization": [
            0.12, 0.18, 0.26, 0.52, 0.61, 0.33, 0.49, 0.44,
            0.14, 0.21, 0.29, 0.54, 0.58, 0.36, 0.47, 0.42,
            0.16, 0.23, 0.31, 0.56, 0.63, 0.39, 0.51, 0.46,
        ],
        "segment": [
            "new", "new", "repeat", "repeat", "vip", "new", "vip", "repeat",
            "new", "repeat", "repeat", "vip", "vip", "new", "vip", "repeat",
            "new", "repeat", "repeat", "vip", "vip", "new", "vip", "repeat",
        ],
        "target": [
            0, 0, 0, 1, 1, 0, 1, 0,
            0, 0, 1, 1, 1, 0, 1, 0,
            0, 0, 1, 1, 1, 0, 1, 0,
        ],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_credit_pd(sample_credit_df: pl.DataFrame) -> pd.DataFrame:
    return sample_credit_df.to_pandas()


@pytest.fixture
def feature_start_aware_df() -> pl.DataFrame:
    rows = []

    for day in _daily_datetimes("2024-01-01", periods=3):
        for _ in range(99):
            rows.append({"biz_dt": day, "segment": "PRE", "x": None, "target": 0})
        rows.append({"biz_dt": day, "segment": "PRE", "x": 0.0, "target": 0})

    for day in _daily_datetimes("2024-02-15", periods=3):
        for _ in range(10):
            rows.append({"biz_dt": day, "segment": "ACTIVE_A", "x": 0.0, "target": 0})
        for _ in range(10):
            rows.append({"biz_dt": day, "segment": "ACTIVE_A", "x": 1.0, "target": 1})

    for day in _daily_datetimes("2024-03-15", periods=3):
        for _ in range(10):
            rows.append({"biz_dt": day, "segment": "ACTIVE_B", "x": 0.0, "target": 0})
        for _ in range(10):
            rows.append({"biz_dt": day, "segment": "ACTIVE_B", "x": 1.0, "target": 1})

    return pl.DataFrame(rows)


@pytest.fixture
def sample_modeling_pd() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    split_specs = [
        ("train", 60, "2024-01-01", 0.0),
        ("val", 30, "2024-02-01", 0.2),
        ("oot1", 30, "2024-03-01", 0.35),
    ]

    for split_name, size, start_date, drift in split_specs:
        dates = _daily_datetimes(start_date, periods=size)
        for idx in range(size):
            x1 = rng.normal(loc=drift, scale=1.0)
            x2 = rng.normal(loc=0.0, scale=1.1)
            x3 = rng.normal(loc=-drift, scale=0.8)
            raw_score = 1.6 * x1 - 1.1 * x2 + 0.7 * x3 + rng.normal(scale=0.4)
            target = int(raw_score > 0.0)
            benchmark_score = 1 / (1 + np.exp(-(0.9 * x1 - 0.6 * x2 + 0.3 * x3)))
            rows.append(
                {
                    "biz_dt": dates[idx],
                    "dataset_flag": split_name,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "segment": "A" if x1 > 0.6 else ("B" if x1 > -0.4 else "C"),
                    "target": target,
                    "benchmark_score": benchmark_score,
                }
            )

    df = pd.DataFrame(rows)
    for split_name in ["train", "val", "oot1"]:
        sub = df[df["dataset_flag"] == split_name]
        assert sub["target"].nunique() == 2
    return df


@pytest.fixture
def sample_modeling_df(sample_modeling_pd: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(sample_modeling_pd)
