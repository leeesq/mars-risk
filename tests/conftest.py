import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
