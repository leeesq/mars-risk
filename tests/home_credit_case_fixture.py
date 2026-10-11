"""小型 Home Credit 形状案例 fixture；合成数据只用于测试，不进入公开产物。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl


def write_case_fixture(directory: Path) -> Path:
    """写出三文件小型 fixture，使用真实角色名测试完整共享流程。

    Parameters
    ----------
    directory : Path
        自动创建的 fixture 目录。

    Returns
    -------
    Path
        含 train/test/description 三文件的目录。

    Examples
    --------
    >>> write_case_fixture(Path("output/test-fixture"))  # doctest: +SKIP
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(22137)
    n = 1800
    x, y = rng.uniform(size=(2, n))
    probability = np.where((x < 0.25) & (y < 0.25), 0.7, 0.02)
    social = rng.integers(0, 10, n)
    frame = pl.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1), "TARGET": (rng.uniform(size=n) < probability).astype(int),
        "AMT_CREDIT": rng.uniform(1000, 10000, n), "EXT_SOURCE_1": rng.uniform(size=n),
        "EXT_SOURCE_2": x, "EXT_SOURCE_3": y,
        "AMT_INCOME_TOTAL": rng.lognormal(10, 0.5, n), "DAYS_BIRTH": -rng.integers(7000, 23000, n),
        "DAYS_EMPLOYED": [-365243 if i % 10 else 365243 for i in range(n)],
        "AMT_ANNUITY": rng.uniform(500, 1000, n), "AMT_GOODS_PRICE": rng.uniform(1000, 5000, n),
        "REGION_RATING_CLIENT": rng.integers(1, 4, n), "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, n),
        "OBS_30_CNT_SOCIAL_CIRCLE": social, "OBS_60_CNT_SOCIAL_CIRCLE": social,
    })
    frame.head(1500).write_csv(directory / "application_train.csv")
    frame.tail(300).drop("TARGET").write_csv(directory / "application_test.csv")
    with (directory / "HomeCredit_columns_description.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "Table", "Row", "Description", "Special"])
        writer.writerows([i, "application_{train|test}.csv", name, f"Fixture definition for {name}", ""] for i, name in enumerate(frame.columns))
    return directory
